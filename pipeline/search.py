"""Hybrid search over a built graph:
``ontosmith search``.

The serving side of the release. The lab proved this stack
as its production retrieval path (``pipeline.query_stack``:
the linker, dense + hybrid chunk retrievers, the graph
traversal arms, one code path behind the pinned-QA readouts);
what was absent is a user-facing entry point and a store
surface a stranger can run. This module is that surface:

- a **serving store** read contract — the reads the search
  layer needs (chunk retrieval, neighborhood, entity lookup,
  attestation records) in the shapes ``pipeline.graph_store``
  established over Neo4j;
- :class:`FileServingStore` — the sample path. It serves a
  built graph straight from the registry's artifact tree
  (a batch directory, or a records file + corpus directory —
  the shipped example's form), with **no Neo4j and no API
  key**: lexical chunk retrieval is a local token scorer,
  the graph arm is the lab's overlap ranking, and the vector
  arm joins only when the build shipped chunk embeddings
  and an embedder is bound;
- :class:`Neo4jServingStore` — the same contract over the
  §4 :class:`~pipeline.graph_store.GraphStore` when a
  ``NEO4J_*`` connection is configured (selected when
  ``NEO4J_URL`` is present in the environment, or forced
  with ``--store``). Chunk retrieval and neighborhood are
  the store's own; attestation records still read from the
  registry artifacts — the registry is the source of truth
  (§4), exactly as the lab's ``GraphData`` loaded them.
  The store holds one graph: the serving store
  resolves it from the WRITE stamp, never from a
  user-supplied scope.

:func:`hybrid_search` is the library API: lexical (+ vector
when available) chunk arms fused at document level with the
lab's RRF definition (``query_stack.hybrid_ranking``), plus
the graph arm (``query_stack.graph_current_ranking``'s math,
over the store's attestation records) and a deterministic
entity link. It returns plain dicts — the CLI renders them
as a report card, the sample API (``api.server``, R17)
serves the same dict as JSON.

Scope honesty: the PPR traversal arm and the LLM answerer
stay in the eval stack (they need scipy / a chat binding);
the product arms are lexical + vector + graph-overlap, and
the result names the arms that actually ran.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from . import connections, query_stack
from .graph_store import GraphStore, chunk_id_for
from .inspect import BATCH_ARTEFACTS
from .registry import index_normalize
from .registry_store import RegistryStore

#: Chunk filter keys, identical to graph_store's contract.
CHUNK_FILTER_KEYS = ("corpus", "batch_id", "doc_id", "tier", "tag")

#: Per-document chunk excerpts kept in a search result.
CHUNKS_PER_DOC = 5


class StoreError(ValueError):
    """A serving store cannot be opened or cannot answer.
    The message names the store, the source, and what was
    looked for — refusals are findings, not tracebacks."""


def _first(rec: dict, keys) -> object:
    """First present, non-None value among ``keys``."""
    for k in keys:
        if rec.get(k) is not None:
            return rec[k]
    return None


_SUBJECT_KEYS = ("subject", "subject_id")
_OBJECT_KEYS = ("object", "object_id")
_RELATION_KEYS = ("relation", "surface_relation", "type")
_QUOTE_KEYS = ("quote", "evidence_sentence", "evidence",
               "sentence", "context")


def _normalize_triple(rec: dict, batch_id: str = "") -> dict | None:
    """Normalize one attestation record, across the
    pipeline's record families, to the serving shape:
    surfaces, resolved-endpoint placeholders, evidence
    quote, gate provenance. ``None`` when the record is
    not triple-shaped. Entity ids are filled in later by
    the store (registry resolution); until then the
    surface name stands in as the id."""
    rel = _first(rec, _RELATION_KEYS)
    subj = _first(rec, _SUBJECT_KEYS)
    obj = _first(rec, _OBJECT_KEYS)
    if rel is None or (subj is None and obj is None):
        return None
    tier = rec.get("tier")
    p_true = rec.get("p_true")
    return {
        "subject_id": str(subj) if subj is not None else "",
        "subject": str(subj) if subj is not None else "",
        "relation": str(rel),
        "object_id": str(obj) if obj is not None else "",
        "object": str(obj) if obj is not None else "",
        "doc_id": str(rec.get("doc_id") or ""),
        "quote": str(_first(rec, _QUOTE_KEYS) or ""),
        "p_true": float(p_true) if p_true is not None else None,
        "tier": int(tier) if tier is not None else None,
        "tag": rec.get("tag") or (
            "VERIFIED" if tier == 1 else "UNVERIFIED"),
        "batch_id": str(rec.get("batch_id") or batch_id),
        "model": rec.get("model") or rec.get("method") or "",
        "gate": rec.get("gate") or rec.get("gate_model") or "",
    }


# ----------------------------------------------------------------------
# artifact-backed records/entities — shared by both stores
# ----------------------------------------------------------------------
class _ArtifactRecords:
    """Attestation records + entity resolution loaded from
    the registry's artifact tree (a records file, or a
    batch's routed rows). Entity ids resolve through the
    SQLite registry when one is present beside the root
    (``data/registry.db``) — canonical ids, names, and
    aliases from the source of truth; without a registry,
    the surface name is the id (the records form's rule,
    and the honest fallback everywhere)."""

    def __init__(self, records: list[dict], corpus: str = "",
                 batch_id: str = "", registry=None):
        self._corpus = corpus
        self._registry = registry
        self._entities: dict[str, dict] = {}
        self._triples: list[dict] = []
        for rec in records:
            row = _normalize_triple(rec, batch_id=batch_id)
            if row is None:
                continue
            for side in ("subject", "object"):
                name = row[side]
                if name:
                    row[f"{side}_id"] = self._resolve(name)
            self._triples.append(row)

    # -- entity resolution --------------------------------------
    def _resolve(self, surface: str) -> str:
        """Entity id for a surface name; registers the
        entity record (id → canonical name + aliases) on
        first sight."""
        if self._registry is not None:
            ent = self._registry.resolve(
                surface, corpus=self._corpus or None)
            if ent is not None:
                self._entities.setdefault(ent.entity_id, {
                    "entity_id": ent.entity_id,
                    "name": ent.name,
                    "aliases": list(ent.aliases or [])})
                return ent.entity_id
        eid = surface
        self._entities.setdefault(eid, {
            "entity_id": eid, "name": surface, "aliases": []})
        return eid

    # -- the read contract --------------------------------------
    def entities(self) -> dict[str, str]:
        """entity_id → canonical name, for every endpoint
        the records attest."""
        return {eid: e["name"]
                for eid, e in self._entities.items()}

    def entity(self, entity_id: str) -> dict | None:
        """One entity record, or None when unknown."""
        e = self._entities.get(entity_id)
        return dict(e) if e is not None else None

    def find_entities(self, query: str,
                      limit: int = 10) -> list[dict]:
        """Deterministic name linking over canonical
        names and aliases, on registry index keys:
        exact key, then token-set containment (a name's
        tokens all in the query, or the query's all in
        the name — the form that links "Alder & Finch"
        inside a question that drops the "&"), then
        substring. Ties break on (name, id) so the
        result is stable across runs and stores."""
        q = index_normalize(query or "")
        if not q:
            return []
        qtoks = _key_tokens(q)
        scored = []
        for eid, e in self._entities.items():
            forms = [e["name"], *e["aliases"]]
            best = None
            for form in forms:
                nf = index_normalize(form)
                ntoks = _key_tokens(nf)
                if nf == q:
                    rank = 0
                elif ntoks and (ntoks <= qtoks
                                or qtoks <= ntoks):
                    rank = 1
                elif nf and (nf in q or q in nf):
                    rank = 2
                else:
                    continue
                best = rank if best is None else min(best, rank)
            if best is not None:
                scored.append((best, e["name"], eid))
        scored.sort()
        return [dict(self._entities[eid])
                for _, _, eid in scored[:limit]]

    def triples(self) -> list[dict]:
        """The attestation records, normalized (copies)."""
        return [dict(t) for t in self._triples]

    def triple_records(self) -> list[dict]:
        """Records in the spelling the R9 viewer
        (``pipeline.inspect.build_report``) normalizes:
        the same rows, keyed for its record families."""
        return [{
            "subject": t["subject"], "relation": t["relation"],
            "object": t["object"], "quote": t["quote"],
            "doc_id": t["doc_id"], "tier": t["tier"],
            "p_true": t["p_true"], "batch_id": t["batch_id"],
            "tag": t["tag"],
        } for t in self._triples]


def _key_tokens(index_key: str) -> set[str]:
    """The content tokens of a registry index key —
    tokens carrying at least one alphanumeric (a bare
    "&" is punctuation, not a name token)."""
    return {t for t in index_key.split(" ")
            if any(c.isalnum() for c in t)}


def _load_registry(root: Path | None):
    """The SQLite entity registry beside ``root``, when one
    exists (``root/data/registry.db``); else None. Read
    access only — serving never mints or merges."""
    if root is None:
        return None
    db = Path(root) / "data" / "registry.db"
    if not db.is_file():
        return None
    from .file_registry import SqliteEntityRegistry
    return SqliteEntityRegistry(db)


def _records_from_dir(d: Path) -> tuple[list[dict], Path] | None:
    """A directory's richest triple artefact, in the R9
    viewer's resolution order (inspect.BATCH_ARTEFACTS),
    plus a bare ``triples.jsonl``. (records, path) or
    None."""
    for name in (*BATCH_ARTEFACTS, "triples.jsonl"):
        cand = d / name
        if cand.is_file():
            return _read_records(cand), cand
    return None


def _read_records(path: Path) -> list[dict]:
    """JSONL (or a single JSON list/object) of records."""
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in text.splitlines()
            if line.strip()]


# ----------------------------------------------------------------------
# file-backed store — the sample path (no Neo4j, no key)
# ----------------------------------------------------------------------
class FileServingStore(_ArtifactRecords):
    """The serving contract over the artifact tree.

    Source forms (first match wins):

    - a **batch directory** (carries ``batch.json``):
      routed attestations + document texts + the tier-2
      serving index (chunk texts + embeddings) when the
      batch's INDEX stage wrote one;
    - a **records directory or file** (the shipped
      example's form): a triple-records JSONL plus a
      corpus of ``.txt`` documents (a ``corpus/`` sibling
      or the same directory); chunks are cut at load with
      the lab's measured chunker, so retrieval runs over
      exactly the chunk universe the build used.

    Lexical chunk scoring is local and deterministic: a
    chunk scores ``Σ (1 + ln tf)`` over the distinct query
    terms it contains — a monotone stand-in for the Lucene
    arm's contract (same rows, same fusion), not a claim
    of Lucene's scores.
    """

    kind = "file"

    def __init__(self, source, root: Path | None = None):
        source = Path(source)
        self.source = source
        self._chunks: list[dict] = []
        self._embeddings: dict[str, np.ndarray] = {}
        self._documents: dict[str, str] = {}
        self._batch_id = ""
        self._corpus = ""
        registry = _load_registry(root)
        if source.is_dir() and (source / "batch.json").is_file():
            records = self._load_batch(source, root)
        elif source.is_dir():
            records = self._load_records_dir(source)
        elif source.is_file():
            records = self._load_records_file(source)
        else:
            raise StoreError(
                f"serving source not found: {source} (looked "
                f"for a batch directory, a records directory, "
                f"or a records file)")
        super().__init__(records, corpus=self._corpus,
                         batch_id=self._batch_id,
                         registry=registry)
        self._finish_chunks()

    # -- source forms -------------------------------------------
    def _load_batch(self, d: Path,
                    root: Path | None) -> list[dict]:
        rs_root = root
        if rs_root is None:
            # <root>/batches/<batch_id> — the store root is
            # two levels up from the batch directory.
            rs_root = d.parent.parent
        rs = RegistryStore(rs_root)
        self._batch_id = d.name
        try:
            self._corpus = rs.corpus(self._batch_id)
        except (OSError, ValueError, KeyError):
            self._corpus = ""
        texts_p = d / "texts.jsonl"
        if texts_p.is_file():
            for line in texts_p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._documents[r["doc_id"]] = r.get(
                        "text", "")
        index = rs.tier2_index(self._batch_id)
        if index is not None:
            self._chunks_from_index(index)
        routed_p = d / "routed.jsonl"
        if routed_p.is_file():
            return _read_records(routed_p)
        found = _records_from_dir(d)
        return found[0] if found else []

    def _load_records_dir(self, d: Path) -> list[dict]:
        found = _records_from_dir(d)
        if found is None:
            raise StoreError(
                f"{d} holds no triple artefact (looked for "
                f"{', '.join(BATCH_ARTEFACTS)}, triples.jsonl)")
        records, _ = found
        self._batch_id = d.name
        corpus_dir = d / "corpus" if (d / "corpus").is_dir() \
            else d
        self._load_corpus(corpus_dir)
        return records

    def _load_records_file(self, p: Path) -> list[dict]:
        records = _read_records(p)
        self._batch_id = p.stem
        corpus_dir = p.parent / "corpus" \
            if (p.parent / "corpus").is_dir() else p.parent
        self._load_corpus(corpus_dir)
        return records

    def _load_corpus(self, corpus_dir: Path) -> None:
        for txt in sorted(corpus_dir.glob("*.txt")):
            self._documents[txt.stem] = txt.read_text()

    def _chunks_from_index(self, index: dict) -> None:
        embs = index.get("embeddings")
        for i, (meta, text) in enumerate(
                zip(index.get("meta", []),
                    index.get("chunks", []), strict=True)):
            doc_id = meta.get("doc_id", "")
            idx = int(meta.get("chunk_idx", i))
            row = {
                "chunk_id": chunk_id_for(
                    meta.get("batch_id", self._batch_id),
                    doc_id, idx),
                "doc_id": doc_id, "idx": idx,
                "tier": meta.get("tier", 2),
                "tag": meta.get("tag", "UNVERIFIED"),
                "text": text,
                "batch_id": meta.get("batch_id",
                                     self._batch_id),
                "corpus": self._corpus,
            }
            self._chunks.append(row)
            if embs is not None and i < len(embs):
                self._embeddings[row["chunk_id"]] = np.asarray(
                    embs[i], dtype=np.float64)

    def _finish_chunks(self) -> None:
        """Cut chunks for documents the index did not cover
        (the records form's whole corpus), with the lab's
        chunker; a document's chunk tier is its best
        attested triple tier — the projection's own rule
        (query_stack.project_serving)."""
        covered = {c["doc_id"] for c in self._chunks}
        best_tier: dict[str, int] = {}
        for t in self._triples_raw():
            if t["tier"] is not None:
                cur = best_tier.get(t["doc_id"], 2)
                best_tier[t["doc_id"]] = min(cur, t["tier"])
        chunker = query_stack.FixedWindowChunker()
        for doc_id, text in sorted(self._documents.items()):
            if doc_id in covered:
                continue
            tier = best_tier.get(doc_id, 2)
            for idx, ch in chunker.chunk(text):
                self._chunks.append({
                    "chunk_id": chunk_id_for(
                        self._batch_id, doc_id, idx),
                    "doc_id": doc_id, "idx": idx,
                    "tier": tier,
                    "tag": "VERIFIED" if tier == 1
                    else "UNVERIFIED",
                    "text": ch, "batch_id": self._batch_id,
                    "corpus": self._corpus,
                })

    def _triples_raw(self) -> list[dict]:
        return self._triples

    # -- store identity -------------------------------------------
    def describe(self) -> dict:
        """Store identity + inventory, for result headers,
        the API health endpoint, and refusal messages."""
        return {
            "kind": self.kind,
            "source": str(self.source),
            "batch_id": self._batch_id,
            "entities": len(self._entities),
            "triples": len(self._triples),
            "documents": len(self._documents),
            "chunks": len(self._chunks),
            "embeddings": bool(self._embeddings),
        }

    def batch_filter(self) -> dict:
        """The chunk-filter pinning retrieval to this
        store's build (the lab's arms all filter so)."""
        return {"batch_id": self._batch_id} \
            if self._batch_id else {}

    # -- graph reads ----------------------------------------------
    def neighborhood(self, entity_ids, hops: int = 1,
                     tier_filter=None, limit: int = 200) -> dict:
        """REL neighborhood (undirected), same shape as
        ``GraphStore.neighborhood``: {nodes, edges}. The
        Cypher store caps *paths*; here ``limit`` caps the
        collected edges — identical on any graph small
        enough for the shapes to meet."""
        if not 1 <= int(hops) <= 4:
            raise ValueError("hops must be 1..4")
        tiers = {int(t) for t in tier_filter} \
            if tier_filter else None
        adjacency: dict[str, list[dict]] = {}
        for t in self._triples:
            if tiers is not None and t["tier"] not in tiers:
                continue
            edge = {"from": t["subject_id"], "to": t["object_id"],
                    "type": t["relation"], "tier": t["tier"],
                    "tag": t["tag"], "p_true": t["p_true"],
                    "batch_id": t["batch_id"],
                    "doc_id": t["doc_id"]}
            adjacency.setdefault(t["subject_id"], []).append(edge)
            adjacency.setdefault(t["object_id"], []).append(edge)
        nodes: dict[str, dict] = {}
        edges: dict[tuple, dict] = {}
        frontier = set(entity_ids)
        for eid in frontier:
            if eid in self._entities:
                nodes[eid] = {"entity_id": eid,
                              "name": self._entities[eid]["name"]}
        for _ in range(int(hops)):
            nxt: set[str] = set()
            for eid in list(frontier):
                for edge in adjacency.get(eid, []):
                    key = (edge["from"], edge["type"],
                           edge["to"], edge["batch_id"],
                           edge["doc_id"])
                    if key not in edges \
                            and len(edges) < int(limit):
                        edges[key] = edge
                    for side in (edge["from"], edge["to"]):
                        if side not in nodes:
                            ent = self._entities.get(side)
                            nodes[side] = {
                                "entity_id": side,
                                "name": ent["name"] if ent
                                else side}
                            nxt.add(side)
            frontier = nxt
            if not frontier:
                break
        return {"nodes": list(nodes.values()),
                "edges": list(edges.values())}

    # -- chunk retrieval ------------------------------------------
    def search_chunks(self, vector=None, text=None,
                      filters=None, k: int = 10) -> list[dict]:
        """The §4 chunk-retrieval contract, locally: vector
        → cosine over the build's shipped embeddings; text
        → the local lexical scorer (module docstring);
        both → union with summed scores. Rows carry the
        GraphStore row keys."""
        filters = {kk: v for kk, v in (filters or {}).items()
                   if kk in CHUNK_FILTER_KEYS}
        rows = [c for c in self._chunks
                if all(c.get(kk) == v
                       for kk, v in filters.items())]
        out: dict[str, dict] = {}
        if vector is not None:
            qv = np.asarray(vector, dtype=np.float64)
            qn = float(np.linalg.norm(qv)) or 1.0
            for c in rows:
                emb = self._embeddings.get(c["chunk_id"])
                if emb is None:
                    continue
                en = float(np.linalg.norm(emb)) or 1.0
                score = float(qv @ emb / (qn * en))
                out[c["chunk_id"]] = {
                    **_chunk_row(c), "score": score,
                    "vector_score": score}
        if text is not None:
            terms = _query_terms(text)
            for c in rows:
                score = _lexical_score(c["text"], terms)
                if score <= 0:
                    continue
                if c["chunk_id"] in out:
                    out[c["chunk_id"]]["text_score"] = score
                    out[c["chunk_id"]]["score"] += score
                else:
                    out[c["chunk_id"]] = {
                        **_chunk_row(c), "score": score,
                        "text_score": score}
        return sorted(out.values(),
                      key=lambda d: -d["score"])[:int(k)]


def _chunk_row(c: dict) -> dict:
    """A chunk's contract row (no score yet)."""
    return {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
            "idx": c["idx"], "tier": c["tier"],
            "tag": c["tag"], "text": c["text"]}


def _query_terms(text: str) -> list[str]:
    """The query's distinct terms. Accepts the lab's
    OR-form (``text_chunk_rows`` builds ``"a OR b"``) or
    raw text, tokenized the lab's way."""
    if " OR " in text:
        raw = text.split(" OR ")
        return list(dict.fromkeys(
            t.strip().lower() for t in raw if t.strip()))
    return list(dict.fromkeys(query_stack.tokenize(text)))


def _lexical_score(chunk_text: str, terms: list[str]) -> float:
    """Σ (1 + ln tf) over the distinct query terms the
    chunk contains; 0 when it contains none."""
    if not terms:
        return 0.0
    toks = query_stack.tokenize(chunk_text)
    if not toks:
        return 0.0
    score = 0.0
    for term in terms:
        tf = toks.count(term)
        if tf:
            score += 1.0 + math.log1p(tf)
    return score


# ----------------------------------------------------------------------
# Neo4j-backed store — same contract, §4 transport
# ----------------------------------------------------------------------
class Neo4jServingStore(_ArtifactRecords):
    """The serving contract over the §4 GraphStore, for
    when a ``NEO4J_*`` connection is configured. Chunk
    retrieval and neighborhood are the store's own (the
    parity is by construction); attestation records and
    entity resolution read from the registry artifacts —
    the registry is the source of truth (§4), the same
    split the lab's GraphData used. A store that cannot
    be reached refuses by name (StoreError) — it never
    silently falls back to the file store.

    One graph: the store holds
    exactly one graph, stamped at WRITE with its
    identity. Serving resolves THAT graph from the stamp
    — there is no user-supplied scope. A ``--batch`` /
    ``--source`` naming a different build refuses by
    name (build it to make its graph the store's, or
    drop the name to serve the store's current graph);
    a store with no stamp serves nothing and says why,
    instead of printing an empty inventory over whatever
    happens to be in the database."""

    kind = "neo4j"

    def __init__(self, source=None, root: Path | None = None,
                 graph_store: GraphStore | None = None):
        self.source = Path(source) if source else None
        self.gs = graph_store or GraphStore()
        records: list[dict] = []
        self._batch_id = ""
        self._corpus = ""
        self._stamp: dict | None = None
        from .neo4j_store import Neo4jClient
        if isinstance(getattr(self.gs, "c", None), Neo4jClient):
            records = self._resolve_store_graph(root)
        elif self.source is not None and self.source.is_dir():
            self._batch_id = self.source.name
            found = _records_from_dir(self.source)
            if found:
                records = found[0]
            rs = RegistryStore(root or self.source.parent.parent)
            try:
                self._corpus = rs.corpus(self._batch_id)
            except (OSError, ValueError, KeyError):
                self._corpus = ""
        super().__init__(records, corpus=self._corpus,
                         batch_id=self._batch_id,
                         registry=_load_registry(root))
        self._documents_count = len(
            {t["doc_id"] for t in self._triples if t["doc_id"]})

    def _resolve_store_graph(self, root: Path | None) -> list[dict]:
        """Resolve the store's one graph from its stamp
        and load that build's attestation records.
        Refusals are StoreError, named — see the class
        docstring for the cases."""
        from .neo4j_store import (
            describe_graph_identity,
            read_graph_stamp,
            store_inventory,
        )
        url = getattr(self.gs.c, "url", "") or \
            os.environ.get("NEO4J_URL", "")
        stamp = self._guard("read the store's graph stamp",
                            lambda: read_graph_stamp(self.gs.c))
        if stamp is None:
            inv = self._guard("inventory the store",
                              lambda: store_inventory(self.gs.c))
            if inv["rel_edges"] or inv["entities"] or inv["chunks"]:
                raise StoreError(
                    f"the Neo4j store at {url} holds "
                    f"{inv['rel_edges']} REL edges, "
                    f"{inv['entities']} entities and "
                    f"{inv['chunks']} chunks that no OntoSmith "
                    f"graph stamp identifies — serving cannot "
                    f"tell which build's graph it would be "
                    f"answering from, so it refuses rather than "
                    f"guess. If the content is disposable, "
                    f"clear the database and re-run the build; "
                    f"setup: docs/connections.md (Neo4j — the "
                    f"graph store).")
            raise StoreError(
                f"the Neo4j store at {url} holds no graph "
                f"yet — no build has written one. Build first "
                f"(`ontosmith batch run <batch>`), or read a "
                f"build's artifacts back with --store file.")
        self._stamp = stamp
        want = stamp["batch_id"]
        if self.source is not None and self.source.name != want:
            raise StoreError(
                f"the named build '{self.source.name}' is not "
                f"the Neo4j store's current graph — the store "
                f"holds {describe_graph_identity(stamp)} (the "
                f"store serves one graph). Build "
                f"'{self.source.name}' to make its graph the "
                f"store's, or drop --batch/--source to serve "
                f"the store's current graph.")
        sroot = root or _state_root()
        bdir = sroot / "batches" / want
        if not bdir.is_dir():
            raise StoreError(
                f"the Neo4j store's current graph was built "
                f"by batch '{want}', whose artifacts are not "
                f"under this state root ({sroot}) — the "
                f"attestation records live with the registry "
                f"that built the graph; run from that "
                f"workspace, or rebuild here.")
        self._batch_id = want
        self.source = bdir
        rs = RegistryStore(sroot)
        try:
            self._corpus = rs.corpus(want)
        except (OSError, ValueError, KeyError):
            self._corpus = ""
        found = _records_from_dir(bdir)
        return found[0] if found else []

    def describe(self) -> dict:
        url = getattr(self.gs.c, "url", "") or \
            os.environ.get("NEO4J_URL", "")
        return {
            "kind": self.kind,
            "source": str(self.source) if self.source else "",
            "neo4j_url": url,
            "batch_id": self._batch_id,
            "graph_stamp": self._stamp,
            "entities": len(self._entities),
            "triples": len(self._triples),
            "documents": self._documents_count,
            "chunks": None,
            "embeddings": None,
        }

    def batch_filter(self) -> dict:
        return {"batch_id": self._batch_id} \
            if self._batch_id else {}

    def _guard(self, what: str, fn):
        try:
            return fn()
        except Exception as e:  # transport refusal, named
            url = getattr(self.gs.c, "url", "") or \
                os.environ.get("NEO4J_URL", "")
            raise StoreError(
                f"Neo4j store at {url} could not answer "
                f"{what}: {e}") from e

    def search_chunks(self, vector=None, text=None,
                      filters=None, k: int = 10) -> list[dict]:
        return self._guard(
            "chunk search",
            lambda: self.gs.search_chunks(
                vector=vector, text=text, filters=filters,
                k=k))

    def neighborhood(self, entity_ids, hops: int = 1,
                     tier_filter=None, limit: int = 200) -> dict:
        return self._guard(
            "neighborhood",
            lambda: self.gs.neighborhood(
                entity_ids, hops=hops,
                tier_filter=tier_filter, limit=limit))


# ----------------------------------------------------------------------
# store opening — the existing surfaces only (no new config)
# ----------------------------------------------------------------------
def _state_root() -> Path:
    """The batch/state root rule (batch_cli.state_root,
    ), imported lazily to keep this module light."""
    from .batch_cli import state_root
    return state_root()


def example_candidates() -> list[Path]:
    """Where the shipped example build may live, in
    resolution order: the state root's tree, the package
    tree, the pip --target layout (data files beside the
    packages), the regular install layout (the
    interpreter's data path) — the packs' own candidate
    discipline."""
    import sysconfig
    pkg_parent = Path(__file__).resolve().parent.parent
    return [
        _state_root() / "examples",
        pkg_parent / "examples",
        pkg_parent / "share" / "ontosmith" / "examples",
        Path(sysconfig.get_path("data")) / "share"
        / "ontosmith" / "examples",
    ]


def default_example() -> Path:
    """The shipped example build's directory; StoreError
    naming every candidate when none resolves."""
    for cand in example_candidates():
        if (cand / "triples.jsonl").is_file():
            return cand
    raise StoreError(
        "the shipped example build was not found (looked "
        "for triples.jsonl under: "
        + ", ".join(str(c) for c in example_candidates())
        + ") — pass --source or --batch explicitly")


def open_store(source=None, batch: str | None = None,
               store: str = "auto",
               env: dict | None = None):
    """Open the serving store for a search/API session.

    Selection (the existing surfaces only): ``store`` =
    ``"neo4j"`` forces the §4 store; ``"file"`` forces the
    artifact store; ``"auto"`` uses Neo4j when ``NEO4J_URL``
    is configured in the environment, else the file store.
    The graph served is ``batch`` (a batch id under the
    state root) or ``source`` (a path); with neither, the
    shipped example build."""
    env = os.environ if env is None else env
    root = _state_root()
    src: Path | None = None
    if batch:
        src = root / "batches" / batch
        if not src.is_dir():
            raise StoreError(
                f"batch not found: {src} (no such batch "
                f"directory under the state root {root})")
    elif source:
        src = Path(source)
    kind = store
    if kind == "auto":
        kind = "neo4j" if env.get("NEO4J_URL") else "file"
    if kind == "neo4j":
        return Neo4jServingStore(source=src, root=root)
    if src is None:
        src = default_example()
    return FileServingStore(src, root=root)


# ----------------------------------------------------------------------
# hybrid search — the R16 library API
# ----------------------------------------------------------------------
def hybrid_search(store, query: str, k: int = 10,
                  embedder=None,
                  fetch_k: int | None = None) -> dict:
    """Hybrid retrieval over the store's build: lexical
    (+ vector, when an embedder is bound and the build
    shipped embeddings) chunk arms fused at document
    level by the lab's RRF definition, the graph arm over
    attestation records, and a deterministic entity link.

    ``embedder`` is any object with ``embed(texts)`` (the
    pipeline's EntityEmbedder shape); its failures
    propagate — a declared binding that cannot run is a
    refusal, not a silent arm-drop. Returns the result
    dict the CLI renders and the API serves."""
    query = (query or "").strip()
    if not query:
        raise ValueError("empty search query")
    fetch = int(fetch_k) if fetch_k else query_stack.CHUNK_FETCH_K
    filt = store.batch_filter()
    arms: list[str] = []
    notes: list[str] = []
    if embedder is not None \
            and store.describe().get("embeddings") is False:
        # The build ships no chunk embeddings: embedding
        # the query could only spend, never match. The arm
        # stays off and the result says why.
        embedder = None
        notes.append(
            "vector arm inactive: the build ships no "
            "chunk embeddings for this store")

    # lexical arm (the lab's text_chunk_rows form)
    toks = [t for t in query_stack.tokenize(query)
            if len(t) > 2]
    text_rows: list[dict] = []
    if toks:
        text_rows = store.search_chunks(
            text=" OR ".join(toks), filters=filt, k=fetch)
        arms.append("lexical")

    # vector arm (dense_chunk_rows), when bound + shipped
    vec_rows: list[dict] = []
    if embedder is not None:
        qvec = embedder.embed([query])[0]
        vec_rows = store.search_chunks(
            vector=[float(x) for x in qvec], filters=filt,
            k=fetch)
        if vec_rows:
            arms.append("vector")
        else:
            notes.append(
                "vector arm inactive: the build ships no "
                "chunk embeddings for this store")

    # fuse chunk arms at document level (hybrid_ranking's
    # RRF definition, scores recomputed on the same
    # 1/(60 + rank) formula query_stack.rrf applies)
    rankings = []
    if vec_rows:
        rankings.append(query_stack.docs_from_chunk_rows(
            vec_rows))
    if text_rows:
        rankings.append(query_stack.docs_from_chunk_rows(
            text_rows))
    doc_scores: dict[str, float] = {}
    for ranking in rankings:
        for i, doc_id in enumerate(ranking):
            doc_scores[doc_id] = doc_scores.get(
                doc_id, 0.0) + 1.0 / (60 + i + 1)
    doc_order = [d for d, _ in sorted(
        doc_scores.items(), key=lambda kv: -kv[1])]

    chunks_by_doc: dict[str, dict[str, dict]] = {}
    for row in (*vec_rows, *text_rows):
        chunks_by_doc.setdefault(row["doc_id"], {})[
            row["chunk_id"]] = row

    documents = []
    for doc_id in doc_order[:int(k)]:
        chunks = sorted(
            chunks_by_doc.get(doc_id, {}).values(),
            key=lambda r: -r["score"])[:CHUNKS_PER_DOC]
        documents.append({
            "doc_id": doc_id,
            "score": round(doc_scores[doc_id], 6),
            "chunks": chunks,
        })

    # graph arm: the lab's overlap ranking over the
    # store's attestation records (graph_current_ranking
    # iterates exactly these rows, with these token sets)
    triples = store.triples()
    token_sets = [set(query_stack.tokenize(
        f"{t['subject']} {t['relation']} {t['object']}"))
        for t in triples]
    graph_docs = query_stack._overlap_ranking(
        triples, token_sets, query)
    if graph_docs:
        arms.append("graph")
    triple_scores: dict[int, float] = {}
    qtoks = {t for t in query_stack.tokenize(query)
             if t not in query_stack.STOP and len(t) > 2}
    for i, (t, ttoks) in enumerate(
            zip(triples, token_sets, strict=True)):
        overlap = len(qtoks & ttoks)
        if overlap >= 2:
            w = (t["p_true"] or 0.3) * (
                1.0 if t["tier"] == 1
                else query_stack.TIER2_EDGE_DISCOUNT)
            triple_scores[i] = w * overlap
    ranked_triples = [
        {**triples[i], "score": round(s, 6)}
        for i, s in sorted(triple_scores.items(),
                           key=lambda kv: -kv[1])[:int(k)]]

    return {
        "query": query,
        "store": store.describe(),
        "arms": arms,
        "notes": notes,
        "entities": store.find_entities(query, limit=5),
        "documents": documents,
        "graph_documents": graph_docs[:int(k)],
        "triples": ranked_triples,
    }


def try_embedder() -> object | None:
    """The embedder for the vector arm, when one resolves
    without spend risk: the hosted embedder seat when its
    key is present. There is no local fallback to load;
    ``--embed`` asks for the connection layer's
    default_embedder() explicitly, which refuses by name
    when the seat's key is unset. Returns None when no
    hosted key is bound."""
    from . import connections
    try:
        binding = connections.seat("embedder")
    except Exception:
        return None
    if not binding.key_or_none():
        return None
    from .embedder import SeatEmbedder
    return SeatEmbedder()


# ----------------------------------------------------------------------
# CLI — ``ontosmith search``
# ----------------------------------------------------------------------
def render_results(result: dict) -> str:
    """The human report card for a search result."""
    s = result["store"]
    lines = [
        f"Search  {result['query']!r}",
        f"  store   {s['kind']} · {s.get('source') or ''} "
        f"({s['triples']} triples, {s['documents']} "
        f"documents)",
        f"  arms    {' · '.join(result['arms']) or '(none)'}",
    ]
    for note in result["notes"]:
        lines.append(f"  note    {note}")
    if result["entities"]:
        lines.append("Entities")
        for e in result["entities"]:
            lines.append(f"  - {e['name']}")
    lines.append("Documents")
    if result["documents"]:
        for i, d in enumerate(result["documents"], 1):
            lines.append(
                f"  {i}. {d['doc_id']}  (score {d['score']})")
            for c in d["chunks"][:2]:
                excerpt = " ".join(c["text"].split())[:160]
                lines.append(f"     {excerpt}")
    else:
        lines.append("  (no chunk matched)")
    lines.append("Triples")
    if result["triples"]:
        for t in result["triples"]:
            lines.append(
                f"  - {t['subject']} — {t['relation']} → "
                f"{t['object']}  (p {t['p_true']}, "
                f"tier {t['tier']}, {t['doc_id']})")
            if t["quote"]:
                lines.append(f"    \"{t['quote']}\"")
    else:
        lines.append("  (no triple matched)")
    return "\n".join(lines)


def main(argv=None) -> int:
    """``ontosmith search <query> [--source PATH]
    [--batch ID] [--store auto|file|neo4j] [--k N]
    [--embed] [--json]``."""
    ap = argparse.ArgumentParser(
        prog="ontosmith search",
        description="Hybrid search over a built graph "
                    ": lexical + vector + "
                    "graph retrieval over the build's "
                    "chunks and triples")
    ap.add_argument("query", help="the search question")
    ap.add_argument("--source", default=None,
                    help="batch directory, records "
                         "directory, or records file "
                         "(default: the shipped example "
                         "build)")
    ap.add_argument("--batch", default=None,
                    help="batch id under the state root's "
                         "batches/ (alternative to --source)")
    ap.add_argument("--store", default="auto",
                    choices=("auto", "file", "neo4j"),
                    help="serving store: auto = Neo4j when "
                         "NEO4J_URL is configured, else the "
                         "file store (default: auto)")
    ap.add_argument("--k", type=int, default=10,
                    help="max documents / triples returned "
                         "(default: 10)")
    ap.add_argument("--embed", action="store_true",
                    help="activate the vector arm via the "
                         "connection layer's default "
                         "embedder (refuses if the embedder "
                         "seat's key is unset)")
    ap.add_argument("--json", action="store_true",
                    help="print the result as JSON instead "
                         "of the report card")
    args = ap.parse_args(argv)

    try:
        store = open_store(source=args.source,
                           batch=args.batch, store=args.store)
        embedder = None
        if args.embed:
            from .embedder import default_embedder
            embedder = default_embedder()
        else:
            embedder = try_embedder()
        result = hybrid_search(store, args.query, k=args.k,
                               embedder=embedder)
    except (ValueError, connections.BindingKeyError) as e:
        print(f"search refused: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=1,
                         ensure_ascii=False))
    else:
        print(render_results(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
