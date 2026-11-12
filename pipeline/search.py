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

:func:`sentence_search` is the CLI's default serving
path (the serving fix): sentence-level evidence serving
over a direct index of every sentence in the build,
with the graph pool, a beam traversal channel over the
entity graph, an optional local cross-encoder rerank of
the merged head, and a whole-unit fill under one token
budget. :func:`hybrid_search` — lexical (+ vector when
available) chunk arms fused at document level with the
lab's RRF definition (``query_stack.hybrid_ranking``),
plus the graph arm (``query_stack.graph_current_ranking``'s
math, over the store's attestation records) and a
deterministic entity link — remains the chunk-serving
library form: the CLI's internal fallback when sentence
serving cannot run (no embedder bound, or a store that
serves no sentence index), and the form the sample API
(``api.server``, R17) serves as JSON. Both return plain
dicts — the CLI renders them as a report card.

Scope honesty: the PPR traversal arm and the LLM answerer
stay in the eval stack (they need scipy / a chat binding);
the product arms are lexical + vector + graph-overlap
plus the sentence/beam serving channels, and the result
names the arms that actually ran.
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
from .sentence_index import build_sentence_rows, ent_key, ws_norm

#: The sentence identity key (dedupe / coverage), shared
#: with the build-time sentence universe.
_ws_norm = ws_norm

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
        self._sent_rows: list[dict] | None = None
        self._sent_emb: np.ndarray | None = None
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
        sents = index.get("sentences")
        sembs = index.get("sentence_embeddings")
        if sents is not None and sembs is not None:
            self._sent_rows = [dict(s) for s in sents]
            self._sent_emb = np.asarray(sembs, dtype=np.float64)

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
            "sentences": len(self._sent_rows)
            if self._sent_rows is not None else None,
        }

    # -- sentence index ---------------------------------------
    def prepare_sentence_index(self, embedder) -> str | None:
        """Make the sentence snapshot available. A build
        whose INDEX stage shipped a sentence index needs
        nothing (returns None). An older build — or the
        records form, which ships no index at all — gets
        the identical universe built at query time from
        its chunks with the bound embedder, memoized on
        the store; the returned note says so, so the
        result never hides that the build predates the
        shipped index. StoreError when the store holds
        no chunks to build from."""
        if self._sent_rows is not None:
            return None
        if not self._chunks:
            raise StoreError(
                f"the file store at {self.source} holds no "
                f"chunks — there is no sentence universe "
                f"to serve")
        meta = [{"doc_id": c["doc_id"],
                 "chunk_idx": c["idx"],
                 "batch_id": c["batch_id"]}
                for c in self._chunks]
        rows = build_sentence_rows(
            [c["text"] for c in self._chunks], meta)
        embs = embedder.embed([r["text"] for r in rows])
        self._sent_rows = rows
        self._sent_emb = np.asarray(embs, dtype=np.float64)
        return (f"sentence index built at query time: "
                f"this build predates the shipped sentence "
                f"index ({len(rows)} sentences embedded "
                f"with the bound embedder)")

    def sentence_snapshot(self):
        """(rows, embeddings) of the sentence universe,
        or None when no index is prepared. Rows carry
        text / doc_id / chunk_id; embeddings align by
        position."""
        if self._sent_rows is None or self._sent_emb is None:
            return None
        return [dict(r) for r in self._sent_rows], self._sent_emb

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
        self._serving: dict | None = None
        self._sent_rows: list[dict] | None = None
        self._sent_emb: np.ndarray | None = None
        from .neo4j_store import Neo4jClient
        if isinstance(getattr(self.gs, "c", None), Neo4jClient):
            records = self._resolve_store_graph(root)
            self._heal_serving(root)
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

    def _heal_serving(self, root: Path | None) -> None:
        """Stores built before the serving projection
        became the INDEX stage's job answer too: at open,
        ensure the serving schema and project the stamped
        batch's chunks when the store holds none for it
        (GraphStore.ensure_serving — idempotent, and no
        provider spend: the projection replays the batch's
        recorded embeddings). The projection reads this
        workspace's artifacts, so it runs through a store
        bound to the same state root the stamp resolution
        just used. A store that cannot be healed refuses
        by name here (StoreError), not at first search."""
        from .graph_store import GraphStore
        gs = GraphStore(client=self.gs.c,
                        registry_store=RegistryStore(
                            root or _state_root()))
        self._serving = self._guard(
            "ensure the serving schema and chunk projection",
            lambda: gs.ensure_serving(self._batch_id))
        if self._serving["projected"] is not None:
            print(f"serving: the store held no chunks for "
                  f"batch '{self._batch_id}' — projected "
                  f"{self._serving['projected']['chunks']} "
                  f"chunks from its recorded index (schema "
                  f"ensured; no provider spend)",
                  file=sys.stderr)

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
            "sentences": len(self._sent_rows)
            if self._sent_rows is not None else None,
            "serving": self._serving,
        }

    # -- sentence index ---------------------------------------
    def prepare_sentence_index(self, embedder) -> str | None:
        """Make the sentence snapshot available: the
        store's projected (:Sentence) nodes when the
        build shipped a sentence index (no spend — the
        projection replayed the recorded embeddings);
        otherwise the identical universe built at query
        time from the store's chunks with the bound
        embedder, memoized, with the returned note
        saying so. StoreError when the store can answer
        neither way."""
        if self._sent_rows is not None:
            return None
        if hasattr(self.gs, "sentence_snapshot"):
            snap = self._guard(
                "sentence snapshot",
                lambda: self.gs.sentence_snapshot(
                    self._batch_id))
            if snap is not None:
                self._sent_rows, self._sent_emb = snap
                return None
        if not hasattr(self.gs, "all_chunks"):
            raise StoreError(
                "the Neo4j serving store cannot serve a "
                "sentence index: it holds no projected "
                "sentences and exposes no chunk texts to "
                "build one from")
        rows = self._guard(
            "chunk texts for the sentence index",
            lambda: self.gs.all_chunks(self._batch_id))
        if not rows:
            raise StoreError(
                "the Neo4j serving store holds no chunks "
                "for its graph — there is no sentence "
                "universe to serve")
        meta = [{"doc_id": r["doc_id"],
                 "chunk_idx": r["idx"],
                 "batch_id": r["batch_id"]}
                for r in rows]
        srows = build_sentence_rows(
            [r["text"] for r in rows], meta)
        embs = embedder.embed([r["text"] for r in srows])
        self._sent_rows = srows
        self._sent_emb = np.asarray(embs, dtype=np.float64)
        return (f"sentence index built at query time: "
                f"this build predates the shipped sentence "
                f"index ({len(srows)} sentences embedded "
                f"with the bound embedder)")

    def sentence_snapshot(self):
        """(rows, embeddings) of the sentence universe,
        or None when no index is prepared."""
        if self._sent_rows is None or self._sent_emb is None:
            return None
        return [dict(r) for r in self._sent_rows], self._sent_emb

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


def _rank_triples(triples, query: str, k: int,
                  token_sets=None,
                  rel_weight: float = 0.0) -> list[dict]:
    """The graph arm's per-triple ranking: content-token
    overlap ≥ 2 against ``subject relation object``,
    weighted by p_true and the tier-2 discount. Shared by
    :func:`hybrid_search` (the displayed triples) and
    :func:`sentence_search` (the candidate pool).

    ``rel_weight`` (the serving-quality program's β)
    adds that multiple of the relation-only overlap to
    the score, so a triple whose relation terms the
    query actually names outranks one carried by name
    overlap alone. 0.0 is the historical scoring."""
    if token_sets is None:
        token_sets = [set(query_stack.tokenize(
            f"{t['subject']} {t['relation']} {t['object']}"))
            for t in triples]
    qtoks = {t for t in query_stack.tokenize(query)
             if t not in query_stack.STOP
             and query_stack.is_content_token(t)}
    scores: dict[int, float] = {}
    for i, (t, ttoks) in enumerate(
            zip(triples, token_sets, strict=True)):
        overlap = len(qtoks & ttoks)
        if overlap >= 2:
            w = (t["p_true"] or 0.3) * (
                1.0 if t["tier"] == 1
                else query_stack.TIER2_EDGE_DISCOUNT)
            boost = 0.0
            if rel_weight:
                rel_toks = {x for x in query_stack.tokenize(
                    t["relation"])
                    if x not in query_stack.STOP
                    and query_stack.is_content_token(x)}
                boost = rel_weight * len(qtoks & rel_toks)
            scores[i] = w * (overlap + boost)
    return [{**triples[i], "score": round(s, 6)}
            for i, s in sorted(scores.items(),
                               key=lambda kv: -kv[1])[:int(k)]]


def _content_tokens(text: str) -> set[str]:
    return {t for t in query_stack.tokenize(text)
            if t not in query_stack.STOP
            and query_stack.is_content_token(t)}


def _tag_name_only(ranked: list[dict], query: str,
                   entity_names) -> tuple[list[dict], int]:
    """Decompose each displayed triple's query overlap
    into the part carried by the linked entities' name
    tokens (``name_overlap``) and the rest — relation
    and value terms (``term_overlap``). A triple whose
    overlap is names alone is a mention, not an answer
    to the query's terms: it is tagged ``name_only``
    and moved after every term-matched triple, each
    side keeping its rank order. Returns the reordered
    list and the name-only count. Ranking itself is
    untouched — this shapes the display, and the pool
    form (sentence serving) keeps the raw ranking."""
    name_toks: set[str] = set()
    for name in entity_names:
        name_toks |= _content_tokens(name)
    qtoks = _content_tokens(query)
    substantive: list[dict] = []
    name_only: list[dict] = []
    for t in ranked:
        overlap = qtoks & _content_tokens(
            f"{t['subject']} {t['relation']} {t['object']}")
        name_ov = len(overlap & name_toks)
        term_ov = len(overlap) - name_ov
        tagged = {**t, "name_overlap": name_ov,
                  "term_overlap": term_ov,
                  "name_only": term_ov == 0}
        (name_only if tagged["name_only"]
         else substantive).append(tagged)
    return substantive + name_only, len(name_only)


def _name_only_note(n_only: int, total: int) -> str:
    if total and n_only == total:
        return ("graph arm: no triple's relation or "
                "value terms matched the query — the "
                f"{total} triple(s) returned match on "
                "entity names only (mentions, not "
                "answers)")
    return (f"graph arm: {n_only} of {total} triples "
            "match on entity names only and are listed "
            "after the term-matched triples")


def _zscores(values: list[float]) -> list[float]:
    """Population z-scores; a constant input scores 0."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var)
    if sd == 0.0:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va)) or 1.0
    nb = float(np.linalg.norm(vb)) or 1.0
    return float(va @ vb / (na * nb))


# ----------------------------------------------------------------------
# hybrid search — the R16 library API
# ----------------------------------------------------------------------
def hybrid_search(store, query: str, k: int = 10,
                  embedder=None,
                  fetch_k: int | None = None,
                  rel_weight: float = 0.0) -> dict:
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
            if query_stack.is_content_token(t)]
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
    ranked_triples = _rank_triples(triples, query, k,
                                   token_sets=token_sets,
                                   rel_weight=rel_weight)
    entities = store.find_entities(query, limit=5)
    ranked_triples, n_only = _tag_name_only(
        ranked_triples, query,
        [n for e in entities
         for n in [e["name"], *e.get("aliases", [])]])
    if n_only:
        notes.append(_name_only_note(
            n_only, len(ranked_triples)))

    return {
        "query": query,
        "store": store.describe(),
        "arms": arms,
        "notes": notes,
        "entities": entities,
        "documents": documents,
        "graph_documents": graph_docs[:int(k)],
        "triples": ranked_triples,
    }


# ----------------------------------------------------------------------
# sentence-level serving — the default search path
# ----------------------------------------------------------------------
#: Token budget a sentence-serving result fills to. One
#: budget for every query, so results stay comparable
#: across questions and arms.
SENTENCE_BUDGET_TOKENS = 4000

#: Triples whose evidence quotes seed the graph pool.
_POOL_TRIPLES = 12

#: Chunk-arm rows anchoring the chunk layer. The fill
#: only ever reaches the top of the ranking; the cap
#: bounds the per-query work, it does not trim the fill.
_SENTENCE_CANDIDATE_CHUNKS = 60

#: Direct-index sentences entering the candidate set.
_DIRECT_TOP = 200

#: Merged units the rerank step reorders — the whole
#: fill population (units past the head never fill).
_RERANK_POOL = 80

#: Beam channel caps, exactly as measured: triples kept
#: per frontier entity / per hop, quote-sentence units
#: the channel may add, traversal depth.
_BEAM_PER_ENTITY = 8
_BEAM_PER_HOP = 24
_BEAM_UNIT_CAP = 16
_BEAM_HOPS = 2


class SentenceServingUnavailable(StoreError):
    """Sentence serving cannot run on this store or
    binding. The CLI catches it and falls back to chunk
    serving (naming the fallback in the result); library
    callers get the named refusal."""


def _link_entities(triples, query: str):
    """Deterministic entity linking for the beam channel:
    entity surfaces (normalized) appearing in the
    normalized question, longest names first,
    non-overlapping spans. Returns (linked entity keys,
    {entity key: [triples]}) — the traversal's adjacency,
    keyed by surface identity."""
    by_key: dict[str, list[dict]] = {}
    for t in triples:
        for side in ("subject", "object"):
            name = (t.get(side) or "").strip()
            if name:
                by_key.setdefault(ent_key(name), []).append(t)
    q = " " + ent_key(query) + " "
    found, taken = [], []
    for key in sorted(by_key, key=len, reverse=True):
        if len(key) < 4:
            continue
        pat = " " + key + " "
        i = q.find(pat)
        while i >= 0:
            span = (i, i + len(pat))
            if not any(span[0] < b and a < span[1]
                       for a, b in taken):
                taken.append(span)
                found.append(key)
                break
            i = q.find(pat, i + 1)
    return found, by_key


def _quote_sentences(store, triples_sel, embedder, filt,
                     chunk_cos, chunk_texts, cand_chunks,
                     cap):
    """Triple quote → owning chunk → its sentences
    carrying the quote. The owning chunk is the triple's
    own document chunk most similar to the quote (the
    build's chunks are fixed windows, so containment is
    by similarity, not offset); the quote itself stands
    in when it does not anchor. Owner chunks register
    their cosine / text and join ``cand_chunks`` — the
    scoring layer's population grows exactly the way the
    measured stack's did."""
    from .proposer import split_sentences
    pool, seen_pool = [], set()
    for t in triples_sel:
        quote = (t.get("quote") or "").strip()
        if not quote:
            continue
        qf = dict(filt)
        if t.get("doc_id"):
            qf["doc_id"] = t["doc_id"]
        qvec_quote = embedder.embed([quote])[0]
        owners = store.search_chunks(
            vector=[float(x) for x in qvec_quote],
            filters=qf, k=1)
        owner_id = owners[0]["chunk_id"] if owners else ""
        if owners and owner_id not in chunk_cos:
            chunk_cos[owner_id] = float(
                owners[0].get("vector_score", 0.0))
            chunk_texts[owner_id] = owners[0]["text"]
            cand_chunks.append(owners[0])
        texts: list[str] = []
        if owners:
            nq = _ws_norm(quote)
            for s in split_sentences(owners[0]["text"]):
                if _ws_norm(s) in nq or nq in _ws_norm(s):
                    texts.append(s)
        if not texts:
            texts = [quote]
        for s in texts:
            key = _ws_norm(s)
            if key and key not in seen_pool:
                seen_pool.add(key)
                pool.append({"text": s, "doc_id": t["doc_id"],
                             "chunk_id": owner_id})
        if len(pool) >= cap:
            break
    return pool[:cap]


def _beam_units(store, by_key, linked, qvec,
                embedder, filt, chunk_cos, chunk_texts,
                cand_chunks):
    """The beam traversal channel, with the measured
    definition: 2 hops over the entity graph from the
    linked entities; each hop prunes the frontier
    entities' triples by triple-verbalization cosine
    (top ``_BEAM_PER_ENTITY`` per entity, at most
    ``_BEAM_PER_HOP`` per hop); the frontier extends to
    the surviving triples' endpoints. The survivors'
    quote sentences (at most ``_BEAM_UNIT_CAP``) join
    the candidate population as via="beam" units.
    Verbalization cosines are computed for frontier
    triples only — the only scores the pruning reads."""
    if not linked:
        return []
    chosen: list[dict] = []
    chosen_ids: set[int] = set()
    frontier = list(linked)
    seen_ent = set(linked)
    for _hop in range(_BEAM_HOPS):
        hop_triples: list[dict] = []
        seen_t: set[int] = set()
        for ek in frontier:
            for t in by_key.get(ek, []):
                if id(t) not in chosen_ids \
                        and id(t) not in seen_t:
                    seen_t.add(id(t))
                    hop_triples.append(t)
        if not hop_triples:
            break
        verbs = [f"{t.get('subject')} {t.get('relation')} "
                 f"{t.get('object')}" for t in hop_triples]
        sims = {id(t): _cosine(qvec, e) for t, e in zip(
            hop_triples, embedder.embed(verbs), strict=True)}
        picked: list[dict] = []
        for ek in frontier:
            ts = [t for t in by_key.get(ek, [])
                  if id(t) in sims and id(t) not in chosen_ids]
            ts.sort(key=lambda t: -sims[id(t)])
            picked.extend(ts[:_BEAM_PER_ENTITY])
        seen_p: set[int] = set()
        uniq: list[dict] = []
        for t in picked:
            if id(t) not in seen_p:
                seen_p.add(id(t))
                uniq.append(t)
        uniq.sort(key=lambda t: -sims[id(t)])
        uniq = uniq[:_BEAM_PER_HOP]
        for t in uniq:
            chosen_ids.add(id(t))
            chosen.append(t)
        nxt: list[str] = []
        for t in uniq:
            for side in ("subject", "object"):
                kk = ent_key(t.get(side))
                if kk and kk not in seen_ent:
                    seen_ent.add(kk)
                    nxt.append(kk)
        frontier = nxt
        if not frontier:
            break
    if not chosen:
        return []
    raw = _quote_sentences(store, chosen, embedder, filt,
                           chunk_cos, chunk_texts, cand_chunks,
                           cap=_BEAM_UNIT_CAP)
    return [dict(u, via="beam", kind="sentence") for u in raw]


def _fill_units(ordered, budget: int) -> list[dict]:
    """Whole-unit fill under the token budget: a unit
    whose sentences a taken unit already covers is
    skipped, a unit that does not fit is skipped, and
    the scan keeps going (the measured fill semantics).
    Nothing is cut mid-sentence."""
    from .proposer import split_sentences
    units: list[dict] = []
    covered: set[str] = set()
    used = 0
    for u in ordered:
        if u["kind"] == "chunk":
            parts = {_ws_norm(s)
                     for s in split_sentences(u["text"])}
            if parts and parts <= covered:
                continue
        elif _ws_norm(u["text"]) in covered:
            continue
        cost = len(query_stack.tokenize(u["text"]))
        if used + cost > int(budget):
            continue
        used += cost
        units.append({**u, "score": round(u["merge"], 6),
                      "tokens": cost})
        if u["kind"] == "chunk":
            covered |= {_ws_norm(s)
                        for s in split_sentences(u["text"])}
        else:
            covered.add(_ws_norm(u["text"]))
    return units


def sentence_search(store, query: str, k: int = 10,
                    embedder=None, reranker=None,
                    budget_tokens: int =
                    SENTENCE_BUDGET_TOKENS,
                    fetch_k: int | None = None,
                    floor_cos: float | None = None,
                    rel_weight: float = 0.0) -> dict:
    """Sentence-level evidence serving — the measured
    final form, wired onto the store contract:

    1. **Direct arm** — the build's sentence index
       (every sentence of every chunk, embedded at
       build time with the build's embedder) contributes
       its top 200 sentences by query cosine. Chunk arms
       are candidate suppliers here, never a gate.
    2. **Graph pool** — the top triples under the graph
       arm's own ranking contribute the sentences that
       carry their evidence quotes.
    3. **Beam channel** — a pruned 2-hop traversal from
       the question's linked entities adds its quote
       sentences as a protected side channel
       (:func:`_beam_units`; the channel that measured
       a positive net contribution where pool mixing
       alone did not).
    4. **Two-layer score** — every candidate sentence
       scores z(sentence cosine) + z(owning-chunk
       cosine), each z within its own layer's candidate
       population; **Z2 merge** — sentences carry 2 x
       their combined score, chunks their z(chunk
       cosine), in one ranked list.
    5. **Rerank + fill** — the merged head (top 80) is
       reordered by the rerank step when one is active
       (``reranker``: an object with ``score(query,
       texts)``; None → the reference local cross-encoder
       when its extra is installed; False → off), then
       whole units fill the token budget.

    The form is embedding-shaped, so ``embedder`` is
    required: a missing binding, or a store that serves
    no sentence index, refuses by name
    (:class:`SentenceServingUnavailable`) rather than
    silently degrading to another serving form. Returns
    the hybrid result's shape plus ``units``."""
    query = (query or "").strip()
    if not query:
        raise ValueError("empty search query")
    if embedder is None:
        raise SentenceServingUnavailable(
            "sentence serving needs an embedder: bind one "
            "with --embed (the connection layer's default "
            "embedder) — the sentence form has no lexical "
            "fallback")
    prepare = getattr(store, "prepare_sentence_index", None)
    snapshot = getattr(store, "sentence_snapshot", None)
    if prepare is None or snapshot is None:
        raise SentenceServingUnavailable(
            "this store serves no sentence index")
    notes: list[str] = []
    built = prepare(embedder)
    if built:
        notes.append(built)
    snap = snapshot()
    if snap is None:
        raise SentenceServingUnavailable(
            "this store serves no sentence index")
    sent_rows, sent_emb = snap
    sent_emb = np.asarray(sent_emb, dtype=np.float64)

    filt = store.batch_filter()
    fetch = int(fetch_k) if fetch_k else query_stack.CHUNK_FETCH_K
    qvec = np.asarray(embedder.embed([query])[0],
                      dtype=np.float64)

    # chunk front: lexical + vector arms, top chunks by
    # vector score — the chunk layer's population.
    toks = [t for t in query_stack.tokenize(query)
            if query_stack.is_content_token(t)]
    rows: dict[str, dict] = {}
    if toks:
        for r in store.search_chunks(
                text=" OR ".join(toks), filters=filt, k=fetch):
            rows[r["chunk_id"]] = r
    for r in store.search_chunks(
            vector=[float(x) for x in qvec], filters=filt,
            k=fetch):
        rows.setdefault(r["chunk_id"], r)
        rows[r["chunk_id"]] = {**rows[r["chunk_id"]], **r}
    cand_chunks = sorted(
        rows.values(),
        key=lambda r: -(r.get("vector_score", -2.0)))
    cand_chunks = cand_chunks[:_SENTENCE_CANDIDATE_CHUNKS]
    chunk_cos: dict[str, float] = {}
    chunk_texts: dict[str, str] = {}
    need = [c for c in cand_chunks if "vector_score" not in c]
    if need:
        es = embedder.embed([c["text"] for c in need])
        for c, e in zip(need, es, strict=True):
            chunk_cos[c["chunk_id"]] = _cosine(qvec, e)
    for c in cand_chunks:
        chunk_texts[c["chunk_id"]] = c["text"]
        if "vector_score" in c:
            chunk_cos[c["chunk_id"]] = float(c["vector_score"])

    shipped_emb = getattr(store, "_embeddings", None)

    def chunk_cos_of(cid: str) -> float:
        if shipped_emb and cid in shipped_emb:
            return _cosine(qvec, shipped_emb[cid])
        text = chunk_texts.get(cid)
        if text:
            return _cosine(qvec, embedder.embed([text])[0])
        return 0.0

    # direct arm: top sentences of the index by cosine
    sims = np.zeros(0)
    direct: list[dict] = []
    sent_by_key: dict[str, int] = {}
    for i, r in enumerate(sent_rows):
        sent_by_key.setdefault(_ws_norm(r["text"]), i)
    if sent_rows:
        enorm = np.linalg.norm(sent_emb, axis=1)
        enorm = np.where(enorm == 0, 1.0, enorm)
        qn = float(np.linalg.norm(qvec)) or 1.0
        sims = (sent_emb @ qvec) / (enorm * qn)
        for i in np.argsort(-sims)[:_DIRECT_TOP]:
            r = sent_rows[int(i)]
            direct.append({"text": r["text"],
                           "doc_id": r["doc_id"],
                           "chunk_id": r["chunk_id"],
                           "_cos": float(sims[int(i)])})

    # graph pool + beam channel
    triples = store.triples()
    ranked_triples = _rank_triples(
        triples, query, max(int(k), _POOL_TRIPLES),
        rel_weight=rel_weight)
    pool = [dict(u, via="graph", kind="sentence")
            for u in _quote_sentences(
                store, ranked_triples[:_POOL_TRIPLES],
                embedder, filt, chunk_cos, chunk_texts,
                cand_chunks, cap=10 ** 9)]
    linked, by_key = _link_entities(triples, query)
    beam = _beam_units(store, by_key, linked, qvec,
                       embedder, filt, chunk_cos, chunk_texts,
                       cand_chunks)

    cands: list[dict] = list(pool) + list(beam)
    seen = {_ws_norm(c["text"]) for c in cands}
    for d in direct:
        key = _ws_norm(d["text"])
        if key not in seen:
            seen.add(key)
            cands.append(dict(d, via="direct",
                              kind="sentence"))
    arms = ["sentences"]
    if pool:
        arms.append("graph-pool")
    if beam:
        arms.append("beam")
    # The name-only tag links names the way the
    # hybrid form does — find_entities over names AND
    # aliases — not by the beam channel's exact-span
    # linking: a query that drops a title ("Elizabeth
    # Bennet" for "Miss Elizabeth Bennet") must still
    # tag name-only matches, or the flag silently
    # clears on exactly the queries that need it.
    entities = store.find_entities(query, limit=5)
    display_triples, n_only = _tag_name_only(
        ranked_triples[:int(k)], query,
        [n for e in entities
         for n in [e["name"], *e.get("aliases", [])]])
    if n_only:
        notes.append(_name_only_note(
            n_only, len(display_triples)))
    base = {"query": query, "store": store.describe(),
            "arms": arms, "notes": notes,
            "entities": entities,
            "triples": display_triples}
    if not cands:
        return {**base, "units": [], "documents": []}

    # two-layer score + Z2 merge
    for c in cands:
        cid = c["chunk_id"]
        if cid and cid not in chunk_cos:
            chunk_cos[cid] = chunk_cos_of(cid)
    zpop = list(chunk_cos)
    sent_cos: list[float] = []
    for c in cands:
        if "_cos" in c:
            sent_cos.append(c["_cos"])
        else:
            i = sent_by_key.get(_ws_norm(c["text"]))
            if i is not None:
                sent_cos.append(float(sims[int(i)]))
            else:
                sent_cos.append(_cosine(
                    qvec, embedder.embed([c["text"]])[0]))
    if floor_cos is not None:
        # Relevance floor (serving-quality program):
        # candidates whose own cosine to the query
        # falls below the floor never enter the merge —
        # the fill stops instead of padding the budget
        # with the least-unrelated units. Withheld
        # counts are reported in the notes, never
        # silent. (z_chunk's population was fixed
        # above; it still includes floored-out chunks'
        # cosines — a normalization detail, not a
        # ranking input for survivors' order beyond
        # their own scores.)
        kept = [i for i, cs in enumerate(sent_cos)
                if cs >= floor_cos]
        withheld = len(cands) - len(kept)
        cands = [cands[i] for i in kept]
        sent_cos = [sent_cos[i] for i in kept]

        def _chunk_cos_of(c):
            if "vector_score" in c:
                return c["vector_score"]
            return chunk_cos.get(c["chunk_id"], 0.0)

        before = len(cand_chunks)
        cand_chunks = [c for c in cand_chunks
                       if _chunk_cos_of(c) >= floor_cos]
        withheld += before - len(cand_chunks)
        if withheld:
            notes.append(
                f"relevance floor (cosine >= "
                f"{floor_cos}): {withheld} candidate "
                "unit(s) below the floor withheld")
    z_sent = _zscores(sent_cos)
    z_chunk = dict(zip(zpop, _zscores(
        [chunk_cos[cid] for cid in zpop]), strict=True))
    for c, zs in zip(cands, z_sent, strict=True):
        c["score"] = zs + z_chunk.get(c["chunk_id"], 0.0)
    merged: list[dict] = [
        {"kind": "sentence", "text": c["text"],
         "doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
         "via": c["via"], "merge": 2.0 * c["score"]}
        for c in cands]
    merged += [
        {"kind": "chunk", "text": c["text"],
         "doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
         "via": "chunk",
         "merge": z_chunk.get(c["chunk_id"], 0.0)}
        for c in cand_chunks]
    merged.sort(key=lambda u: -u["merge"])
    head = merged[:_RERANK_POOL]

    # rerank step: reorders the head; the fill reads the
    # reranked order and nothing past the head.
    rr = reranker
    if rr is None:
        from .reranker import try_reranker
        rr = try_reranker()
    if rr is False:
        notes.append("rerank step off (disabled by the "
                     "caller) — units fill in merge order")
    elif rr is None:
        notes.append(
            "rerank step inactive: the local cross-encoder "
            "(the `rerank` extra) is not installed — units "
            "fill in merge order")
    else:
        scores = rr.score(query, [u["text"] for u in head])
        for u, sc in zip(head, scores, strict=True):
            u["merge"] = float(sc)
        head.sort(key=lambda u: -u["merge"])

    units = _fill_units(head, int(budget_tokens))
    by_doc: dict[str, dict] = {}
    for u in units:
        d = by_doc.setdefault(
            u["doc_id"], {"doc_id": u["doc_id"], "units": 0})
        d["units"] += 1
    return {**base, "units": units,
            "documents": list(by_doc.values())}


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
    if "units" in result:
        # sentence serving: the budgeted evidence units,
        # in fill order — what a reader actually receives
        lines.append("Evidence units")
        if result["units"]:
            for i, u in enumerate(result["units"], 1):
                lines.append(
                    f"  {i}. [{u['via']}/{u['kind']}] "
                    f"(score {u['score']}, {u['tokens']} "
                    f"tokens, {u['doc_id']})")
                excerpt = " ".join(u["text"].split())[:200]
                lines.append(f"     {excerpt}")
        else:
            lines.append("  (no unit matched)")
    else:
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
        description="Search a built graph: sentence-"
                    "level evidence serving by default "
                    "(direct sentence index + graph pool "
                    "+ beam traversal channel), with "
                    "chunk-level hybrid search as the "
                    "internal fallback")
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
        try:
            result = sentence_search(store, args.query,
                                     k=args.k,
                                     embedder=embedder)
        except SentenceServingUnavailable as e:
            # the named internal fallback: chunk serving
            # still answers, and the result says so
            result = hybrid_search(store, args.query,
                                   k=args.k,
                                   embedder=embedder)
            result["notes"] = [
                f"sentence serving unavailable ({e}); "
                f"fell back to chunk-level hybrid search",
                *result["notes"]]
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
