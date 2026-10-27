"""GraphStore — the production storage interface over Neo4j.

This is the only way anything touches
the graph. It wraps the existing Neo4jClient (pipeline/neo4j_store.py —
transport selection, tx vs Query API v2, stays inside that client)
and exposes the method set:

  upsert_entities / upsert_edges   idempotent MERGE writes
  neighborhood(ids, hops, tier)    tier-weighted traversal (§7 input)
  search_chunks(vector, text, …)   vector + full-text chunk retrieval
  project_from_registry(batch_id)  rebuild a batch's REL projection
                                   from its WRITE manifest (§4, E3)
  verify_counts(batch_id)          artifact recount vs live graph

plus the §6 (E5) serving-index projection:

  project_chunks(batch_id)         tier2_index.pkl → (:Chunk) nodes
                                   with embeddings, (:Document) nodes,
                                   HAS_CHUNK and MENTIONS edges
  ensure_serving(batch_id)         the serving contract in one
                                   call: ensure_schema + the
                                   batch's chunk projection when
                                   the store holds none for it —
                                   what the INDEX stage runs and
                                   what serving heals through, so
                                   a built graph answers search
                                   with no manual step

Schema (idempotent, ensure_schema): unique constraints on
Entity.entity_id / Chunk.chunk_id / Document.doc_id; a vector index
on Chunk.embedding (text-embedding-3-large, 3072d, cosine); Lucene
full-text indexes on Chunk.text and Entity.name.

Note: pipeline.registry.GraphStore is the legacy ABC (write_triples /
traverse) implemented by Neo4jGraphStore in neo4j_store.py; this class
is the deployable interface and supersedes it for new code. The
REL edge write semantics (attestation-level MERGE key — see
upsert_edges) are identical to Neo4jGraphStore's, so both
writers produce the same projection.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from .neo4j_store import Neo4jClient
from .registry import attestation_id, normalize_name
from .registry_store import RegistryStore

EMBED_DIMS = 3072  # text-embedding-3-large (§4)

VECTOR_INDEX = "chunk_embedding"
CHUNK_TEXT_INDEX = "chunk_text_ft"
ENTITY_NAME_INDEX = "entity_name_ft"

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS "
    "FOR (c:Chunk) ON (c.embedding) OPTIONS {indexConfig: "
    f"{{`vector.dimensions`: {EMBED_DIMS}, "
    "`vector.similarity_function`: 'cosine'}}",
    f"CREATE FULLTEXT INDEX {CHUNK_TEXT_INDEX} IF NOT EXISTS "
    "FOR (c:Chunk) ON EACH [c.text]",
    f"CREATE FULLTEXT INDEX {ENTITY_NAME_INDEX} IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name]",
]

_CHUNK_FILTER_KEYS = ("corpus", "batch_id", "doc_id", "tier", "tag")


def chunk_id_for(batch_id: str, doc_id: str, idx: int) -> str:
    """Deterministic, batch-scoped chunk id (E5)."""
    return f"{batch_id}:{doc_id}:{idx}"


class GraphStore:
    """§4 GraphStore over Neo4jClient (local Bolt/HTTP or Aura)."""

    def __init__(self, client: Neo4jClient | None = None,
                 registry_store: RegistryStore | None = None,
                 entity_registry=None):
        self.c = client or Neo4jClient()
        self.rs = registry_store or RegistryStore()
        # Optional EntityRegistry: when attached,
        # project_from_registry takes entity records (names, aliases,
        # corpus) from the registry — the source of truth — instead of
        # reading them back out of the graph being rebuilt.
        self.entity_registry = entity_registry

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def ensure_schema(self) -> list[dict]:
        """Create constraints + indexes (all IF NOT EXISTS; safe to
        re-run). Returns the post-state of the §4 indexes."""
        for stmt in SCHEMA_STATEMENTS:
            self.c.run([{"statement": stmt, "parameters": {}}])
        self.c.run([{"statement": "CALL db.awaitIndexes(300)",
                     "parameters": {}}])
        rows = self.c.run([{
            "statement":
                "SHOW INDEXES YIELD name, type, state, labelsOrTypes, "
                "properties WHERE name IN $names "
                "RETURN name, type, state, labelsOrTypes, properties",
            "parameters": {"names": [VECTOR_INDEX, CHUNK_TEXT_INDEX,
                                     ENTITY_NAME_INDEX,
                                     "entity_id_unique",
                                     "chunk_id_unique",
                                     "doc_id_unique"]}}])[0]
        return [{"name": r["row"][0], "type": r["row"][1],
                 "state": r["row"][2], "labels": r["row"][3],
                 "properties": r["row"][4]} for r in rows]

    def chunk_count(self, batch_id: str) -> int:
        """(:Chunk) nodes the store currently holds for one
        batch — the serving projection's presence check."""
        return int(self.c.one(
            "MATCH (c:Chunk {batch_id: $b}) RETURN count(c)",
            {"b": batch_id}) or 0)

    def ensure_serving(self, batch_id: str) -> dict:
        """The serving contract for one batch, in one call:
        the §4 schema exists (constraints + the vector index
        + both full-text indexes — ensure_schema, IF NOT
        EXISTS throughout) and the batch's chunk projection
        is in the store, projected from its recorded
        tier2_index.pkl when the store holds no chunks for
        it. Idempotent end to end: the schema statements are
        no-ops when present and the projection is MERGE-based
        with a presence check in front, so a second call over
        a served store writes nothing. No provider spend —
        the projection replays embeddings the INDEX stage
        already recorded. Returns {schema, chunks_before,
        projected (the project_chunks report, or None when
        the store already held the batch's chunks / the batch
        recorded no index), chunks}."""
        schema = self.ensure_schema()
        before = self.chunk_count(batch_id)
        projected = None
        if before == 0 and self.rs.tier2_index(batch_id) is not None:
            projected = self.project_chunks(batch_id)
        return {"schema": schema, "chunks_before": before,
                "projected": projected,
                "chunks": self.chunk_count(batch_id)}

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def upsert_entities(self, entities, corpus: str | None = None) -> int:
        """MERGE entities by entity_id. Accepts dicts
        {entity_id, name, label?, aliases?, corpus?} or
        pipeline.registry.Entity objects. Returns rows written."""
        rows = []
        for e in entities:
            if not isinstance(e, dict):
                e = {"entity_id": e.entity_id, "name": e.name,
                     "label": e.label, "aliases": e.aliases}
            rows.append({
                "id": e["entity_id"], "name": e["name"],
                "label": e.get("label") or "",
                "aliases": list(e.get("aliases") or []),
                "corpus": e.get("corpus") or corpus})
        n = 0
        for i in range(0, len(rows), 500):
            batch = rows[i:i + 500]
            n += self.c.one(
                "UNWIND $rows AS row "
                "MERGE (e:Entity {entity_id: row.id}) "
                "SET e.name = row.name, e.aliases = row.aliases, "
                "e.label = CASE WHEN row.label = '' THEN e.label "
                "ELSE row.label END, "
                "e.corpus = coalesce(row.corpus, e.corpus) "
                "RETURN count(e)",
                {"rows": batch}) or 0
        return int(n)

    def upsert_edges(self, triples) -> int:
        """MERGE REL edges at ATTESTATION level: the key is
        {type, batch_id, run_id, doc_id, att_id}, where att_id is
        the content-derived attestation identity
        (pipeline.registry.attestation_id over endpoints, doc, and —
        when the writer carries them — the extractor surfaces,
        quote, and model). A fact attested in N documents is N
        edges, each carrying its own doc_id, p_true, tier, and
        surfaces, so the store alone can answer "which documents
        attest this fact" and can be scored per attestation without
        a sidecar. The alternative — one fact edge plus ATTESTED_IN
        edges to Document nodes — was rejected: it adds a hop to
        every graph-arm traversal and splits provenance across two
        edge types, while attestation edges cost only the duplicate
        (fact, doc) rows themselves (+9.4% edges on the flagship
        projection: 18,710 vs 17,105). Writers without surface material (the
        pipeline) get att_id over (fact, doc) — their previous
        semantics for exact duplicates, which they dedupe upstream.

        Accepts dicts {subject_id, object_id, relation, batch_id,
        run_id, doc_id, p_true?, tier?, tag?, model?, gate?,
        surface_s?, surface_o?, quote?, valid_at?, invalid_at?} or
        pipeline.registry.Triple objects. Entities must already exist
        (registry-first). Returns rows actually written (MATCH
        drops unknown ids).

        valid_at / invalid_at (OSS borrow #6, Graphiti bi-temporal
        schema): nullable world-time validity of the fact. A null
        value removes the property (Cypher SET semantics), so edges
        written without them stay byte-identical to before; the
        fields are schema support only — no behavior change."""
        rows = []
        for t in triples:
            if not isinstance(t, dict):
                t = {"subject_id": t.subject_id, "object_id": t.object_id,
                     "relation": t.relation, "batch_id": t.batch_id,
                     "run_id": t.run_id, "doc_id": t.doc_id,
                     "p_true": t.p_true, "tier": t.tier, "tag": t.tag,
                     "model": t.model, "gate": t.gate,
                     "surface_s": getattr(t, "surface_s", None),
                     "surface_o": getattr(t, "surface_o", None),
                     "quote": getattr(t, "quote", None),
                     "valid_at": getattr(t, "valid_at", None),
                     "invalid_at": getattr(t, "invalid_at", None)}
            ss, so = t.get("surface_s"), t.get("surface_o")
            quote = t.get("quote")
            rows.append({
                "s": t["subject_id"], "o": t["object_id"],
                "rel": t["relation"], "batch_id": t["batch_id"],
                "run_id": t["run_id"], "doc_id": t.get("doc_id", ""),
                "att": attestation_id(
                    t["subject_id"], t["relation"], t["object_id"],
                    t.get("doc_id", ""), ss or "", so or "",
                    quote or "", t.get("model", "") or ""),
                "ss": ss, "so": so, "quote": quote,
                "p": t.get("p_true"), "tier": t.get("tier", 1),
                "tag": t.get("tag", "VERIFIED"),
                "model": t.get("model", ""), "gate": t.get("gate", ""),
                "va": t.get("valid_at"), "ia": t.get("invalid_at")})
        n = 0
        for i in range(0, len(rows), 500):
            batch = rows[i:i + 500]
            n += self.c.one(
                "UNWIND $rows AS row "
                "MATCH (a:Entity {entity_id: row.s}), "
                "(b:Entity {entity_id: row.o}) "
                "MERGE (a)-[r:REL {type: row.rel, "
                "batch_id: row.batch_id, run_id: row.run_id, "
                "doc_id: row.doc_id, att_id: row.att}]->(b) "
                "SET r.p_true = row.p, r.tier = row.tier, "
                "r.tag = row.tag, r.model = row.model, "
                "r.gate = row.gate, r.surface_s = row.ss, "
                "r.surface_o = row.so, r.quote = row.quote, "
                "r.valid_at = row.va, r.invalid_at = row.ia "
                "RETURN count(r)",
                {"rows": batch}) or 0
        return int(n)

    # ------------------------------------------------------------------
    # traversal
    # ------------------------------------------------------------------
    def neighborhood(self, entity_ids, hops: int = 1,
                     tier_filter=None, limit: int = 200) -> dict:
        """REL neighborhood of the seed entities (undirected), as
        {nodes: [{entity_id, name}], edges: [{from, to, type, tier,
        tag, p_true, batch_id, doc_id}]}. tier_filter = iterable of
        tier ints applied to every relationship on a path."""
        if not 1 <= int(hops) <= 4:
            raise ValueError("hops must be 1..4")
        params: dict = {"ids": list(entity_ids), "limit": int(limit)}
        conds = ["a.entity_id IN $ids"]
        if tier_filter:
            conds.append("all(x IN r WHERE x.tier IN $tiers)")
            params["tiers"] = [int(t) for t in tier_filter]
        rows = self.c.run([{
            "statement": (
                f"MATCH p=(a:Entity)-[r:REL*1..{int(hops)}]-(b:Entity) "
                "WHERE " + " AND ".join(conds) + " "
                "RETURN [n IN nodes(p) | n.entity_id], "
                "[n IN nodes(p) | n.name], "
                "[x IN r | startNode(x).entity_id], "
                "[x IN r | endNode(x).entity_id], "
                "[x IN r | x.type], [x IN r | x.tier], "
                "[x IN r | x.tag], [x IN r | x.p_true], "
                "[x IN r | x.batch_id], [x IN r | x.doc_id] "
                "LIMIT $limit"),
            "parameters": params}])[0]
        nodes: dict[str, dict] = {}
        edges: dict[tuple, dict] = {}
        for r in rows:
            ids, names = r["row"][0], r["row"][1]
            for eid, name in zip(ids, names):
                nodes.setdefault(eid, {"entity_id": eid, "name": name})
            for j, frm in enumerate(r["row"][2]):
                key = (frm, r["row"][4][j], r["row"][3][j],
                       r["row"][8][j], r["row"][9][j])
                edges[key] = {
                    "from": frm, "to": r["row"][3][j],
                    "type": r["row"][4][j], "tier": r["row"][5][j],
                    "tag": r["row"][6][j], "p_true": r["row"][7][j],
                    "batch_id": r["row"][8][j], "doc_id": r["row"][9][j]}
        return {"nodes": list(nodes.values()),
                "edges": list(edges.values())}

    # ------------------------------------------------------------------
    # chunk retrieval
    # ------------------------------------------------------------------
    def search_chunks(self, vector=None, text=None, filters=None,
                      k: int = 10) -> list[dict]:
        """Chunk retrieval over the §4 indexes. vector → cosine over
        Chunk.embedding; text → Lucene full-text over Chunk.text;
        both → union with summed scores (naive fusion; ranking
        research belongs to E6). filters may carry corpus, batch_id,
        doc_id, tier, tag. Rows: {chunk_id, doc_id, idx, tier, tag,
        text, score, vector_score?, text_score?}."""
        filters = {k2: v for k2, v in (filters or {}).items()
                   if k2 in _CHUNK_FILTER_KEYS}
        where = ""
        params: dict = {"k": int(k)}
        if filters:
            where = " WHERE " + " AND ".join(
                f"node.{k2} = ${k2}" for k2 in filters)
            params.update(filters)
        fetch = int(k) * 5 if filters else int(k)
        out: dict[str, dict] = {}
        if vector is not None:
            rows = self.c.run([{
                "statement": (
                    "CALL db.index.vector.queryNodes($index, $fetch, "
                    "$vec) YIELD node, score" + where +
                    " RETURN node.chunk_id, node.doc_id, node.idx, "
                    "node.tier, node.tag, node.text, score "
                    "ORDER BY score DESC LIMIT $k"),
                "parameters": {**params, "index": VECTOR_INDEX,
                               "fetch": fetch,
                               "vec": [float(x) for x in vector]}}])[0]
            for r in rows:
                row = r["row"]
                out[row[0]] = {"chunk_id": row[0], "doc_id": row[1],
                               "idx": row[2], "tier": row[3],
                               "tag": row[4], "text": row[5],
                               "score": row[6], "vector_score": row[6]}
        if text is not None:
            rows = self.c.run([{
                "statement": (
                    "CALL db.index.fulltext.queryNodes($index, $q) "
                    "YIELD node, score" + where +
                    " RETURN node.chunk_id, node.doc_id, node.idx, "
                    "node.tier, node.tag, node.text, score "
                    "ORDER BY score DESC LIMIT $k"),
                "parameters": {**params, "index": CHUNK_TEXT_INDEX,
                               "q": text, "fetch": fetch}}])[0]
            for r in rows:
                row = r["row"]
                if row[0] in out:
                    out[row[0]]["text_score"] = row[6]
                    out[row[0]]["score"] += row[6]
                else:
                    out[row[0]] = {"chunk_id": row[0], "doc_id": row[1],
                                   "idx": row[2], "tier": row[3],
                                   "tag": row[4], "text": row[5],
                                   "score": row[6], "text_score": row[6]}
        return sorted(out.values(), key=lambda d: -d["score"])[:k]

    # ------------------------------------------------------------------
    # E5: serving-index projection (pkl → Chunk nodes)
    # ------------------------------------------------------------------
    def project_chunks(self, batch_id: str, with_mentions: bool = True,
                       batch_size: int = 50) -> dict:
        """Load the batch's tier2_index.pkl into (:Chunk) nodes with
        embeddings + (:Document) nodes + HAS_CHUNK edges (§6).
        MENTIONS edges (chunk → entity) are derived from ner.jsonl:
        a mention surface attested verbatim in the chunk text and
        resolved to a graph entity by normalized name/alias match.
        Idempotent; batch-scoped by chunk_id + batch_id properties.
        Existing Entity/REL data is never altered."""
        idx = self.rs.tier2_index(batch_id)
        if idx is None:
            raise FileNotFoundError(
                f"{batch_id}: no tier2_index.pkl (INDEX stage output)")
        corpus = self.rs.corpus(batch_id)
        # One-graph store: the store's chunk projection
        # is the current build's — another batch's chunks are
        # cleared before this batch's are loaded, so a chunk
        # search can never range over a replaced graph's
        # documents. The graph stamp itself is WRITE's to set;
        # chunk projection does not move it.
        chunks_cleared = 0
        if isinstance(self.c, Neo4jClient):
            from .neo4j_store import clear_store_chunks
            chunks_cleared = clear_store_chunks(
                self.c, keep_batch_id=batch_id)["chunks"]
        meta, chunks, embs = idx["meta"], idx["chunks"], idx["embeddings"]

        # Documents: texts.jsonl metadata, else derived from pkl keys.
        docs = {d["doc_id"]: d["source_path"]
                for d in self.rs.documents(batch_id)}
        for m in meta:
            docs.setdefault(m["doc_id"], "")
        doc_rows = [{"doc_id": d, "source_path": s, "corpus": corpus}
                    for d, s in docs.items()]
        for i in range(0, len(doc_rows), 500):
            self.c.run([{
                "statement":
                    "UNWIND $rows AS row "
                    "MERGE (d:Document {doc_id: row.doc_id}) "
                    "SET d.source_path = row.source_path, "
                    "d.corpus = row.corpus",
                "parameters": {"rows": doc_rows[i:i + 500]}}])

        rows = [{
            "chunk_id": chunk_id_for(batch_id, m["doc_id"],
                                     int(m["chunk_idx"])),
            "doc_id": m["doc_id"], "idx": int(m["chunk_idx"]),
            "text": text, "tier": int(m.get("tier", 2)),
            "tag": m.get("tag", "UNVERIFIED"), "batch_id": batch_id,
            "corpus": corpus,
            "embedding": [float(x) for x in embs[i]],
        } for i, (m, text) in enumerate(zip(meta, chunks))]
        n_chunks = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            n_chunks += self.c.one(
                "UNWIND $rows AS row "
                "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
                "SET c.doc_id = row.doc_id, c.idx = row.idx, "
                "c.text = row.text, c.tier = row.tier, "
                "c.tag = row.tag, c.batch_id = row.batch_id, "
                "c.corpus = row.corpus, c.embedding = row.embedding "
                "WITH row, c "
                "MATCH (d:Document {doc_id: row.doc_id}) "
                "MERGE (d)-[:HAS_CHUNK]->(c) "
                "RETURN count(c)",
                {"rows": batch}) or 0

        result = {"batch_id": batch_id, "corpus": corpus,
                  "documents": len(doc_rows), "chunks": int(n_chunks),
                  "other_chunks_cleared": int(chunks_cleared),
                  "model": idx.get("model", ""),
                  "dims": int(embs.shape[1]) if len(embs) else 0,
                  "mentions": 0}
        if with_mentions:
            result["mentions"] = self._project_mentions(
                batch_id, meta, chunks)
        self.c.run([{"statement": "CALL db.awaitIndexes(300)",
                     "parameters": {}}])
        return result

    def _project_mentions(self, batch_id: str, meta, chunks) -> int:
        """MENTIONS projection (see project_chunks). Resolution map
        is built from the live graph's Entity names + aliases
        (normalized); a pair is written only when the mention surface
        is attested in the chunk text (case-sensitive, then
        case-insensitive containment)."""
        mentions = self.rs.ner_mentions(batch_id)
        if not mentions:
            return 0
        rows = self.c.run([{
            "statement": "MATCH (e:Entity) "
                         "RETURN e.entity_id, e.name, e.aliases",
            "parameters": {}}])[0]
        by_norm: dict[str, str] = {}
        for r in rows:  # canonical names win over aliases
            by_norm.setdefault(normalize_name(r["row"][1]), r["row"][0])
        for r in rows:
            for alias in r["row"][2] or []:
                by_norm.setdefault(normalize_name(alias), r["row"][0])

        by_doc: dict[str, list[tuple[str, str]]] = {}
        for i, m in enumerate(meta):
            by_doc.setdefault(m["doc_id"], []).append(
                (chunk_id_for(batch_id, m["doc_id"], int(m["chunk_idx"])),
                 chunks[i]))
        pairs: set[tuple[str, str]] = set()
        for doc_id, doc_chunks in by_doc.items():
            surfaces: dict[str, str] = {}  # entity_id → surface
            for ment in mentions.get(doc_id, []):
                eid = by_norm.get(normalize_name(ment["text"]))
                if eid:
                    surfaces.setdefault(eid, ment["text"])
            if not surfaces:
                continue
            for cid, text in doc_chunks:
                low = text.lower()
                for eid, surface in surfaces.items():
                    if surface in text or surface.lower() in low:
                        pairs.add((cid, eid))
        pair_rows = [{"chunk_id": c, "eid": e} for c, e in pairs]
        n = 0
        for i in range(0, len(pair_rows), 1000):
            n += self.c.one(
                "UNWIND $rows AS row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id}), "
                "(e:Entity {entity_id: row.eid}) "
                "MERGE (c)-[:MENTIONS]->(e) RETURN count(*)",
                {"rows": pair_rows[i:i + 1000]}) or 0
        return int(n)

    # ------------------------------------------------------------------
    # E3: projection rebuild + verification
    # ------------------------------------------------------------------
    def project_from_registry(self, batch_id: str,
                              id_remap: dict | None = None) -> dict:
        """Rebuild the batch's REL projection from its WRITE manifest
        (written_triples.jsonl): delete the batch's REL edges, then
        re-upsert entities and edges from the manifest. p_true is not
        in the manifest — it is carried over from the pre-existing
        edge when the identical edge is already in the graph, and
        left unset (counted) otherwise. Raises FileNotFoundError for
        pre-manifest batches (projection not reconcilable from
        artifacts — the stage_verify precedent).

        id_remap: {merged_away_id: surviving_id} — an
        ER settlement's fold, as recorded in the registry's
        id_mapping. Manifest endpoints resolve through the map
        (chains followed), and the p_true carry-over lookup keys
        are normalized through it too, so a triple a settlement
        re-pointed projects onto the survivor and inherits its
        pre-merge edge's p_true — the score is part of the
        attestation's evidence chain and must not be dropped."""
        manifest = self.rs.manifest_triples(batch_id)
        if manifest is None:
            raise FileNotFoundError(
                f"{batch_id}: no written_triples.jsonl "
                "(batch predates the WRITE manifest)")
        meta = self.rs.batch_meta(batch_id)
        corpus = meta.get("source") or batch_id
        run_id = meta.get("run_id", "")
        model = meta.get("config", {}).get("gate_model", "")

        # One-graph store: a projection asserts this
        # batch's graph as the store's graph. When the stamp
        # names a different graph, clear it first — the clear
        # keeps this batch's own edges, so the p_true
        # carry-over below still reads them — and report the
        # replacement; content no stamp identifies stops the
        # projection by name (check_store_write). The
        # registry keeps the replaced graph's full history.
        replaced = None
        if isinstance(self.c, Neo4jClient):
            from .neo4j_store import (
                batch_graph_identity,
                check_store_write,
                clear_store_graph,
            )
            _action, _current = check_store_write(
                self.c, batch_graph_identity(meta))
            if _action == "replace" and _current is not None:
                _removed = clear_store_graph(
                    self.c, keep_batch_id=batch_id)
                replaced = {
                    "previous_batch": _current["batch_id"],
                    "previous_pack": _current["pack_id"],
                    "previous_pack_version":
                        _current["pack_version"],
                    "previous_triples": _current["triples"],
                    "removed": _removed}

        remap = dict(id_remap or {})

        def _rid(eid: str) -> str:
            seen = set()
            while eid in remap and eid not in seen:
                seen.add(eid)
                eid = remap[eid]
            return eid

        ent_ids = sorted({_rid(t["subject_id"]) for t in manifest}
                         | {_rid(t["object_id"]) for t in manifest})
        # Entity records: the attached registry first (G-13 — the
        # source of truth, so a rebuild never depends on the graph it
        # is rebuilding), then the live graph (the Phase-1 registry
        # view); ids found nowhere fall back to an id-slug name.
        known: dict[str, dict] = {}
        from_registry = 0
        if self.entity_registry is not None:
            for eid in ent_ids:
                ent = self.entity_registry.get(eid)
                if ent is not None:
                    from_registry += 1
                    known[eid] = {
                        "entity_id": ent.entity_id, "name": ent.name,
                        "label": ent.label or "",
                        "aliases": list(ent.aliases or []),
                        "corpus": ent.corpus or None}
        missing_ids = [e for e in ent_ids if e not in known]
        for i in range(0, len(missing_ids), 1000):
            rows = self.c.run([{
                "statement":
                    "MATCH (e:Entity) WHERE e.entity_id IN $ids "
                    "RETURN e.entity_id, e.name, e.label, e.aliases",
                "parameters": {"ids": missing_ids[i:i + 1000]}}])[0]
            for r in rows:
                known[r["row"][0]] = {
                    "entity_id": r["row"][0], "name": r["row"][1],
                    "label": r["row"][2] or "",
                    "aliases": r["row"][3] or []}
        entities, missing_names = [], 0
        for eid in ent_ids:
            if eid in known:
                entities.append({**known[eid],
                                 "corpus": known[eid].get("corpus")
                                 or corpus})
            else:
                missing_names += 1
                slug = eid.split(":", 1)[-1].replace("_", " ")
                entities.append({"entity_id": eid, "name": slug,
                                 "label": "", "aliases": [],
                                 "corpus": corpus})
        # Carry p_true over from identical pre-existing edges
        # (attestation-level: keyed with doc_id, G-28, so one doc's
        # score never lands on another doc's attestation).
        prev: dict[tuple, float] = {}
        rows = self.c.run([{
            "statement":
                "MATCH (a:Entity)-[r:REL {batch_id: $b}]->(bb:Entity) "
                "RETURN a.entity_id, r.type, bb.entity_id, r.p_true, "
                "r.doc_id",
            "parameters": {"b": batch_id}}])[0]
        for r in rows:
            if r["row"][3] is not None:
                prev[(_rid(r["row"][0]), r["row"][1],
                      _rid(r["row"][2]),
                      r["row"][4] or "")] = r["row"][3]

        deleted = self._delete_batch_rels(batch_id)
        n_ent = self.upsert_entities(entities, corpus=corpus)
        triples, carried, unset = [], 0, 0
        for t in manifest:
            key = (_rid(t["subject_id"]), t["relation"],
                   _rid(t["object_id"]), t.get("doc_id", ""))
            p = prev.get(key)
            if p is None:
                unset += 1
            else:
                carried += 1
            triples.append({
                "subject_id": _rid(t["subject_id"]),
                "object_id": _rid(t["object_id"]),
                "relation": t["relation"], "batch_id": batch_id,
                "run_id": run_id, "doc_id": t.get("doc_id", ""),
                "p_true": p, "tier": 1, "tag": "VERIFIED",
                "model": model, "gate": "triple"})
        n_edges = self.upsert_edges(triples)
        # The projection is complete: (re)stamp the store
        # with this graph's identity, exactly as a
        # build's WRITE does.
        if isinstance(self.c, Neo4jClient):
            from .neo4j_store import (
                batch_graph_identity,
                store_inventory,
                write_graph_stamp,
            )
            _inv = store_inventory(self.c)
            write_graph_stamp(self.c, {
                **batch_graph_identity(meta),
                "triples": _inv["rel_edges"],
                "entities": _inv["entities"],
                "chunks": _inv["chunks"],
                "stamped_at": datetime.now(UTC).isoformat()})
        out = {"batch_id": batch_id, "entities": n_ent,
               "entity_names_from_registry": from_registry,
               "entity_names_from_graph": len(known) - from_registry,
               "entity_names_from_id_slug": missing_names,
               "rels_deleted": int(deleted or 0), "edges": n_edges,
               "p_true_carried": carried, "p_true_unset": unset}
        if replaced is not None:
            out["replaced"] = replaced
        return out

    def _delete_batch_rels(self, batch_id: str) -> int:
        rows = self.c.run([{
            "statement":
                "MATCH ()-[r:REL {batch_id: $b}]->() "
                "WITH collect(r) AS rs, count(r) AS n "
                "FOREACH (x IN rs | DELETE x) RETURN n",
            "parameters": {"b": batch_id}}])[0]
        return int(rows[0]["row"][0]) if rows else 0

    def verify_counts(self, batch_id: str) -> dict:
        """Recount the batch from its artifacts (RegistryStore) and
        from the live graph (batch-scoped, the migrate_to_aura
        precedent: REL relationships with this batch_id; entities =
        Entity nodes incident to those relationships), and compare.
        With a WRITE manifest the comparison is exact on the distinct
        (subject_id, type, object_id) edge set; without one, the
        artifact side reports routed Tier 1 surface counts only and
        match is None (not reconcilable — stage_verify precedent)."""
        graph_edges = self.c.one(
            "MATCH ()-[r:REL {batch_id: $b}]->() RETURN count(r)",
            {"b": batch_id}) or 0
        graph_entities = self.c.one(
            "MATCH (e:Entity) WHERE EXISTS { (e)-[r:REL]-() "
            "WHERE r.batch_id = $b } RETURN count(e)",
            {"b": batch_id}) or 0
        graph_triple_set = self.c.one(
            "MATCH (a:Entity)-[r:REL {batch_id: $b}]->(bb:Entity) "
            "WITH DISTINCT a.entity_id AS s, r.type AS t, "
            "bb.entity_id AS o RETURN count(*)",
            {"b": batch_id}) or 0
        out: dict = {
            "batch_id": batch_id,
            "graph": {"rel_edges": int(graph_edges),
                      "entities": int(graph_entities),
                      "distinct_triples": int(graph_triple_set)},
            "artifacts": {}, "match": None}
        manifest = self.rs.manifest_triples(batch_id)
        if manifest is not None:
            triples = {(t["subject_id"], t["relation"],
                        t["object_id"]) for t in manifest}
            ents = ({t["subject_id"] for t in manifest}
                    | {t["object_id"] for t in manifest})
            out["artifacts"] = {
                "manifest_rows": len(manifest),
                "distinct_triples": len(triples),
                "entities": len(ents)}
            out["match"] = (len(triples) == int(graph_triple_set)
                            and len(ents) == int(graph_entities))
        else:
            tier1 = self.rs.routed_rows(batch_id, tier=1)
            surfaces = ({r["subject"].lower() for r in tier1}
                        | {r["object"].lower() for r in tier1})
            out["artifacts"] = {
                "manifest_rows": None,
                "routed_tier1_triples": len(tier1),
                "surface_endpoints": len(surfaces),
                "note": "no written_triples.jsonl — canonical "
                        "entity/edge counts are not reconstructible "
                        "from surface artifacts (WRITE-time ER "
                        "mapping); graph side is the recount"}
        return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="§4 GraphStore / §6 chunk projection CLI")
    ap.add_argument("cmd", choices=["schema", "project-chunks",
                                    "project", "verify"])
    ap.add_argument("batch_id", nargs="?",
                    help="batch id (all commands except schema)")
    ap.add_argument("--no-mentions", action="store_true",
                    help="project-chunks: skip MENTIONS projection")
    args = ap.parse_args()
    gs = GraphStore()
    if args.cmd == "schema":
        print(json.dumps(gs.ensure_schema(), indent=1))
        return
    if not args.batch_id:
        ap.error(f"{args.cmd} requires a batch_id")
    if args.cmd == "project-chunks":
        print(json.dumps(gs.project_chunks(
            args.batch_id, with_mentions=not args.no_mentions),
            indent=1))
    elif args.cmd == "project":
        print(json.dumps(gs.project_from_registry(args.batch_id),
                         indent=1))
    elif args.cmd == "verify":
        print(json.dumps(gs.verify_counts(args.batch_id), indent=1))


if __name__ == "__main__":
    main()
