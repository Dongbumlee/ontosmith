"""One-graph store identity (RR-2 store ruling, 2026-10-23).

The Neo4j store holds exactly ONE graph — the
materialization of the workspace's completed, current
ontology, stamped at WRITE with its identity. A build
under a different ontology REPLACES the store's graph
(named in the preflight and at WRITE, recorded in the
batch metrics); a build never overwrites content no
stamp identifies; serving resolves the store's one
graph from the stamp, scope-free.

The store side runs against an in-memory subclass of
the real Neo4jClient (statement dispatch over a node /
edge / chunk model), so the production Cypher — stamp
reads and writes, scoped clears, projections, search —
is what executes; only the transport is fake.
"""
import io
import json
import pickle
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, batch_dir, save_batch
from pipeline.file_registry import SqliteEntityRegistry
from pipeline.graph_store import GraphStore as Section4GraphStore
from pipeline.neo4j_store import (
    GRAPH_STAMP_FIELDS,
    Neo4jClient,
    Neo4jGraphStore,
    StoreIdentityError,
    check_store_write,
    read_graph_stamp,
)
from pipeline.registry import Entity
from pipeline.registry_store import RegistryStore
from pipeline.search import (
    Neo4jServingStore,
    StoreError,
    hybrid_search,
)


# ----------------------------------------------------------------------
# the in-memory transport
# ----------------------------------------------------------------------
class _InMemoryNeo4jClient(Neo4jClient):
    """The real client's interface over an in-memory
    model. run() dispatches on the Cypher text; rows come
    back in the client's own shape ({"row": [...]})."""

    def __init__(self):
        self.url = "neo4j://in-memory:7687"
        self.transport = "tx"
        self.db = "neo4j"
        self.auth = ""
        self.entities: dict[str, dict] = {}
        self.rels: list[dict] = []
        self.chunks: dict[str, dict] = {}
        self.documents: dict[str, dict] = {}
        self.has_chunk: set[tuple[str, str]] = set()
        self.mentions: set[tuple[str, str]] = set()
        self.stamp: dict | None = None

    # -- transport -------------------------------------------------
    def run(self, statements, timeout=None):
        return [self._exec(s["statement"], s.get("parameters") or {})
                for s in statements]

    def _exec(self, stmt: str, p: dict) -> list[dict]:
        s = " ".join(stmt.split())
        # --- stamp -------------------------------------------------
        if "MATCH (g:OntoSmithGraph) RETURN" in s:
            if self.stamp is None:
                return []
            return [{"row": [self.stamp.get(f) for f in
                             GRAPH_STAMP_FIELDS]}]
        if "MERGE (g:OntoSmithGraph)" in s:
            self.stamp = {f: p.get(f) for f in GRAPH_STAMP_FIELDS}
            return []
        # --- housekeeping ------------------------------------------
        if s.startswith("RETURN 1"):
            return [{"row": [1]}]
        if "db.awaitIndexes" in s or "SHOW INDEXES" in s:
            return []
        if s.startswith(("CREATE ", "DROP ")):
            return []
        # --- writes --------------------------------------------------
        if "MERGE (e:Entity {entity_id: row.id})" in s:
            for row in p["rows"]:
                e = self.entities.setdefault(row["id"], {
                    "name": "", "label": "", "aliases": [],
                    "corpus": None})
                e["name"] = row["name"]
                e["aliases"] = list(row["aliases"] or [])
                if row["label"]:
                    e["label"] = row["label"]
                if row["corpus"] is not None:
                    e["corpus"] = row["corpus"]
            return [{"row": [len(p["rows"])]}]
        if "MERGE (a)-[r:REL" in s:
            n = 0
            for row in p["rows"]:
                key = (row["s"], row["rel"], row["o"],
                       row["batch_id"], row["run_id"],
                       row["doc_id"], row["att"])
                hit = next((r for r in self.rels
                            if (r["s"], r["type"], r["o"],
                                r["batch_id"], r["run_id"],
                                r["doc_id"], r["att_id"]) == key),
                           None)
                props = {"s": row["s"], "o": row["o"],
                         "type": row["rel"],
                         "batch_id": row["batch_id"],
                         "run_id": row["run_id"],
                         "doc_id": row["doc_id"],
                         "att_id": row["att"],
                         "p_true": row.get("p"),
                         "tier": row.get("tier"),
                         "tag": row.get("tag"),
                         "model": row.get("model"),
                         "gate": row.get("gate")}
                if hit is None:
                    self.rels.append(props)
                else:
                    hit.update(props)
                n += 1
            return [{"row": [n]}] if "RETURN count" in s else []
        if "MERGE (d:Document {doc_id: row.doc_id})" in s:
            for row in p["rows"]:
                self.documents[row["doc_id"]] = {
                    "source_path": row["source_path"],
                    "corpus": row["corpus"]}
            return []
        if "MERGE (c:Chunk {chunk_id: row.chunk_id})" in s:
            for row in p["rows"]:
                self.chunks[row["chunk_id"]] = {
                    "doc_id": row["doc_id"], "idx": row["idx"],
                    "text": row["text"], "tier": row["tier"],
                    "tag": row["tag"], "batch_id": row["batch_id"],
                    "corpus": row["corpus"],
                    "embedding": row["embedding"]}
                self.has_chunk.add((row["doc_id"], row["chunk_id"]))
            return [{"row": [len(p["rows"])]}]
        if "MERGE (c)-[:MENTIONS]->(e)" in s:
            n = 0
            for row in p["rows"]:
                if row["chunk_id"] in self.chunks \
                        and row["eid"] in self.entities:
                    self.mentions.add((row["chunk_id"], row["eid"]))
                    n += 1
            return [{"row": [n]}]
        # --- scoped clears (RR-2) ------------------------------------
        if "DELETE r" in s and "coalesce(r.batch_id" in s:
            keep = p["keep"]
            self.rels = [r for r in self.rels
                         if (r["batch_id"] or "") == keep]
            return []
        if "WITH collect(r) AS rs" in s:
            n = sum(1 for r in self.rels
                    if r["batch_id"] == p["b"])
            self.rels = [r for r in self.rels
                         if r["batch_id"] != p["b"]]
            return [{"row": [n]}]
        if "DETACH DELETE c" in s and "coalesce(c.batch_id" in s:
            keep = p["keep"]
            dead = {cid for cid, c in self.chunks.items()
                    if (c["batch_id"] or "") != keep}
            for cid in dead:
                del self.chunks[cid]
            self.has_chunk = {(d, c) for d, c in self.has_chunk
                              if c not in dead}
            self.mentions = {(c, e) for c, e in self.mentions
                             if c not in dead}
            return []
        if "DELETE d" in s and "HAS_CHUNK" in s:
            linked = {d for d, _ in self.has_chunk}
            self.documents = {d: v for d, v in
                              self.documents.items() if d in linked}
            return []
        if "DETACH DELETE e" in s and "MENTIONS" in s:
            touching = ({r["s"] for r in self.rels}
                        | {r["o"] for r in self.rels})
            mentioned = {e for _, e in self.mentions}
            self.entities = {eid: v for eid, v in
                             self.entities.items()
                             if eid in touching or eid in mentioned}
            return []
        # --- counts ----------------------------------------------------
        if "RETURN count(r)" in s and "REL" in s:
            if "{batch_id: $b}" in s:
                n = sum(1 for r in self.rels
                        if r["batch_id"] == p["b"])
            elif "coalesce(r.batch_id" in s:
                n = sum(1 for r in self.rels
                        if (r["batch_id"] or "") != p["keep"])
            else:
                n = len(self.rels)
            return [{"row": [n]}]
        if "RETURN count(e)" in s:
            if "WHERE EXISTS" in s:
                ids = ({r["s"] for r in self.rels
                        if r["batch_id"] == p["b"]}
                       | {r["o"] for r in self.rels
                          if r["batch_id"] == p["b"]})
                n = sum(1 for eid in ids if eid in self.entities)
            elif "NOT (e)-[:REL]-()" in s:
                touching = ({r["s"] for r in self.rels}
                            | {r["o"] for r in self.rels})
                mentioned = {e for _, e in self.mentions}
                n = sum(1 for eid in self.entities
                        if eid not in touching
                        and eid not in mentioned)
            else:
                n = len(self.entities)
            return [{"row": [n]}]
        if "RETURN count(c)" in s or "RETURN count(*)" in s:
            if "coalesce(c.batch_id" in s:
                n = sum(1 for c in self.chunks.values()
                        if (c["batch_id"] or "") != p["keep"])
            elif "WITH DISTINCT" in s:
                n = len({(r["s"], r["type"], r["o"])
                         for r in self.rels
                         if r["batch_id"] == p["b"]})
            else:
                n = len(self.chunks)
            return [{"row": [n]}]
        if "RETURN count(d)" in s:
            if "HAS_CHUNK" in s:
                linked = {d for d, _ in self.has_chunk}
                n = sum(1 for d in self.documents
                        if d not in linked)
            else:
                n = len(self.documents)
            return [{"row": [n]}]
        # --- reads -----------------------------------------------------
        if "RETURN a.entity_id, bb.entity_id" in s:
            return [{"row": [r["s"], r["o"]]} for r in self.rels
                    if r["batch_id"] == p["b"]]
        if "RETURN a.entity_id, r.type, bb.entity_id, r.p_true" in s:
            return [{"row": [r["s"], r["type"], r["o"],
                             r["p_true"], r["doc_id"]]}
                    for r in self.rels if r["batch_id"] == p["b"]]
        if "WHERE e.entity_id IN $ids" in s:
            return [{"row": [eid, self.entities[eid]["name"],
                             self.entities[eid]["label"],
                             self.entities[eid]["aliases"]]}
                    for eid in p["ids"] if eid in self.entities]
        if "MATCH (e:Entity) RETURN e.entity_id, e.name, e.aliases" in s:
            return [{"row": [eid, e["name"], e["aliases"]]}
                    for eid, e in self.entities.items()]
        if "MATCH p=(a:Entity)-[r:REL" in s:
            seeds = set(p["ids"])
            rows = []
            for r in self.rels:
                if r["s"] in seeds or r["o"] in seeds:
                    rows.append({"row": [
                        [r["s"], r["o"]],
                        [self.entities[r["s"]]["name"],
                         self.entities[r["o"]]["name"]],
                        [r["s"]], [r["o"]], [r["type"]],
                        [r["tier"]], [r["tag"]], [r["p_true"]],
                        [r["batch_id"]], [r["doc_id"]]]})
            return rows[:p.get("limit", 200)]
        if "db.index.fulltext.queryNodes" in s:
            toks = set(re.findall(r"[a-z0-9]+", p["q"].lower()))
            scored = []
            for cid, c in self.chunks.items():
                if not self._filters_ok(c, p):
                    continue
                words = re.findall(r"[a-z0-9]+", c["text"].lower())
                score = sum(1 for w in words if w in toks)
                if score:
                    scored.append((score, cid, c))
            scored.sort(key=lambda t: -t[0])
            return [{"row": [cid, c["doc_id"], c["idx"],
                             c["tier"], c["tag"], c["text"], sc]}
                    for sc, cid, c in scored[:p.get("k", 10)]]
        if "db.index.vector.queryNodes" in s:
            import math
            vec = p["vec"]
            scored = []
            for cid, c in self.chunks.items():
                emb = c.get("embedding")
                if not emb or not self._filters_ok(c, p):
                    continue
                dot = sum(a * b for a, b in
                          zip(vec, emb, strict=True))
                na = math.sqrt(sum(a * a for a in vec)) or 1.0
                nb = math.sqrt(sum(b * b for b in emb)) or 1.0
                scored.append((dot / (na * nb), cid, c))
            scored.sort(key=lambda t: -t[0])
            return [{"row": [cid, c["doc_id"], c["idx"],
                             c["tier"], c["tag"], c["text"], sc]}
                    for sc, cid, c in scored[:p.get("k", 10)]]
        raise AssertionError(f"unhandled Cypher in fake: {s[:120]}")

    @staticmethod
    def _filters_ok(chunk: dict, p: dict) -> bool:
        for key in ("batch_id", "doc_id", "tier", "tag", "corpus"):
            if key in p and chunk.get(key) != p[key]:
                return False
        return True


# ----------------------------------------------------------------------
# build driver — stage_write against the fake store, the
# test_er_settlement shape (seeded artifacts, real registry)
# ----------------------------------------------------------------------
PACK_A = {"pack_id": "core", "pack_version": "0.4.0",
          "pack_payload_sha256": "sha256:aaaa"}
PACK_B = {"pack_id": "legal-contract", "pack_version": "0.2.0",
          "pack_payload_sha256": "sha256:bbbb"}

ROWS_A = [
    {"doc_id": "doc-a", "subject": "Alder & Finch",
     "relation": "founded by", "object": "Mara Alder",
     "context": "Alder & Finch is a ceramics studio founded "
                "by Mara Alder.",
     "tier": 1, "p_true": 0.95},
    {"doc_id": "doc-a", "subject": "Alder & Finch",
     "relation": "based in", "object": "Lisbon",
     "context": "Alder & Finch is based in Lisbon.",
     "tier": 1, "p_true": 0.9},
]
ROWS_B = [
    {"doc_id": "doc-b", "subject": "Northbeam Trade",
     "relation": "led by", "object": "Priya Nair",
     "context": "Northbeam Trade is led by Priya Nair.",
     "tier": 1, "p_true": 0.93},
    {"doc_id": "doc-b", "subject": "Northbeam Trade",
     "relation": "supplies", "object": "Hearthline",
     "context": "Northbeam Trade supplies Hearthline.",
     "tier": 1, "p_true": 0.88},
]
TEXT_A = ("Alder & Finch is a ceramics studio in Lisbon "
          "founded by Mara Alder. The studio glazes every "
          "stoneware piece by hand in small batches.")
TEXT_B = ("Northbeam Trade is led by Priya Nair and "
          "supplies Hearthline with green coffee beans "
          "under a multi-year agreement.")


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "data").mkdir(parents=True)
        self.client = _InMemoryNeo4jClient()
        self.registry = SqliteEntityRegistry(
            self.root / "data" / "registry.db")
        self.addCleanup(self.registry.close)

    def _build(self, batch_id, pack, rows, text):
        b = Batch(batch_id=batch_id, run_id="run-1",
                  source="corpus", collection="docs",
                  doc_ids=[rows[0]["doc_id"]],
                  config=BatchConfig(max_cost_usd=None),
                  pack_provenance=dict(pack))
        d = batch_dir(self.root, batch_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "routed.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        (d / "texts.jsonl").write_text(json.dumps({
            "doc_id": rows[0]["doc_id"], "text": text,
            "source_path": rows[0]["doc_id"] + ".txt"}) + "\n")
        (d / "ner.jsonl").write_text(json.dumps({
            "doc_id": rows[0]["doc_id"],
            "entities": [{"text": rows[0]["subject"],
                          "label": "organization"},
                         {"text": rows[0]["object"],
                          "label": "person"}]}) + "\n")
        save_batch(b, self.root)
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run.stage_write(b, self.root, [], self.registry,
                                  Neo4jGraphStore(self.client))
        return b, out.getvalue()

    def _project_chunks(self, batch_id, text):
        d = batch_dir(self.root, batch_id)
        meta = [{"doc_id": batch_id.replace("batch-", "doc-"),
                 "chunk_idx": 0, "tag": "UNVERIFIED", "tier": 2,
                 "batch_id": batch_id}]
        with open(d / "tier2_index.pkl", "wb") as f:
            pickle.dump({"embeddings": np.array([[1.0, 0.0]]),
                         "meta": meta, "chunks": [text],
                         "model": "openai/text-embedding-3-large"}, f)
        gs = Section4GraphStore(
            client=self.client,
            registry_store=RegistryStore(self.root))
        return gs.project_chunks(batch_id)

    def _serving(self, **kw):
        gs = Section4GraphStore(
            client=self.client,
            registry_store=RegistryStore(self.root))
        return Neo4jServingStore(root=self.root, graph_store=gs,
                                 **kw)


class TestCheckStoreWrite(unittest.TestCase):
    def setUp(self):
        self.client = _InMemoryNeo4jClient()

    def test_empty_store_is_fresh(self):
        action, current = check_store_write(
            self.client, {"batch_id": "b1",
                          "pack_payload_sha256": "sha256:aaaa",
                          "profile_hash": ""})
        self.assertEqual(action, "fresh")
        self.assertIsNone(current)

    def test_unstamped_content_stops_by_name(self):
        self.client.entities["ent:x"] = {
            "name": "X", "label": "", "aliases": [], "corpus": None}
        self.client.rels.append({
            "s": "ent:x", "o": "ent:x", "type": "links",
            "batch_id": "other", "run_id": "r", "doc_id": "d",
            "att_id": "a", "p_true": 0.5, "tier": 1,
            "tag": "VERIFIED", "model": "", "gate": ""})
        with self.assertRaises(StoreIdentityError) as cm:
            check_store_write(self.client, {"batch_id": "b1"})
        msg = str(cm.exception)
        self.assertIn("1 REL edges", msg)
        self.assertIn("no OntoSmith graph stamp identifies", msg)
        self.assertIn("MATCH (n) DETACH DELETE n", msg)

    def test_same_and_replace_verdicts(self):
        from pipeline.neo4j_store import write_graph_stamp
        write_graph_stamp(self.client, {
            "batch_id": "batch-a", "run_id": "run-1",
            "pack_id": "core", "pack_version": "0.4.0",
            "pack_payload_sha256": "sha256:aaaa",
            "triples": 2, "entities": 3, "chunks": 1})
        action, current = check_store_write(self.client, {
            "batch_id": "batch-a",
            "pack_payload_sha256": "sha256:aaaa",
            "profile_hash": ""})
        self.assertEqual(action, "same")
        action, current = check_store_write(self.client, {
            "batch_id": "batch-b",
            "pack_payload_sha256": "sha256:bbbb",
            "profile_hash": ""})
        self.assertEqual(action, "replace")
        self.assertEqual(current["batch_id"], "batch-a")
        self.assertEqual(current["pack_id"], "core")


if __name__ == "__main__":
    unittest.main()


class TestTwoBuildsOneStore(_StoreCase):
    """The RR-2 closing scenario: two builds under two
    different ontologies into ONE store — the store ends
    holding only the second graph, and search answers
    only from it."""

    def test_second_build_replaces_first(self):
        self._build("batch-a", PACK_A, ROWS_A, TEXT_A)
        stamp = read_graph_stamp(self.client)
        self.assertEqual(stamp["batch_id"], "batch-a")
        self.assertEqual(stamp["pack_id"], "core")
        self.assertEqual(stamp["triples"], 2)
        self._project_chunks("batch-a", TEXT_A)

        # Serving over build A: its content answers.
        store = self._serving()
        self.assertEqual(store.describe()["batch_id"], "batch-a")
        hits = hybrid_search(store, "Alder ceramics")
        self.assertTrue(any(
            d["doc_id"] == "doc-a" for d in hits["documents"]))

        # The preflight for build B names build A's graph
        # before anything is spent.
        b_b = Batch(batch_id="batch-b", run_id="run-1",
                    source="corpus", collection="docs",
                    doc_ids=["doc-b"],
                    config=BatchConfig(max_cost_usd=None),
                    pack_provenance=dict(PACK_B))
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run._preflight_store(
                b_b, [], Neo4jGraphStore(self.client))
        notice = out.getvalue()
        self.assertIn("currently holds", notice)
        self.assertIn("pack 'core' 0.4.0", notice)
        self.assertIn("batch-a", notice)
        self.assertIn("2 triples", notice)
        self.assertIn("replaces it at WRITE", notice)

        # Build B replaces: named in the WRITE output and
        # recorded in the batch metrics.
        b_b, write_out = self._build("batch-b", PACK_B,
                                     ROWS_B, TEXT_B)
        self.assertIn("replacing the store's current graph",
                      write_out)
        self.assertIn("pack 'core' 0.4.0", write_out)
        replaced = b_b.metrics["write"]["store_replaced"]
        self.assertEqual(replaced["previous_batch"], "batch-a")
        self.assertEqual(replaced["previous_pack"], "core")
        self.assertEqual(replaced["previous_triples"], 2)
        self.assertEqual(replaced["removed"]["rel_edges"], 2)

        # The store now holds ONLY build B's graph.
        self.assertTrue(self.client.rels)
        self.assertEqual(
            {r["batch_id"] for r in self.client.rels}, {"batch-b"})
        names = {e["name"] for e in self.client.entities.values()}
        self.assertEqual(names, {"Northbeam Trade", "Priya Nair",
                                 "Hearthline"})
        stamp = read_graph_stamp(self.client)
        self.assertEqual(stamp["batch_id"], "batch-b")
        self.assertEqual(stamp["pack_id"], "legal-contract")

        # WRITE's replacement already cleared build A's
        # chunks from the store.
        self.assertEqual(self.client.chunks, {})
        # Chunk projection enforces the same one-graph
        # rule: an out-of-order projection of the replaced
        # build's chunks is itself replaced by the current
        # build's projection.
        self._project_chunks("batch-a", TEXT_A)
        self.assertEqual(
            {c["batch_id"] for c in self.client.chunks.values()},
            {"batch-a"})
        res = self._project_chunks("batch-b", TEXT_B)
        self.assertEqual(res["other_chunks_cleared"], 1)
        self.assertEqual(
            {c["batch_id"] for c in self.client.chunks.values()},
            {"batch-b"})

        # Search over the replaced store returns only the
        # current graph's content — no scope given, none
        # needed.
        store = self._serving()
        desc = store.describe()
        self.assertEqual(desc["batch_id"], "batch-b")
        self.assertEqual(desc["graph_stamp"]["pack_id"],
                         "legal-contract")
        hits = hybrid_search(store, "Northbeam coffee")
        self.assertTrue(any(
            d["doc_id"] == "doc-b" for d in hits["documents"]))
        hits = hybrid_search(store, "Alder ceramics Lisbon")
        self.assertEqual(hits["documents"], [])
        self.assertEqual(hits["triples"], [])

    def test_stale_batch_name_refuses_by_name(self):
        self._build("batch-a", PACK_A, ROWS_A, TEXT_A)
        self._build("batch-b", PACK_B, ROWS_B, TEXT_B)
        with self.assertRaises(StoreError) as cm:
            self._serving(source=self.root / "batches" / "batch-a")
        msg = str(cm.exception)
        self.assertIn("'batch-a'", msg)
        self.assertIn("not the Neo4j store's current graph", msg)
        self.assertIn("batch-b", msg)

    def test_same_build_rewrite_is_idempotent(self):
        self._build("batch-a", PACK_A, ROWS_A, TEXT_A)
        rels_before = len(self.client.rels)
        ents_before = len(self.client.entities)
        b_a2, out = self._build("batch-a", PACK_A, ROWS_A, TEXT_A)
        self.assertNotIn("replacing the store's current graph", out)
        self.assertNotIn("store_replaced", b_a2.metrics["write"])
        self.assertEqual(len(self.client.rels), rels_before)
        self.assertEqual(len(self.client.entities), ents_before)
        gs = Section4GraphStore(
            client=self.client,
            registry_store=RegistryStore(self.root))
        counts = gs.verify_counts("batch-a")
        self.assertTrue(counts["match"], counts)

    def test_project_from_registry_replaces(self):
        self._build("batch-a", PACK_A, ROWS_A, TEXT_A)
        self._build("batch-b", PACK_B, ROWS_B, TEXT_B)
        gs = Section4GraphStore(
            client=self.client,
            registry_store=RegistryStore(self.root),
            entity_registry=self.registry)
        out = gs.project_from_registry("batch-a")
        self.assertEqual(out["replaced"]["previous_batch"],
                         "batch-b")
        self.assertEqual(
            {r["batch_id"] for r in self.client.rels}, {"batch-a"})
        # No p_true carry-over: replacement deleted build
        # A's edges, and the manifest never carried scores
        # (the projection contract) — the re-projected
        # edges are unset, counted honestly in the report.
        self.assertEqual(out["p_true_unset"], 2)
        stamp = read_graph_stamp(self.client)
        self.assertEqual(stamp["batch_id"], "batch-a")


class TestServingRefusals(_StoreCase):
    def test_empty_store_refuses_by_name(self):
        with self.assertRaises(StoreError) as cm:
            self._serving()
        self.assertIn("holds no graph yet", str(cm.exception))

    def test_unstamped_store_refuses_by_name(self):
        self.client.entities["ent:x"] = {
            "name": "X", "label": "", "aliases": [], "corpus": None}
        self.client.rels.append({
            "s": "ent:x", "o": "ent:x", "type": "links",
            "batch_id": "other", "run_id": "r", "doc_id": "d",
            "att_id": "a", "p_true": 0.5, "tier": 1,
            "tag": "VERIFIED", "model": "", "gate": ""})
        with self.assertRaises(StoreError) as cm:
            self._serving()
        self.assertIn("no OntoSmith graph stamp identifies",
                      str(cm.exception))

    def test_preflight_stops_on_unstamped_store(self):
        self.client.rels.append({
            "s": "ent:x", "o": "ent:x", "type": "links",
            "batch_id": "other", "run_id": "r", "doc_id": "d",
            "att_id": "a", "p_true": 0.5, "tier": 1,
            "tag": "VERIFIED", "model": "", "gate": ""})
        b = Batch(batch_id="batch-b", run_id="run-1",
                  source="corpus", collection="docs",
                  doc_ids=["doc-b"],
                  config=BatchConfig(max_cost_usd=None),
                  pack_provenance=dict(PACK_B))
        with self.assertRaises(StoreIdentityError):
            batch_run._preflight_store(
                b, [], Neo4jGraphStore(self.client))
        # Nothing was overwritten.
        self.assertEqual(len(self.client.rels), 1)


class _PlainGraph:
    """stage_write's graph double (the test_er_settlement
    shape): no Neo4j client, so the identity gate must
    stay entirely out of its way."""

    def __init__(self):
        self.entities = []
        self.triples = []

    def upsert_entities(self, records):
        self.entities.extend(records)
        return len(records)

    def write_triples(self, triples):
        self.triples.extend(triples)
        return len(triples)


class TestWriteHoldNamesRemedy(unittest.TestCase):
    """RR-6: the WRITE hold line names the er-review
    command, the way er-review names er-decide."""

    def test_hold_line_names_er_review(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        reg = SqliteEntityRegistry(root / "registry.db")
        self.addCleanup(reg.close)
        reg.create(Entity(
            entity_id="ent:survivor",
            name="Hearthline Coffee Roasters",
            label="Coffee Shop", batch_id="b0", run_id="run-0",
            doc_id="d1", corpus="corpus"))

        def gate(name_a, name_b, label, context, model=None):
            return 0.71, {"cost": 0.0}

        b = Batch(batch_id="b2", run_id="run-2", source="corpus",
                  collection="coll", doc_ids=["d2"],
                  config=BatchConfig(max_cost_usd=None))
        d = batch_dir(root, "b2")
        d.mkdir(parents=True, exist_ok=True)
        row = {"doc_id": "d2", "subject": "Hearthline",
               "relation": "buys from",
               "object": "Northbeam Trade",
               "context": "Hearthline buys from Northbeam Trade.",
               "tier": 1, "p_true": 0.9}
        (d / "routed.jsonl").write_text(json.dumps(row) + "\n")
        save_batch(b, root)
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run.stage_write(b, root, [], reg, _PlainGraph(),
                                  er_decide_fn=gate)
        text = out.getvalue()
        self.assertEqual(b.metrics["write"]["er_review_pairs"], 1)
        self.assertIn("held for human settlement", text)
        self.assertIn("ontosmith batch er-review b2", text)
