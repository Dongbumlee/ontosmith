"""Negative findings in the graph (0.2.2, freeze item 1).

0.2.1 materialised pack-sweep absences as citable records
(negative_findings.jsonl) readable only through the store
layer. 0.2.2 projects them into Neo4j as (:NegativeFinding)
nodes — first-class, flagged absence records, never REL
edges — at WRITE and on a projection rebuild, and serving
reads them back from the graph, so "no such clause" is
answered from the store's current graph itself.

The store side runs against an in-memory subclass of the
real Neo4jClient (statement dispatch over a node / edge /
finding model), so the production Cypher is what executes;
only the transport is fake. No live Neo4j, no live models.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, batch_dir, save_batch
from pipeline.file_registry import SqliteEntityRegistry
from pipeline.graph_store import GraphStore
from pipeline.neo4j_store import (
    GRAPH_STAMP_FIELDS,
    Neo4jClient,
    Neo4jGraphStore,
    batch_graph_identity,
    clear_store_graph,
    clear_store_negative_findings,
    write_graph_stamp,
)
from pipeline.registry_store import RegistryStore
from pipeline.search import FileServingStore, Neo4jServingStore

FINDINGS = [
    {"kind": "negative_finding",
     "finding_id": "nf-doc-1-cap-on-liability",
     "batch_id": "b1", "run_id": "run-1", "doc_id": "doc-1",
     "subject": "Services Agreement",
     "category": "Cap On Liability", "status": "absent",
     "method": "clause-llm"},
    {"kind": "negative_finding",
     "finding_id": "nf-doc-1-parties",
     "batch_id": "b1", "run_id": "run-1", "doc_id": "doc-1",
     "subject": "Services Agreement", "category": "Parties",
     "status": "absent", "method": "clause-llm"},
]

ROUTED_ROW = {"doc_id": "doc-1", "subject": "Services Agreement",
              "relation": "Governing Law",
              "object": "State of New York",
              "context": "This agreement is governed by the "
                         "laws of the State of New York.",
              "tier": 1, "p_true": 0.99}

MANIFEST_ROW = {"subject_id": "ent:agreement",
                "relation": "governed_by",
                "surface_relation": "Governing Law",
                "object_id": "ent:new-york", "doc_id": "doc-1"}


# ----------------------------------------------------------------------
# the in-memory transport
# ----------------------------------------------------------------------
class _InMemoryClient(Neo4jClient):
    """The real client's interface over an in-memory model
    (the test_store_identity pattern): run() dispatches on
    the Cypher text; rows come back in the client's own
    shape ({"row": [...]})."""

    def __init__(self):
        self.url = "neo4j://in-memory:7687"
        self.transport = "tx"
        self.db = "neo4j"
        self.auth = ""
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.findings: dict[str, dict] = {}
        self.stamp: dict | None = None

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
                e = self.nodes.setdefault(row["id"], {
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
                if row["s"] not in self.nodes \
                        or row["o"] not in self.nodes:
                    continue
                key = (row["s"], row["rel"], row["o"],
                       row["batch_id"], row["run_id"],
                       row["doc_id"], row["att"])
                hit = next((r for r in self.edges
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
                         "p_true": row.get("p")}
                if hit is None:
                    self.edges.append(props)
                else:
                    hit.update(props)
                n += 1
            return [{"row": [n]}] if "RETURN count" in s else []
        if "MERGE (f:NegativeFinding" in s:
            for row in p["rows"]:
                self.findings[row["finding_id"]] = {
                    "kind": row["kind"],
                    "batch_id": row["batch_id"],
                    "run_id": row["run_id"],
                    "doc_id": row["doc_id"],
                    "subject": row["subject"],
                    "category": row["category"],
                    "status": row["status"],
                    "method": row["method"]}
            return [{"row": [len(p["rows"])]}]
        # --- deletes -------------------------------------------------
        if "WITH collect(r) AS rs" in s:
            n = sum(1 for r in self.edges
                    if r["batch_id"] == p["b"])
            self.edges = [r for r in self.edges
                          if r["batch_id"] != p["b"]]
            return [{"row": [n]}]
        if "DELETE f" in s and "{batch_id: $b}" in s:
            self.findings = {fid: f for fid, f in
                             self.findings.items()
                             if f["batch_id"] != p["b"]}
            return []
        if "DELETE f" in s and "coalesce(f.batch_id" in s:
            keep = p["keep"]
            self.findings = {fid: f for fid, f in
                             self.findings.items()
                             if (f["batch_id"] or "") == keep}
            return []
        if "DELETE r" in s and "coalesce(r.batch_id" in s:
            keep = p["keep"]
            self.edges = [r for r in self.edges
                          if (r["batch_id"] or "") == keep]
            return []
        if "DETACH DELETE e" in s:
            touching = ({r["s"] for r in self.edges}
                        | {r["o"] for r in self.edges})
            self.nodes = {eid: v for eid, v in self.nodes.items()
                          if eid in touching}
            return []
        # --- counts ----------------------------------------------------
        if "RETURN count(r)" in s:
            if "coalesce(r.batch_id" in s:
                n = sum(1 for r in self.edges
                        if (r["batch_id"] or "") != p["keep"])
            elif "{batch_id: $b}" in s:
                n = sum(1 for r in self.edges
                        if r["batch_id"] == p["b"])
            else:
                n = len(self.edges)
            return [{"row": [n]}]
        if "RETURN count(e)" in s:
            if "NOT (e)-[:REL]-()" in s:
                touching = ({r["s"] for r in self.edges}
                            | {r["o"] for r in self.edges})
                n = sum(1 for eid in self.nodes
                        if eid not in touching)
            else:
                n = len(self.nodes)
            return [{"row": [n]}]
        if "RETURN count(c)" in s:
            return [{"row": [0]}]
        if "RETURN count(d)" in s:
            return [{"row": [0]}]
        if "RETURN count(f)" in s:
            if "coalesce(f.batch_id" in s:
                n = sum(1 for f in self.findings.values()
                        if (f["batch_id"] or "") != p["keep"])
            else:
                n = sum(1 for f in self.findings.values()
                        if f["batch_id"] == p["b"])
            return [{"row": [n]}]
        # --- reads -----------------------------------------------------
        if "WHERE e.entity_id IN $ids" in s:
            return [{"row": [eid, self.nodes[eid]["name"],
                             self.nodes[eid]["label"],
                             self.nodes[eid]["aliases"]]}
                    for eid in p["ids"] if eid in self.nodes]
        if "RETURN a.entity_id, r.type, bb.entity_id" in s:
            return [{"row": [r["s"], r["type"], r["o"],
                             r["p_true"], r["doc_id"]]}
                    for r in self.edges
                    if r["batch_id"] == p["b"]]
        if "RETURN f.finding_id" in s:
            rows = sorted(
                (fid, f) for fid, f in self.findings.items()
                if f["batch_id"] == p["b"])
            return [{"row": [fid, f["kind"], f["batch_id"],
                             f["run_id"], f["doc_id"],
                             f["subject"], f["category"],
                             f["status"], f["method"]]}
                    for fid, f in rows]
        raise AssertionError(f"unhandled Cypher in fake: {s[:120]}")


# ----------------------------------------------------------------------
# artifact fixtures
# ----------------------------------------------------------------------
def _batch_artifacts(root: Path, batch_id: str = "b1",
                     findings=FINDINGS, manifest=True) -> Path:
    d = root / "batches" / batch_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "batch.json").write_text(json.dumps({
        "batch_id": batch_id, "run_id": "run-1",
        "source": "cuad", "collection": "docs",
        "doc_ids": ["doc-1"], "config": {"gate_model": "m"}}))
    (d / "routed.jsonl").write_text(json.dumps(ROUTED_ROW) + "\n")
    if manifest:
        (d / "written_triples.jsonl").write_text(
            json.dumps(MANIFEST_ROW) + "\n")
    if findings is not None:
        (d / "negative_findings.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in findings))
    return d


class _Case(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.client = _InMemoryClient()
        self.gs = GraphStore(
            client=self.client,
            registry_store=RegistryStore(self.root))


class TestProjection(_Case):
    def test_project_from_registry_projects_findings(self):
        _batch_artifacts(self.root)
        report = self.gs.project_from_registry("b1")
        self.assertEqual(report["edges"], 1)
        self.assertEqual(report["negative_findings"], 2)
        # The absences are nodes, not edges: exactly the
        # manifest's one REL exists, and two finding nodes.
        self.assertEqual(len(self.client.edges), 1)
        self.assertEqual(len(self.client.findings), 2)
        # Read back from the graph, in the records' shape.
        got = self.gs.negative_findings("b1")
        self.assertEqual(got, sorted(
            FINDINGS, key=lambda f: f["finding_id"]))
        for f in got:
            self.assertEqual(f["kind"], "negative_finding")
            self.assertEqual(f["status"], "absent")

    def test_reprojection_replaces_the_stale_set(self):
        _batch_artifacts(self.root)
        self.gs.project_from_registry("b1")
        _batch_artifacts(self.root, findings=FINDINGS[:1])
        report = self.gs.project_from_registry("b1")
        self.assertEqual(report["negative_findings"], 1)
        got = self.gs.negative_findings("b1")
        self.assertEqual([f["finding_id"] for f in got],
                         ["nf-doc-1-cap-on-liability"])
        self.assertEqual(len(self.client.findings), 1)

    def test_batch_without_findings_projects_zero(self):
        _batch_artifacts(self.root, findings=None)
        self.assertEqual(
            self.gs.project_negative_findings("b1"), 0)
        self.assertEqual(self.gs.negative_findings("b1"), [])
        self.assertEqual(self.client.findings, {})


class TestClear(_Case):
    def _seed(self):
        for fid, batch in (("nf-a", "b-old"), ("nf-b", "b-old"),
                           ("nf-c", "b-new")):
            self.client.findings[fid] = {
                "kind": "negative_finding", "batch_id": batch,
                "run_id": "run-1", "doc_id": "doc-1",
                "subject": "S", "category": "C",
                "status": "absent", "method": "clause-llm"}

    def test_clear_store_negative_findings_keeps_batch(self):
        self._seed()
        n = clear_store_negative_findings(
            self.client, keep_batch_id="b-new")
        self.assertEqual(n, 2)
        self.assertEqual(list(self.client.findings), ["nf-c"])

    def test_clear_store_graph_takes_findings_with_it(self):
        self._seed()
        removed = clear_store_graph(
            self.client, keep_batch_id="b-new")
        self.assertEqual(removed["negative_findings"], 2)
        self.assertEqual(list(self.client.findings), ["nf-c"])


class TestStageWriteProjects(_Case):
    """The build's own WRITE projects the findings, over
    the same client the triples are written through."""

    def test_write_projects_findings(self):
        (self.root / "data").mkdir(parents=True)
        registry = SqliteEntityRegistry(
            self.root / "data" / "registry.db")
        self.addCleanup(registry.close)
        b = Batch(batch_id="b1", run_id="run-1", source="cuad",
                  collection="docs", doc_ids=["doc-1"],
                  config=BatchConfig(max_cost_usd=None))
        d = batch_dir(self.root, "b1")
        d.mkdir(parents=True, exist_ok=True)
        (d / "routed.jsonl").write_text(
            json.dumps(ROUTED_ROW) + "\n")
        (d / "texts.jsonl").write_text(json.dumps({
            "doc_id": "doc-1",
            "text": "Services Agreement. This agreement is "
                    "governed by the laws of the State of "
                    "New York.",
            "source_path": "contract.txt"}) + "\n")
        (d / "negative_findings.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in FINDINGS))
        save_batch(b, self.root)
        batch_run.stage_write(b, self.root, [], registry,
                              Neo4jGraphStore(self.client))
        self.assertEqual(len(self.client.findings), 2)
        for f in self.client.findings.values():
            self.assertEqual(f["batch_id"], "b1")
            self.assertEqual(f["status"], "absent")
        self.assertEqual(self.client.stamp["batch_id"], "b1")


class TestServingReadPath(_Case):
    def test_file_store_reads_findings_from_artifacts(self):
        d = _batch_artifacts(self.root)
        store = FileServingStore(d, root=self.root)
        self.assertEqual(store.negative_findings(), FINDINGS)
        # Absences never enter the triple records.
        self.assertEqual(len(store.triples()), 1)

    def test_neo4j_store_artifact_path_reads_findings(self):
        d = _batch_artifacts(self.root)

        class _StubGraphStore:
            pass

        store = Neo4jServingStore(
            source=d, root=self.root,
            graph_store=_StubGraphStore())
        self.assertEqual(store.negative_findings(), FINDINGS)

    def test_neo4j_store_reads_findings_from_the_graph(self):
        # The artifact carries ONE finding; the graph was
        # projected with TWO. Serving must answer from the
        # graph — the store's current state — not from the
        # artifact the build left behind.
        d = _batch_artifacts(self.root, findings=FINDINGS)
        self.gs.project_negative_findings("b1")
        # The artifact then diverges (one finding): the
        # graph's projected set is the one serving reads.
        _batch_artifacts(self.root, findings=FINDINGS[:1])
        meta = json.loads((d / "batch.json").read_text())
        write_graph_stamp(self.client, {
            **batch_graph_identity(meta),
            "triples": 1, "entities": 2, "chunks": 0,
            "stamped_at": "2026-11-20T00:00:00+00:00"})
        store = Neo4jServingStore(root=self.root,
                                  graph_store=self.gs)
        got = store.negative_findings()
        self.assertEqual(got, sorted(
            FINDINGS, key=lambda f: f["finding_id"]))


if __name__ == "__main__":
    unittest.main()
