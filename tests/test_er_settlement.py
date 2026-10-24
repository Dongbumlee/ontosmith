"""ER review-band settlement tests.

The settlement surface: held pairs (er_decisions rows with
outcome='review') can be listed with their evidence, decided
by a human (merge / keep-separate — append-only decision
rows, the registry's own pattern), and applied: a merge
folds the pair in the registry (source of truth) and
`batch er-apply` re-projects the graph from the amended
WRITE manifest through the existing projection machinery,
carrying each re-pointed edge's p_true over so no triple
loses its evidence chain. A keep-separate decision is
honored by later WRITE runs: the pair is never re-held and
the ER gate is never re-consulted for it.
"""
import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from pipeline import batch_cli
from pipeline.batch import Batch, BatchConfig, batch_dir, save_batch
from pipeline.file_registry import SqliteEntityRegistry
from pipeline.registry import Entity

SURVIVOR = "ent:survivor"
LOSER = "ent:loser"


def _registry(tmp: Path) -> SqliteEntityRegistry:
    reg = SqliteEntityRegistry(tmp / "data" / "registry.db")
    reg.create(Entity(entity_id=SURVIVOR,
                      name="Hearthline Coffee Roasters",
                      label="Coffee Shop", batch_id="b1",
                      run_id="run-0", doc_id="d1", corpus="corpus"))
    reg.create(Entity(entity_id=LOSER, name="Hearthline",
                      label="Coffee Shop", batch_id="b1",
                      run_id="run-0", doc_id="d2", corpus="corpus"))
    return reg


def _held_pair(reg: SqliteEntityRegistry) -> None:
    reg.record_er_decision(
        "b1", "run-1", "Hearthline", SURVIVOR,
        "Hearthline Coffee Roasters", model="typesafe/jev-1.13",
        p_same=0.71, outcome="review")


class TestMergeEntities(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.reg = _registry(self.tmp)
        self.addCleanup(self.reg.close)
        _held_pair(self.reg)

    def test_fold_merges_aliases_records_decision_and_mapping(self):
        report = self.reg.merge_entities(
            SURVIVOR, LOSER, batch_id="b1", run_id="run-2",
            model="human", p_same=0.71,
            reason="human settlement of review pair #1")
        survivor = self.reg.get(SURVIVOR)
        self.assertIsNotNone(survivor)
        # pick_survivor convention: the fuller name is canonical,
        # the folded name becomes an alias.
        self.assertEqual(survivor.name, "Hearthline Coffee Roasters")
        self.assertIn("Hearthline", survivor.aliases)
        self.assertIsNone(self.reg.get(LOSER))
        # The folded name now resolves to the survivor.
        self.assertEqual(self.reg.resolve("Hearthline").entity_id,
                         SURVIVOR)
        self.assertEqual(self.reg.mapping(), {LOSER: SURVIVOR})
        self.assertEqual(report["merged_id"], LOSER)
        self.assertEqual(report["surviving_id"], SURVIVOR)
        rows = [r for r in self.reg.er_decisions()
                if r["outcome"] == "merged"
                and r["surviving_id"] == SURVIVOR
                and r["merged_name"] == "Hearthline"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "human")
        self.assertEqual(rows[0]["p_same"], 0.71)
        self.assertIn("human settlement", rows[0]["reason"])

    def test_review_pair_reports_settled_after_merge(self):
        pairs = self.reg.er_review_pairs("b1")
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(pairs[0]["settled"])
        self.reg.merge_entities(
            SURVIVOR, LOSER, batch_id="b1", run_id="run-2",
            model="human", p_same=0.71,
            reason="human settlement of review pair #1")
        pairs = self.reg.er_review_pairs("b1")
        self.assertEqual(pairs[0]["settled"]["outcome"], "merged")

    def test_unknown_and_same_id_refused(self):
        with self.assertRaises(KeyError):
            self.reg.merge_entities(SURVIVOR, "ent:nope")
        with self.assertRaises(KeyError):
            self.reg.merge_entities("ent:nope", LOSER)
        with self.assertRaises(ValueError):
            self.reg.merge_entities(SURVIVOR, SURVIVOR)


class TestSettledDecisionLookup(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.reg = _registry(self.tmp)
        self.addCleanup(self.reg.close)
        _held_pair(self.reg)

    def test_keep_separate_is_a_human_settlement(self):
        self.assertIsNone(
            self.reg.settled_er_decision("Hearthline", SURVIVOR))
        self.reg.record_er_decision(
            "b1", "run-2", "Hearthline", SURVIVOR,
            "Hearthline Coffee Roasters", model="human",
            outcome="kept_separate",
            reason="human settlement of review pair #1")
        settled = self.reg.settled_er_decision("Hearthline", SURVIVOR)
        self.assertIsNotNone(settled)
        self.assertEqual(settled["outcome"], "kept_separate")
        pairs = self.reg.er_review_pairs("b1")
        self.assertEqual(pairs[0]["settled"]["outcome"],
                         "kept_separate")

    def test_machine_rows_are_not_human_settlements(self):
        # A WRITE-time 'merged' row (no human-settlement reason)
        # must never be read back as a human decision.
        self.reg.record_er_decision(
            "b1", "run-2", "Hearthline", SURVIVOR,
            "Hearthline Coffee Roasters", model="",
            outcome="merged")
        self.assertIsNone(
            self.reg.settled_er_decision("Hearthline", SURVIVOR))


class _FakeGraph:
    """The stage_write graph double (upsert_entities /
    write_triples), same shape as the bindings tests'."""

    def __init__(self):
        self.entities = []
        self.triples = []

    def upsert_entities(self, records):
        self.entities.extend(records)
        return len(records)

    def write_triples(self, triples):
        self.triples.extend(triples)
        return len(triples)


class TestWriteHonorsKeepSeparate(unittest.TestCase):
    """A pair a human settled keep-separate is never re-held:
    a later batch's WRITE does not consult the ER gate for it
    and records no new review row; the mention becomes its
    own entity, as a held pair's mention does."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.reg = SqliteEntityRegistry(self.tmp / "registry.db")
        self.addCleanup(self.reg.close)
        self.reg.create(Entity(
            entity_id=SURVIVOR, name="Hearthline Coffee Roasters",
            label="Coffee Shop", batch_id="b0", run_id="run-0",
            doc_id="d1", corpus="corpus"))
        self.gate_calls = []

    def _gate(self, name_a, name_b, label, context, model=None):
        self.gate_calls.append((name_a, name_b))
        return 0.71, {"cost": 0.0}

    def _run_write(self):
        from pipeline import batch_run
        b = Batch(batch_id="b2", run_id="run-2", source="corpus",
                  collection="coll", doc_ids=["d2"],
                  config=BatchConfig(max_cost_usd=None))
        d = batch_dir(self.tmp, "b2")
        d.mkdir(parents=True, exist_ok=True)
        row = {"doc_id": "d2", "subject": "Hearthline",
               "relation": "buys from",
               "object": "Northbeam Trade",
               "context": "Hearthline buys from Northbeam Trade.",
               "tier": 1, "p_true": 0.9}
        (d / "routed.jsonl").write_text(json.dumps(row) + "\n")
        save_batch(b, self.tmp)
        graph = _FakeGraph()
        batch_run.stage_write(b, self.tmp, [], self.reg, graph,
                              er_decide_fn=self._gate)
        return b

    def _review_rows(self):
        return [r for r in self.reg.er_decisions()
                if r["outcome"] == "review"]

    def test_unsettled_pair_consults_gate_and_holds(self):
        self._run_write()
        self.assertEqual(self.gate_calls,
                         [("Hearthline", "Hearthline Coffee Roasters")])
        self.assertEqual(len(self._review_rows()), 1)

    def test_settled_pair_never_reheld(self):
        self.reg.record_er_decision(
            "b1", "run-1", "Hearthline", SURVIVOR,
            "Hearthline Coffee Roasters", model="human",
            outcome="kept_separate",
            reason="human settlement of review pair #1")
        self._run_write()
        self.assertEqual(self.gate_calls, [])
        self.assertEqual(self._review_rows(), [])
        # The mention still became its own entity (not merged).
        ent = self.reg.resolve("Hearthline")
        self.assertIsNotNone(ent)
        self.assertNotEqual(ent.entity_id, SURVIVOR)


class _FakeNeo4jClient:
    """In-memory Neo4jClient double for GraphStore: nodes +
    REL edges, dispatching on the statements the projection
    path issues."""

    def __init__(self, nodes, edges):
        self.nodes = {n["entity_id"]: dict(n) for n in nodes}
        self.edges = [dict(e) for e in edges]

    def run(self, statements, timeout=None):
        out = []
        for s in statements:
            stmt, params = s["statement"], s.get("parameters") or {}
            if "RETURN 1 AS ok" in stmt:
                out.append([{"row": [1]}])
            elif "WHERE e.entity_id IN $ids" in stmt:
                rows = [{"row": [eid, self.nodes[eid]["name"],
                                 self.nodes[eid].get("label", ""),
                                 self.nodes[eid].get("aliases", [])]}
                        for eid in params["ids"] if eid in self.nodes]
                out.append(rows)
            elif "RETURN a.entity_id, r.type, bb.entity_id" in stmt:
                rows = [{"row": [e["s"], e["rel"], e["o"],
                                 e.get("p"), e.get("doc_id", "")]}
                        for e in self.edges
                        if e["batch_id"] == params["b"]]
                out.append(rows)
            elif "collect(r) AS rs" in stmt:
                kept = [e for e in self.edges
                        if e["batch_id"] != params["b"]]
                n = len(self.edges) - len(kept)
                self.edges = kept
                out.append([{"row": [n]}])
            else:
                raise AssertionError(f"unexpected statement: {stmt}")
        return out

    def one(self, statement, parameters=None):
        params = parameters or {}
        if "MERGE (e:Entity" in statement:
            for row in params["rows"]:
                self.nodes[row["id"]] = {
                    "entity_id": row["id"], "name": row["name"],
                    "label": row.get("label") or "",
                    "aliases": row.get("aliases") or []}
            return len(params["rows"])
        if "MERGE (a)-[r:REL" in statement:
            n = 0
            for row in params["rows"]:
                if row["s"] in self.nodes and row["o"] in self.nodes:
                    self.edges.append({
                        "s": row["s"], "o": row["o"],
                        "rel": row["rel"], "batch_id": row["batch_id"],
                        "run_id": row["run_id"],
                        "doc_id": row["doc_id"], "p": row["p"],
                        "att": row["att"]})
                    n += 1
            return n
        raise AssertionError(f"unexpected one(): {statement}")


class TestProjectFromRegistryRemap(unittest.TestCase):
    """project_from_registry with an id_remap: manifest rows
    still naming a merged-away entity project onto the
    survivor, and a re-pointed edge inherits the p_true of
    its pre-merge self — the attestation's score is part of
    its evidence chain and must not be dropped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rs_root = self.tmp
        d = self.rs_root / "batches" / "b1"
        d.mkdir(parents=True)
        (d / "batch.json").write_text(json.dumps({
            "batch_id": "b1", "run_id": "run-1",
            "source": "corpus", "config": {"gate_model": "m"}}))
        manifest = [
            {"subject_id": LOSER, "relation": "buys_from",
             "surface_relation": "buys from",
             "object_id": "ent:supplier", "doc_id": "d2"},
            {"subject_id": SURVIVOR, "relation": "founded_by",
             "surface_relation": "founded by",
             "object_id": "ent:person", "doc_id": "d1"},
        ]
        (d / "written_triples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in manifest))
        reg = _registry(self.tmp / "regroot")
        self.addCleanup(reg.close)
        # The registry already folded the pair.
        reg.merge_entities(SURVIVOR, LOSER, batch_id="b1",
                           run_id="run-2", model="human",
                           p_same=0.71,
                           reason="human settlement of review pair #1")
        reg.create(Entity(entity_id="ent:supplier", name="Cascadia",
                          label="", batch_id="b1", doc_id="d2",
                          corpus="corpus"))
        reg.create(Entity(entity_id="ent:person", name="Mara",
                          label="", batch_id="b1", doc_id="d1",
                          corpus="corpus"))
        self.reg = reg

    def test_remap_repoints_and_carries_p_true(self):
        from pipeline.graph_store import GraphStore
        client = _FakeNeo4jClient(
            nodes=[{"entity_id": SURVIVOR,
                    "name": "Hearthline Coffee Roasters"},
                   {"entity_id": LOSER, "name": "Hearthline"},
                   {"entity_id": "ent:supplier", "name": "Cascadia"},
                   {"entity_id": "ent:person", "name": "Mara"}],
            edges=[{"s": LOSER, "o": "ent:supplier",
                    "rel": "buys_from", "batch_id": "b1",
                    "run_id": "run-1", "doc_id": "d2", "p": 0.87},
                   {"s": SURVIVOR, "o": "ent:person",
                    "rel": "founded_by", "batch_id": "b1",
                    "run_id": "run-1", "doc_id": "d1", "p": 0.93}])
        from pipeline.registry_store import RegistryStore
        gs = GraphStore(client=client,
                        registry_store=RegistryStore(self.rs_root),
                        entity_registry=self.reg)
        report = gs.project_from_registry(
            "b1", id_remap={LOSER: SURVIVOR})
        self.assertEqual(report["edges"], 2)
        self.assertEqual(report["p_true_carried"], 2)
        self.assertEqual(report["p_true_unset"], 0)
        by_rel = {e["rel"]: e for e in client.edges}
        self.assertEqual(by_rel["buys_from"]["s"], SURVIVOR)
        self.assertEqual(by_rel["buys_from"]["p"], 0.87)
        self.assertEqual(by_rel["founded_by"]["p"], 0.93)


# ----------------------------------------------------------------------
# CLI surface
# ----------------------------------------------------------------------

class TestErCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env = mock.patch.dict(
            os.environ, {"ONTOSMITH_ROOT": str(self.tmp)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.reg = _registry(self.tmp)
        self.addCleanup(self.reg.close)
        _held_pair(self.reg)
        b = Batch(batch_id="b1", run_id="run-1", source="corpus",
                  collection="coll", doc_ids=["d1", "d2"],
                  config=BatchConfig(max_cost_usd=None))
        save_batch(b, self.tmp)
        d = batch_dir(self.tmp, "b1")
        routed = [
            {"doc_id": "d1", "subject": "Hearthline Coffee Roasters",
             "relation": "founded by", "object": "Mara Ellison",
             "context": "Hearthline Coffee Roasters is a specialty "
                        "coffee shop founded by Mara Ellison.",
             "tier": 1, "p_true": 0.93},
            {"doc_id": "d2", "subject": "Hearthline",
             "relation": "buys from",
             "object": "Cascadia Coffee Importers",
             "context": "Hearthline buys from Cascadia Coffee "
                        "Importers and Northbeam.",
             "tier": 1, "p_true": 0.87},
        ]
        (d / "routed.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in routed))
        manifest = [
            {"subject_id": SURVIVOR, "relation": "founded_by",
             "surface_relation": "founded by",
             "object_id": "ent:person", "doc_id": "d1"},
            {"subject_id": LOSER, "relation": "buys_from",
             "surface_relation": "buys from",
             "object_id": "ent:supplier", "doc_id": "d2"},
        ]
        (d / "written_triples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in manifest))

    def _pair_id(self) -> int:
        import sqlite3
        con = sqlite3.connect(self.tmp / "data" / "registry.db")
        return con.execute(
            "SELECT id FROM er_decisions WHERE outcome='review'"
        ).fetchone()[0]

    def test_review_lists_pair_with_evidence(self):
        out = io.StringIO()
        with redirect_stdout(out):
            batch_cli.cmd_er_review(argparse.Namespace(
                batch_id="b1", all=False))
        text = out.getvalue()
        self.assertIn("Hearthline", text)
        self.assertIn("Hearthline Coffee Roasters", text)
        self.assertIn("0.71", text)
        self.assertIn("buys from Cascadia", text)  # sample context

    def test_decide_keep_separate_then_refuses_second(self):
        args = argparse.Namespace(
            batch_id="b1", pair=self._pair_id(), merge=False,
            keep_separate=True, by="tester", reason="")
        out = io.StringIO()
        with redirect_stdout(out):
            batch_cli.cmd_er_decide(args)
        settled = self.reg.settled_er_decision("Hearthline", SURVIVOR)
        self.assertEqual(settled["outcome"], "kept_separate")
        self.assertEqual(settled["model"], "tester")
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                batch_cli.cmd_er_decide(args)
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("already settled", err.getvalue())

    def test_decide_unknown_or_foreign_pair_refused(self):
        err = io.StringIO()
        with redirect_stderr(err):
            with self.assertRaises(SystemExit):
                batch_cli.cmd_er_decide(argparse.Namespace(
                    batch_id="b1", pair=999, merge=True,
                    keep_separate=False, by="tester", reason=""))
        self.assertIn("no held pair", err.getvalue())

    def test_decide_merge_folds_and_apply_reprojects(self):
        args = argparse.Namespace(
            batch_id="b1", pair=self._pair_id(), merge=True,
            keep_separate=False, by="tester", reason="")
        out = io.StringIO()
        with redirect_stdout(out):
            batch_cli.cmd_er_decide(args)
        self.assertIsNone(self.reg.get(LOSER))
        self.assertIn("Hearthline",
                      self.reg.get(SURVIVOR).aliases)

        captured = {}

        class _FakeGS:
            def __init__(self, **kw):
                self.c = _FakeNeo4jClient([], [])
                orig_run = self.c.run

                def run(statements, timeout=None):
                    stmt = statements[0]["statement"]
                    if "DETACH DELETE" in stmt:
                        captured.setdefault("deleted", []).append(
                            statements[0]["parameters"]["id"])
                        return [[{"row": []}]]
                    return orig_run(statements, timeout)
                self.c.run = run
                # Orphan check answers: no REL edges remain.
                self.c.one = lambda statement, parameters=None: 0

            def project_from_registry(self, batch_id, id_remap=None):
                captured["batch_id"] = batch_id
                captured["id_remap"] = id_remap
                return {"entities": 2, "edges": 2,
                        "p_true_carried": 2, "p_true_unset": 0,
                        "rels_deleted": 2}

        with mock.patch("pipeline.graph_store.GraphStore", _FakeGS):
            out = io.StringIO()
            with redirect_stdout(out):
                batch_cli.cmd_er_apply(argparse.Namespace(
                    batch_id="b1"))
        self.assertEqual(captured["id_remap"], {LOSER: SURVIVOR})
        self.assertEqual(captured["deleted"], [LOSER])
        # The manifest itself was amended: no row names the loser.
        rows = [json.loads(line) for line in
                (batch_dir(self.tmp, "b1")
                 / "written_triples.jsonl").read_text().splitlines()]
        self.assertTrue(all(
            LOSER not in (r["subject_id"], r["object_id"])
            for r in rows))
        # A second apply has nothing left to do.
        out = io.StringIO()
        with redirect_stdout(out):
            batch_cli.cmd_er_apply(argparse.Namespace(batch_id="b1"))
        self.assertIn("nothing to apply", out.getvalue())


if __name__ == "__main__":
    unittest.main()
