"""Drift spot audit (G-23): pipeline.drift_check + the VERIFY hook.

Regression for the installed-copy defect: stage_verify used to
load the audit implementation from scripts/drift_check.py by
repo-relative file path. The wheel ships no scripts/ tree, so
any installed batch run with a manifest and a graph store died
at VERIFY with FileNotFoundError. The implementation now lives
in the package as pipeline.drift_check; these tests exercise it
the way an installed copy runs it — module import, manifest
join against a workspace registry, store reads through the
store's own client, and the stage_verify hook end to end
against a fake store (no Neo4j server, no API calls).

Also guarded here: the workspace-resolution discipline the fix
follows (state root, never the package tree) for the sibling
surfaces aligned in the same pass — RegistryStore's default
root, docred's data dir, gate_cli's results dir.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import drift_check
from pipeline.batch import Batch, BatchConfig, batch_dir
from pipeline.batch_run import stage_verify
from pipeline.file_registry import SqliteEntityRegistry
from pipeline.registry import Entity

KEY = ("ent-s", "supplies", "ent-o", "doc-1")


class FakeClient:
    """Neo4jClient-shaped stub: run() dispatches on the statement.

    edges: (subject_id, type, object_id, doc_id, p_true, tier,
    att_id) rows, as the store would return them.
    """

    def __init__(self, edges):
        self.edges = list(edges)

    def run(self, statements):
        stmt = statements[0]["statement"]
        if "r.att_id" in stmt:  # the drift audit's store read
            rows = [{"row": list(e)} for e in self.edges]
        elif "r.type" in stmt:  # VERIFY check 5: edge set
            rows = [{"row": [e[0], e[1], e[2]]} for e in self.edges]
        else:  # VERIFY check 1: endpoint pairs
            rows = [{"row": [e[0], e[2]]} for e in self.edges]
        return [rows]

    def one(self, statement, parameters=None):
        return 0


class FakeGraph:
    def __init__(self, edges):
        self.c = FakeClient(edges)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class DriftWorkspace(unittest.TestCase):
    """A tmp state root: registry + one batch with one Tier-1 row."""

    batch_id = "drift-test"
    stages = ("INGEST", "NER", "CANDIDATES", "GATES", "ROUTE",
              "WRITE", "INDEX")

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.registry = SqliteEntityRegistry(
            self.root / "data" / "registry.db")
        self.registry.create(Entity(
            entity_id="ent-s", name="Acme Corp", label="organization",
            corpus="corp"))
        self.registry.create(Entity(
            entity_id="ent-o", name="Globex", label="organization",
            corpus="corp"))
        d = batch_dir(self.root, self.batch_id)
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"source": "corp"}))
        _write_jsonl(d / "routed.jsonl", [{
            "subject": "Acme Corp", "object": "Globex",
            "relation": "supplies", "doc_id": "doc-1",
            "tier": 1, "p_true": 0.9}])
        _write_jsonl(d / "written_triples.jsonl", [{
            "subject_id": "ent-s", "relation": "supplies",
            "object_id": "ent-o", "doc_id": "doc-1",
            "surface_relation": "supplies"}])
        _write_jsonl(d / "review_queue.jsonl", [])

    def make_batch(self) -> Batch:
        return Batch(
            batch_id=self.batch_id, run_id="run-1", source="corp",
            collection="docs", doc_ids=["doc-1"],
            config=BatchConfig(),
            metrics={"route": {"tier1": 1, "tier2": 0, "dropped": 0}})

    def clean_edges(self):
        return [("ent-s", "supplies", "ent-o", "doc-1", 0.9, 1, "att-1")]


class TestAuditCore(unittest.TestCase):
    def test_clean(self):
        exp = {KEY: {"p_true": 0.9, "tier": 1}}
        rep = drift_check.audit(exp, dict(exp), sample_n=200)
        self.assertTrue(rep["drift_free"])
        self.assertEqual(rep["expected_total"], 1)
        self.assertEqual(rep["sampled_expected"], 1)

    def test_missing_and_extra(self):
        exp = {KEY: {"p_true": 0.9, "tier": 1}}
        other = ("ent-x", "supplies", "ent-o", "doc-1")
        rep = drift_check.audit(exp, {other: {"p_true": 0.9, "tier": 1}},
                                sample_n=200)
        self.assertFalse(rep["drift_free"])
        self.assertEqual(rep["missing"], 1)
        self.assertEqual(rep["extra"], 1)
        self.assertEqual(rep["missing_sample"], [list(KEY)])

    def test_property_drift_tolerance_and_tier(self):
        exp = {KEY: {"p_true": 0.9, "tier": 1}}
        within = {KEY: {"p_true": 0.9 + 5e-10, "tier": 1}}
        self.assertTrue(drift_check.audit(exp, within)["drift_free"])
        moved = {KEY: {"p_true": 0.9 + 1e-6, "tier": 1}}
        rep = drift_check.audit(exp, moved)
        self.assertEqual(rep["property_drift"], 1)
        self.assertEqual(rep["property_drift_sample"][0]["field"],
                         "p_true")
        retiered = {KEY: {"p_true": 0.9, "tier": 2}}
        rep = drift_check.audit(exp, retiered)
        self.assertEqual(rep["property_drift"], 1)
        self.assertEqual(rep["property_drift_sample"][0]["field"],
                         "tier")

    def test_none_properties_are_never_drift(self):
        exp = {KEY: {"p_true": None, "tier": None}}
        act = {KEY: {"p_true": 0.1, "tier": 2}}
        self.assertTrue(drift_check.audit(exp, act)["drift_free"])

    def test_sampling_is_deterministic_and_capped(self):
        exp = {(f"s{i}", "rel", f"o{i}", "d"): {"p_true": 0.5,
                                                "tier": 1}
               for i in range(50)}
        rep1 = drift_check.audit(exp, dict(exp), sample_n=10)
        rep2 = drift_check.audit(exp, dict(exp), sample_n=10)
        self.assertEqual(rep1, rep2)
        self.assertEqual(rep1["sampled_expected"], 10)
        self.assertEqual(rep1["expected_total"], 50)


class TestExpectedFromManifest(DriftWorkspace):
    def test_join(self):
        exp, join = drift_check.expected_from_manifest(
            self.batch_id, root=self.root, registry=self.registry)
        self.assertEqual(exp, {KEY: {"p_true": 0.9, "tier": 1}})
        self.assertEqual(join, {"manifest_rows": 1,
                                "attestation_keys": 1, "joined": 1,
                                "ambiguous": 0, "existence_only": 0,
                                "corpus": "corp",
                                "registry": str(self.root / "data"
                                                / "registry.db")})

    def test_corpus_from_batch_json_not_state(self):
        # Run 32 (audit F2): batch.json's `source` is the
        # authoritative corpus. A state.json without `source`
        # (everything save_state wrote before run 32) must not
        # zero the corpus and void the join.
        d = batch_dir(self.root, self.batch_id)
        (d / "state.json").write_text(json.dumps({}))
        (d / "batch.json").write_text(json.dumps(
            {"source": "corp"}))
        exp, join = drift_check.expected_from_manifest(
            self.batch_id, root=self.root, registry=self.registry)
        self.assertEqual(join["corpus"], "corp")
        self.assertEqual(join["joined"], 1)
        self.assertEqual(exp, {KEY: {"p_true": 0.9, "tier": 1}})

    def test_batch_dir_registry_wins_over_default(self):
        # Run 32 (audit F2): with no handle passed, a registry
        # file in the batch's own directory is resolved before
        # the state-root default — the default here is a fresh
        # registry that knows none of the batch's entities, so
        # the pre-fix resolution joined 0.
        d = batch_dir(self.root, self.batch_id)
        own = SqliteEntityRegistry(d / "registry_own.db")
        own.create(Entity(entity_id="ent-s", name="Acme Corp",
                          label="organization", corpus="corp"))
        own.create(Entity(entity_id="ent-o", name="Globex",
                          label="organization", corpus="corp"))
        exp, join = drift_check.expected_from_manifest(
            self.batch_id, root=self.root)
        self.assertEqual(join["joined"], 1)
        self.assertEqual(join["registry"],
                         str(d / "registry_own.db"))
        self.assertEqual(exp, {KEY: {"p_true": 0.9, "tier": 1}})

    def test_ambiguous_join_stays_existence_only(self):
        d = batch_dir(self.root, self.batch_id)
        rows = [json.loads(line)
                for line in open(d / "routed.jsonl")]
        rows.append(dict(rows[0], p_true=0.7))  # a namesake candidate
        _write_jsonl(d / "routed.jsonl", rows)
        exp, join = drift_check.expected_from_manifest(
            self.batch_id, root=self.root, registry=self.registry)
        self.assertEqual(exp[KEY], {"p_true": None, "tier": None})
        self.assertEqual(join["joined"], 0)
        self.assertEqual(join["ambiguous"], 1)
        self.assertEqual(join["existence_only"], 1)


class TestActualFromStore(unittest.TestCase):
    def test_row_shaping(self):
        client = FakeClient([
            ("ent-s", "supplies", "ent-o", "doc-1", 0.9, 1, "att-1"),
            ("ent-s", "supplies", "ent-o", "doc-2", None, None, "")])
        act = drift_check.actual_from_store("b", client=client)
        self.assertEqual(act[KEY], {"p_true": 0.9, "tier": 1})
        self.assertEqual(act[("ent-s", "supplies", "ent-o", "doc-2")],
                         {"p_true": None, "tier": None})

    def test_attestation_keys(self):
        client = FakeClient([
            ("ent-s", "supplies", "ent-o", "doc-1", 0.9, 1, "att-1")])
        act = drift_check.actual_from_store(
            "b", client=client, attestation_keys=True)
        self.assertIn((*KEY, "att-1"), act)


class TestVerifyHook(DriftWorkspace):
    def test_stage_verify_runs_drift_audit_clean(self):
        b = self.make_batch()
        stage_verify(b, self.root, list(self.stages), self.registry,
                     FakeGraph(self.clean_edges()))
        checks = b.metrics["verify"]["checks"]
        audit = checks["drift_spot_audit"]
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["sample_checked"], 1)
        self.assertEqual(audit["missing"], 0)
        self.assertEqual(audit["extra"], 0)
        self.assertEqual(audit["property_drift"], 0)
        self.assertEqual(audit["join"]["joined"], 1)
        report = json.loads(
            (batch_dir(self.root, self.batch_id)
             / "verify_report.json").read_text())
        self.assertTrue(report["all_pass"])

    def test_stage_verify_flags_property_drift(self):
        drifted = [("ent-s", "supplies", "ent-o", "doc-1", 0.5, 1,
                    "att-1")]
        b = self.make_batch()
        with self.assertRaises(RuntimeError):
            stage_verify(b, self.root, list(self.stages), self.registry,
                         FakeGraph(drifted))
        report = json.loads(
            (batch_dir(self.root, self.batch_id)
             / "verify_report.json").read_text())
        audit = report["checks"]["drift_spot_audit"]
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["property_drift"], 1)

    def test_stage_verify_zero_join_blocks_verified(self):
        # Run 32 guard (audit F2): the routed surfaces resolve in
        # a corpus the registry does not hold, so the property
        # join is 0 — the drift check must FAIL (VERIFIED
        # blocked), never pass on key existence alone. This is
        # the BioRED Stage D failure shape in miniature.
        d = batch_dir(self.root, self.batch_id)
        _write_jsonl(d / "routed.jsonl", [{
            "subject": "Unknown Co", "object": "Neverheard Ltd",
            "relation": "supplies", "doc_id": "doc-1",
            "tier": 1, "p_true": 0.9}])
        b = self.make_batch()
        with self.assertRaises(RuntimeError):
            stage_verify(b, self.root, list(self.stages),
                         self.registry, FakeGraph(self.clean_edges()))
        report = json.loads((d / "verify_report.json").read_text())
        audit = report["checks"]["drift_spot_audit"]
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["join"]["joined"], 0)
        self.assertIn("joined 0", audit["note"])
        self.assertFalse(report["all_pass"])

    def test_stage_verify_without_manifest_skips(self):
        (batch_dir(self.root, self.batch_id)
         / "written_triples.jsonl").unlink()
        b = self.make_batch()
        # Check 5/6 both degrade to pass=None without a manifest;
        # the tier-count and queue checks still bind, so zero the
        # routed expectations to keep the stage green.
        b.metrics["route"] = {"tier1": 0, "tier2": 0, "dropped": 0}
        _write_jsonl(batch_dir(self.root, self.batch_id)
                     / "routed.jsonl", [])
        stage_verify(b, self.root, list(self.stages), self.registry,
                     FakeGraph([]))
        audit = b.metrics["verify"]["checks"]["drift_spot_audit"]
        self.assertIsNone(audit["pass"])
        # The report must say plainly that the property audit
        # did not run (run 32) — a skipped audit is not a passed
        # one, and the record must not read as if it were.
        self.assertIn("property audit did not run", audit["note"])


class TestWorkspaceResolution(unittest.TestCase):
    """The state-root discipline: workspace data resolves under
    $ONTOSMITH_ROOT / the state root, never the package tree."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ,
                              {"ONTOSMITH_ROOT": str(self.tmp)})
        env.start()
        self.addCleanup(env.stop)

    def test_registry_store_default_root(self):
        from pipeline.registry_store import RegistryStore
        self.assertEqual(RegistryStore().root, self.tmp)
        explicit = self.tmp / "elsewhere"
        self.assertEqual(RegistryStore(explicit).root, explicit)

    def test_docred_data_dir(self):
        from pipeline import docred
        self.assertEqual(docred.data_dir(), self.tmp / "data")

    def test_gate_cli_results_dir(self):
        from pipeline import gate_cli
        self.assertEqual(gate_cli._default_results_dir(),
                         self.tmp / "results")


if __name__ == "__main__":
    unittest.main()
