"""Ship gate: full-population verification at the write
boundary (0.2.1, freeze item 2).

Every store-bound row is audited — the fake audit transport
answers per triple, so the whole population's trail is
asserted, not a sample: pass, retype-then-pass, and
second-failure exclusion, plus the batch accounting identity
candidates = vetoed + review + dropped + shipped + excluded.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline import adjudication as adj
from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir

TEXT = ("Acme Corp is headquartered in New York. "
        "Mara Voss founded Acme Corp in 2019. "
        "The studio sells stoneware bowls to cafes.")


def _row(**kw):
    r = {"doc_id": "doc-1", "subject": "Acme Corp",
         "relation": "is headquartered in", "object": "New York",
         "quote": "Acme Corp is headquartered in New York",
         "method": "llm", "context": "", "p_true": 0.9, "tier": 1}
    r.update(kw)
    return r


def _audit_chat(behavior):
    """behavior(subject, relation, object) -> verdict dict."""
    def chat(prompt, model=None):
        import re
        m = re.search(r"Triple: \((.*?) \| (.*?) \| (.*?)\)\n",
                      prompt)
        verdict = behavior(*m.groups())
        return json.dumps(verdict), 0.001
    return chat


PASS = {"verdict": "pass", "failure_class": None, "reason": "ok",
        "corrected": None}


class TestAuditStoreRow(unittest.TestCase):
    def test_clean_row_ships(self):
        outcome, _row_out, record, _ = adj.audit_store_row(
            _audit_chat(lambda s, r, o: PASS), _row(), TEXT)
        self.assertEqual(outcome, "shipped")
        self.assertEqual(record["rounds"],
                         [{"round": 1, "result": "pass",
                           "failure_class": None}])

    def test_judge_fail_with_correction_ships_retyped(self):
        def behavior(s, r, o):
            if o == "New York City":
                return PASS
            return {"verdict": "fail",
                    "failure_class": "object_overreach",
                    "reason": "quote states New York",
                    "corrected": {
                        "subject": "Acme Corp",
                        "relation": "is headquartered in",
                        "object": "New York City",
                        "quote": "Acme Corp is headquartered in "
                                 "New York"}}
        # The correction's object is not in its quote either,
        # so it must NOT validate — use a truthful one instead.
        def behavior2(s, r, o):
            if r == "is headquartered in" and o == "New York":
                return PASS
            return {"verdict": "fail",
                    "failure_class": "relation_unattested",
                    "reason": "wrong relation",
                    "corrected": {
                        "subject": "Acme Corp",
                        "relation": "is headquartered in",
                        "object": "New York",
                        "quote": "Acme Corp is headquartered in "
                                 "New York"}}
        row = _row(relation="is based in")
        outcome, shipped, record, _ = adj.audit_store_row(
            _audit_chat(behavior2), row, TEXT)
        self.assertEqual(outcome, "shipped_retyped")
        self.assertEqual(shipped["relation"], "is headquartered in")
        self.assertEqual(record["remedy"], "retype_to_quote")
        self.assertEqual(len(record["rounds"]), 2)

    def test_invalid_correction_does_not_ship(self):
        def behavior(s, r, o):
            return {"verdict": "fail",
                    "failure_class": "object_overreach",
                    "reason": "overreach",
                    "corrected": {
                        "subject": "Acme Corp",
                        "relation": "is headquartered in",
                        "object": "New York City",
                        "quote": "Acme Corp is headquartered in "
                                 "New York"}}
        outcome, shipped, _record, _ = adj.audit_store_row(
            _audit_chat(behavior), _row(), TEXT)
        self.assertEqual(outcome, "excluded")
        self.assertIsNone(shipped)

    def test_deterministic_failure_extended_then_ships(self):
        row = _row(quote="Acme Corp is headquartered in New")
        outcome, shipped, record, _ = adj.audit_store_row(
            _audit_chat(lambda s, r, o: PASS), row, TEXT)
        self.assertEqual(outcome, "shipped_retyped")
        self.assertEqual(record["remedy"], "extend_quote")
        self.assertIn("New York.", shipped["quote"])

    def test_unrepairable_row_excluded_after_two_rounds(self):
        row = _row(object="Boston",
                   quote="Acme Corp is headquartered in New York")
        outcome, _shipped, record, _ = adj.audit_store_row(
            _audit_chat(lambda s, r, o: {
                "verdict": "fail",
                "failure_class": "object_overreach",
                "reason": "Boston is not stated",
                "corrected": None}), row, TEXT)
        self.assertEqual(outcome, "excluded")
        self.assertEqual(record["excluded_class"], "object_overreach")


class TestStageShipGate(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.b = Batch(batch_id="gate-test", run_id="run-1",
                       source="corp", collection="docs",
                       doc_ids=["doc-1"], config=BatchConfig(),
                       status=BatchStatus.ROUTED,
                       profile_family="testfam")
        self.b.metrics["route"] = {"tier1": 3, "tier2": 1,
                                   "dropped": 0}
        d = batch_dir(self.root, "gate-test")
        d.mkdir(parents=True)
        (d / "setup_config.resolved.json").write_text(json.dumps(
            {"full_verification_gate": True}))
        (d / "texts.jsonl").write_text(json.dumps(
            {"doc_id": "doc-1", "text": TEXT,
             "source_path": "doc-1.txt"}) + "\n")
        rows = [
            _row(),
            _row(subject="Mara Voss", relation="founded",
                 object="Acme Corp",
                 quote="Mara Voss founded Acme Corp in 2019"),
            _row(subject="The studio", relation="sells",
                 object="bowls and cups",
                 quote="The studio sells stoneware bowls to cafes"),
            _row(subject="Mara Voss", relation="founded",
                 object="Acme Corp", tier=2,
                 quote="Mara Voss founded Acme Corp in 2019"),
        ]
        with open(d / "routed.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(d / "candidates.jsonl", "w") as f:
            for _i in range(4):
                f.write(json.dumps({"doc_id": "doc-1"}) + "\n")

    def _chat(self, prompt, model=None):
        import re
        m = re.search(r"Triple: \((.*?) \| (.*?) \| (.*?)\)\n",
                      prompt)
        _s, _r, o = m.groups()
        if o == "bowls and cups":
            return json.dumps({
                "verdict": "fail",
                "failure_class": "object_overreach",
                "reason": "quote states stoneware bowls only",
                "corrected": {
                    "subject": "The studio", "relation": "sells",
                    "object": "stoneware bowls",
                    "quote": "The studio sells stoneware bowls "
                             "to cafes"}}), 0.001
        return json.dumps(PASS), 0.001

    def test_stage_audits_whole_tier1_population(self):
        batch_run.stage_ship_gate(self.b, self.root, [], self._chat)
        d = batch_dir(self.root, "gate-test")
        ship = [json.loads(line) for line in open(d / "ship_set.jsonl")]
        self.assertEqual(len(ship), 3)
        corrected = [r for r in ship
                     if r["object"] == "stoneware bowls"]
        self.assertEqual(len(corrected), 1)
        trail = [json.loads(line)
                 for line in open(d / "verification_gate.jsonl")]
        self.assertEqual(len(trail), 3)  # tier-1 only, all of it
        outcomes = sorted(r["outcome"] for r in trail)
        self.assertEqual(outcomes, ["shipped", "shipped",
                                    "shipped_retyped"])
        self.assertEqual(self.b.status, BatchStatus.SHIP_GATED)
        self.assertEqual(self.b.metrics["accounting"], {
            "candidates": 4, "vetoed": 0, "review": 1,
            "dropped": 0, "shipped": 3, "excluded": 0})
        # routed.jsonl is never mutated.
        routed = [json.loads(line) for line in open(d / "routed.jsonl")]
        self.assertEqual(len(routed), 4)

    def test_exclusion_closes_accounting(self):
        def chat(prompt, model=None):
            import re
            m = re.search(r"Triple: \((.*?) \| (.*?) \| (.*?)\)\n",
                          prompt)
            if m.group(3) == "bowls and cups":
                return json.dumps({
                    "verdict": "fail",
                    "failure_class": "object_overreach",
                    "reason": "not stated", "corrected": None}), 0.001
            return json.dumps(PASS), 0.001
        batch_run.stage_ship_gate(self.b, self.root, [], chat)
        acct = self.b.metrics["accounting"]
        self.assertEqual(acct["shipped"], 2)
        self.assertEqual(acct["excluded"], 1)
        self.assertEqual(sum(v for k, v in acct.items()
                             if k != "candidates"),
                         acct["candidates"])
        self.assertEqual(
            self.b.metrics["ship_gate"]["excluded_classes"],
            {"object_overreach": 1})

    def test_declared_check_and_unwired_refusal(self):
        self.assertTrue(batch_run._declared_ship_gate(
            self.b, self.root))
        d = batch_dir(self.root, "gate-test")
        (d / "state.json").write_text(json.dumps(
            {"stages_completed": ["INGEST", "NER", "CANDIDATES",
                                  "GATES", "ROUTE"]}))
        with self.assertRaises(RuntimeError) as cm:
            batch_run.run_batch(self.b, self.root, None)
        self.assertIn("SHIP_GATE", str(cm.exception))

    def test_write_consumes_ship_set(self):
        batch_run.stage_ship_gate(self.b, self.root, [], self._chat)
        from pipeline.file_registry import SqliteEntityRegistry

        class FakeGraph:
            def __init__(self):
                self.written = []

            def write_triples(self, triples):
                self.written = list(triples)
                return len(triples)

        registry = SqliteEntityRegistry(
            self.root / "data" / "registry.db")
        graph = FakeGraph()
        self.b.status = BatchStatus.SHIP_GATED
        batch_run.stage_write(self.b, self.root, ["SHIP_GATE"],
                              registry, graph)
        relations = sorted(t.relation for t in graph.written)
        self.assertEqual(len(graph.written), 3)
        self.assertIn("sells", relations)
        d = batch_dir(self.root, "gate-test")
        written = [json.loads(line)
                   for line in open(d / "written_triples.jsonl")]
        self.assertEqual(len(written), 3)


if __name__ == "__main__":
    unittest.main()
