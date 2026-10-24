"""Cell-runner tests: the pre-registered adoption-cell
lifecycle (prepare gate, stop-line during execution, ledger
resume, gate verdict) against a fake executor — no spend.
"""
import tempfile
import unittest
from pathlib import Path

from harness.verification.cell_runner import (
    CellSpec,
    judge_cell,
    prepare_cell,
    run_cell,
    run_registered_cell,
)
from harness.verification.config import (
    CountedProjection,
    GateDefinition,
    SpendLedger,
    StopLine,
    StopLineExceeded,
)

ITEMS = [{"item_id": f"d{i}"} for i in range(5)]


def executor(item):
    return ({"echo": item["item_id"]}, 0.10)


def make_spec(**kw):
    kw.setdefault("stop_line", StopLine(1.00, 0.95))
    return CellSpec(
        name="cell-x",
        gates=[GateDefinition("G-ETA", "eta", ">=", 0.30)],
        default_item_cost_usd=0.10, **kw)


class TestPrepare(unittest.TestCase):
    def test_projection_over_stop_line_refuses_start(self):
        spec = make_spec(projection=CountedProjection(
            unit_costs={"pair": 0.01},
            scenarios={"A": {"pair": 200}}))
        with self.assertRaises(StopLineExceeded):
            prepare_cell(spec)

    def test_projection_within_line_starts(self):
        spec = make_spec(projection=CountedProjection(
            unit_costs={"pair": 0.01},
            scenarios={"A": {"pair": 50}, "B": {"pair": 80}}))
        rec = prepare_cell(spec)
        self.assertTrue(rec["started"])
        self.assertEqual(rec["projection"]["governing_usd"], 0.8)


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ledger_path = self.tmp / "ledger.jsonl"

    def test_run_executes_and_ledgers(self):
        spec = make_spec()
        ledger = SpendLedger(self.ledger_path, stop=spec.stop_line)
        rec = run_cell(spec, ITEMS, executor, ledger)
        self.assertEqual(rec["executed"], 5)
        self.assertEqual(rec["spent_usd"], 0.5)
        self.assertIsNone(rec["stopped"])
        self.assertEqual(rec["results"]["d3"], {"echo": "d3"})

    def test_stop_line_halts_before_crossing(self):
        spec = make_spec(stop_line=StopLine(0.35, 0.35))
        ledger = SpendLedger(self.ledger_path, stop=spec.stop_line)
        rec = run_cell(spec, ITEMS, executor, ledger)
        # 3 items at $0.10 fit; the 4th start would reach $0.40
        self.assertEqual(rec["executed"], 3)
        self.assertEqual(rec["stopped"]["item_id"], "d3")
        self.assertEqual(rec["stopped"]["reason"], "stop_line")

    def test_resume_skips_ledgered_items(self):
        spec = make_spec()
        ledger = SpendLedger(self.ledger_path, stop=spec.stop_line)
        run_cell(spec, ITEMS[:2], executor, ledger)
        ledger2 = SpendLedger(self.ledger_path, stop=spec.stop_line)
        rec = run_cell(spec, ITEMS, executor, ledger2)
        self.assertEqual(rec["skipped"], 2)
        self.assertEqual(rec["executed"], 3)
        self.assertEqual(rec["spent_usd"], 0.5)

    def test_executor_error_recorded_not_completed(self):
        def flaky(item):
            if item["item_id"] == "d1":
                raise RuntimeError("boom")
            return (None, 0.10)

        spec = make_spec()
        ledger = SpendLedger(self.ledger_path, stop=spec.stop_line)
        rec = run_cell(spec, ITEMS, flaky, ledger)
        self.assertEqual(rec["executed"], 4)
        self.assertNotIn("d1", ledger.completed_ids())


class TestJudgeAndLifecycle(unittest.TestCase):
    def test_judge_cell_adoption(self):
        spec = make_spec()
        ok = judge_cell(spec, {"eta": 0.31})
        self.assertTrue(ok["adopted"])
        no = judge_cell(spec, {"eta": 0.29})   # the J3F2 shape
        self.assertFalse(no["adopted"])
        self.assertEqual(no["gates"][0]["value"], 0.29)

    def test_registered_cell_end_to_end(self):
        tmp = Path(tempfile.mkdtemp())
        spec = make_spec()
        out = run_registered_cell(
            spec, ITEMS, executor, tmp / "ledger.jsonl",
            metrics_of=lambda run: {"eta": 0.5})
        self.assertTrue(out["prepare"]["started"])
        self.assertEqual(out["run"]["executed"], 5)
        self.assertTrue(out["verdict"]["adopted"])


if __name__ == "__main__":
    unittest.main()
