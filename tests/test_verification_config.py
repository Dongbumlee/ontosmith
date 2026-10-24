"""Verification config tests: gates, stop-lines, projections,
ledgers, escalation policy (plan R7/R12 as data structures).

Behavior targets are the lab cells' recorded discipline:
the governing projection is the MAXIMUM scenario; a cell
whose projection exceeds its stop-line does not start; the
ledger is append-only with resume; the judging seat never
escalates (R12), whatever the configuration says.
"""
import json
import tempfile
import unittest
from pathlib import Path

from harness.verification.config import (
    CountedProjection,
    FrontierEscalationConfig,
    GateDefinition,
    SpendLedger,
    StopLine,
    StopLineExceeded,
    VerificationConfig,
    all_passed,
    evaluate_gates,
)


class TestGates(unittest.TestCase):
    def test_gate_evaluation(self):
        g = GateDefinition("G1", "eta", ">=", 0.30, "not adopted")
        self.assertTrue(g.evaluate({"eta": 0.31}).passed)
        self.assertFalse(g.evaluate({"eta": 0.29}).passed)

    def test_missing_metric_fails_closed(self):
        g = GateDefinition("G1", "eta", ">=", 0.30)
        r = g.evaluate({})
        self.assertFalse(r.passed)
        self.assertIn("eta", r.note)

    def test_bad_comparator_rejected(self):
        with self.assertRaises(ValueError):
            GateDefinition("G1", "eta", "=>", 0.30)

    def test_evaluate_gates_and_all_passed(self):
        gates = [GateDefinition("A", "x", ">=", 1.0),
                 GateDefinition("B", "y", "<", 2.0)]
        results = evaluate_gates(gates, {"x": 1.5, "y": 1.0})
        self.assertTrue(all_passed(results))
        results = evaluate_gates(gates, {"x": 0.5, "y": 1.0})
        self.assertFalse(all_passed(results))


class TestStopLineAndProjection(unittest.TestCase):
    def test_stop_line_defaults_to_cap(self):
        self.assertEqual(StopLine(3.10).stop_line, 3.10)
        self.assertEqual(StopLine(3.10, 3.05).stop_line, 3.05)

    def test_stop_line_above_cap_rejected(self):
        with self.assertRaises(ValueError):
            StopLine(1.0, 2.0)

    def test_may_start_includes_reserve(self):
        stop = StopLine(10.0, reserve_usd=1.0)
        self.assertTrue(stop.may_start(8.0, 1.0))
        self.assertFalse(stop.may_start(8.0, 1.01))

    def test_governing_is_max_scenario(self):
        proj = CountedProjection(
            unit_costs={"pair": 0.001, "call": 0.5},
            scenarios={"A": {"pair": 1000}, "B": {"call": 3}})
        self.assertEqual(proj.totals(), {"A": 1.0, "B": 1.5})
        self.assertEqual(proj.governing(), 1.5)

    def test_unpriced_unit_fails_closed(self):
        proj = CountedProjection(unit_costs={},
                                 scenarios={"A": {"pair": 5}})
        with self.assertRaises(ValueError):
            proj.governing()

    def test_projection_over_stop_line_stops(self):
        proj = CountedProjection(
            unit_costs={"pair": 0.01}, scenarios={"A": {"pair": 400}})
        with self.assertRaises(StopLineExceeded):
            proj.assert_within(StopLine(3.10, 3.05))


class TestSpendLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "ledger.jsonl"

    def test_append_cumulative_and_resume(self):
        led = SpendLedger(self.path)
        led.append({"item_id": "d1", "cost_usd": 0.5})
        led.append({"item_id": "d2", "cost_usd": 0.25})
        self.assertEqual(led.total(), 0.75)
        resumed = SpendLedger(self.path)
        self.assertEqual(resumed.total(), 0.75)
        self.assertEqual(resumed.completed_ids(), {"d1", "d2"})

    def test_error_entries_are_not_completed(self):
        led = SpendLedger(self.path)
        led.append({"item_id": "d1", "status": "error",
                    "cost_usd": 0.0})
        self.assertEqual(led.completed_ids(), set())

    def test_append_over_stop_line_raises_and_writes_nothing(self):
        led = SpendLedger(self.path, stop=StopLine(1.0))
        led.append({"item_id": "d1", "cost_usd": 0.9})
        with self.assertRaises(StopLineExceeded):
            led.append({"item_id": "d2", "cost_usd": 0.2})
        lines = self.path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["item_id"], "d1")

    def test_in_memory_ledger(self):
        led = SpendLedger(None)
        led.append({"item_id": "x", "cost_usd": 1.5})
        self.assertEqual(led.total(), 1.5)


class TestEscalation(unittest.TestCase):
    def test_default_off(self):
        cfg = FrontierEscalationConfig()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.should_escalate("audit", 0.4))

    def test_band_only(self):
        cfg = FrontierEscalationConfig(enabled=True)
        self.assertTrue(cfg.should_escalate("audit", 0.4))
        self.assertFalse(cfg.should_escalate("audit", 0.9))
        self.assertFalse(cfg.should_escalate("audit", 0.1))
        self.assertFalse(cfg.should_escalate("audit", None))

    def test_judging_seat_never_escalates(self):
        cfg = FrontierEscalationConfig(enabled=True)
        for role in ("judge", "pair_judge", "gate", "jev",
                     "er_gate", "triple_gate"):
            self.assertFalse(cfg.should_escalate(role, 0.4), role)

    def test_invalid_band_rejected(self):
        with self.assertRaises(ValueError):
            FrontierEscalationConfig(band_lo=0.6, band_hi=0.5)


class TestVerificationConfigRoundTrip(unittest.TestCase):
    def test_json_round_trip(self):
        cfg = VerificationConfig(
            gates=[GateDefinition("G", "eta", ">=", 0.3, "no")],
            stop_line=StopLine(3.10, 3.05), seed=42)
        d = json.loads(json.dumps(cfg.to_dict()))
        back = VerificationConfig.from_dict(d)
        self.assertEqual(back.gates[0].threshold, 0.3)
        self.assertEqual(back.stop_line.stop_line, 3.05)
        self.assertFalse(back.escalation.enabled)
        self.assertEqual(back.seed, 42)


if __name__ == "__main__":
    unittest.main()
