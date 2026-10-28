"""ER finish-line accounting tests (issue #51).

The defect: a build could finish VERIFIED with most of its
ER decisions held for human settlement and nothing at the
finish line said so — WRITE's stage metrics carried the
count, but the batch's own completion report never totaled
ER decisions by terminal state (multilingual pilot,
2026-10-27: 49 decisions, 44 held, finish line silent).

The hold itself is the product's measured design (a human
decision, never an auto-merge) and is NOT changed here.
What is pinned: at completion the batch accounts merged /
settled / held-unactioned in its metrics and log, and
unactioned holds produce a loud PENDING line naming the
settlement command; once settled, the line is gone.
"""
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, batch_dir, save_batch
from pipeline.file_registry import SqliteEntityRegistry

BID = "er-finish"


def _registry(tmp: Path) -> SqliteEntityRegistry:
    reg = SqliteEntityRegistry(tmp / "data" / "registry.db")
    # One automatic merge (gate / deterministic channel).
    reg.record_er_decision(
        BID, "run-1", "Auto Merged Ltd", "ent:a", "Auto Merged",
        model="typesafe/jev-1.13", p_same=0.91, outcome="merged")
    # Held pair 1: never settled.
    reg.record_er_decision(
        BID, "run-1", "Held One", "ent:b", "Held One Canonical",
        model="typesafe/jev-1.13", p_same=0.97, outcome="review")
    # Held pair 2: settled keep-separate.
    reg.record_er_decision(
        BID, "run-1", "Held Two", "ent:c", "Held Two Canonical",
        model="typesafe/jev-1.13", p_same=0.66, outcome="review")
    reg.record_er_decision(
        BID, "run-1", "Held Two", "ent:c", "Held Two Canonical",
        model="human", p_same=0.66, outcome="kept_separate",
        reason="human settlement of review pair #3")
    # Held pair 3: settled merge (the fold's own merged row
    # carries the settlement marker as its reason).
    reg.record_er_decision(
        BID, "run-1", "Held Three", "ent:d",
        "Held Three Canonical", model="typesafe/jev-1.13",
        p_same=0.98, outcome="review")
    reg.record_er_decision(
        BID, "run-1", "Held Three", "ent:d",
        "Held Three Canonical", model="human", p_same=0.98,
        outcome="merged",
        reason="human settlement of review pair #5")
    return reg


class TestErTerminalAccount(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "data").mkdir(parents=True)
        self.reg = _registry(self.tmp)
        self.addCleanup(self.reg.close)

    def test_counts_by_terminal_state(self):
        account = batch_run._er_terminal_account(self.reg, BID)
        self.assertEqual(account, {
            "merged": 2, "held_total": 3, "held_settled_merge": 1,
            "held_settled_keep_separate": 1, "held_unactioned": 1})

    def test_other_batch_is_not_counted(self):
        account = batch_run._er_terminal_account(self.reg, "nope")
        self.assertEqual(account["merged"], 0)
        self.assertEqual(account["held_total"], 0)
        self.assertEqual(account["held_unactioned"], 0)

    def test_registry_without_decision_store_returns_none(self):
        self.assertIsNone(
            batch_run._er_terminal_account(object(), BID))


class TestFinishSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "data").mkdir(parents=True)
        self.reg = _registry(self.tmp)
        self.addCleanup(self.reg.close)
        self.b = Batch(batch_id=BID, run_id="run-1",
                       source="corpus", collection="docs",
                       doc_ids=[],
                       config=BatchConfig(max_cost_usd=None))
        batch_dir(self.tmp, BID).mkdir(parents=True, exist_ok=True)
        save_batch(self.b, self.tmp)

    def _finish(self) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run._finish_summary(self.b, self.tmp, self.reg)
        return out.getvalue()

    def test_unactioned_holds_are_loud_and_in_metrics(self):
        text = self._finish()
        self.assertIn("ER SETTLEMENT PENDING: 1 held pair(s)",
                      text)
        self.assertIn(f"ontosmith batch er-review {BID}", text)
        er = self.b.metrics["finish"]["er"]
        self.assertEqual(er["held_unactioned"], 1)
        self.assertEqual(er["merged"], 2)

    def test_settling_the_last_hold_silences_the_flag(self):
        self._finish()
        self.reg.record_er_decision(
            BID, "run-1", "Held One", "ent:b",
            "Held One Canonical", model="human", p_same=0.97,
            outcome="kept_separate",
            reason="human settlement of review pair #2")
        text = self._finish()
        self.assertNotIn("SETTLEMENT PENDING", text)
        self.assertEqual(
            self.b.metrics["finish"]["er"]["held_unactioned"], 0)


if __name__ == "__main__":
    unittest.main()
