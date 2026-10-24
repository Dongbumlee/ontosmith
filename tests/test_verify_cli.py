"""verify CLI smoke tests: score / audit-sets / project /
report over tiny JSONL fixtures (the R10 one-command forms).
"""
import json
import tempfile
import unittest
from pathlib import Path

from harness.verification.verify_cli import main

PRED = [{"doc_id": "d1", "subject": "A", "relation": "r",
         "object": "B"}]
GOLD = [{"doc_id": "d1", "head_names": ["A"], "tail_names": ["B"],
         "relation": "r"},
        {"doc_id": "d1", "head_names": ["A"], "tail_names": ["C"],
         "relation": "r"}]


class TestVerifyCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pred = self.tmp / "pred.jsonl"
        self.gold = self.tmp / "gold.jsonl"
        self.pred.write_text("\n".join(json.dumps(r) for r in PRED))
        self.gold.write_text("\n".join(json.dumps(r) for r in GOLD))

    def test_score(self, capsys=None):
        rc = main(["score", "--predicted", str(self.pred),
                   "--gold", str(self.gold)])
        self.assertEqual(rc, 0)

    def test_audit_sets_writes_files(self):
        out = self.tmp / "audit"
        rc = main(["audit-sets", "--predicted", str(self.pred),
                   "--gold", str(self.gold), "--out-dir", str(out),
                   "--seed", "3"])
        self.assertEqual(rc, 0)
        frame = json.loads((out / "frame.json").read_text())
        self.assertEqual(frame["matched_predicted"], 1)
        self.assertTrue((out / "set_A.jsonl").exists())

    def test_project_within_and_over(self):
        rc = main(["project", "--counts", "pair=100",
                   "--unit-cost", "pair=0.01", "--cap", "3.10",
                   "--stop-line", "3.05"])
        self.assertEqual(rc, 0)

    def test_report_to_file(self):
        out = self.tmp / "report.md"
        rc = main(["report", "--predicted", str(self.pred),
                   "--gold", str(self.gold), "--out", str(out)])
        self.assertEqual(rc, 0)
        self.assertIn("Evaluation report", out.read_text())


if __name__ == "__main__":
    unittest.main()
