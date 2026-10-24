"""Audit-core tests: sampling determinism, A/B/C set
construction, escalation verdict resolution, quote verification.
"""
import unittest

from harness.verification.audit import (
    build_audit_sets,
    clean_quote,
    decisive_counts,
    final_verdicts,
    quote_verified,
    stratified_sample,
)
from harness.verification.scoring import match_records  # noqa: F401

PRED = [
    {"doc_id": f"d{i % 3}", "subject": f"S{i}", "relation": "r",
     "object": f"O{i}"}
    for i in range(12)
]
GOLD = [
    {"doc_id": "d0", "head_names": ["S0"], "tail_names": ["O0"],
     "relation": "r"},
    {"doc_id": "d1", "head_names": ["Sx"], "tail_names": ["Ox"],
     "relation": "r"},
]


class TestSampling(unittest.TestCase):
    def test_deterministic_for_seed(self):
        a = stratified_sample(PRED, 6, seed=7)
        b = stratified_sample(PRED, 6, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)

    def test_whole_pool_when_capped(self):
        self.assertEqual(len(stratified_sample(PRED, 99, seed=1)), 12)

    def test_spread_across_documents(self):
        sample = stratified_sample(PRED, 3, seed=3)
        docs = {r["doc_id"] for r in sample}
        self.assertEqual(docs, {"d0", "d1", "d2"})


class TestAuditSets(unittest.TestCase):
    def test_sets_partition(self):
        sets = build_audit_sets(PRED, GOLD, seed=5)
        self.assertEqual(sets["frame"]["matched_predicted"], 1)
        self.assertEqual(len(sets["A"]), 11)   # unmatched predicted
        self.assertEqual(len(sets["B"]), 2)    # all gold
        self.assertEqual(len(sets["C"]), 1)    # matched control
        self.assertEqual(sets["A"][0]["item_id"], "A-001")
        self.assertEqual(sets["C"][0]["record"]["subject"], "S0")

    def test_sizes_cap_sets(self):
        sets = build_audit_sets(PRED, GOLD,
                                sizes={"A": 4, "B": 1}, seed=5)
        self.assertEqual(len(sets["A"]), 4)
        self.assertEqual(len(sets["B"]), 1)
        self.assertEqual(len(sets["C"]), 1)  # uncapped = whole pool


class TestVerdicts(unittest.TestCase):
    def test_escalation_supersedes_decisive_only(self):
        primary = [{"item_id": "A-1", "verdict": "not_supported"},
                   {"item_id": "A-2", "verdict": "supported"},
                   {"item_id": "A-3", "verdict": "unknown"}]
        escalation = [{"item_id": "A-1", "verdict": "supported"},
                      {"item_id": "A-2", "verdict": "unclear"}]
        final = final_verdicts(primary, escalation)
        self.assertEqual(final, {"A-1": "supported",
                                 "A-2": "supported"})
        self.assertEqual(decisive_counts(final),
                         {"decisive": 2, "positive": 2})

    def test_quote_verification(self):
        self.assertTrue(quote_verified(
            '"the cat  sat"', "the cat sat down"))
        self.assertEqual(clean_quote("'abc'"), "abc")
        self.assertFalse(quote_verified("dog", "the cat sat"))
        self.assertFalse(quote_verified("", "anything"))


if __name__ == "__main__":
    unittest.main()
