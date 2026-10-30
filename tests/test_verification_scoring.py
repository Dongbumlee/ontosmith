"""Scoring-core tests.

Expected values are hand-computed from the reference
instrument's formulas (Wilson interval, TN-1/TN-2
canonicalization, the corrected-precision /
ceiling-adjusted-recall arithmetic, the stratified estimator
with finite-population correction); the run's MANIFEST entry
records the direct equivalence check against the original
scorer module itself.
"""
import unittest

from harness.verification.report import build_report, render_markdown
from harness.verification.scoring import (
    NormalizationTable,
    ceiling_adjusted_recall,
    corrected_precision,
    match_records,
    norm_text,
    rate,
    stratified_estimate,
    wilson,
)

PRED = [
    {"doc_id": "d1", "subject": "Acme", "relation": "employs",
     "object": "Smith"},
    {"doc_id": "d1", "subject": "Acme", "relation": "employs",
     "object": "Jones"},
    {"doc_id": "d2", "subject": "Beta", "relation": "owns",
     "object": "Gamma"},
    # duplicate key of row 1 — counts once (distinctness rule)
    {"doc_id": "d1", "subject": "acme ", "relation": "employs",
     "object": " smith"},
]
GOLD = [
    {"doc_id": "d1", "head_names": ["Acme Corp", "Acme"],
     "tail_names": ["Smith"], "relation": "employs"},
    {"doc_id": "d1", "head_names": ["Acme"], "tail_names": ["Lee"],
     "relation": "employs"},
    {"doc_id": "d2", "head_names": ["Beta"], "tail_names": ["Gamma"],
     "relation": "owns"},
]


class TestPrimitives(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(norm_text("  A  B\tC "), "a b c")
        # verbatim lab edge behavior (see scoring.norm_text):
        # a missing value normalizes to "none"
        self.assertEqual(norm_text(None), "none")

    def test_wilson_known_value(self):
        # lab instrument: wilson(40, 100) = (0.3094, 0.4980)
        self.assertEqual(wilson(40, 100), (0.3094, 0.498))
        self.assertEqual(wilson(0, 0), (None, None))

    def test_rate_shape(self):
        r = rate(1, 4)
        self.assertEqual(r["rate"], 0.25)
        self.assertEqual(r["k"], 1)
        self.assertEqual(r["n"], 4)


class TestMatching(unittest.TestCase):
    def test_exact_match_partition(self):
        m = match_records(PRED, GOLD)
        self.assertEqual(m.predicted_total, 3)   # duplicate folded
        self.assertEqual(m.gold_total, 3)
        self.assertEqual(m.matched_predicted, 2)
        self.assertEqual(m.matched_gold, 2)
        self.assertEqual(len(m.unmatched_predicted), 1)
        self.assertEqual(m.unmatched_gold_indices, [1])
        raw = m.raw_scores()
        self.assertEqual(raw["precision"]["rate"], 0.6667)
        self.assertEqual(raw["recall"]["rate"], 0.6667)

    def test_tn1_symmetric(self):
        table = NormalizationTable(symmetric=frozenset({"married_to"}))
        pred = [{"doc_id": "d", "subject": "B", "relation": "married_to",
                 "object": "A"}]
        gold = [{"doc_id": "d", "head_names": ["A"],
                 "tail_names": ["B"], "relation": "married_to"}]
        m = match_records(pred, gold, table)
        self.assertEqual(m.matched_gold, 1)
        m0 = match_records(pred, gold)
        self.assertEqual(m0.matched_gold, 0)

    def test_tn2_inverse(self):
        table = NormalizationTable(
            inverses={"employed_by": "employs"})
        pred = [{"doc_id": "d", "subject": "Smith",
                 "relation": "employed_by", "object": "Acme"}]
        gold = [{"doc_id": "d", "head_names": ["Acme"],
                 "tail_names": ["Smith"], "relation": "employs"}]
        m = match_records(pred, gold, table)
        self.assertEqual(m.matched_gold, 1)

    def test_alias_surfaces_match(self):
        # The guard canonicalized the subject to the fuller
        # name and recorded both attested surfaces; the gold
        # cluster knows only the shorter one.
        pred = [{"doc_id": "d", "subject": "Wilfried Schneider",
                 "relation": "place of birth", "object": "Mediaș",
                 "canonicalized": {"subject": ["Willi Schneider",
                                               "Wilfried Schneider"]}}]
        gold = [{"doc_id": "d", "head_names": ["Willi Schneider"],
                 "tail_names": ["Mediaș"],
                 "relation": "place of birth"}]
        m = match_records(pred, gold)
        self.assertEqual(m.matched_gold, 1)
        bare = [{"doc_id": "d", "subject": "Wilfried Schneider",
                 "relation": "place of birth", "object": "Mediaș"}]
        self.assertEqual(match_records(bare, gold).matched_gold, 0)

    def test_alias_both_endpoints_match(self):
        # BOTH endpoints canonicalized (the guard rewrote the
        # subject and the object): gold knows only the short
        # surface of each, so matching needs the alias cross
        # product, not single-endpoint substitution.
        pred = [{"doc_id": "d", "subject": "Digital Bangladesh",
                 "relation": "head of government",
                 "object": "Prime Minister Sheikh Hasina",
                 "canonicalized": {
                     "subject": ["Bangladesh", "Digital Bangladesh"],
                     "object": ["Sheikh Hasina",
                                "Prime Minister Sheikh Hasina"]}}]
        gold = [{"doc_id": "d", "head_names": ["Bangladesh"],
                 "tail_names": ["Sheikh Hasina"],
                 "relation": "head of government"}]
        self.assertEqual(match_records(pred, gold).matched_gold, 1)
        # A crossed pair that was never attested as one entity
        # still cannot match: aliases widen surfaces per
        # endpoint, they do not invent pairings with other
        # records' endpoints.
        gold_other = [{"doc_id": "d", "head_names": ["Bangladesh"],
                       "tail_names": ["Mediaș"],
                       "relation": "head of government"}]
        self.assertEqual(
            match_records(pred, gold_other).matched_gold, 0)

    def test_tn3_relation_group(self):
        table = NormalizationTable(
            relation_groups={"country": "CONTAINMENT",
                             "located in": "CONTAINMENT"})
        pred = [{"doc_id": "d", "subject": "Vienna",
                 "relation": "located in", "object": "Austria"}]
        gold = [{"doc_id": "d", "head_names": ["Vienna"],
                 "tail_names": ["Austria"], "relation": "country"}]
        self.assertEqual(match_records(pred, gold, table).matched_gold, 1)
        self.assertEqual(match_records(pred, gold).matched_gold, 0)

    def test_tn3_group_keeps_tn2_pairing(self):
        # Grouping the TN-2 primary must not break matching
        # against a record stated under its inverse label.
        table = NormalizationTable(
            inverses={"subsidiary": "parent organization"},
            relation_groups={"parent organization": "OWNERSHIP",
                             "owned by": "OWNERSHIP"})
        gold = [{"doc_id": "d", "head_names": ["HoldCo"],
                 "tail_names": ["SubCo"],
                 "relation": "parent organization"}]
        pred_inv = [{"doc_id": "d", "subject": "SubCo",
                     "relation": "subsidiary", "object": "HoldCo"}]
        self.assertEqual(
            match_records(pred_inv, gold, table).matched_gold, 1)
        pred_grp = [{"doc_id": "d", "subject": "SubCo",
                     "relation": "owned by", "object": "HoldCo"}]
        gold_inv = [{"doc_id": "d", "head_names": ["SubCo"],
                     "tail_names": ["HoldCo"], "relation": "owned by"}]
        self.assertEqual(
            match_records(pred_grp, gold_inv, table).matched_gold, 1)

    def test_tn4_inverse_expansion(self):
        table = NormalizationTable(
            inverse_expansions={"father": ("child",),
                                "mother": ("child",),
                                "child": ("father", "mother")})
        pred = [{"doc_id": "d", "subject": "Paul Jr.",
                 "relation": "child", "object": "Paul Sr."}]
        gold = [{"doc_id": "d", "head_names": ["Paul Sr."],
                 "tail_names": ["Paul Jr."], "relation": "father"}]
        self.assertEqual(match_records(pred, gold, table).matched_gold, 1)
        self.assertEqual(match_records(pred, gold).matched_gold, 0)

    def test_tn5_subsumption(self):
        table = NormalizationTable(
            inverses={"capital of": "capital"},
            subsumptions={"located in": ("capital of",)})
        pred = [{"doc_id": "d", "subject": "Phnom Penh",
                 "relation": "capital of", "object": "Cambodia"}]
        gold = [{"doc_id": "d", "head_names": ["Phnom Penh"],
                 "tail_names": ["Cambodia"], "relation": "located in"}]
        self.assertEqual(match_records(pred, gold, table).matched_gold, 1)
        # No reverse entailment: a merely-located city does not
        # satisfy a capital-of gold item.
        gold2 = [{"doc_id": "d", "head_names": ["Phnom Penh"],
                  "tail_names": ["Cambodia"], "relation": "capital of"}]
        pred2 = [{"doc_id": "d", "subject": "Phnom Penh",
                  "relation": "located in", "object": "Cambodia"}]
        self.assertEqual(match_records(pred2, gold2, table).matched_gold, 0)


class TestCorrections(unittest.TestCase):
    def test_corrected_precision_formula(self):
        # (matched + judged supported) / predicted
        self.assertEqual(corrected_precision(90, 200, 12.5), 0.5125)
        self.assertIsNone(corrected_precision(0, 0, 0.0))

    def test_ceiling_adjusted_recall_formula(self):
        # matched / (gold x support) = 50 / (200 x 0.5) = 0.5
        self.assertEqual(ceiling_adjusted_recall(50, 200, 0.5), 0.5)
        self.assertIsNone(ceiling_adjusted_recall(50, 200, 0.0))

    def test_stratified_estimate_single_stratum(self):
        est = stratified_estimate([{
            "name": "only", "population": 100, "sampled": 10,
            "labels": ["supported"] * 8 + ["not_supported"] * 2}])
        self.assertEqual(est["estimate"], 0.8)
        self.assertEqual(est["estimated_supported_count"], 80.0)
        self.assertEqual(est["decisive"], 10)
        lo, hi = est["wilson95"]
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)

    def test_stratified_estimate_unlabeled_excluded(self):
        est = stratified_estimate([{
            "name": "s", "population": 50, "sampled": 10,
            "labels": ["supported"] * 5 + ["unlabeled"] * 5}])
        self.assertEqual(est["estimate"], 1.0)
        self.assertEqual(est["unlabeled"], 5)
        self.assertEqual(est["decisive"], 5)

    def test_stratified_estimate_weighting(self):
        est = stratified_estimate([
            {"name": "big", "population": 900, "sampled": 10,
             "labels": ["supported"] * 9 + ["not_supported"]},
            {"name": "small", "population": 100, "sampled": 10,
             "labels": ["supported"] * 1 + ["not_supported"] * 9}])
        # 0.9 x 0.9 + 0.1 x 0.1 = 0.82
        self.assertEqual(est["estimate"], 0.82)
        self.assertEqual(est["estimated_supported_count"], 820.0)


class TestReport(unittest.TestCase):
    def test_report_anatomy_sums_to_gold(self):
        rep = build_report(PRED, GOLD,
                           gold_fates={"1": "never_enumerated"})
        self.assertEqual(rep["anatomy"],
                         {"matched": 2, "never_enumerated": 1})
        self.assertEqual(rep["raw"]["precision"]["rate"], 0.6667)

    def test_report_with_corrections(self):
        rep = build_report(
            PRED, GOLD,
            unmatched_support={"groups": [{
                "name": "u", "population": 1, "sampled": 1,
                "labels": ["supported"]}]},
            gold_support={"groups": [{
                "name": "g", "population": 3, "sampled": 3,
                "labels": ["supported"] * 3}]})
        self.assertIn("corrected", rep)
        # precision: (2 + 1.0) / 3 = 1.0
        self.assertEqual(rep["corrected"]["precision"], 1.0)
        # ceiling recall: 2 / (3 x 1.0) = 0.6667
        self.assertEqual(
            rep["corrected"]["ceiling_adjusted_recall"], 0.6667)

    def test_render_markdown(self):
        md = render_markdown(build_report(PRED, GOLD))
        self.assertIn("# Evaluation report", md)
        self.assertIn("Failure anatomy", md)
        self.assertIn("| precision |", md)


if __name__ == "__main__":
    unittest.main()
