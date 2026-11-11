"""Probe gate decision logic (pipeline/probe_gate).

Pinned here:
- the probe draw is deterministic under the policy seed and is
  a duplicate-free subset of the census;
- a census at or under the probe size is its own probe;
- the floor comparison is >= : a probe at exactly the floor
  opens, one hit below it closes;
- an empty probe never opens;
- hit-rate arithmetic counts every non-NONE verdict as a hit.
"""
import unittest

from pipeline.probe_gate import (
    PROBE_MIN_HIT_RATE,
    PROBE_N,
    hit_rate,
    probe_decision,
    select_probe,
)


def _census(n):
    return [{"subject": f"S{i}", "object": f"O{i}"} for i in range(n)]


class TestSelectProbe(unittest.TestCase):

    def test_deterministic_under_seed(self):
        census = _census(500)
        self.assertEqual(select_probe(census), select_probe(census))

    def test_subset_without_duplicates(self):
        census = _census(500)
        probe = select_probe(census)
        self.assertEqual(len(probe), PROBE_N)
        keys = [(p["subject"], p["object"]) for p in probe]
        self.assertEqual(len(set(keys)), PROBE_N)
        self.assertTrue(set(keys) <= {(p["subject"], p["object"])
                                      for p in census})

    def test_small_census_is_its_own_probe(self):
        census = _census(28)
        self.assertEqual(select_probe(census), census)

    def test_census_exactly_probe_size_is_whole(self):
        census = _census(PROBE_N)
        self.assertEqual(select_probe(census), census)

    def test_empty_census(self):
        self.assertEqual(select_probe([]), [])

    def test_bad_probe_n_rejected(self):
        with self.assertRaises(ValueError):
            select_probe(_census(10), probe_n=0)


class TestHitRate(unittest.TestCase):

    def test_arithmetic(self):
        rels = ["Parties"] * 7 + ["NONE"] * 33
        self.assertAlmostEqual(hit_rate(rels), 0.175)

    def test_none_is_case_insensitive(self):
        self.assertEqual(hit_rate(["none", "None", "NONE"]), 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(hit_rate([]), 0.0)

    def test_all_positive(self):
        self.assertEqual(hit_rate(["A", "B"]), 1.0)


class TestProbeDecision(unittest.TestCase):

    def test_exactly_at_floor_opens(self):
        # 2/40 = 0.05 = the floor: >= opens (boundary probes
        # measured on the open side).
        rels = ["Parties"] * 2 + ["NONE"] * 38
        d = probe_decision(rels)
        self.assertTrue(d["open"])
        self.assertEqual(d["hits"], 2)
        self.assertAlmostEqual(d["hit_rate"], PROBE_MIN_HIT_RATE)

    def test_one_below_floor_closes(self):
        rels = ["Parties"] * 1 + ["NONE"] * 39
        self.assertFalse(probe_decision(rels)["open"])

    def test_zero_hits_closes(self):
        self.assertFalse(probe_decision(["NONE"] * 40)["open"])

    def test_empty_probe_never_opens(self):
        d = probe_decision([])
        self.assertFalse(d["open"])
        self.assertEqual(d["probe_n"], 0)

    def test_small_census_full_probe(self):
        # Census 28 judged in full: 0 hits closes (the fiction
        # case); the floor still applies at full-probe size —
        # 1/28 = 3.6% closes, 2/28 = 7.1% opens.
        self.assertFalse(probe_decision(["NONE"] * 28)["open"])
        self.assertFalse(probe_decision(["is a"] + ["NONE"] * 27)["open"])
        self.assertTrue(
            probe_decision(["is a", "is a"] + ["NONE"] * 26)["open"])


if __name__ == "__main__":
    unittest.main()
