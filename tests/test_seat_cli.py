"""Seat envelopes + the `seat` command's judge loop
(identity-seat wiring, 2026-10-30): the envelope builder
assembles exactly the material the measured seat ruled
from; the judge loop writes only complete/stand verdicts
to the decisions file, omits abstentions and failures as
no-opinion, and stops at the cost cap.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline import proposer, seat_cli, seat_envelope

DOC_A = ("Alpha Report\n"
         "Maria Chen leads the Chen Lab at MIT. "
         "Chen published the report. "
         "The Chen Lab released it.")
DOC_B = "Beta\nJonas Beck works at Beck Studio. Beck spoke."


def _points():
    a = proposer.identity_doc_key(DOC_A)
    b = proposer.identity_doc_key(DOC_B)
    return [
        {"doc": a, "ep": "chen", "cand": "chen lab",
         "fragment": "Chen", "candidate": "Chen Lab"},
        # consulted twice — deduplicated by the builder
        {"doc": a, "ep": "chen", "cand": "chen lab",
         "fragment": "Chen", "candidate": "Chen Lab"},
        {"doc": a, "ep": "chen", "cand": "maria chen",
         "fragment": "Chen", "candidate": "Maria Chen"},
        {"doc": b, "ep": "beck", "cand": "beck studio",
         "fragment": "Beck", "candidate": "Beck Studio"},
    ]


class TestEnvelopeBuilder(unittest.TestCase):
    def setUp(self):
        pts = _points()
        self.a = pts[0]["doc"]
        self.envs = seat_envelope.build_envelopes(
            pts,
            {self.a: DOC_A, pts[3]["doc"]: DOC_B},
            {self.a: [{"s": "Chen", "r": "leads",
                       "o": "Chen Lab"},
                      {"s": "MIT", "r": "hosts",
                       "o": "Chen Lab"}],
             pts[3]["doc"]: [{"s": "Beck", "r": "works at",
                              "o": "Beck Studio"}]})

    def test_dedup_and_shape(self):
        self.assertEqual(len(self.envs), 3)
        env = self._env("chen", "chen lab")
        self.assertEqual(env["point"]["fragment"], "Chen")
        self.assertEqual(env["document"]["headline"],
                         "Alpha Report")
        self.assertEqual(env["document"]["text"], DOC_A)

    def _env(self, ep, cand):
        return next(e for e in self.envs
                    if e["point"]["ep"] == ep
                    and e["point"]["cand"] == cand)

    def test_items_only_fragment_triples(self):
        env = self._env("chen", "chen lab")
        self.assertEqual(len(env["items"]), 1)
        self.assertEqual(env["items"][0]["s"], "Chen")

    def test_occurrences(self):
        env = self._env("chen", "chen lab")
        frag_sents = [o["sentence"]
                      for o in env["occurrences"]["fragment"]]
        self.assertTrue(any("Chen published" in s
                            for s in frag_sents))
        cand_sents = [o["sentence"]
                      for o in env["occurrences"]["candidate"]]
        self.assertTrue(any("Chen Lab" in s
                            for s in cand_sents))

    def test_registry_is_per_document(self):
        env = self._env("chen", "chen lab")
        self.assertEqual(
            env["registry"]["fragment_pairs"],
            [{"candidate": "Maria Chen", "points": 1}])
        self.assertEqual(
            env["registry"]["candidate_pairs"], [])


class TestJudgeLoop(unittest.TestCase):
    def _env(self, idx, ep, cand):
        return {"idx": idx,
                "point": {"doc": f"doc{idx}", "ep": ep,
                          "cand": cand, "fragment": ep,
                          "candidate": cand},
                "document": {"headline": "H", "text": "T"},
                "items": [],
                "occurrences": {"fragment": [],
                                "candidate": []},
                "registry": {"fragment_pairs": [],
                             "candidate_pairs": []}}

    def test_verdicts_abstain_error_handling(self):
        envs = [self._env(0, "a", "a b"),
                self._env(1, "c", "c d"),
                self._env(2, "e", "e f"),
                self._env(3, "g", "g h")]
        answers = iter([
            ('{"decision": "complete", "reason": "x"}', 0.01),
            ('{"decision": "abstain", "reason": "x"}', 0.01),
            ("not json", 0.01),
            ('{"decision": "stand", "reason": "x"}', 0.01)])
        rows, record = seat_cli.judge_envelopes(
            envs, lambda user: next(answers), concurrency=1)
        self.assertEqual(
            [(r["ep"], r["decision"]) for r in rows],
            [("a", "complete"), ("g", "stand")])
        self.assertEqual(len(record["abstain"]), 1)
        self.assertEqual(len(record["errors"]), 1)
        self.assertEqual(record["total_cost_usd"], 0.04)
        # the written rows load through the product's
        # fail-closed decisions loader
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "decisions.jsonl"
            p.write_text("".join(
                json.dumps(r) + "\n" for r in rows))
            loaded = proposer.load_identity_decisions(p)
        self.assertEqual(loaded[("doc0", "a", "a b")], True)
        self.assertEqual(loaded[("doc3", "g", "g h")], False)

    def test_cost_cap_stops_and_marks_unjudged(self):
        envs = [self._env(i, f"f{i}", f"f{i} x")
                for i in range(4)]
        rows, record = seat_cli.judge_envelopes(
            envs,
            lambda user: (
                '{"decision": "stand", "reason": "x"}', 0.6),
            max_cost_usd=1.0, concurrency=1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(record["cap_reached"])
        self.assertEqual(len(record["unjudged"]), 2)

    def test_frozen_prompt_carries_the_article_principle(self):
        self.assertIn("an article is not part of an "
                      "entity's name", seat_cli.SYSTEM_PROMPT)
        self.assertIn("abstain", seat_cli.SYSTEM_PROMPT)

    def test_frozen_prompt_carries_the_document_title_policy(self):
        self.assertIn("the generic word for the document itself",
                      seat_cli.SYSTEM_PROMPT)
        self.assertIn("this document's own title, complete",
                      seat_cli.SYSTEM_PROMPT)
        self.assertIn("the title of a different document",
                      seat_cli.SYSTEM_PROMPT)
        self.assertIn("distinguishing words beyond the generic "
                      "word itself", seat_cli.SYSTEM_PROMPT)
        self.assertIn("is not the title and stands",
                      seat_cli.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
