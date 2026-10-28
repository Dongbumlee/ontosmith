"""Extraction density tests (issue #52).

The diagnosis: the free proposer chunks documents fully
(20,000-char windows — coverage is not the defect), but its
prompt caps proposals at MAX_PER_CHUNK = 10 per chunk, a
CUAD-recipe constant applied to every pack and document
shape, and nothing reported the cap's saturation or the
per-document candidate distribution. In the multilingual
pilot the cap was the binding constraint (135 proposals
against a 160 ceiling over 16 one-chunk documents).

Pinned here:
- the result counts chunks and chunks that reached the cap
  (the saturation signal);
- a declared per-profile ceiling threads into the prompt,
  while the undeclared path stays the core default — the
  existing packs' prompts do not move;
- the profile accessor fails closed on a malformed value;
- the per-document distribution counts every built
  document, including zero-candidate ones.
"""
import json
import unittest
from unittest import mock

from pipeline import batch_run, profile_config, proposer

_ITEMS = [{"subject": f"Entity {i}", "relation": "relates to",
           "object": f"Other {i}", "quote": f"quote {i}"}
          for i in range(10)]


def _fake_chat(captured):
    def chat(content, model, max_tokens, key):
        captured.append(content)
        return json.dumps(_ITEMS), {"cost": 0.0}
    return chat


class TestProposerSaturation(unittest.TestCase):
    TEXT = "x" * 45000  # 3 proposer chunks (STEP 19500)

    def test_default_cap_counts_saturation(self):
        captured = []
        with mock.patch.object(proposer, "_chat",
                               _fake_chat(captured)):
            res = proposer.propose_triples_for_doc(
                "doc", self.TEXT, "key")
        self.assertEqual(res["chunks"], 3)
        self.assertEqual(res["proposed"], 30)
        self.assertEqual(res["chunks_at_cap"], 3)
        self.assertTrue(all("at most 10 objects" in c
                            for c in captured))

    def test_declared_cap_threads_into_prompt(self):
        captured = []
        with mock.patch.object(proposer, "_chat",
                               _fake_chat(captured)):
            res = proposer.propose_triples_for_doc(
                "doc", self.TEXT, "key", max_per_chunk=25)
        self.assertTrue(all("at most 25 objects" in c
                            for c in captured))
        # 10 returned per chunk is below the raised ceiling:
        # nothing saturates.
        self.assertEqual(res["chunks_at_cap"], 0)
        self.assertEqual(res["proposed"], 30)


class TestProposerMaxPerChunkAccessor(unittest.TestCase):
    def test_absent_is_none(self):
        self.assertIsNone(profile_config.proposer_max_per_chunk(
            None))
        self.assertIsNone(profile_config.proposer_max_per_chunk(
            {}))
        self.assertIsNone(profile_config.proposer_max_per_chunk(
            {"proposer_framing": "x"}))

    def test_declared_value(self):
        self.assertEqual(profile_config.proposer_max_per_chunk(
            {"proposer_max_per_chunk": 25}), 25)

    def test_malformed_fails_closed(self):
        for bad in (0, -3, "25", 2.5, True):
            with self.assertRaises(ValueError):
                profile_config.proposer_max_per_chunk(
                    {"proposer_max_per_chunk": bad})


class TestPerDocDistribution(unittest.TestCase):
    def test_counts_every_document_including_zero(self):
        cands = ([{"doc_id": "a"}] * 3 + [{"doc_id": "b"}])
        dist = batch_run._per_doc_distribution(
            cands, ["a", "b", "c"])
        self.assertEqual(dist, {"documents": 3, "min": 0,
                                "median": 1, "max": 3})

    def test_no_documents(self):
        self.assertEqual(
            batch_run._per_doc_distribution([], []),
            {"documents": 0, "min": 0, "median": 0, "max": 0})


if __name__ == "__main__":
    unittest.main()
