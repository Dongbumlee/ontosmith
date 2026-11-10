"""Multi-draw proposer passes: declared extra framings and
the complement pass (2026-11-02 draw measurement).

Pinned here:
- the two declarations validate fail-closed;
- the complement prompt block is generated from the
  chunk's own prior items and narrows the task — and the
  no-prior prompt is byte-identical to before;
- at the stage, the passes union before the guards: a
  cross-pass duplicate is blocked once as a duplicate,
  per-pass counts and pass provenance land in the metrics
  and on the candidate records (base draw = no stamp).
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from pipeline import batch_run, pair_proposer, proposer
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir, save_batch
from pipeline.profile_config import proposer_complement_draw, proposer_draw_framings


class TestDrawDeclarations(unittest.TestCase):

    def test_absent_means_single_base_draw(self):
        self.assertEqual(proposer_draw_framings(None), [])
        self.assertEqual(proposer_draw_framings({}), [])
        self.assertFalse(proposer_complement_draw(None))
        self.assertFalse(proposer_complement_draw({}))

    def test_valid_declarations(self):
        self.assertEqual(
            proposer_draw_framings(
                {"proposer_draw_framings": ["angle two"]}),
            ["angle two"])
        self.assertTrue(proposer_complement_draw(
            {"proposer_complement_draw": True}))

    def test_invalid_declarations_fail_closed(self):
        for bad in ("angle", [""], ["ok", 3], [], ["x"] * 5):
            with self.assertRaises(ValueError):
                proposer_draw_framings(
                    {"proposer_draw_framings": bad})
        with self.assertRaises(ValueError):
            proposer_complement_draw(
                {"proposer_complement_draw": "yes"})

    def test_create_time_validation_accepts_and_rejects(self):
        from pipeline.profile_config import setup_config_errors
        ok = {"proposer_draw_framings": ["angle two"],
              "proposer_complement_draw": True}
        self.assertEqual(setup_config_errors(ok), [])
        bad = setup_config_errors(
            {"proposer_draw_framings": "angle"})
        self.assertTrue(any("proposer_draw_framings" in e
                            for e in bad))
        bad = setup_config_errors(
            {"proposer_complement_draw": "yes"})
        self.assertTrue(any("proposer_complement_draw" in e
                            for e in bad))


class TestComplementPrompt(unittest.TestCase):

    def test_prior_block_lists_items_and_narrows(self):
        block = proposer._prior_block([
            {"subject": "A", "relation": "founded",
             "object": "B"}])
        self.assertIn("A | founded | B", block)
        self.assertIn("ALREADY been extracted", block)
        self.assertIn("NOT", block)

    def test_prompt_without_prior_is_unchanged(self):
        plain = proposer._build_triple_prompt("some text")
        with_prior = proposer._build_triple_prompt(
            "some text", prior_block="BLOCK")
        self.assertTrue(with_prior.startswith(plain))
        self.assertTrue(with_prior.endswith("BLOCK"))

    def test_prior_reaches_the_chunk_prompt(self):
        seen = {}

        def chat(content, model, max_tokens, key):
            seen["prompt"] = content
            return "[]", {"cost": 0.0}

        prior = {0: [{"subject": "A", "relation": "founded",
                      "object": "B"}]}
        with mock.patch.object(proposer, "_chat", chat):
            proposer.propose_triples_for_doc(
                "d", "Alpha founded Beta in 1901.", "key",
                prior=prior)
        self.assertIn("A | founded | B", seen["prompt"])


_BASE = {"subject": "Alpha", "relation": "founded",
         "object": "Beta",
         "quote": "Alpha founded Beta in 1901"}
_SECOND = {"subject": "Beta", "relation": "is located in",
           "object": "Gamma City",
           "quote": "Beta is located in Gamma City"}
_COMPLEMENT = {"subject": "Alpha", "relation": "employed",
               "object": "Delta",
               "quote": "Alpha employed Delta as counsel"}


def _run_stage(root, batch_id):
    cfg = BatchConfig(max_cost_usd=None)
    b = Batch(batch_id=batch_id, run_id="run-0001",
              source="src", collection="coll", doc_ids=["d1"],
              config=cfg, status=BatchStatus.NER_DONE,
              profile_family="draws-test")
    d = batch_dir(root, batch_id)
    d.mkdir(parents=True)
    text = " ".join(it["quote"] + "." for it in
                    (_BASE, _SECOND, _COMPLEMENT))
    (d / "texts.jsonl").write_text(json.dumps(
        {"doc_id": "d1", "text": text,
         "source_path": "d1.txt"}) + "\n")
    (d / "setup_config.resolved.json").write_text(json.dumps({
        "proposer_draw_framings": ["SECOND ANGLE"],
        "proposer_complement_draw": True}))

    def chat(content, model, max_tokens, key):
        if "ALREADY been extracted" in content:
            items = [_COMPLEMENT]
        elif "SECOND ANGLE" in content:
            items = [_SECOND, dict(_BASE)]
        else:
            items = [_BASE]
        return json.dumps(items), {"cost": 0.0}

    save_batch(b, root)
    out = io.StringIO()
    with mock.patch.dict(os.environ,
                         {"OPENROUTER_API_KEY": "stub-key"}), \
            mock.patch.object(proposer, "_chat", chat), \
            redirect_stdout(out):
        batch_run.stage_candidates(b, root, ["INGEST", "NER"])
    return b, d


class TestDrawUnionAtStage(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_passes_union_and_stamp(self):
        b, d = _run_stage(self.root, "draws-on")
        cands = [json.loads(line) for line in
                 open(d / "candidates.jsonl")]
        keys = {(c["subject"], c["relation"]) for c in cands}
        self.assertEqual(keys, {("Alpha", "founded"),
                                ("Beta", "is located in"),
                                ("Alpha", "employed")})
        by_rel = {c["relation"]: c for c in cands}
        self.assertNotIn("draw", by_rel["founded"])
        self.assertEqual(by_rel["is located in"]["draw"], 1)
        self.assertEqual(by_rel["employed"]["draw"], 2)
        # The second draw's copy of the base item is the
        # union's one duplicate.
        blocked = [json.loads(line) for line in
                   open(d / "candidates_blocked.jsonl")]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["reason"], "duplicate")
        self.assertEqual(blocked[0]["draw"], 1)
        prop = b.metrics["candidates"]["proposer"]
        self.assertEqual(prop["draws"], 3)
        self.assertEqual(prop["per_draw_proposed"], [1, 2, 1])


class TestComplementInertUnderPairEnumeration(
        unittest.TestCase):
    """The core complement default reaches a
    pair-enumeration pack's resolved config — and stops
    there: the CANDIDATES pair branch replaces the free
    proposer wholesale, so the complement pass (a
    free-proposer pass) can never run under it. Pinned at
    the stage: with pair_enumeration on and the complement
    switch on, the free proposer is never called and the
    pair census runs (here over a one-mention roster,
    which enumerates no pairs and asks nothing)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_pair_branch_never_calls_free_proposer(self):
        cfg = BatchConfig(max_cost_usd=None)
        b = Batch(batch_id="pair-enum-complement",
                  run_id="run-0001", source="src",
                  collection="coll", doc_ids=["d1"],
                  config=cfg, status=BatchStatus.NER_DONE,
                  profile_family="pair-test")
        d = batch_dir(self.root, b.batch_id)
        d.mkdir(parents=True)
        text = "Alpha founded Beta in 1901."
        (d / "texts.jsonl").write_text(json.dumps(
            {"doc_id": "d1", "text": text,
             "source_path": "d1.txt"}) + "\n")
        (d / "setup_config.resolved.json").write_text(
            json.dumps({
                "pair_enumeration": True,
                "pair_window": "sentence",
                "proposer_complement_draw": True}))
        (d / "roster.jsonl").write_text(json.dumps(
            {"doc_id": "d1",
             "mentions": [{"surface": "Alpha",
                           "type": "PER"}]}) + "\n")
        save_batch(b, self.root)
        out = io.StringIO()
        with mock.patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "stub-key"}), \
                mock.patch.object(
                    proposer, "propose_triples_for_doc",
                    side_effect=AssertionError(
                        "free proposer called under pair "
                        "enumeration")), \
                mock.patch.object(
                    pair_proposer, "_chat",
                    side_effect=AssertionError(
                        "pair judge called with no "
                        "enumerated pairs")), \
                redirect_stdout(out):
            batch_run.stage_candidates(
                b, self.root, ["INGEST", "NER"])
        pair_metrics = b.metrics["candidates"]["pair_enum"]
        self.assertEqual(pair_metrics["enumerated"], 0)
        prop = b.metrics["candidates"]["proposer"]
        self.assertEqual(prop["proposed"], 0)
        cands_path = d / "candidates.jsonl"
        cands = ([json.loads(line) for line in
                  open(cands_path)] if cands_path.exists()
                 else [])
        self.assertEqual(cands, [])


if __name__ == "__main__":
    unittest.main()
