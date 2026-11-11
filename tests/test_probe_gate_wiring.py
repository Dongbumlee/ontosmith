"""Probe-gate channel wiring at CANDIDATES.

Pinned here:
- the declaration validates fail-closed
  (profile_config.pair_probe_gate_enabled);
- with the gate declared and a closed relation inventory,
  the pair census runs over the free channel's own endpoints,
  already-recorded pairs are excluded, a probe sample is
  judged first, and the full census runs only when the
  probe's hit rate clears the policy floor — judged items
  pass the same guard and merge behind the free channel
  with method "pair-probe", verdicts joining the sidecar
  with a channel mark;
- a closed probe (hit rate under the floor) judges the
  probe sample and nothing more;
- an open-inventory profile leaves the gate inert, counted;
- undeclared = no probe-gate metrics block at all (the
  legacy path is untouched).
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
from pipeline.profile_config import pair_probe_gate_enabled


class TestGateDeclaration(unittest.TestCase):

    def test_absent_means_off(self):
        self.assertIs(pair_probe_gate_enabled(None), False)
        self.assertIs(pair_probe_gate_enabled({}), False)

    def test_valid_declarations(self):
        self.assertIs(
            pair_probe_gate_enabled({"pair_probe_gate": True}),
            True)
        self.assertIs(
            pair_probe_gate_enabled({"pair_probe_gate": False}),
            False)

    def test_invalid_declarations_fail_closed(self):
        for bad in ("yes", 1, ["true"]):
            with self.assertRaises(ValueError):
                pair_probe_gate_enabled({"pair_probe_gate": bad})


def _stage(root, batch_id, text, resolved, free_items,
           pair_answer):
    cfg = BatchConfig(max_cost_usd=None)
    b = Batch(batch_id=batch_id, run_id="run-0001",
              source="src", collection="coll", doc_ids=["d1"],
              config=cfg, status=BatchStatus.NER_DONE,
              profile_family="probe-test")
    d = batch_dir(root, batch_id)
    d.mkdir(parents=True)
    (d / "texts.jsonl").write_text(json.dumps(
        {"doc_id": "d1", "text": text,
         "source_path": "d1.txt"}) + "\n")
    (d / "setup_config.resolved.json").write_text(
        json.dumps(resolved))
    pair_calls = {"n": 0}

    def free_chat(content, model, max_tokens, key):
        return json.dumps(free_items), {"cost": 0.0}

    def pair_chat(content, model, max_tokens, key):
        pair_calls["n"] += 1
        return json.dumps(pair_answer), {"cost": 0.0}

    save_batch(b, root)
    out = io.StringIO()
    with mock.patch.dict(os.environ,
                         {"OPENROUTER_API_KEY": "stub-key"}), \
            mock.patch.object(proposer, "_chat", free_chat), \
            mock.patch.object(pair_proposer, "_chat", pair_chat), \
            redirect_stdout(out):
        batch_run.stage_candidates(b, root, ["INGEST", "NER"])
    return b, d, pair_calls


class TestProbeGateAtStage(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_gate_opens_and_merges_behind_free(self):
        # The R3 sentence split treats the block before the
        # first blank line as the title (sentence 0); the body
        # below splits into three sentences.
        text = ("Summit Records\n\n"
                "Alpha employed Beta in 1901. "
                "Gamma advised Delta on the merger. "
                "Alpha hired Gamma as counsel.")
        free_items = [
            {"subject": "Alpha", "relation": "employs",
             "object": "Beta",
             "quote": "Alpha employed Beta in 1901"},
            {"subject": "Gamma", "relation": "advises",
             "object": "Delta",
             "quote": "Gamma advised Delta on the merger"}]
        b, d, pair_calls = _stage(
            self.root, "probe-open", text,
            {"pair_probe_gate": True,
             "relation_types": ["employs"]},
            free_items,
            {"relation": "employs",
             "quote": "Alpha hired Gamma as counsel"})
        cands = [json.loads(line) for line in
                 open(d / "candidates.jsonl")]
        probe_recs = [c for c in cands
                      if c["method"] == "pair-probe"]
        self.assertEqual(len(probe_recs), 1)
        rec = probe_recs[0]
        self.assertEqual((rec["subject"], rec["relation"],
                          rec["object"]),
                         ("Alpha", "employs", "Gamma"))
        # The free channel's own records are untouched and
        # keep their method.
        self.assertEqual(
            sorted(c["method"] for c in cands),
            ["llm", "llm", "pair-probe"])
        pg = b.metrics["candidates"]["probe_gate"]
        self.assertEqual(pg["docs"], 1)
        self.assertEqual(pg["docs_open"], 1)
        self.assertEqual(pg["census_pairs"], 3)
        self.assertEqual(pg["fresh_pairs"], 1)
        self.assertEqual(pg["probe_judged"], 1)
        self.assertEqual(pg["probe_hits"], 1)
        self.assertEqual(pg["rest_judged"], 0)
        self.assertEqual(pg["merged"], 1)
        self.assertEqual(pair_calls["n"], 1)
        # Verdicts join the sidecar with the channel mark.
        verdicts = [json.loads(line) for line in
                    open(d / "pair_verdicts.jsonl")]
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0]["channel"], "probe")
        self.assertEqual(verdicts[0]["outcome"], "ok")

    def test_closed_probe_judges_sample_only(self):
        # 11 names, none a substring of another (occurrence
        # matching is normalized-substring). The free channel
        # records a 10-link chain over them, so all 11 names
        # are roster endpoints; the one body sentence holds
        # them all.
        names = ["Alpha", "Bravo", "Celia", "Dario", "Elif",
                 "Farid", "Greta", "Hana", "Idris", "Jia", "Kofi"]
        sentence = " ".join(names) + " attended the summit."
        text = "Summit Records\n\n" + sentence
        free_items = [
            {"subject": names[i], "relation": "attended",
             "object": names[i + 1], "quote": sentence}
            for i in range(len(names) - 1)]
        b, d, pair_calls = _stage(
            self.root, "probe-closed", text,
            {"pair_probe_gate": True,
             "relation_types": ["employs"]},
            free_items,
            {"relation": "NONE"})
        pg = b.metrics["candidates"]["probe_gate"]
        # 11 endpoints in one sentence: C(11,2) = 55 census
        # pairs, 10 already recorded -> 45 fresh; the probe
        # judges the 40-pair policy sample and, with zero
        # hits, the remaining 5 are never judged.
        self.assertEqual(pg["census_pairs"], 55)
        self.assertEqual(pg["fresh_pairs"], 45)
        self.assertEqual(pg["probe_judged"], 40)
        self.assertEqual(pg["probe_hits"], 0)
        self.assertEqual(pg["docs_open"], 0)
        self.assertEqual(pg["rest_judged"], 0)
        self.assertEqual(pair_calls["n"], 40)
        cands = [json.loads(line) for line in
                 open(d / "candidates.jsonl")]
        self.assertEqual(
            [c for c in cands if c["method"] == "pair-probe"],
            [])

    def test_open_inventory_is_inert_and_counted(self):
        text = "Alpha employed Beta in 1901."
        free_items = [
            {"subject": "Alpha", "relation": "employs",
             "object": "Beta",
             "quote": "Alpha employed Beta in 1901"}]
        b, _d, pair_calls = _stage(
            self.root, "probe-inert", text,
            {"pair_probe_gate": True},
            free_items,
            {"relation": "employs", "quote": "x"})
        pg = b.metrics["candidates"]["probe_gate"]
        self.assertEqual(pg["docs"], 1)
        self.assertEqual(pg["docs_inert"], 1)
        self.assertEqual(pg["census_pairs"], 0)
        self.assertEqual(pair_calls["n"], 0)

    def test_undeclared_leaves_no_trace(self):
        text = "Alpha employed Beta in 1901."
        free_items = [
            {"subject": "Alpha", "relation": "employs",
             "object": "Beta",
             "quote": "Alpha employed Beta in 1901"}]
        b, _d, pair_calls = _stage(
            self.root, "probe-off", text, {},
            free_items,
            {"relation": "employs", "quote": "x"})
        self.assertNotIn("probe_gate",
                         b.metrics["candidates"])
        self.assertEqual(pair_calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
