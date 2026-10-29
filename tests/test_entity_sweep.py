"""Entity sweep (issue #58): the profile-gated additive
second proposal pass inside CANDIDATES.

The pass exists because the gold re-examination measured
the never-proposed population at ~30% of verified labels
— entities the text names (verbatim, checked) that the
chunk pass never built a triple around. The design
contract pinned here:

the flag accessor (absent = off; non-boolean fails
closed; the key is registered); the deterministic census
(same text -> same roster; an extracted surface that
fails verbatim attestation is dropped AND counted); the
anchor view (mention sentences ± 1, ceiling-counted);
flag-off byte-identity (candidate files, prompts, and
finish record are the pre-sweep build's); the merge
semantics (a sweep item whose key already stands — chunk
channel first, then earlier-merged sweep items — is a
counted duplicate, never a second record; chunk
provenance wins); the partial-compliance stress (a
proposer answering mostly off-anchor or malformed items,
or failing outright, degrades the sweep to zero yield
with the chunk channel untouched and the attrition
visible in the finish counts — never a fail-closed cliff
against the baseline); and the finish-accounting
restatement (candidates_sweep with shipped Tier-1
attributed from the routed records).
"""
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from pipeline import batch_run, profile_config, proposer
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir, save_batch

DOC = (
    "Alpha Corp is based in Toronto. "
    "Gamma Reyes works with Delta Bank. "
    "Delta Bank is based in Toronto."
)

CHUNK_ITEMS = [
    {"subject": "Alpha Corp", "relation": "is based in",
     "object": "Toronto",
     "quote": "Alpha Corp is based in Toronto"},
]


def _sweep_items_for(anchor):
    """The scripted sweep answers, by anchor. Gamma Reyes
    and Delta Bank share one fact (each anchor's view
    contains the sentence), so the fact is proposed under
    both anchors — the canonical duplicate."""
    if anchor == "Gamma Reyes":
        return [{"subject": "Gamma Reyes",
                 "relation": "works with",
                 "object": "Delta Bank",
                 "quote": "Gamma Reyes works with Delta Bank"}]
    if anchor == "Delta Bank":
        return [{"subject": "Gamma Reyes",
                 "relation": "works with",
                 "object": "Delta Bank",
                 "quote": "Gamma Reyes works with Delta Bank"},
                {"subject": "Delta Bank",
                 "relation": "is based in",
                 "object": "Toronto",
                 "quote": "Delta Bank is based in Toronto"}]
    return []


def _make_chat(sweep_fn=_sweep_items_for, chunk_items=CHUNK_ITEMS,
               record=None):
    """A proposer._chat stub: chunk prompts get the chunk
    items; sweep prompts (they carry the anchor rule) get
    sweep_fn(anchor). `record`, if given, collects every
    prompt the stage built."""
    def chat(content, model, max_tokens, key):
        if record is not None:
            record.append(content)
        if "Additional rule for this pass" in content:
            m = re.search(r'must be the entity "([^"]+)"',
                          content)
            return json.dumps(sweep_fn(m.group(1))), \
                {"cost": 0.0}
        return json.dumps(chunk_items), {"cost": 0.0}
    return chat


def _run_candidates_stage(root, batch_id, chat, sweep):
    """One scripted document through the real CANDIDATES
    stage (candidate_source=llm, proposer chat mocked).
    With `sweep`, the batch declares the pass via its
    frozen setup_config.resolved.json — the channel a real
    batch executes with. Returns (batch, stage_output)."""
    cfg = BatchConfig(candidate_source="llm", max_cost_usd=None)
    b = Batch(batch_id=batch_id, run_id="run-0001",
              source="src", collection="coll", doc_ids=["d1"],
              config=cfg, status=BatchStatus.NER_DONE,
              profile_family="sweep-test" if sweep else "")
    d = batch_dir(root, batch_id)
    d.mkdir(parents=True)
    (d / "texts.jsonl").write_text(json.dumps(
        {"doc_id": "d1", "text": DOC,
         "source_path": "d1.txt"}) + "\n")
    if sweep:
        (d / "setup_config.resolved.json").write_text(
            json.dumps({"entity_sweep": True}))
    save_batch(b, root)
    out = io.StringIO()
    with mock.patch.dict(os.environ,
                         {"OPENROUTER_API_KEY": "stub-key"}), \
            mock.patch.object(proposer, "_chat", chat), \
            redirect_stdout(out):
        batch_run.stage_candidates(b, root, ["INGEST", "NER"])
    return b, out.getvalue()


class TestSweepAccessor(unittest.TestCase):
    def test_absent_is_off(self):
        self.assertFalse(profile_config.entity_sweep_enabled(None))
        self.assertFalse(profile_config.entity_sweep_enabled({}))
        self.assertFalse(profile_config.entity_sweep_enabled(
            {"entity_sweep": False}))

    def test_declared(self):
        self.assertTrue(profile_config.entity_sweep_enabled(
            {"entity_sweep": True}))

    def test_malformed_fails_closed(self):
        for bad in ("yes", 1, 0, ["true"]):
            with self.assertRaises(ValueError):
                profile_config.entity_sweep_enabled(
                    {"entity_sweep": bad})

    def test_key_registered(self):
        self.assertEqual(profile_config.setup_config_errors(
            {"entity_sweep": True}), [])


class TestCensus(unittest.TestCase):
    def test_deterministic_and_attested(self):
        a = proposer.census_entities(DOC)
        b = proposer.census_entities(DOC)
        self.assertEqual(a, b)
        norms = [e["norm"] for e in a["entities"]]
        self.assertEqual(
            norms,
            ["alpha corp", "toronto", "gamma reyes",
             "delta bank"])
        self.assertEqual(a["unattested"], 0)
        self.assertEqual(a["extracted"], 6)

    def test_connector_names_join(self):
        c = proposer.census_entities(
            "The Bank of America hired Ana Ruiz. "
            "Ana Ruiz joined in March.")
        norms = [e["norm"] for e in c["entities"]]
        self.assertIn("bank of america", norms)
        self.assertIn("ana ruiz", norms)

    def test_unattested_surface_dropped_and_counted(self):
        # "U.S. Army" joins to "U.S Army"; its norm
        # ("u.s army") is not a substring of the document
        # norm ("u.s. army"), so the surface is dropped —
        # and the drop is counted, not silent.
        c = proposer.census_entities(
            "The U.S. Army opened a base. "
            "The base opened in June.")
        norms = [e["norm"] for e in c["entities"]]
        self.assertNotIn("u.s army", norms)
        self.assertEqual(c["unattested"], 1)

    def test_anchor_contexts_windows_and_cap(self):
        text = ("Alpha Corp is based in Toronto. "
                "Gamma Reyes works with Delta Bank. "
                "Delta Bank is based in Toronto. "
                "Toronto is a large city.")
        v = proposer.anchor_contexts(text, "Toronto")
        self.assertEqual(v["mention_sentences"], 3)
        self.assertIn("Alpha Corp is based in Toronto.",
                      v["context"])
        self.assertIn("Toronto is a large city.",
                      v["context"])
        # Ceiling: many separated mention runs, each long,
        # cannot all fit — the overflow is counted.
        long_text = " ".join(
            f"Anchor Name appears here with quite a lot of "
            f"surrounding prose to fill the view {i}. "
            f"Filler sentence number {i} follows with more "
            f"words to say. "
            f"Another unrelated sentence closes block {i}."
            for i in range(40))
        v2 = proposer.anchor_contexts(long_text, "Anchor Name")
        self.assertLessEqual(len(v2["context"]),
                             proposer.SWEEP_CONTEXT_MAX)
        self.assertGreater(v2["windows_dropped"], 0)


class TestSweepPrompt(unittest.TestCase):
    def test_prompt_is_chunk_prompt_plus_one_rule(self):
        view = "Gamma Reyes works with Delta Bank."
        base = proposer._build_triple_prompt(view)
        sweep = proposer._build_sweep_prompt(view, "Gamma Reyes")
        self.assertTrue(sweep.startswith(base))
        self.assertIn('must be the entity "Gamma Reyes"',
                      sweep[len(base):])

    def test_chunk_prompt_has_no_anchor_rule(self):
        self.assertNotIn(
            "Additional rule for this pass",
            proposer._build_triple_prompt("Any text."))


class TestSweepStage(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_merge_semantics(self):
        b, out = _run_candidates_stage(
            self.root, "sweep-merge", _make_chat(), sweep=True)
        m = b.metrics["candidates"]
        sw = m["sweep"]
        # Anchors: the census entities that are no endpoint
        # of the chunk item (Alpha Corp / Toronto are).
        self.assertEqual(sw["anchors"], 2)
        self.assertEqual(sw["proposed"], 3)
        self.assertEqual(sw["off_anchor"], 0)
        # The shared fact is proposed under both anchors:
        # merged once, counted duplicate once. The chunk
        # record is untouched and singular.
        self.assertEqual(sw["merged"], 2)
        self.assertEqual(sw["duplicates"], 1)
        self.assertEqual(sw["blocked"], 0)
        d = batch_dir(self.root, "sweep-merge")
        rows = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertEqual(len(rows), 3)
        shared = [r for r in rows
                  if r["subject"] == "Gamma Reyes"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["method"], "llm-sweep")
        chunk = [r for r in rows
                 if r["subject"] == "Alpha Corp"]
        self.assertEqual(len(chunk), 1)
        self.assertEqual(chunk[0]["method"], "llm")
        self.assertIn("entity sweep: 2 anchor(s)", out)

    def test_flag_off_byte_identity(self):
        prompts = []
        b, _ = _run_candidates_stage(
            self.root, "sweep-off",
            _make_chat(record=prompts), sweep=False)
        m = b.metrics["candidates"]
        self.assertNotIn("sweep", m)
        # Only the chunk call happened; its prompt is the
        # pre-sweep construction (no anchor rule anywhere).
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("Additional rule for this pass",
                         prompts[0])
        d = batch_dir(self.root, "sweep-off")
        rows = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "llm")
        # No finish noise either.
        with redirect_stdout(io.StringIO()):
            batch_run._finish_summary(b, self.root, None)
        self.assertNotIn("finish", b.metrics)

    def test_sweep_proposing_nothing_is_byte_identical(self):
        # The additive invariant: a sweep whose anchors all
        # answer [] changes no candidate byte.
        b_off, _ = _run_candidates_stage(
            self.root, "sweep-empty-off", _make_chat(),
            sweep=False)
        b_on, _ = _run_candidates_stage(
            self.root, "sweep-empty-on",
            _make_chat(sweep_fn=lambda a: []), sweep=True)
        a = (batch_dir(self.root, "sweep-empty-off")
             / "candidates.jsonl").read_bytes()
        c = (batch_dir(self.root, "sweep-empty-on")
             / "candidates.jsonl").read_bytes()
        self.assertEqual(a, c)
        self.assertEqual(
            b_on.metrics["candidates"]["sweep"]["merged"], 0)
        self.assertEqual(b_off.metrics["candidates"]["total"],
                         b_on.metrics["candidates"]["total"])


class TestSweepPartialCompliance(unittest.TestCase):
    """The cut-2 lesson, applied to the sweep: a proposer
    that answers off-anchor, malformed, or not at all must
    degrade the pass to (near-)zero yield with the chunk
    channel untouched and the stage alive — the attrition
    visible in the counts, never a cliff against the
    baseline."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_mostly_off_anchor_degrades_to_zero(self):
        def drift(anchor):
            # Well-formed triples that ignore the anchor
            # rule entirely (neither endpoint is the
            # anchor), plus one malformed item.
            return [{"subject": "Alpha Corp",
                     "relation": "is based in",
                     "object": "Toronto",
                     "quote": "Alpha Corp is based in Toronto"},
                    {"subject": "Nobody", "relation": "x"}]
        b, _ = _run_candidates_stage(
            self.root, "sweep-drift", _make_chat(sweep_fn=drift),
            sweep=True)
        m = b.metrics["candidates"]
        sw = m["sweep"]
        self.assertEqual(sw["off_anchor"], 2)
        self.assertEqual(sw["merged"], 0)
        self.assertEqual(sw["duplicates"], 0)
        # The chunk channel is untouched: its one candidate
        # stands, nothing blocked by the sweep's drift.
        self.assertEqual(m["total"], 1)
        d = batch_dir(self.root, "sweep-drift")
        rows = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertEqual(rows[0]["method"], "llm")
        self.assertFalse(
            (d / "candidates_blocked.jsonl").exists())

    def test_sweep_call_failure_leaves_chunk_channel(self):
        def boom(content, model, max_tokens, key):
            if "Additional rule for this pass" in content:
                raise RuntimeError("provider down")
            return json.dumps(CHUNK_ITEMS), {"cost": 0.0}
        b, _ = _run_candidates_stage(
            self.root, "sweep-boom", boom, sweep=True)
        m = b.metrics["candidates"]
        self.assertEqual(m["total"], 1)
        self.assertEqual(m["sweep"]["errors"], 2)
        self.assertEqual(m["sweep"]["merged"], 0)


class TestSweepFinishAccounting(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_finish_restates_sweep_with_shipped(self):
        b, _ = _run_candidates_stage(
            self.root, "sweep-finish", _make_chat(), sweep=True)
        # Routed records: the merged sweep candidate and
        # the based-in sweep candidate reach Tier 1; the
        # chunk candidate is Tier 2 here — attribution is
        # by method on the routed rows.
        d = batch_dir(self.root, "sweep-finish")
        routed = [
            {"doc_id": "d1", "subject": "Gamma Reyes",
             "relation": "works with", "object": "Delta Bank",
             "method": "llm-sweep", "tier": 1},
            {"doc_id": "d1", "subject": "Delta Bank",
             "relation": "is based in", "object": "Toronto",
             "method": "llm-sweep", "tier": 1},
            {"doc_id": "d1", "subject": "Alpha Corp",
             "relation": "is based in", "object": "Toronto",
             "method": "llm", "tier": 2},
        ]
        (d / "routed.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in routed))
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run._finish_summary(b, self.root, None)
        self.assertEqual(
            b.metrics["finish"]["candidates_sweep"],
            {"anchors": 2, "proposed": 3, "duplicates": 1,
             "off_anchor": 0,
             "blocked": {"total": 0, "by_reason": {}},
             "shipped_tier1": 2})
        self.assertIn("finish: CANDIDATES entity sweep",
                      out.getvalue())


if __name__ == "__main__":
    unittest.main()
