"""Mention-layer tests (issue #54).

The diagnosis: entities register and link by noun surface
forms, so facts stated via pronouns, definite descriptions,
or cross-sentence evidence have no citable path to an
entity, and drops record only a score. The layer's first
cut (profile-gated by `mention_layer`, English packs
first):

- the proposer emits per-endpoint mention spans + types +
  verbatim antecedent resolutions;
- the guards verify BOTH spans verbatim — an unverifiable
  span blocks by name (mention_missing / mention_span /
  mention_antecedent), never silently;
- GATES judges each resolution as its own Jev decision
  before the fact: clear -> judged as usual; review band
  -> judged but capped at Tier-2 review by ROUTE;
  rejected -> never fact-judged, dropped with reason
  class resolution_rejected, distinct from fact_rejected.

Pinned here: the accessor fails closed; the undeclared
path is byte-identical (legacy prompt, legacy records);
the span discipline; the three resolution outcomes; the
route cap and the drop reason classes.

Batch-level guards added after the Phase 2 verdict (the
A/B failed on an interaction the unit tests above never
simulated — see the verdict record in the changelog):
the finish record restates guard-block counts by reason,
so a mention_missing cliff cannot be silent at the
finish; a scripted partial-compliance proposer (27%
field omission, the measured Phase 2 rate) characterizes
the fail-closed cliff exactly; and fully compliant input
survives identically with the layer on and off.
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

from pipeline import batch_run, profile_config, proposer
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir, save_batch

DOC = (
    "Paul Desmarais Jr. leads Power Corp. "
    "They have four children: Paul, Alexandre, Nicolas, and Charles. "
    "He leads the company."
)


def _item(**kw):
    base = {"subject": "Paul Desmarais Jr.", "relation": "leads",
            "object": "Power Corp",
            "quote": "Paul Desmarais Jr. leads Power Corp",
            "chunk_idx": 0, "chunk_norm": proposer.norm_ws(DOC)}
    base.update(kw)
    return base


def _named_mentions():
    return {"subject": {"span": "Paul Desmarais Jr.", "type": "named"},
            "object": {"span": "Power Corp", "type": "named"}}


class TestMentionAccessor(unittest.TestCase):
    def test_absent_is_off(self):
        self.assertFalse(profile_config.mention_layer_enabled(None))
        self.assertFalse(profile_config.mention_layer_enabled({}))
        self.assertFalse(profile_config.mention_layer_enabled(
            {"mention_layer": False}))

    def test_declared(self):
        self.assertTrue(profile_config.mention_layer_enabled(
            {"mention_layer": True}))

    def test_malformed_fails_closed(self):
        for bad in ("yes", 1, 0, ["true"]):
            with self.assertRaises(ValueError):
                profile_config.mention_layer_enabled(
                    {"mention_layer": bad})

    def test_key_registered(self):
        self.assertEqual(profile_config.setup_config_errors(
            {"mention_layer": True}), [])


class TestMentionPromptAndParse(unittest.TestCase):
    def test_legacy_prompt_has_no_mention_fields(self):
        p = proposer._build_triple_prompt("chunk text")
        self.assertNotIn("mention", p)
        self.assertEqual(p, proposer._build_triple_prompt(
            "chunk text", mentions=False))

    def test_mention_prompt_asks_for_spans(self):
        p = proposer._build_triple_prompt("chunk text", mentions=True)
        for field in ("subject_mention", "subject_mention_type",
                      "subject_antecedent", "object_mention",
                      "object_antecedent"):
            self.assertIn(field, p)
        self.assertIn("pronominal", p)

    def test_parse_carries_mentions_only_in_mention_mode(self):
        items = [{"subject": "Paul Desmarais Jr.", "relation": "has",
                  "object": "Alexandre",
                  "quote": "They have four children: Paul, Alexandre",
                  "subject_mention": "They",
                  "subject_mention_type": "Pronominal",
                  "subject_antecedent": "Paul Desmarais Jr.",
                  "object_mention": "Alexandre",
                  "object_mention_type": "named",
                  "object_antecedent": ""}]
        chat = lambda content, model, max_tokens, key: (  # noqa: E731
            json.dumps(items), {"cost": 0.0})
        with mock.patch.object(proposer, "_chat", chat):
            res = proposer.propose_triples_for_doc(
                "d", DOC, "key", mentions=True)
        m = res["items"][0]["mentions"]
        self.assertEqual(m["subject"]["span"], "They")
        self.assertEqual(m["subject"]["type"], "pronominal")
        self.assertEqual(m["subject"]["antecedent"],
                         "Paul Desmarais Jr.")
        self.assertNotIn("antecedent", m["object"])
        with mock.patch.object(proposer, "_chat", chat):
            res = proposer.propose_triples_for_doc("d", DOC, "key")
        self.assertNotIn("mentions", res["items"][0])


class TestMentionGuards(unittest.TestCase):
    def guard(self, item, mentions=True):
        return proposer.guard_triples([item], DOC, mentions=mentions)

    def test_named_mentions_kept(self):
        kept, blocked = self.guard(_item(mentions=_named_mentions()))
        self.assertEqual(len(kept), 1)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["mentions"]["subject"]["type"],
                         "named")

    def test_pronominal_with_verbatim_antecedent_kept(self):
        it = _item(
            relation="has", object="Alexandre",
            quote="They have four children: Paul, Alexandre, "
                  "Nicolas, and Charles",
            mentions={
                "subject": {"span": "They", "type": "pronominal",
                            "antecedent": "Paul Desmarais Jr."},
                "object": {"span": "Alexandre", "type": "named"}})
        kept, _blocked = self.guard(it)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["mentions"]["subject"]["antecedent"],
                         "Paul Desmarais Jr.")

    def test_descriptive_with_antecedent_kept(self):
        it = _item(
            quote="He leads the company",
            mentions={
                "subject": {"span": "He", "type": "pronominal",
                            "antecedent": "Paul Desmarais Jr."},
                "object": {"span": "the company",
                           "type": "descriptive",
                           "antecedent": "Power Corp"}})
        kept, _blocked = self.guard(it)
        self.assertEqual(len(kept), 1)

    def test_missing_mentions_blocked_by_name(self):
        kept, blocked = self.guard(_item())
        self.assertEqual(kept, [])
        self.assertEqual(blocked[0]["reason"], "mention_missing")

    def test_bad_type_is_mention_missing(self):
        m = _named_mentions()
        m["subject"] = {"span": "Paul Desmarais Jr.", "type": "pronoun"}
        _kept, blocked = self.guard(_item(mentions=m))
        self.assertEqual(blocked[0]["reason"], "mention_missing")

    def test_span_not_in_quote_blocked(self):
        m = _named_mentions()
        m["subject"] = {"span": "He", "type": "pronominal",
                        "antecedent": "Paul Desmarais Jr."}
        _kept, blocked = self.guard(_item(mentions=m))
        self.assertEqual(blocked[0]["reason"], "mention_span")

    def test_antecedent_absent_blocked(self):
        m = _named_mentions()
        m["subject"] = {"span": "Paul Desmarais Jr.",
                        "type": "descriptive"}
        _kept, blocked = self.guard(_item(
            quote="Paul Desmarais Jr. leads Power Corp", mentions=m))
        # descriptive without an antecedent: unresolved.
        self.assertEqual(blocked[0]["reason"], "mention_antecedent")

    def test_antecedent_not_verbatim_blocked(self):
        it = _item(
            relation="has", object="Alexandre",
            quote="They have four children: Paul, Alexandre",
            mentions={
                "subject": {"span": "They", "type": "pronominal",
                            "antecedent": "Barack Obama"},
                "object": {"span": "Alexandre", "type": "named"}})
        _kept, blocked = self.guard(it)
        self.assertEqual(blocked[0]["reason"], "mention_antecedent")

    def test_antecedent_naming_another_entity_blocked(self):
        it = _item(
            relation="has", object="Alexandre",
            quote="They have four children: Paul, Alexandre",
            mentions={
                "subject": {"span": "They", "type": "pronominal",
                            "antecedent": "Power Corp"},
                "object": {"span": "Alexandre", "type": "named"}})
        _kept, blocked = self.guard(it)
        self.assertEqual(blocked[0]["reason"], "mention_antecedent")

    def test_legacy_mode_ignores_mentions(self):
        kept, _blocked = self.guard(_item(), mentions=False)
        self.assertEqual(len(kept), 1)
        self.assertNotIn("mentions", kept[0])


def _gates_batch(root, candidates):
    cfg = BatchConfig(candidate_source="llm", max_cost_usd=None,
                      triple_threshold=0.25, review_band_hi=0.85)
    b = Batch(batch_id="mention-gates", run_id="run-0001",
              source="src", collection="coll", doc_ids=["d1"],
              config=cfg, status=BatchStatus.CANDIDATES_DONE)
    d = batch_dir(root, b.batch_id)
    d.mkdir(parents=True)
    (d / "candidates.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in candidates))
    save_batch(b, root)
    return b


def _cand_with_resolution(subject_p=0.97):
    return {"doc_id": "d1", "subject": "Paul Desmarais Jr.",
            "relation": "has", "object": "Alexandre",
            "context": "Paul Desmarais Jr. ... They have four "
                       "children: Paul, Alexandre",
            "quote": "They have four children: Paul, Alexandre",
            "method": "llm", "attest": "chunk",
            "mentions": {
                "subject": {"span": "They", "type": "pronominal",
                            "antecedent": "Paul Desmarais Jr."},
                "object": {"span": "Alexandre", "type": "named"}}}


_PLAIN = {"doc_id": "d1", "subject": "Power Corp",
          "relation": "is located in", "object": "Canada",
          "context": "Power Corp is located in Canada.",
          "method": "llm"}


class TestGatesResolution(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fact_calls = []

    def _fact(self, ctx, s, r, o, model=None):
        self.fact_calls.append((s, r, o))
        return 0.99, {"cost": 0.001}

    def _run(self, res_p, with_fn=True):
        b = _gates_batch(self.root,
                         [_cand_with_resolution(), dict(_PLAIN)])
        res_fn = None
        if with_fn:
            def res_fn(ctx, mention, antecedent, entity, model=None):
                return res_p, {"cost": 0.0005}
        with redirect_stdout(io.StringIO()):
            batch_run.stage_gates(
                b, self.root, ["INGEST", "NER", "CANDIDATES"],
                self._fact, resolution_decide_fn=res_fn)
        d = batch_dir(self.root, b.batch_id)
        return [json.loads(line) for line in open(d / "judgments.jsonl")], b

    def test_clear_resolution_fact_judged(self):
        js, b = self._run(0.97)
        j = next(x for x in js if x["subject"] == "Paul Desmarais Jr.")
        self.assertEqual(j["resolution_status"], "clear")
        self.assertEqual(j["p_resolution"], 0.97)
        self.assertEqual(j["p_true"], 0.99)
        self.assertIn(("Paul Desmarais Jr.", "has", "Alexandre"),
                      self.fact_calls)
        self.assertEqual(b.metrics["gates"]["resolutions"],
                         {"clear": 1})

    def test_review_resolution_still_fact_judged(self):
        js, _ = self._run(0.60)
        j = next(x for x in js if x["subject"] == "Paul Desmarais Jr.")
        self.assertEqual(j["resolution_status"], "review")
        self.assertEqual(j["p_true"], 0.99)

    def test_rejected_resolution_never_fact_judged(self):
        js, _ = self._run(0.10)
        j = next(x for x in js if x["subject"] == "Paul Desmarais Jr.")
        self.assertEqual(j["resolution_status"], "rejected")
        self.assertEqual(j["drop_reason"], "resolution_rejected")
        self.assertEqual(j["fact_judged"], False)
        self.assertNotIn(("Paul Desmarais Jr.", "has", "Alexandre"),
                         self.fact_calls)

    def test_plain_candidate_has_no_resolution_fields(self):
        js, _ = self._run(0.97)
        j = next(x for x in js if x["subject"] == "Power Corp")
        self.assertNotIn("resolution_status", j)
        self.assertNotIn("p_resolution", j)

    def test_missing_resolution_judge_fails_closed(self):
        b = _gates_batch(self.root, [_cand_with_resolution()])
        with self.assertRaises(RuntimeError):
            with redirect_stdout(io.StringIO()):
                batch_run.stage_gates(
                    b, self.root, ["INGEST", "NER", "CANDIDATES"],
                    self._fact)


class TestRouteResolution(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def _route(self, judgments):
        cfg = BatchConfig(candidate_source="llm", max_cost_usd=None,
                          triple_threshold=0.25, review_band_hi=0.85)
        b = Batch(batch_id="mention-route", run_id="run-0001",
                  source="src", collection="coll", doc_ids=["d1"],
                  config=cfg, status=BatchStatus.GATED)
        d = batch_dir(self.root, b.batch_id)
        d.mkdir(parents=True)
        (d / "judgments.jsonl").write_text(
            "".join(json.dumps(j) + "\n" for j in judgments))
        save_batch(b, self.root)
        with redirect_stdout(io.StringIO()):
            batch_run.stage_route(b, self.root,
                                  ["INGEST", "NER", "CANDIDATES", "GATES"])
        routed = [json.loads(line) for line in open(d / "routed.jsonl")]
        review = [json.loads(line) for line in open(d / "review_queue.jsonl")]
        dropped = [json.loads(line) for line in open(d / "dropped_queue.jsonl")]
        return b, routed, review, dropped

    def _j(self, **kw):
        base = {"doc_id": "d1", "subject": "S", "relation": "r",
                "object": "O", "p_true": 0.99, "cost": 0, "error": None}
        base.update(kw)
        return base

    def test_review_resolution_caps_at_tier2(self):
        b, routed, review, _ = self._route(
            [self._j(resolution_status="review", p_resolution=0.6)])
        self.assertEqual(routed[0]["tier"], 2)
        self.assertTrue(routed[0]["resolution_capped"])
        self.assertEqual(review[0]["resolution_review"], True)
        self.assertEqual(b.metrics["route"]["resolution_review_caps"], 1)

    def test_clear_resolution_reaches_tier1(self):
        _, routed, _, _ = self._route(
            [self._j(resolution_status="clear", p_resolution=0.97)])
        self.assertEqual(routed[0]["tier"], 1)

    def test_resolution_rejected_drop_reason_class(self):
        _, routed, _, dropped = self._route([self._j(
            p_true=0.0, resolution_status="rejected",
            drop_reason="resolution_rejected", fact_judged=False)])
        self.assertEqual(routed[0]["tier"], 0)
        self.assertEqual(dropped[0]["reason_class"],
                         "resolution_rejected")

    def test_fact_rejected_drop_reason_class(self):
        _, _, _, dropped = self._route([self._j(
            p_true=0.05, resolution_status="clear",
            p_resolution=0.97)])
        self.assertEqual(dropped[0]["reason_class"], "fact_rejected")

    def test_legacy_drop_has_no_reason_class(self):
        _, _, _, dropped = self._route([self._j(p_true=0.05)])
        self.assertNotIn("reason_class", dropped[0])


# --------------------------------------- batch-level guards
# Post-verdict (Phase 2, issue #54): the A/B's Tier-1
# recall fell 0.1066 -> 0.0772 because the proposer omitted
# the mention fields on a large share of proposals and the
# fail-closed guard culled them before judging — an
# interaction every test above missed, because every
# fixture proposer is perfectly compliant. The tests below
# run the real CANDIDATES stage over scripted proposers so
# the failure class itself is pinned: its visibility at
# the finish, its exact cliff shape under partial
# compliance, and its absence under full compliance. The
# guard POLICY is not changed here — a future policy
# change (e.g. a legacy-evidence fallback for a missing
# mention block) must rewrite the cliff test on purpose.


def _scripted_items(n, omit):
    """n proposals over generated one-fact sentences. The
    first `omit` carry NO mention fields (the proposer's
    field-omission failure shape); the rest carry
    well-formed named mentions for both endpoints."""
    items = []
    for i in range(n):
        it = {"subject": f"Person {i}", "relation": "leads",
              "object": f"Company {i}",
              "quote": f"Person {i} leads Company {i}"}
        if i >= omit:
            it.update({
                "subject_mention": f"Person {i}",
                "subject_mention_type": "named",
                "object_mention": f"Company {i}",
                "object_mention_type": "named"})
        items.append(it)
    return items


def _run_candidates_stage(root, batch_id, items, mention):
    """One scripted document through the real CANDIDATES
    stage (candidate_source=llm, proposer chat mocked).
    With `mention`, the batch declares the layer via its
    frozen setup_config.resolved.json — the channel a real
    batch executes with. Returns (batch, stage_output)."""
    cfg = BatchConfig(candidate_source="llm", max_cost_usd=None)
    b = Batch(batch_id=batch_id, run_id="run-0001",
              source="src", collection="coll", doc_ids=["d1"],
              config=cfg, status=BatchStatus.NER_DONE,
              profile_family="mention-guard" if mention else "")
    d = batch_dir(root, batch_id)
    d.mkdir(parents=True)
    text = " ".join(it["quote"] + "." for it in items)
    (d / "texts.jsonl").write_text(json.dumps(
        {"doc_id": "d1", "text": text,
         "source_path": "d1.txt"}) + "\n")
    if mention:
        (d / "setup_config.resolved.json").write_text(
            json.dumps({"mention_layer": True}))
    save_batch(b, root)

    def chat(content, model, max_tokens, key):
        return json.dumps(items), {"cost": 0.0}

    out = io.StringIO()
    with mock.patch.dict(os.environ,
                         {"OPENROUTER_API_KEY": "stub-key"}), \
            mock.patch.object(proposer, "_chat", chat), \
            redirect_stdout(out):
        batch_run.stage_candidates(b, root, ["INGEST", "NER"])
    return b, out.getvalue()


class TestMentionFinishAccounting(unittest.TestCase):
    """A mention_missing cliff appears BY NAME AND COUNT
    in the finish record an operator signs off on — the
    #51 rule (the finish must state what the stages
    decided) applied to guard blocks. The surface pinned:
    batch.metrics["finish"]["candidates_blocked"]
    ({"total", "by_reason"}) plus the finish log line."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_block_counts_reach_the_finish_record(self):
        b, stage_out = _run_candidates_stage(
            self.root, "mention-finish",
            _scripted_items(100, omit=27), mention=True)
        # The stage's own accounting (pre-existing surface).
        cm = b.metrics["candidates"]
        self.assertEqual(cm["blocked"], 27)
        self.assertEqual(cm["blocked_reasons"],
                         {"mention_missing": 27})
        self.assertIn("mention_missing", stage_out)
        # The finish record restates it — before this, the
        # finish carried only the ER account and a guard
        # cliff was silent there.
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run._finish_summary(b, self.root, None)
        self.assertEqual(
            b.metrics["finish"]["candidates_blocked"],
            {"total": 27,
             "by_reason": {"mention_missing": 27}})
        self.assertIn("finish: CANDIDATES blocked 27",
                      out.getvalue())
        self.assertIn("mention_missing", out.getvalue())

    def test_no_blocks_no_finish_noise(self):
        b, _ = _run_candidates_stage(
            self.root, "mention-finish-clean",
            _scripted_items(12, omit=0), mention=True)
        self.assertEqual(b.metrics["candidates"]["blocked"], 0)
        with redirect_stdout(io.StringIO()):
            batch_run._finish_summary(b, self.root, None)
        self.assertNotIn("finish", b.metrics)


class TestMentionPartialCompliance(unittest.TestCase):
    """The cliff, characterized exactly. The fixture's
    omission rate is the measured Phase 2 rate: the A/B
    run recorded 69 mention_missing blocks out of 257
    proposals (≈27%; cap-recall-measurement.md, Phase 2
    section). Here 27 of 100 scripted proposals omit the
    mention fields; under the current fail-closed policy
    every one of them blocks by name before judging and
    every compliant proposal survives. This test is the
    reviewed artifact of that behavior — changing the
    policy means rewriting it deliberately."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_cliff_matches_the_omission_count(self):
        b, _ = _run_candidates_stage(
            self.root, "mention-cliff",
            _scripted_items(100, omit=27), mention=True)
        m = b.metrics["candidates"]
        self.assertEqual(m["proposer"]["proposed"], 100)
        self.assertEqual(m["blocked"], 27)
        self.assertEqual(m["blocked_reasons"],
                         {"mention_missing": 27})
        self.assertEqual(m["total"], 73)
        d = batch_dir(self.root, "mention-cliff")
        blocked = [json.loads(line)
                   for line in open(d / "candidates_blocked.jsonl")]
        self.assertEqual(len(blocked), 27)
        self.assertTrue(all(r["reason"] == "mention_missing"
                            for r in blocked))
        # The culled set is exactly the non-compliant slice.
        self.assertEqual({r["subject"] for r in blocked},
                         {f"Person {i}" for i in range(27)})
        kept = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertEqual(len(kept), 73)
        self.assertTrue(all("mentions" in r for r in kept))


class TestMentionCompliantParity(unittest.TestCase):
    """Fully compliant input: the guard never over-blocks
    well-formed mentions — identical proposals survive
    identically with the layer on and off."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_equal_survival_on_compliant_input(self):
        items = _scripted_items(40, omit=0)
        bm, _ = _run_candidates_stage(
            self.root, "mention-parity-on", items, mention=True)
        bl, _ = _run_candidates_stage(
            self.root, "mention-parity-off", items, mention=False)
        mm, lm = (bm.metrics["candidates"],
                  bl.metrics["candidates"])
        self.assertEqual(mm["blocked"], 0)
        self.assertEqual(lm["blocked"], 0)
        self.assertEqual(mm["total"], lm["total"])
        self.assertEqual(mm["total"], 40)
        # The record-shape difference is the layer's own:
        # kept mention-mode candidates carry the verified
        # mention records; legacy candidates never do.
        d = batch_dir(self.root, "mention-parity-on")
        rows = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertTrue(all("mentions" in r for r in rows))
        d = batch_dir(self.root, "mention-parity-off")
        rows = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertTrue(all("mentions" not in r for r in rows))


if __name__ == "__main__":
    unittest.main()
