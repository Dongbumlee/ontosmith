"""Mention-layer tests (issue #54).

The diagnosis: entities register and link by noun surface
forms, so facts stated via pronouns, definite descriptions,
or cross-sentence evidence have no citable path to an
entity, and drops record only a score. The layer's first
cut (profile-gated by `mention_layer`, English packs
first):

- the proposer emits per-endpoint mention spans + types +
  verbatim antecedent resolutions;
- the guards verify every DECLARED span verbatim — a
  present-but-invalid mention blocks by name
  (mention_missing / mention_span / mention_antecedent),
  never silently; an endpoint whose mention data is
  absent falls back to the legacy evidence path and is
  judged on its quote (cut 2 — the layer is additive:
  only a declared-but-invalid mention can block), and
  every fallback is counted by endpoint in the batch
  accounting;
- GATES judges each resolution as its own Jev decision
  before the fact: clear -> judged as usual; review band
  -> judged but capped at Tier-2 review by ROUTE;
  rejected -> never fact-judged, dropped with reason
  class resolution_rejected, distinct from fact_rejected.

Pinned here: the accessor fails closed; the undeclared
path is byte-identical (legacy prompt, legacy records);
the span discipline; the fallback policy; the three
resolution outcomes; the route cap and the drop reason
classes.

Batch-level guards added after the Phase 2 verdict (the
A/B failed on an interaction the unit tests above never
simulated — see the verdict record in the changelog):
the finish record restates guard-block counts by reason
and mention fallback counts by endpoint, so neither a
block cliff nor a silent fallback can hide at the
finish; a scripted partial-compliance proposer (27%
field omission, the measured Phase 2 rate) survives via
fallback with the fallback count equal to the omission
count; and fully compliant input survives identically
with the layer on and off.
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

    def test_missing_mentions_fall_back(self):
        # Cut 2: no mention data at all is NOT a block.
        # Both endpoints fall back to the legacy evidence
        # path, the fallback is named on the kept record,
        # and the record carries no mentions — exactly the
        # legacy shape.
        kept, blocked = self.guard(_item())
        self.assertEqual(blocked, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["mention_fallback"],
                         ["subject", "object"])
        self.assertNotIn("mentions", kept[0])
        # The parse layer's shape for omitted fields —
        # an empty record per endpoint — falls back too.
        kept, blocked = self.guard(_item(
            mentions={"subject": {}, "object": {}}))
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["mention_fallback"],
                         ["subject", "object"])
        self.assertNotIn("mentions", kept[0])

    def test_partial_absence_falls_back_per_endpoint(self):
        # Object declared and valid, subject absent: only
        # the subject falls back; the verified object
        # mention rides into candidacy.
        m = {"object": {"span": "Power Corp", "type": "named"}}
        kept, blocked = self.guard(_item(mentions=m))
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["mention_fallback"],
                         ["subject"])
        self.assertEqual(set(kept[0]["mentions"]), {"object"})

    def test_fallback_does_not_excuse_an_invalid_mention(self):
        # Subject absent (would fall back), object declared
        # with a span outside the quote: the declared-but-
        # invalid mention still blocks the item by name.
        m = {"object": {"span": "the company", "type": "named"}}
        _kept, blocked = self.guard(_item(mentions=m))
        self.assertEqual(blocked[0]["reason"], "mention_span")

    def test_bad_type_is_mention_missing(self):
        # A span WAS declared; its malformed type makes the
        # mention present-but-invalid, not absent — it
        # blocks, it does not fall back.
        m = _named_mentions()
        m["subject"] = {"span": "Paul Desmarais Jr.", "type": "pronoun"}
        kept, blocked = self.guard(_item(mentions=m))
        self.assertEqual(kept, [])
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
        self.assertNotIn("mention_fallback", kept[0])


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
# the finish, its exact shape under partial compliance,
# and its absence under full compliance. Cut 2 changed
# the guard policy (absence falls back to the legacy
# evidence path; only a declared-but-invalid mention
# blocks), and these tests were rewritten with it on
# purpose: the omission slice survives via fallback, and
# the fallback count is pinned against the omission count
# at the stage and at the finish.


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


def _scripted_invalid_items(n):
    """n proposals that DECLARE mention data, badly: the
    subject's mention span ("He") does not appear in the
    quote, so the mention is present-but-invalid — the
    shape that still blocks by name under cut 2
    (mention_span)."""
    items = []
    for i in range(n):
        items.append({
            "subject": f"Person {i}", "relation": "leads",
            "object": f"Company {i}",
            "quote": f"Person {i} leads Company {i}",
            "subject_mention": "He",
            "subject_mention_type": "pronominal",
            "subject_antecedent": f"Person {i}",
            "object_mention": f"Company {i}",
            "object_mention_type": "named"})
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
    """Guard blocks and mention fallbacks appear BY NAME
    AND COUNT in the finish record an operator signs off
    on — the #51 rule (the finish must state what the
    stages decided) applied to both. The surfaces pinned:
    batch.metrics["finish"]["candidates_blocked"]
    ({"total", "by_reason"}) and ["candidates_fallback"]
    ({"endpoints", "by_endpoint"}), plus the finish log
    lines."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_block_counts_reach_the_finish_record(self):
        # Declared-but-invalid mentions (span outside the
        # quote): the shape that still blocks under cut 2.
        b, stage_out = _run_candidates_stage(
            self.root, "mention-finish",
            _scripted_invalid_items(27), mention=True)
        # The stage's own accounting (pre-existing surface).
        cm = b.metrics["candidates"]
        self.assertEqual(cm["blocked"], 27)
        self.assertEqual(cm["blocked_reasons"],
                         {"mention_span": 27})
        self.assertIn("mention_span", stage_out)
        # The finish record restates it — before this, the
        # finish carried only the ER account and a guard
        # cliff was silent there.
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run._finish_summary(b, self.root, None)
        self.assertEqual(
            b.metrics["finish"]["candidates_blocked"],
            {"total": 27,
             "by_reason": {"mention_span": 27}})
        self.assertNotIn("candidates_fallback",
                         b.metrics["finish"])
        self.assertIn("finish: CANDIDATES blocked 27",
                      out.getvalue())
        self.assertIn("mention_span", out.getvalue())

    def test_fallback_counts_reach_the_finish_record(self):
        # The Phase 2 omission shape under cut 2: nothing
        # blocks, and the finish must say how much of the
        # run was judged on the legacy evidence path — a
        # fallback with no finish line would read as an
        # ordinary fully-compliant mention run.
        b, stage_out = _run_candidates_stage(
            self.root, "mention-finish-fallback",
            _scripted_items(100, omit=27), mention=True)
        self.assertEqual(b.metrics["candidates"]["blocked"], 0)
        self.assertIn("mention fallback: 54", stage_out)
        out = io.StringIO()
        with redirect_stdout(out):
            batch_run._finish_summary(b, self.root, None)
        self.assertEqual(
            b.metrics["finish"]["candidates_fallback"],
            {"endpoints": 54,
             "by_endpoint": {"subject": 27, "object": 27}})
        self.assertNotIn("candidates_blocked",
                         b.metrics["finish"])
        self.assertIn("finish: CANDIDATES mention fallback "
                      "54 endpoint(s)", out.getvalue())

    def test_no_blocks_no_finish_noise(self):
        b, _ = _run_candidates_stage(
            self.root, "mention-finish-clean",
            _scripted_items(12, omit=0), mention=True)
        cm = b.metrics["candidates"]
        self.assertEqual(cm["blocked"], 0)
        self.assertEqual(cm["proposer"]
                         ["mention_fallback_subject"], 0)
        self.assertEqual(cm["proposer"]
                         ["mention_fallback_object"], 0)
        with redirect_stdout(io.StringIO()):
            batch_run._finish_summary(b, self.root, None)
        self.assertNotIn("finish", b.metrics)


class TestMentionPartialCompliance(unittest.TestCase):
    """Partial compliance, characterized exactly. The
    fixture's omission rate is the measured Phase 2 rate:
    the v1 A/B run recorded 69 mention_missing blocks out
    of 257 proposals (≈27%; cap-recall-measurement.md,
    Phase 2 section) — under v1's fail-closed policy the
    omission slice was culled before judging and Tier-1
    recall fell 0.1066 -> 0.0772. Cut 2 makes the layer
    additive, and this test is the reviewed artifact of
    the new behavior: here 27 of 100 scripted proposals
    omit the mention fields, EVERY proposal survives —
    the 27 omission items via fallback, the 73 compliant
    ones with verified mentions — and the fallback count
    equals the omission count exactly (27 items x 2
    endpoints). The v1 cliff is now a measured fallback
    rate; changing either means rewriting this test
    deliberately."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_omission_slice_falls_back_and_is_counted(self):
        b, _ = _run_candidates_stage(
            self.root, "mention-cliff",
            _scripted_items(100, omit=27), mention=True)
        m = b.metrics["candidates"]
        self.assertEqual(m["proposer"]["proposed"], 100)
        self.assertEqual(m["blocked"], 0)
        self.assertEqual(m["blocked_reasons"], {})
        self.assertEqual(m["total"], 100)
        self.assertEqual(
            m["proposer"]["mention_fallback_subject"], 27)
        self.assertEqual(
            m["proposer"]["mention_fallback_object"], 27)
        d = batch_dir(self.root, "mention-cliff")
        self.assertFalse(
            (d / "candidates_blocked.jsonl").exists())
        kept = [json.loads(line)
                for line in open(d / "candidates.jsonl")]
        self.assertEqual(len(kept), 100)
        # The fallback set is exactly the non-compliant
        # slice, named on each record, and those records
        # are otherwise the legacy shape (no mentions).
        fb = [r for r in kept if "mention_fallback" in r]
        self.assertEqual(len(fb), 27)
        self.assertEqual({r["subject"] for r in fb},
                         {f"Person {i}" for i in range(27)})
        self.assertTrue(all(
            r["mention_fallback"] == ["subject", "object"]
            for r in fb))
        self.assertTrue(all("mentions" not in r for r in fb))
        withm = [r for r in kept if "mentions" in r]
        self.assertEqual(len(withm), 73)


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
        # Fully compliant input never falls back.
        self.assertEqual(
            mm["proposer"]["mention_fallback_subject"], 0)
        self.assertEqual(
            mm["proposer"]["mention_fallback_object"], 0)
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
