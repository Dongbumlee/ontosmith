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
"""
import io
import json
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


if __name__ == "__main__":
    unittest.main()
