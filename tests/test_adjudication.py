"""Pass D adjudication (0.2.1 two-pass build, freeze item 1).

The deterministic layer is tested directly: R5 quote
completeness classes, the extend_quote repair, R4's mandatory
veto record (including the veto -> retype conversion), and
the garbled-relation keep ban. The stage is tested with a
fake chat transport — no live model anywhere: the fake reads
the candidate blocks out of the prompt and answers from a
per-test decision map.
"""
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline import adjudication as adj
from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir
from pipeline.profile_config import (
    adjudication_layer_enabled,
    canonical_key_for,
    canonical_vocabularies,
    full_verification_gate_enabled,
    setup_config_errors,
)

TEXT = ("Acme Corp is headquartered in New York. "
        "The agreement is governed by the laws of the State of "
        "New York. Mara Voss founded Acme Corp in 2019.")


def _cand(**kw):
    c = {"doc_id": "doc-1", "subject": "Acme Corp",
         "relation": "is headquartered in", "object": "New York",
         "quote": "Acme Corp is headquartered in New York",
         "method": "llm", "context": ""}
    c.update(kw)
    return c


class TestQuoteFailureClass(unittest.TestCase):
    def test_complete_quote_passes(self):
        self.assertIsNone(adj.quote_failure_class(
            "Acme Corp is headquartered in New York",
            "Acme Corp", "is headquartered in", "New York", TEXT))

    def test_truncated_quote(self):
        self.assertEqual(
            adj.quote_failure_class(
                "Acme Corp is headquartered in New",
                "Acme Corp", "is headquartered in", "New York",
                TEXT),
            "object_overreach")

    def test_quote_not_in_excerpt_is_truncated(self):
        self.assertEqual(
            adj.quote_failure_class(
                "Acme Corp is headquartered in Boston",
                "Acme Corp", "is headquartered in", "Boston", TEXT),
            "truncated_quote")

    def test_subject_not_named(self):
        self.assertEqual(
            adj.quote_failure_class(
                "The agreement is governed by the laws of the "
                "State of New York",
                "Acme Corp", "is governed by", "New York", TEXT),
            "subject_not_named")

    def test_relation_unattested(self):
        self.assertEqual(
            adj.quote_failure_class(
                "Acme Corp is headquartered in New York",
                "Acme Corp", "acquired", "New York", TEXT),
            "relation_unattested")

    def test_garbled_relation(self):
        for rel in ("is", "is the", "is a", "are"):
            self.assertTrue(adj.relation_garbled(rel), rel)
        self.assertFalse(adj.relation_garbled("is headquartered in"))

    def test_doc_subject_waives_subject_and_relation(self):
        # Clause-slot candidate: subject is the document,
        # relation is the declared category label.
        self.assertIsNone(adj.quote_failure_class(
            "governed by the laws of the State of New York",
            "contract.txt", "Governing Law", "State of New York",
            TEXT, doc_subject=True))


class TestExtendSpan(unittest.TestCase):
    def test_extends_to_attesting_sentence(self):
        span = adj.find_extending_span(
            TEXT, "governed by the laws of the State of",
            "The agreement", "New York")
        self.assertIsNotNone(span)
        self.assertIn("State of New York", span)
        self.assertIn(span, TEXT)

    def test_no_attesting_span(self):
        self.assertIsNone(adj.find_extending_span(
            TEXT, "Acme Corp is headquartered in New York",
            "Acme Corp", "Boston"))


class TestEnforceDecision(unittest.TestCase):
    def test_clean_keep(self):
        row = adj.enforce_decision(
            {"decision": "keep", "clause": None, "reason": "ok",
             "r5_remedy": "none"}, _cand(), "cand-0", TEXT)
        self.assertEqual(row["decision"], "keep")
        self.assertEqual(row["r5_remedy"], "none")
        self.assertIsNone(row["enforcement"])

    def test_keep_repaired_by_extension(self):
        cand = _cand(quote="Acme Corp is headquartered in New")
        cand["object"] = "New York"
        row = adj.enforce_decision(
            {"decision": "keep",
             "final_quote": "Acme Corp is headquartered in New"},
            cand, "cand-0", TEXT)
        self.assertEqual(row["decision"], "keep")
        self.assertEqual(row["r5_remedy"], "extend_quote")
        self.assertEqual(row["enforcement"], "quote_extended")
        self.assertIn("New York.", row["final_quote"])

    def test_keep_unrepairable_becomes_veto_with_r4_record(self):
        cand = _cand(object="Boston",
                     quote="Acme Corp is headquartered in Boston")
        row = adj.enforce_decision(
            {"decision": "keep"}, cand, "cand-0", TEXT)
        self.assertEqual(row["decision"], "veto")
        self.assertEqual(row["enforcement"], "quote_incomplete")
        self.assertTrue(row["retype_attempt"])
        self.assertTrue(row["retype_failure"]["missing"])

    def test_garbled_keep_is_invalid(self):
        cand = _cand(relation="is")
        row = adj.enforce_decision(
            {"decision": "keep"}, cand, "cand-0", TEXT)
        self.assertEqual(row["decision"], "veto")
        self.assertEqual(row["enforcement"], "garbled_relation_keep")

    def test_veto_with_truthful_attempt_converts_to_retype(self):
        cand = _cand(relation="is located in")
        row = adj.enforce_decision(
            {"decision": "veto", "clause": 1,
             "reason": "loose phrasing",
             "retype_attempt": {
                 "subject": "Acme Corp",
                 "relation": "is headquartered in",
                 "object": "New York"},
             "retype_failure": {"clause": 1, "missing": "n/a"}},
            cand, "cand-0", TEXT)
        self.assertEqual(row["decision"], "retype")
        self.assertEqual(row["final_relation"], "is headquartered in")
        self.assertEqual(row["enforcement"],
                         "veto_converted_to_retype")

    def test_veto_missing_r4_fields_is_completed_and_named(self):
        row = adj.enforce_decision(
            {"decision": "veto", "clause": 2,
             "reason": "fabricated"},
            _cand(), "cand-0", TEXT)
        self.assertEqual(row["decision"], "veto")
        self.assertIn("unconstructible", row["retype_attempt"])
        self.assertIn("mandatory R4", row["retype_failure"]["missing"])

    def test_unrecognised_decision_fails_closed(self):
        row = adj.enforce_decision(
            {"decision": "shrug"}, _cand(), "cand-0", TEXT)
        self.assertEqual(row["decision"], "veto")
        self.assertEqual(row["enforcement"], "unrecognised_decision")


def _fake_chat(decisions_by_id, cost=0.001):
    """A chat transport answering from a decision map keyed by
    candidate id; blocks are parsed back out of the prompt so
    keep finals echo the candidate as typed."""
    def chat(prompt, model=None):
        out = []
        for block in re.split(r"\nCandidate ", prompt)[1:]:
            cid = block.split(":", 1)[0].strip()
            m = re.search(r"Triple: \((.*?) \| (.*?) \| (.*?)\)\n"
                          r"Evidence quote: (.*)\n", block)
            s, r, o, q = m.groups()
            d = dict(decisions_by_id.get(cid, {"decision": "keep"}))
            d.setdefault("candidate_id", cid)
            d.setdefault("final_subject", s)
            d.setdefault("final_relation", r)
            d.setdefault("final_object", o)
            d.setdefault("final_quote", q)
            d.setdefault("r5_remedy", "none")
            d.setdefault("clause", None)
            d.setdefault("reason", "")
            out.append(d)
        return json.dumps(out), cost
    return chat


class TestAdjudicateCandidates(unittest.TestCase):
    def test_mixed_outcomes(self):
        cands = [
            _cand(),
            _cand(subject="Mara Voss", relation="founded",
                  object="Acme Corp",
                  quote="Mara Voss founded Acme Corp in 2019"),
            _cand(subject="Acme Corp", relation="acquired",
                  object="Globex",
                  quote="Acme Corp is headquartered in New York"),
        ]
        chat = _fake_chat({
            "cand-2": {"decision": "veto", "clause": 1,
                       "reason": "manufactured object",
                       "retype_attempt": {
                           "unconstructible": "Globex is never "
                                              "mentioned"},
                       "retype_failure": {
                           "clause": 1,
                           "missing": "no acquisition is stated"}},
        })
        rows, survivors, cost = adj.adjudicate_candidates(
            chat, cands, {"doc-1": TEXT})
        self.assertEqual([r["decision"] for r in rows],
                         ["keep", "keep", "veto"])
        self.assertEqual(len(survivors), 2)
        self.assertEqual(survivors[0]["adjudication"]["decision"],
                         "keep")
        self.assertAlmostEqual(cost, 0.001)

    def test_missing_decision_is_a_named_veto(self):
        def chat(prompt, model=None):
            return json.dumps([]), 0.0
        rows, survivors, _ = adj.adjudicate_candidates(
            chat, [_cand()], {"doc-1": TEXT})
        self.assertEqual(rows[0]["decision"], "veto")
        self.assertEqual(rows[0]["enforcement"], "no_decision")
        self.assertEqual(survivors, [])

    def test_unparseable_response_raises(self):
        def chat(prompt, model=None):
            return "not json", 0.0
        with self.assertRaises(RuntimeError):
            adj.adjudicate_candidates(chat, [_cand()],
                                      {"doc-1": TEXT})


class TestDeclarations(unittest.TestCase):
    def test_switches_default_off_and_fail_closed(self):
        self.assertFalse(adjudication_layer_enabled(None))
        self.assertFalse(adjudication_layer_enabled({}))
        self.assertTrue(adjudication_layer_enabled(
            {"adjudication_layer": True}))
        with self.assertRaises(ValueError):
            adjudication_layer_enabled({"adjudication_layer": "yes"})
        self.assertFalse(full_verification_gate_enabled({}))
        with self.assertRaises(ValueError):
            full_verification_gate_enabled(
                {"full_verification_gate": 1})

    def test_setup_config_validation(self):
        self.assertEqual(setup_config_errors(
            {"adjudication_layer": True,
             "full_verification_gate": False}), [])
        errs = setup_config_errors({"adjudication_layer": "yes"})
        self.assertTrue(any("adjudication_layer" in e for e in errs))
        errs = setup_config_errors(
            {"canonical_vocabularies": {"Governing Law": {}}})
        self.assertTrue(any("canonical_vocabularies" in e
                            for e in errs))
        self.assertEqual(setup_config_errors(
            {"canonical_vocabularies":
             {"Governing Law": {"New York": "new_york"}}}), [])

    def test_canonical_lookup_is_surface_insensitive(self):
        cfg = {"canonical_vocabularies": {
            "Governing Law": {"State of New York": "new_york"}}}
        self.assertEqual(canonical_vocabularies(cfg)["Governing Law"]
                         ["State of New York"], "new_york")
        self.assertEqual(canonical_key_for(
            cfg, "Governing Law", "  state of  new york "),
            "new_york")
        self.assertIsNone(canonical_key_for(
            cfg, "Governing Law", "Delaware"))
        self.assertIsNone(canonical_key_for(cfg, "Other", "x"))


class TestStageAdjudicate(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.b = Batch(batch_id="adj-test", run_id="run-1",
                       source="corp", collection="docs",
                       doc_ids=["doc-1"], config=BatchConfig(),
                       status=BatchStatus.CANDIDATES_DONE,
                       profile_family="testfam")
        d = batch_dir(self.root, "adj-test")
        d.mkdir(parents=True)
        (d / "setup_config.resolved.json").write_text(json.dumps(
            {"adjudication_layer": True}))
        (d / "texts.jsonl").write_text(json.dumps(
            {"doc_id": "doc-1", "text": TEXT,
             "source_path": "doc-1.txt"}) + "\n")
        cands = [
            _cand(),
            _cand(subject="Acme Corp", relation="acquired",
                  object="Globex",
                  quote="Acme Corp is headquartered in New York"),
        ]
        with open(d / "candidates.jsonl", "w") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")

    def test_stage_writes_decisions_and_survivors(self):
        chat = _fake_chat({
            "cand-1": {"decision": "veto", "clause": 1,
                       "reason": "manufactured",
                       "retype_attempt": {
                           "unconstructible": "Globex unmentioned"},
                       "retype_failure": {"clause": 1,
                                          "missing": "no statement"}},
        })
        batch_run.stage_adjudicate(self.b, self.root, [], chat)
        d = batch_dir(self.root, "adj-test")
        decisions = [json.loads(line)
                     for line in open(d / "adjudication.jsonl")]
        self.assertEqual([r["decision"] for r in decisions],
                         ["keep", "veto"])
        survivors = [json.loads(line) for line in
                     open(d / "candidates_adjudicated.jsonl")]
        self.assertEqual(len(survivors), 1)
        self.assertEqual(self.b.status, BatchStatus.ADJUDICATED)
        self.assertEqual(self.b.metrics["adjudication"]["veto"], 1)
        # Pass C's record is untouched.
        cands = [json.loads(line) for line in open(d / "candidates.jsonl")]
        self.assertEqual(len(cands), 2)

    def test_gates_judges_adjudicated_survivors(self):
        chat = _fake_chat({
            "cand-1": {"decision": "veto", "clause": 1,
                       "reason": "manufactured",
                       "retype_attempt": {
                           "unconstructible": "Globex unmentioned"},
                       "retype_failure": {"clause": 1,
                                          "missing": "no statement"}},
        })
        batch_run.stage_adjudicate(self.b, self.root, [], chat)

        def decide(context, subject, relation, obj, model=None):
            return 0.9, {"cost": 0.001}
        batch_run.stage_gates(self.b, self.root, ["ADJUDICATE"],
                              decide)
        d = batch_dir(self.root, "adj-test")
        judgments = [json.loads(line)
                     for line in open(d / "judgments.jsonl")]
        self.assertEqual(len(judgments), 1)
        self.assertEqual(judgments[0]["object"], "New York")

    def test_declared_check(self):
        self.assertTrue(batch_run._declared_adjudication(
            self.b, self.root))
        bare = Batch(batch_id="bare", run_id="run-1", source="corp",
                     collection="docs", doc_ids=[],
                     config=BatchConfig())
        self.assertFalse(batch_run._declared_adjudication(
            bare, self.root))

    def test_run_batch_refuses_unwired_declared_layer(self):
        d = batch_dir(self.root, "adj-test")
        (d / "state.json").write_text(json.dumps(
            {"stages_completed": ["INGEST", "NER", "CANDIDATES"]}))
        with self.assertRaises(RuntimeError) as cm:
            batch_run.run_batch(self.b, self.root, None)
        self.assertIn("ADJUDICATE", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
