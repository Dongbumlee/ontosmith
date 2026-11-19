"""Recording-time normalisation (0.2.1, freeze item 3).

Declared canonical vocabularies resolve verbatim slot values
to canonical keys on the candidate record at CANDIDATES;
WRITE stores the object under the key, keeps the verbatim
value as an alias and on the written record; unmapped values
are named, counted build events. First case: the legal
pack's Governing Law vocabulary (pack 0.2.1) — the
'Commonwealth of Virginia' fold gap is closed at record
time, not at derivation time.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir
from pipeline.profile_config import (
    canonical_key_for,
    resolve_setup_config,
)

ROOT = Path(__file__).resolve().parent.parent

CFG = {"canonical_vocabularies": {"Governing Law": {
    "State of New York": "new_york",
    "New York": "new_york",
    "Commonwealth of Virginia": "virginia",
    "Virginia": "virginia"}}}


def _slot(object_, **kw):
    c = {"doc_id": "doc-1", "subject": "contract.txt",
         "relation": "Governing Law", "object": object_,
         "method": "clause-llm", "quote": "x"}
    c.update(kw)
    return c


class TestApplyCanonical(unittest.TestCase):
    def test_mapped_and_unmapped_accounted(self):
        cands = [_slot("Commonwealth of Virginia"),
                 _slot("State of New York"),
                 _slot("Atlantis"),
                 _slot("Virginia", relation="Parties")]
        stats = batch_run._apply_canonical_vocabularies(cands, CFG)
        self.assertEqual(cands[0]["canonical_object"], "virginia")
        self.assertEqual(cands[1]["canonical_object"], "new_york")
        self.assertNotIn("canonical_object", cands[2])
        self.assertNotIn("canonical_object", cands[3])
        self.assertEqual(stats["mapped"], 2)
        self.assertEqual(stats["unmapped"],
                         {"Governing Law": {"Atlantis": 1}})

    def test_no_declaration_is_a_no_op(self):
        cands = [_slot("Virginia")]
        stats = batch_run._apply_canonical_vocabularies(cands, {})
        self.assertFalse(stats["declared"])
        self.assertNotIn("canonical_object", cands[0])

    def test_idempotent(self):
        cands = [_slot("Virginia")]
        batch_run._apply_canonical_vocabularies(cands, CFG)
        stats = batch_run._apply_canonical_vocabularies(cands, CFG)
        self.assertEqual(stats["mapped"], 0)
        self.assertEqual(cands[0]["canonical_object"], "virginia")


class TestLegalPackVocabulary(unittest.TestCase):
    def test_governing_law_keys_resolve(self):
        cfg = resolve_setup_config(
            ROOT, ROOT / "packs" / "legal-contract")
        for surface, key in (
                ("Commonwealth of Virginia", "virginia"),
                ("Virginia", "virginia"),
                ("State of New York", "new_york"),
                ("the laws of the State of California",
                 "california"),
                ("Commonwealth of Massachusetts",
                 "massachusetts"),
                ("Delaware", "delaware")):
            self.assertEqual(
                canonical_key_for(cfg, "Governing Law", surface),
                key, surface)
        self.assertIsNone(canonical_key_for(
            cfg, "Governing Law", "the Moon"))
        self.assertIsNone(canonical_key_for(
            cfg, "Parties", "Virginia"))


class TestWriteCanonicalObject(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.b = Batch(batch_id="canon-test", run_id="run-1",
                       source="cuad", collection="docs",
                       doc_ids=["doc-1"], config=BatchConfig(),
                       status=BatchStatus.ROUTED,
                       profile_family="testfam")
        d = batch_dir(self.root, "canon-test")
        d.mkdir(parents=True)
        (d / "setup_config.resolved.json").write_text(json.dumps(
            {"label_categories": ["Governing Law"]}))
        rows = [{
            "doc_id": "doc-1", "subject": "contract.txt",
            "relation": "Governing Law",
            "object": "Commonwealth of Virginia",
            "canonical_object": "virginia",
            "quote": "governed by the laws of the Commonwealth "
                     "of Virginia",
            "context": "", "p_true": 0.9, "tier": 1}]
        with open(d / "routed.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_write_stores_key_and_keeps_verbatim(self):
        from pipeline.file_registry import SqliteEntityRegistry

        class FakeGraph:
            def __init__(self):
                self.written = []

            def write_triples(self, triples):
                self.written = list(triples)
                return len(triples)

        registry = SqliteEntityRegistry(
            self.root / "data" / "registry.db")
        graph = FakeGraph()
        batch_run.stage_write(self.b, self.root, [], registry, graph)
        self.assertEqual(len(graph.written), 1)
        obj_ent = registry.get(graph.written[0].object_id)
        self.assertEqual(obj_ent.name, "virginia")
        self.assertIn("Commonwealth of Virginia", obj_ent.aliases)
        d = batch_dir(self.root, "canon-test")
        written = [json.loads(line) for line in
                   open(d / "written_triples.jsonl")]
        self.assertEqual(written[0]["canonical_object"], "virginia")
        self.assertEqual(written[0]["verbatim_object"],
                         "Commonwealth of Virginia")
        self.assertEqual(
            self.b.metrics["write"]["canonicalized_objects"], 1)


class TestAdjudicationRederivation(unittest.TestCase):
    def test_retyped_object_gets_its_own_key(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        b = Batch(batch_id="rederive", run_id="run-1", source="cuad",
                  collection="docs", doc_ids=["doc-1"],
                  config=BatchConfig(),
                  status=BatchStatus.CANDIDATES_DONE,
                  profile_family="testfam")
        d = batch_dir(root, "rederive")
        d.mkdir(parents=True)
        (d / "setup_config.resolved.json").write_text(json.dumps(
            {"adjudication_layer": True, **CFG}))
        text = ("This agreement is governed by the laws of the "
                "Commonwealth of Virginia.")
        (d / "texts.jsonl").write_text(json.dumps(
            {"doc_id": "doc-1", "text": text,
             "source_path": "contract.txt"}) + "\n")
        cand = {"doc_id": "doc-1", "subject": "contract.txt",
                "relation": "Governing Law",
                "object": "State of Virginia",
                "canonical_object": "virginia",
                "quote": "governed by the laws of the "
                         "Commonwealth of Virginia",
                "method": "clause-llm", "context": ""}
        with open(d / "candidates.jsonl", "w") as f:
            f.write(json.dumps(cand) + "\n")

        def chat(prompt, model=None):
            return json.dumps([{
                "candidate_id": "cand-0", "decision": "retype",
                "final_subject": "contract.txt",
                "final_relation": "Governing Law",
                "final_object": "Commonwealth of Virginia",
                "final_quote": "governed by the laws of the "
                               "Commonwealth of Virginia",
                "r5_remedy": "none", "clause": 5,
                "reason": "stated form"}]), 0.001

        batch_run.stage_adjudicate(b, root, [], chat)
        survivors = [json.loads(line) for line in
                     open(d / "candidates_adjudicated.jsonl")]
        self.assertEqual(len(survivors), 1)
        self.assertEqual(survivors[0]["object"],
                         "Commonwealth of Virginia")
        self.assertEqual(survivors[0]["canonical_object"],
                         "virginia")


if __name__ == "__main__":
    unittest.main()
