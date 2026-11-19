"""Absence materialisation (0.2.1, freeze item 4).

The clause sweep's record is complete only when an unfilled
declared category is recorded as a finding, not just a
filled one as a candidate. stage_candidates is driven with
the proposer functions mocked at their seam (no live model):
one category filled, two absent, and the Document Name
carrier exempt. The findings land in negative_findings.jsonl
as a distinct, citable record kind and are read back through
the store layer (RegistryStore.negative_findings).
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, BatchStatus, batch_dir
from pipeline.registry_store import RegistryStore

TEXT = ("Services Agreement. This agreement is governed by "
        "the laws of the State of New York.")


def _batch(root, batch_id, categories):
    b = Batch(batch_id=batch_id, run_id="run-1", source="cuad",
              collection="docs", doc_ids=["doc-1"],
              config=BatchConfig(), status=BatchStatus.NER_DONE,
              profile_family="testfam")
    d = batch_dir(root, batch_id)
    d.mkdir(parents=True)
    (d / "setup_config.resolved.json").write_text(json.dumps(
        {"label_categories": categories} if categories is not None
        else {}))
    (d / "texts.jsonl").write_text(json.dumps(
        {"doc_id": "doc-1", "text": TEXT,
         "source_path": "contract.txt"}) + "\n")
    return b


def _triples_empty(doc_id, text, key, **kw):
    return {"items": [], "proposed": 0, "calls": 0, "errors": 0,
            "chunks": 1, "chunks_at_cap": 0, "cost": 0.0}


def _slots(doc_id, text, categories, key, **kw):
    return {"fills": [
        {"category": "Document Name", "value": "Services Agreement",
         "quote": "Services Agreement"},
        {"category": "Governing Law", "value": "State of New York",
         "quote": "governed by the laws of the State of New York"},
    ], "proposed": 2, "calls": 1, "errors": 0, "cost": 0.0}


class TestAbsenceMaterialisation(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def _run_candidates(self, b):
        with mock.patch.dict(os.environ,
                             {"OPENROUTER_API_KEY": "sk-test"}), \
                mock.patch("pipeline.proposer.propose_triples_for_doc",
                           _triples_empty), \
                mock.patch("pipeline.proposer.propose_slots_for_doc",
                           _slots):
            batch_run.stage_candidates(b, self.root, ["INGEST", "NER"])

    def test_absent_categories_become_findings(self):
        b = _batch(self.root, "nf-test",
                   ["Document Name", "Governing Law", "Parties",
                    "Cap On Liability"])
        self._run_candidates(b)
        d = batch_dir(self.root, "nf-test")
        findings = [json.loads(line) for line in
                    open(d / "negative_findings.jsonl")]
        self.assertEqual(len(findings), 2)
        cats = sorted(f["category"] for f in findings)
        self.assertEqual(cats, ["Cap On Liability", "Parties"])
        for f in findings:
            self.assertEqual(f["kind"], "negative_finding")
            self.assertEqual(f["status"], "absent")
            self.assertEqual(f["subject"], "Services Agreement")
            self.assertTrue(f["finding_id"].startswith("nf-doc-1-"))
        ids = {f["finding_id"] for f in findings}
        self.assertEqual(ids, {"nf-doc-1-parties",
                               "nf-doc-1-cap-on-liability"})
        # The filled category is a candidate, not a finding.
        cands = [json.loads(line) for line in
                 open(d / "candidates.jsonl")]
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["relation"], "Governing Law")
        self.assertEqual(b.metrics["candidates"]
                         ["negative_findings"], 2)

    def test_store_layer_reads_findings_alongside_triples(self):
        b = _batch(self.root, "nf-test",
                   ["Document Name", "Governing Law", "Parties"])
        self._run_candidates(b)
        store = RegistryStore(self.root)
        findings = store.negative_findings("nf-test")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["category"], "Parties")
        # A batch with no findings file reads as empty.
        self.assertEqual(store.negative_findings("no-such"), [])

    def test_no_slot_sweep_no_findings(self):
        b = _batch(self.root, "nf-bare", None)
        self._run_candidates(b)
        d = batch_dir(self.root, "nf-bare")
        self.assertFalse((d / "negative_findings.jsonl").exists())
        self.assertNotIn("negative_findings",
                         b.metrics.get("candidates", {}))


if __name__ == "__main__":
    unittest.main()
