"""Serving coverage tests (issue #50).

The defect: INDEX chunked only documents holding a
Tier-2-routed candidate, so a Tier-1-heavy build's serving
index silently covered a subset of the corpus — and VERIFY
recorded green, because no check compared the index against
the documents built (multilingual pilot, 2026-10-27: 1 of 16
documents served, hybrid recall 0.00 even on English
controls).

These tests pin both halves of the fix:
- stage_index chunks EVERY built document, each chunk
  tagged by its document's best routed tier (the flagship
  projection's tagging);
- VERIFY's serving_coverage check FAILS BY NAME when the
  index's document set differs from the build's chunkable
  document set — a green VERIFY means coverage is complete.
"""
import io
import json
import pickle
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline import batch_run
from pipeline.batch import Batch, BatchConfig, batch_dir, save_batch

STAGES = ["INGEST", "NER", "CANDIDATES", "GATES", "ROUTE", "WRITE"]

TEXT_1 = ("Alder & Finch is a ceramics studio in Lisbon. "
          "The studio fires stoneware in a wood kiln and sells "
          "tableware to restaurants across Portugal. ") * 2
TEXT_2 = ("Bexhill Roasters imports green coffee from Huila. "
          "The roastery supplies cafes in Brighton and London "
          "with washed Colombian lots each spring. ") * 2
TEXT_3 = ("Casa Mirlo is a small hotel above the harbour. "
          "It has nine rooms, a tiled courtyard, and a roof "
          "terrace facing the lighthouse. ") * 2


class _StubEmbedder:
    """INDEX's embedder seat, stubbed (the test_store_identity
    pattern): zero vectors, no provider call, no spend."""

    def __init__(self):
        self.usage = {"input_tokens": 0, "cost": 0.0}

    def embed(self, chunks):
        return np.zeros((len(chunks), 3))


class _NoStoreGraph:
    """A graph double with no Neo4j client: VERIFY's
    store-side checks degrade to pass=None, and the serving
    projection is skipped — the file-side invariant is what
    is under test."""


def _batch(root: Path, batch_id: str = "cov-3docs") -> Batch:
    b = Batch(batch_id=batch_id, run_id="run-1", source="corpus",
              collection="docs",
              doc_ids=["doc-1", "doc-2", "doc-3"],
              config=BatchConfig(max_cost_usd=None))
    d = batch_dir(root, batch_id)
    d.mkdir(parents=True, exist_ok=True)
    texts = {"doc-1": TEXT_1, "doc-2": TEXT_2, "doc-3": TEXT_3}
    with open(d / "texts.jsonl", "w") as f:
        for doc_id, text in texts.items():
            f.write(json.dumps({"doc_id": doc_id, "text": text,
                                "source_path": doc_id + ".txt"})
                    + "\n")
    # doc-1 routes Tier 1 only; doc-2 Tier 2; doc-3 has no
    # routed rows at all (nothing survived to ROUTE).
    routed = [
        {"doc_id": "doc-1", "tier": 1, "tag": "VERIFIED"},
        {"doc_id": "doc-1", "tier": 1, "tag": "VERIFIED"},
        {"doc_id": "doc-2", "tier": 2, "tag": "UNVERIFIED"},
    ]
    with open(d / "routed.jsonl", "w") as f:
        for r in routed:
            f.write(json.dumps(r) + "\n")
    with open(d / "review_queue.jsonl", "w") as f:
        f.write(json.dumps({"doc_id": "doc-2"}) + "\n")
    b.metrics["route"] = {"tier1": 2, "tier2": 1, "dropped": 0}
    save_batch(b, root)
    return b


def _run_index(b: Batch, root: Path) -> None:
    with mock.patch.object(batch_run, "_index_embedder",
                           lambda model: _StubEmbedder()):
        with redirect_stdout(io.StringIO()):
            batch_run.stage_index(b, root, list(STAGES),
                                  graph=_NoStoreGraph())


class _CoverageCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.b = _batch(self.root)


class TestIndexChunksAllDocuments(_CoverageCase):
    def test_tier1_heavy_build_serves_every_document(self):
        _run_index(self.b, self.root)
        d = batch_dir(self.root, self.b.batch_id)
        with open(d / "tier2_index.pkl", "rb") as f:
            idx = pickle.load(f)
        docs = {m["doc_id"] for m in idx["meta"]}
        self.assertEqual(docs, {"doc-1", "doc-2", "doc-3"})
        self.assertEqual(self.b.metrics["index"]["documents"], 3)
        self.assertEqual(
            self.b.metrics["index"]["documents_total"], 3)

    def test_chunks_tagged_by_best_routed_tier(self):
        _run_index(self.b, self.root)
        d = batch_dir(self.root, self.b.batch_id)
        with open(d / "tier2_index.pkl", "rb") as f:
            idx = pickle.load(f)
        by_doc = {}
        for m in idx["meta"]:
            by_doc.setdefault(m["doc_id"], set()).add(
                (m["tier"], m["tag"]))
        self.assertEqual(by_doc["doc-1"], {(1, "VERIFIED")})
        self.assertEqual(by_doc["doc-2"], {(2, "UNVERIFIED")})
        # No routed rows: defaults to tier 2 / UNVERIFIED.
        self.assertEqual(by_doc["doc-3"], {(2, "UNVERIFIED")})


class TestVerifyCoverageInvariant(_CoverageCase):
    def _verify(self):
        with redirect_stdout(io.StringIO()):
            batch_run.stage_verify(
                self.b, self.root, [*STAGES, "INDEX"],
                registry=None, graph=_NoStoreGraph())

    def test_full_coverage_passes(self):
        _run_index(self.b, self.root)
        self._verify()
        cov = self.b.metrics["verify"]["checks"]["serving_coverage"]
        self.assertTrue(cov["pass"])
        self.assertEqual(cov["documents_expected"], 3)
        self.assertEqual(cov["documents_indexed"], 3)

    def test_dropped_document_fails_verify_by_name(self):
        _run_index(self.b, self.root)
        d = batch_dir(self.root, self.b.batch_id)
        pkl_p = d / "tier2_index.pkl"
        with open(pkl_p, "rb") as f:
            idx = pickle.load(f)
        keep = [i for i, m in enumerate(idx["meta"])
                if m["doc_id"] != "doc-2"]
        idx["meta"] = [idx["meta"][i] for i in keep]
        idx["chunks"] = [idx["chunks"][i] for i in keep]
        idx["embeddings"] = idx["embeddings"][keep]
        with open(pkl_p, "wb") as f:
            pickle.dump(idx, f)
        # Keep the artifact-vs-metrics check green so the
        # failure is attributable to coverage alone.
        self.b.metrics["index"]["chunks"] = len(idx["chunks"])
        save_batch(self.b, self.root)
        with self.assertRaises(RuntimeError) as ctx:
            self._verify()
        self.assertIn("serving_coverage", str(ctx.exception))
        cov = self.b.metrics["verify"]["checks"]["serving_coverage"]
        self.assertFalse(cov["pass"])
        self.assertEqual(cov["missing_documents"], ["doc-2"])


if __name__ == "__main__":
    unittest.main()
