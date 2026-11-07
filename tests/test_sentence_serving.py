"""Serving fix: sentence-level serving (repair 1) and the
beam traversal channel (repair 2).

Pins, in order:
- the shipped sentence index round-trip: a batch whose
  tier2_index.pkl carries sentences + sentence_embeddings
  loads them into the store (describe counts them;
  prepare is a no-op; the snapshot rows match the build's
  sentence universe);
- the query-time build: an older batch without the
  sentence keys gets the identical universe built from
  its chunks, with the note saying so;
- sentence_search end to end over a fake embedder:
  direct arm serves index sentences, the graph pool
  claims its quote sentence first (via="graph"), and a
  2-hop quote sentence that exists in NO chunk and NO
  index row enters only through the beam channel
  (via="beam") — the measured attribution shape;
- the beam caps (per-entity / per-hop) on a star graph,
  exercised through _beam_units directly;
- the rerank step's toggle: an injected reranker
  reorders the head; reranker=False fills in merge
  order and the note says the step was caller-disabled;
- the named refusal without an embedder, and the CLI's
  default path (sentence serving) plus its named
  fallback to chunk serving when no embedder resolves.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline import query_stack, search
from pipeline.sentence_index import build_sentence_rows

DOC_A = ("Alpha Corp founded Beta Labs in 2019. The Alpha "
         "Corp founders kept a majority stake in Beta Labs "
         "for many years afterwards.")
DOC_B = ("Cobalt Parts machines titanium housings for "
         "aerospace clients. The Cobalt Parts plant runs "
         "three shifts every weekday of the year.")

# The beam-only evidence: triple t2's quote exists in no
# document, so no chunk and no sentence-index row carries
# it; only the traversal can surface it.
BEAM_QUOTE = "Beta Labs supplies Cobalt Parts regularly."

TRIPLES = [
    {"subject": "Alpha Corp", "relation": "founded",
     "object": "Beta Labs", "doc_id": "doc-a",
     "quote": "Alpha Corp founded Beta Labs in 2019.",
     "p_true": 0.97, "tier": 1},
    {"subject": "Beta Labs", "relation": "supplies",
     "object": "Cobalt Parts", "doc_id": "doc-b",
     "quote": BEAM_QUOTE, "p_true": 0.9, "tier": 1},
]

QUERY = "What did Alpha Corp found?"


class _HashEmbedder:
    """Deterministic bag-of-words embedder: token-hash
    buckets, L2-normalized — sentences sharing tokens
    with the query score high, others low."""

    DIMS = 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.DIMS))
        for i, t in enumerate(texts):
            for tok in query_stack.tokenize(t):
                h = int(hashlib.sha1(tok.encode()).hexdigest(),
                        16)
                out[i, h % self.DIMS] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


def _write_batch(root: Path, with_sentences: bool) -> Path:
    """A minimal batch directory in the test_search form,
    optionally carrying the shipped sentence index."""
    bdir = root / "batches" / "fixbatch"
    bdir.mkdir(parents=True)
    (bdir / "batch.json").write_text(json.dumps({
        "batch_id": "fixbatch", "run_id": "r1",
        "source": "fixture", "collection": "fixture",
        "doc_ids": ["doc-a", "doc-b"], "config": {},
        "metrics": {}}))
    (bdir / "texts.jsonl").write_text(
        json.dumps({"doc_id": "doc-a", "text": DOC_A}) + "\n"
        + json.dumps({"doc_id": "doc-b", "text": DOC_B})
        + "\n")
    (bdir / "routed.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in TRIPLES))
    (bdir / "written_triples.jsonl").write_text(
        json.dumps({"subject_id": "ent-1",
                    "relation": "founded",
                    "object_id": "ent-2",
                    "doc_id": "doc-a"}) + "\n")
    chunker = query_stack.FixedWindowChunker()
    meta, chunks = [], []
    for doc_id, text in (("doc-a", DOC_A), ("doc-b", DOC_B)):
        for idx, ch in chunker.chunk(text):
            meta.append({"doc_id": doc_id, "chunk_idx": idx,
                         "tag": "VERIFIED", "tier": 1,
                         "batch_id": "fixbatch"})
            chunks.append(ch)
    emb = _HashEmbedder()
    index = {"embeddings": emb.embed(chunks),
             "meta": meta, "chunks": chunks,
             "model": "fixture"}
    if with_sentences:
        rows = build_sentence_rows(chunks, meta)
        index["sentences"] = rows
        index["sentence_embeddings"] = emb.embed(
            [r["text"] for r in rows])
    with open(bdir / "tier2_index.pkl", "wb") as f:
        pickle.dump(index, f)
    return bdir


class _BatchCase(unittest.TestCase):
    WITH_SENTENCES = True

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bdir = _write_batch(self.root, self.WITH_SENTENCES)
        self.store = search.FileServingStore(
            self.bdir, root=self.root)
        self.emb = _HashEmbedder()

    def tearDown(self):
        self._tmp.cleanup()


class TestShippedSentenceIndex(_BatchCase):

    def test_describe_counts_sentences(self):
        # 2 sentences per document, all distinct
        self.assertEqual(self.store.describe()["sentences"], 4)

    def test_prepare_is_noop_and_snapshot_matches(self):
        self.assertIsNone(
            self.store.prepare_sentence_index(self.emb))
        rows, embs = self.store.sentence_snapshot()
        self.assertEqual(len(rows), 4)
        self.assertEqual(embs.shape, (4, _HashEmbedder.DIMS))
        self.assertEqual(rows[0]["text"],
                         "Alpha Corp founded Beta Labs in 2019.")
        self.assertEqual(rows[0]["doc_id"], "doc-a")
        self.assertTrue(rows[0]["chunk_id"].endswith("doc-a:0"))


class TestQueryTimeBuild(_BatchCase):
    WITH_SENTENCES = False

    def test_old_build_has_no_snapshot_until_prepared(self):
        self.assertIsNone(self.store.describe()["sentences"])
        self.assertIsNone(self.store.sentence_snapshot())
        note = self.store.prepare_sentence_index(self.emb)
        self.assertIn("built at query time", note)
        rows, _embs = self.store.sentence_snapshot()
        self.assertEqual(len(rows), 4)
        # memoized: a second prepare is a no-op
        self.assertIsNone(
            self.store.prepare_sentence_index(self.emb))


class TestSentenceSearch(_BatchCase):

    def test_arms_and_attribution(self):
        r = search.sentence_search(self.store, QUERY,
                                   embedder=self.emb,
                                   reranker=False)
        self.assertEqual(r["arms"],
                         ["sentences", "graph-pool", "beam"])
        by_text = {u["text"]: u for u in r["units"]}
        # graph pool claims its quote sentence first
        pool_u = by_text["Alpha Corp founded Beta Labs in 2019."]
        self.assertEqual(pool_u["via"], "graph")
        # the 2-hop quote exists in no chunk / index row:
        # it can only have arrived through the beam channel
        beam_u = by_text[BEAM_QUOTE]
        self.assertEqual(beam_u["via"], "beam")
        self.assertEqual(beam_u["kind"], "sentence")
        # direct arm served at least one index sentence
        self.assertTrue(any(u["via"] == "direct"
                            for u in r["units"]))
        # every unit carries the fill bookkeeping
        for u in r["units"]:
            self.assertIn("tokens", u)
            self.assertIn("score", u)

    def test_budget_is_respected(self):
        r = search.sentence_search(self.store, QUERY,
                                   embedder=self.emb,
                                   reranker=False,
                                   budget_tokens=30)
        total = sum(u["tokens"] for u in r["units"])
        self.assertLessEqual(total, 30)

    def test_refusal_without_embedder(self):
        with self.assertRaises(
                search.SentenceServingUnavailable):
            search.sentence_search(self.store, QUERY,
                                   embedder=None)


class _ReverseReranker:
    """Scores by position: later head units outrank
    earlier ones, reversing the head."""

    def score(self, query, texts):
        return [float(i) for i in range(len(texts))]


class TestRerankToggle(_BatchCase):

    def test_injected_reranker_reorders_head(self):
        plain = search.sentence_search(self.store, QUERY,
                                       embedder=self.emb,
                                       reranker=False)
        ranked = search.sentence_search(
            self.store, QUERY, embedder=self.emb,
            reranker=_ReverseReranker())
        self.assertNotEqual(
            [u["text"] for u in plain["units"]],
            [u["text"] for u in ranked["units"]])

    def test_disabled_rerank_is_noted(self):
        r = search.sentence_search(self.store, QUERY,
                                   embedder=self.emb,
                                   reranker=False)
        self.assertTrue(any("disabled by the caller" in n
                            for n in r["notes"]))


class TestBeamCaps(unittest.TestCase):
    """A star graph: one hub entity with 30 triples. The
    beam keeps at most _BEAM_PER_ENTITY of them at hop 1,
    so its quote units are capped there too."""

    def test_per_entity_cap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bdir = root / "batches" / "star"
            bdir.mkdir(parents=True)
            (bdir / "batch.json").write_text(json.dumps({
                "batch_id": "star", "run_id": "r1",
                "source": "fixture", "collection": "fixture",
                "doc_ids": ["doc-a"], "config": {},
                "metrics": {}}))
            (bdir / "texts.jsonl").write_text(
                json.dumps({"doc_id": "doc-a",
                            "text": DOC_A}) + "\n")
            triples = [{
                "subject": "Hub Co", "relation": f"rel{i}",
                "object": f"Spoke {i}", "doc_id": "doc-a",
                "quote": f"Hub Co rel{i} Spoke {i} uniquely.",
                "p_true": 0.9, "tier": 1} for i in range(30)]
            (bdir / "routed.jsonl").write_text(
                "".join(json.dumps(t) + "\n" for t in triples))
            (bdir / "written_triples.jsonl").write_text("")
            chunker = query_stack.FixedWindowChunker()
            meta, chunks = [], []
            for idx, ch in chunker.chunk(DOC_A):
                meta.append({"doc_id": "doc-a",
                             "chunk_idx": idx, "tag": "VERIFIED",
                             "tier": 1, "batch_id": "star"})
                chunks.append(ch)
            emb = _HashEmbedder()
            with open(bdir / "tier2_index.pkl", "wb") as f:
                pickle.dump({"embeddings": emb.embed(chunks),
                             "meta": meta, "chunks": chunks,
                             "model": "fixture"}, f)
            store = search.FileServingStore(bdir, root=root)
            store.prepare_sentence_index(emb)
            ts = store.triples()
            linked, by_key = search._link_entities(
                ts, "What does Hub Co do?")
            self.assertEqual(linked, ["hub co"])
            qvec = emb.embed(["What does Hub Co do?"])[0]
            units = search._beam_units(
                store, ts, by_key, linked, qvec, emb,
                store.batch_filter(), {}, {}, [])
            self.assertTrue(units)
            self.assertLessEqual(len(units),
                                 search._BEAM_PER_ENTITY)
            self.assertTrue(all(u["via"] == "beam"
                                for u in units))


class TestCliDefaultPath(_BatchCase):

    def _run_main(self, embedder):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(search, "try_embedder",
                               lambda: embedder):
            with redirect_stdout(out), redirect_stderr(err):
                rc = search.main(
                    ["--source", str(self.bdir), QUERY])
        return rc, out.getvalue(), err.getvalue()

    def test_sentence_serving_is_the_default(self):
        rc, out, _err = self._run_main(self.emb)
        self.assertEqual(rc, 0)
        self.assertIn("Evidence units", out)
        self.assertIn("beam", out)

    def test_fallback_to_chunk_serving_is_named(self):
        rc, out, _err = self._run_main(None)
        self.assertEqual(rc, 0)
        self.assertIn("fell back to chunk-level hybrid search",
                      out)
        self.assertIn("Documents", out)


if __name__ == "__main__":
    unittest.main()
