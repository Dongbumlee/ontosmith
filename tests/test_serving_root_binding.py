"""Serving root binding (0.2.2): a batch directory's own
tree is authoritative for that batch.

The defect class under test: the serving stores bound
their registry reads (corpus, negative findings, the
tier-2 index, the SQLite entity registry) to the root
the caller passed — ``open_store`` always passes the
process state root — so a batch served from another
tree resolved those reads under the state root's
``batches/<same id>`` instead: silently empty, or a
DIFFERENT batch that happens to share the id. A
wire-level MCP check caught it (service-layer tests,
which construct the store with the fixture's own root,
could not see it). These tests reproduce it at the
``open_store`` level and pin the fix: the served
directory's tree wins, in both stores and every source
form; the canonical state-root layout is unchanged.
"""
import json
import pickle

import numpy as np

from pipeline import query_stack, search


def _write_batch_dir(d, *, corpus, finding_id, marker):
    """A minimal built batch in ``d``: manifest, texts,
    one routed triple, one recorded absence, and a
    tier-2 index whose single chunk carries ``marker``
    (and the corpus name only via batch.json, so the
    corpus binding is observable through chunk rows)."""
    d.mkdir(parents=True)
    text = (f"Acme Corp acquired Beta LLC in 2021. "
            f"The {marker} schedule was attached as an "
            f"annex to the agreement, and the parties "
            f"recorded the {marker} terms in the closing "
            f"memorandum for the transaction.")
    (d / "batch.json").write_text(json.dumps({
        "batch_id": d.name, "run_id": "r1",
        "source": corpus, "collection": "fixture",
        "doc_ids": ["doc-a"], "config": {}, "metrics": {}}))
    (d / "texts.jsonl").write_text(
        json.dumps({"doc_id": "doc-a", "text": text}) + "\n")
    (d / "routed.jsonl").write_text(json.dumps({
        "subject": "Acme Corp", "relation": "acquired",
        "object": "Beta LLC", "doc_id": "doc-a",
        "quote": "Acme Corp acquired Beta LLC in 2021.",
        "p_true": 0.97, "tier": 1}) + "\n")
    (d / "negative_findings.jsonl").write_text(json.dumps({
        "kind": "negative_finding", "finding_id": finding_id,
        "batch_id": d.name, "run_id": "r1", "doc_id": "doc-a",
        "subject": "Acme Master Agreement",
        "category": "Termination Clause", "status": "absent",
        "method": "clause-llm"}) + "\n")
    chunker = query_stack.FixedWindowChunker()
    meta, chunks = [], []
    for idx, ch in chunker.chunk(text):
        meta.append({"doc_id": "doc-a", "chunk_idx": idx,
                     "tag": "VERIFIED", "tier": 1,
                     "batch_id": d.name})
        chunks.append(ch)
    index = {
        "embeddings": np.tile(
            np.array([[1.0, 0.0]]), (len(chunks), 1)),
        "meta": meta, "chunks": chunks, "model": "fixture",
        "sentences": [{"text": "Acme Corp acquired Beta LLC "
                               "in 2021.",
                       "doc_id": "doc-a",
                       "chunk_id": "doc-a:0"}],
        "sentence_embeddings": np.array([[1.0, 0.0]]),
    }
    with open(d / "tier2_index.pkl", "wb") as f:
        pickle.dump(index, f)
    return d


def _elsewhere_batch(tmp_path):
    return _write_batch_dir(
        tmp_path / "elsewhere" / "batches" / "fixbatch",
        corpus="elsewhere-corpus",
        finding_id="nf-doc-a-termination-clause",
        marker="zephyr")


def test_source_tree_wins_over_empty_state_root(
        tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("ONTOSMITH_ROOT", str(state))
    d = _elsewhere_batch(tmp_path)
    store = search.open_store(source=str(d), store="file",
                              env={})
    # The absence record resolves from the source tree.
    assert [f["finding_id"]
            for f in store.negative_findings()] == [
                "nf-doc-a-termination-clause"]
    assert store.triples(), "the routed triple serves"
    # The tier-2 index resolves from the source tree too:
    # shipped embeddings and the sentence index are live,
    # and chunk rows carry the source tree's corpus.
    desc = store.describe()
    assert desc["embeddings"] is True
    assert desc["sentences"] == 1
    rows = store.search_chunks(
        text="zephyr",
        filters={"corpus": "elsewhere-corpus"})
    assert rows and "zephyr" in rows[0]["text"]


def test_same_id_collision_source_tree_wins(
        tmp_path, monkeypatch):
    state = tmp_path / "state"
    _write_batch_dir(
        state / "batches" / "fixbatch",
        corpus="state-corpus",
        finding_id="nf-state-termination-clause",
        marker="stateword")
    monkeypatch.setenv("ONTOSMITH_ROOT", str(state))
    d = _elsewhere_batch(tmp_path)
    store = search.open_store(source=str(d), store="file",
                              env={})
    # Nothing from the state root's same-id batch leaks
    # into the served answers.
    assert [f["finding_id"]
            for f in store.negative_findings()] == [
                "nf-doc-a-termination-clause"]
    rows = store.search_chunks(
        text="zephyr",
        filters={"corpus": "elsewhere-corpus"})
    assert rows and "zephyr" in rows[0]["text"]
    assert store.search_chunks(
        text="stateword",
        filters={"corpus": "state-corpus"}) == []


def test_batch_under_state_root_unchanged(
        tmp_path, monkeypatch):
    state = tmp_path / "state"
    d = _write_batch_dir(
        state / "batches" / "fixbatch",
        corpus="state-corpus",
        finding_id="nf-doc-a-termination-clause",
        marker="zephyr")
    monkeypatch.setenv("ONTOSMITH_ROOT", str(state))
    for store in (search.open_store(batch="fixbatch",
                                    env={}),
                  search.open_store(source=str(d), env={})):
        assert [f["finding_id"]
                for f in store.negative_findings()] == [
                    "nf-doc-a-termination-clause"]
        rows = store.search_chunks(
            text="zephyr",
            filters={"corpus": "state-corpus"})
        assert rows and "zephyr" in rows[0]["text"]


def test_standalone_batch_dir_reads_itself(tmp_path):
    # A batch directory outside the canonical layout
    # (not under any batches/ tree) has no containing
    # registry tree: its own files are the whole record,
    # and a foreign root passed alongside must not bind.
    foreign = tmp_path / "foreign"
    _write_batch_dir(
        foreign / "batches" / "fixbatch",
        corpus="foreign-corpus",
        finding_id="nf-foreign-termination-clause",
        marker="foreignword")
    d = _write_batch_dir(
        tmp_path / "loose" / "fixbatch",
        corpus="loose-corpus",
        finding_id="nf-loose-termination-clause",
        marker="looseword")
    for root in (foreign, None):
        store = search.FileServingStore(d, root=root)
        assert [f["finding_id"]
                for f in store.negative_findings()] == [
                    "nf-loose-termination-clause"]
        rows = store.search_chunks(
            text="looseword",
            filters={"corpus": "loose-corpus"})
        assert rows and "looseword" in rows[0]["text"]


def test_neo4j_artifact_fallback_binds_source_tree(
        tmp_path, monkeypatch):
    state = tmp_path / "state"
    _write_batch_dir(
        state / "batches" / "fixbatch",
        corpus="state-corpus",
        finding_id="nf-state-termination-clause",
        marker="stateword")
    monkeypatch.setenv("ONTOSMITH_ROOT", str(state))
    d = _elsewhere_batch(tmp_path)

    class _StubGraphStore:
        pass

    store = search.Neo4jServingStore(
        source=d, root=state, graph_store=_StubGraphStore())
    assert [f["finding_id"]
            for f in store.negative_findings()] == [
                "nf-doc-a-termination-clause"]
