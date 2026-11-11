"""MCP surface tests: the envelope discipline and the
five tools' service layer over the shipped example
build (file store). SDK-dependent boot is covered
separately and skips when the extra is absent."""
import pytest

from pipeline import mcp_envelope as env
from pipeline import mcp_server
from pipeline import search as _search


@pytest.fixture(scope="module")
def service():
    store = _search.open_store(store="file")
    return mcp_server.McpService(store)


# ----------------------------------------------------------------------
# Envelope
# ----------------------------------------------------------------------

def test_triple_id_is_stable_and_key_bound():
    a = env.triple_id("S", "rel", "O")
    assert a == env.triple_id("S", "rel", "O")
    assert a != env.triple_id("S", "rel", "O2")
    assert a.startswith("t_") and len(a) == 14


def test_cursor_round_trip_and_refusal():
    assert env.decode_cursor(env.encode_cursor(7)) == 7
    assert env.decode_cursor(None) == 0
    with pytest.raises(ValueError, match="invalid cursor"):
        env.decode_cursor("not-a-cursor!!")


def test_envelope_budget_truncates_at_item_boundary():
    builder = env.EnvelopeBuilder()
    items = [{"kind": "text", "text": "x" * 400,
              "doc": builder.doc_ref("d1"),
              "cite": builder.cite_key("d1")}
             for _ in range(10)]
    out = builder.build(items, total=10, max_tokens=200)
    assert out["truncated"] is True
    assert out["returned"] < 10
    assert env.decode_cursor(out["cursor"]) == \
        out["returned"]
    assert out["docs"] == {"d1": {"doc_id": "d1"}}


def test_envelope_complete_result_has_no_cursor():
    builder = env.EnvelopeBuilder()
    out = builder.build([{"kind": "text", "text": "hi"}],
                        total=1)
    assert out["truncated"] is False
    assert "cursor" not in out


# ----------------------------------------------------------------------
# Tools over the example build
# ----------------------------------------------------------------------

def test_entity_resolves_with_counts(service):
    out = service.entity("Jane Bennet")
    assert out["resolved"] is True
    item = out["items"][0]
    assert item["entity_id"] == "Jane Bennet"
    assert item["triples_as_subject"] + \
        item["triples_as_object"] > 0


def test_entity_unknown_is_a_state_not_an_error(service):
    out = service.entity("Zebedee Nonexistent")
    assert out["resolved"] is False
    assert out["items"] == [] and out["total"] == 0


def test_entity_ambiguous_returns_candidates(service):
    out = service.entity("Bennet")
    if out.get("ambiguous"):
        assert out["resolved"] is False
        assert len(out["items"]) >= 2
        assert all("distinguisher" in i
                   for i in out["items"])
    else:  # an exact "Bennet" entity exists in the build
        assert out["resolved"] is True


def test_neighborhood_items_carry_evidence(service):
    out = service.neighborhood("Jane Bennet")
    assert out["total"] > 0
    for item in out["items"]:
        assert item["kind"] == "triple"
        assert item["evidence"], "no bare triples"
        for ev in item["evidence"]:
            assert ev["text"] and ev["cite"]
            assert ev["doc"] in out["docs"]


def test_neighborhood_relation_filter(service):
    full = service.neighborhood("Jane Bennet", limit=100)
    rel = full["items"][0]["relation"]
    filtered = service.neighborhood(
        "Jane Bennet", relation=rel, limit=100)
    assert 0 < filtered["total"] <= full["total"]
    assert all(i["relation"] == rel
               for i in filtered["items"])


def test_neighborhood_unknown_id_refuses_correctively(
        service):
    with pytest.raises(mcp_server.McpToolError,
                       match="entity tool"):
        service.neighborhood("no-such-entity")


def test_triple_round_trip_full_evidence(service):
    nb = service.neighborhood("Jane Bennet")
    tid = nb["items"][0]["triple_id"]
    out = service.triple(tid)
    item = out["items"][0]
    assert item["triple_id"] == tid
    assert item["evidence"]
    assert "provenance" in item


def test_triple_unknown_id_refuses(service):
    with pytest.raises(mcp_server.McpToolError,
                       match="triple_id"):
        service.triple("t_000000000000")


def test_connect_finds_path_or_states_none(service):
    out = service.connect("Jane Bennet", "Lydia Bennet")
    assert out["total"] >= 1
    path = out["items"][0]
    assert path["kind"] == "path"
    assert path["hops"] == len(path["steps"])
    for step in path["steps"]:
        assert step["evidence"], "path steps are evidenced"


def test_connect_no_path_is_a_state(service):
    out = service.connect(
        "Jane Bennet", "Jane Bennet", max_hops=1)
    # Same endpoint: either a trivial/loop answer or
    # none — never an exception.
    assert "items" in out


def test_search_envelope_and_arms(service):
    out = service.search("Jane Bennet sister")
    assert out["total"] > 0
    assert "graph" in out["arms"]
    triples = [i for i in out["items"]
               if i["kind"] == "triple"]
    assert triples, "graph-arm claims are bound to evidence"
    for item in triples:
        assert item["evidence"]


class _FakeEmbedder:
    """Deterministic bag-of-words embedder: enough
    for the sentence form to run end to end with no
    API key and no spend."""

    DIM = 256

    def embed(self, texts):
        import numpy as np
        out = []
        for text in texts:
            vec = np.zeros(self.DIM)
            for tok in text.lower().split():
                vec[hash(tok) % self.DIM] += 1.0
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append(vec / norm)
        return out


def test_search_name_only_mentions_are_flagged(service):
    # No income fact exists for Elizabeth: every
    # returned triple matched her name alone. The
    # envelope must flag them and say so in notes —
    # an agent must not read mentions as answers.
    out = service.search(
        "What is Elizabeth Bennet's income?")
    triples = [i for i in out["items"]
               if i["kind"] == "triple"]
    assert triples
    assert all(i.get("name_only") for i in triples)
    assert any("mentions, not answers" in n
               for n in out.get("notes", []))


def test_search_term_matched_triples_unflagged(service):
    out = service.search("What is Mr. Darcy's income?")
    triples = [i for i in out["items"]
               if i["kind"] == "triple"]
    assert triples
    assert not any(i.get("name_only") for i in triples)


def test_search_sentence_form_when_embedder_bound():
    store = _search.open_store(store="file")
    svc = mcp_server.McpService(
        store, embedder=_FakeEmbedder())
    out = svc.search("ceramics studio founded")
    assert "sentences" in out["arms"]
    assert not any("hybrid document form" in n
                   for n in out.get("notes", []))
    units = [i for i in out["items"]
             if i["kind"] in ("sentence", "text")]
    assert units
    assert all("via" in u for u in units)
    assert {u["arm"] for u in units} <= {
        "vector", "graph", "both"}


def _both_sides_store(tmp_path):
    """A one-document batch whose chunk carries a
    triple's quote PLUS a further sentence, so a
    query can hit the graph arm and the text arm at
    once — the shipped example build cannot (its
    graph is one corpus, its chunks another)."""
    import json
    import pickle

    import numpy as np

    from pipeline import query_stack

    text = ("Acme Corp acquired Beta LLC in 2021. "
            "The Acme Corp board approved the Beta "
            "LLC acquisition unanimously.")
    bdir = tmp_path / "fixbatch"
    bdir.mkdir()
    (bdir / "batch.json").write_text(json.dumps({
        "batch_id": "fixbatch", "run_id": "r1",
        "source": "fixture", "collection": "fixture",
        "doc_ids": ["doc-a"], "config": {},
        "metrics": {}}))
    (bdir / "texts.jsonl").write_text(
        json.dumps({"doc_id": "doc-a", "text": text})
        + "\n")
    (bdir / "routed.jsonl").write_text(json.dumps({
        "subject": "Acme Corp", "relation": "acquired",
        "object": "Beta LLC", "doc_id": "doc-a",
        "quote": "Acme Corp acquired Beta LLC in "
                 "2021.",
        "p_true": 0.97, "tier": 1}) + "\n")
    (bdir / "written_triples.jsonl").write_text(
        json.dumps({"subject_id": "ent-1",
                    "relation": "acquired",
                    "object_id": "ent-2",
                    "doc_id": "doc-a"}) + "\n")
    chunker = query_stack.FixedWindowChunker()
    meta, chunks = [], []
    for idx, ch in chunker.chunk(text):
        meta.append({"doc_id": "doc-a", "chunk_idx": idx,
                     "tag": "VERIFIED", "tier": 1,
                     "batch_id": "fixbatch"})
        chunks.append(ch)
    index = {"embeddings": np.array([[1.0, 0.0]]),
             "meta": meta, "chunks": chunks,
             "model": "fixture"}
    with open(bdir / "tier2_index.pkl", "wb") as f:
        pickle.dump(index, f)
    return _search.open_store(source=str(bdir))


def test_search_returns_both_arms_in_one_result(
        tmp_path):
    svc = mcp_server.McpService(
        _both_sides_store(tmp_path))
    out = svc.search("Acme Corp acquired Beta")
    kinds = {i["kind"] for i in out["items"]}
    assert kinds == {"triple", "text"}
    # The passage carries a sentence beyond the
    # triple's quote, so it is NOT deduped away.
    passage = next(i for i in out["items"]
                   if i["kind"] == "text")
    assert "board approved" in passage["text"]


def test_budget_cursor_resumes(service):
    first = service.neighborhood(
        "Jane Bennet", depth=2, limit=100,
        max_tokens=200)
    assert first["truncated"] is True
    second = service.neighborhood(
        "Jane Bennet", depth=2, limit=100,
        max_tokens=200, cursor=first["cursor"])
    first_ids = {i["triple_id"] for i in first["items"]}
    second_ids = {i["triple_id"] for i in second["items"]}
    assert first_ids and second_ids
    assert first_ids.isdisjoint(second_ids)
    assert first["total"] == second["total"]


def test_parameter_ranges_refuse_correctively(service):
    with pytest.raises(mcp_server.McpToolError,
                       match=r"depth must be 1\.\.2"):
        service.neighborhood("Jane Bennet", depth=9)
    with pytest.raises(mcp_server.McpToolError,
                       match=r"max_hops must be 1\.\.4"):
        service.connect("Jane Bennet", "Lydia Bennet",
                        max_hops=9)


# ----------------------------------------------------------------------
# SDK shell (extra-gated)
# ----------------------------------------------------------------------

def test_server_boots_with_five_tools(service):
    pytest.importorskip("mcp")
    server = mcp_server.build_server(service)
    import asyncio
    tools = asyncio.run(server.list_tools())
    assert sorted(t.name for t in tools) == [
        "connect", "entity", "neighborhood", "search",
        "triple"]
    for t in tools:
        assert t.description and len(t.description) > 80
