"""MCP search absence exposure (0.2.2): the store
contract's negative findings — declared slots the
build's sweep found ABSENT — surface in ``search`` as
flagged ``negative_finding`` items alongside the
claims. The discipline under test: matching is the
claim side's own name/term decomposition (the slot
category carries the term part; a subject mention
alone never matches), and the items are additional —
they neither consume ``k`` nor displace a ranked
claim, trailing the results inside the same envelope
accounting."""
import json

from pipeline import mcp_server
from pipeline import search as _search

_TEXT_A = ("Acme Corp acquired Beta LLC in 2021. "
           "The Acme Corp board approved the Beta "
           "LLC acquisition unanimously.")
_TEXT_B = ("Beta LLC provides logistics services to "
           "Acme Corp under a services schedule. "
           "The Beta LLC services schedule renews "
           "annually.")

_FINDINGS = [
    {"kind": "negative_finding",
     "finding_id": "nf-doc-a-termination-clause",
     "batch_id": "fixbatch", "run_id": "r1",
     "doc_id": "doc-a", "subject": "Acme Master Agreement",
     "category": "Termination Clause",
     "status": "absent", "method": "clause-llm"},
    {"kind": "negative_finding",
     "finding_id": "nf-doc-a-governing-law",
     "batch_id": "fixbatch", "run_id": "r1",
     "doc_id": "doc-a", "subject": "Acme Master Agreement",
     "category": "Governing Law",
     "status": "absent", "method": "clause-llm"},
    {"kind": "negative_finding",
     "finding_id": "nf-doc-b-termination-clause",
     "batch_id": "fixbatch", "run_id": "r1",
     "doc_id": "doc-b", "subject": "Beta Services Contract",
     "category": "Termination Clause",
     "status": "absent", "method": "clause-llm"},
]


def _fixture_store(tmp_path, *, findings=True):
    """A two-document batch in the canonical layout
    (``<root>/batches/<id>/``), with one attested
    triple and, optionally, the sweep's absence
    records beside it."""
    bdir = tmp_path / "batches" / "fixbatch"
    bdir.mkdir(parents=True)
    (bdir / "batch.json").write_text(json.dumps({
        "batch_id": "fixbatch", "run_id": "r1",
        "source": "fixture", "collection": "fixture",
        "doc_ids": ["doc-a", "doc-b"], "config": {},
        "metrics": {}}))
    (bdir / "texts.jsonl").write_text(
        json.dumps({"doc_id": "doc-a", "text": _TEXT_A})
        + "\n"
        + json.dumps({"doc_id": "doc-b", "text": _TEXT_B})
        + "\n")
    (bdir / "routed.jsonl").write_text(json.dumps({
        "subject": "Acme Corp", "relation": "acquired",
        "object": "Beta LLC", "doc_id": "doc-a",
        "quote": "Acme Corp acquired Beta LLC in "
                 "2021.",
        "p_true": 0.97, "tier": 1}) + "\n")
    (bdir / "written_triples.jsonl").write_text(
        json.dumps({"subject_id": "Acme Corp",
                    "relation": "acquired",
                    "object_id": "Beta LLC",
                    "doc_id": "doc-a"}) + "\n")
    if findings:
        (bdir / "negative_findings.jsonl").write_text(
            "".join(json.dumps(f) + "\n"
                    for f in _FINDINGS))
    # Constructed directly with the fixture root, the
    # projection tests' pattern: open_store would bind
    # the process state root instead.
    return _search.FileServingStore(bdir, root=tmp_path)


def _absence(out):
    return [i for i in out["items"]
            if i["kind"] == "negative_finding"]


def test_category_query_returns_flagged_absence_items(
        tmp_path):
    svc = mcp_server.McpService(
        _fixture_store(tmp_path))
    out = svc.search("termination clause")
    absence = _absence(out)
    # Both documents' termination-clause absences; the
    # governing-law finding shares no category token
    # with the query and stays out. Order is the
    # deterministic one: equal overlaps, finding_id
    # breaks the tie.
    assert [i["finding_id"] for i in absence] == [
        "nf-doc-a-termination-clause",
        "nf-doc-b-termination-clause"]
    for item in absence:
        assert item["status"] == "absent"
        assert item["category"] == "Termination Clause"
        assert item["subject"]
        assert item["doc"] in out["docs"]
        assert item["cite"].startswith(item["doc"] + ":")
    assert any("absence records" in n
               for n in out.get("notes", []))


def test_linked_subject_gates_other_documents_absence(
        tmp_path):
    svc = mcp_server.McpService(
        _fixture_store(tmp_path))
    out = svc.search(
        "What did Acme Corp record about the "
        "termination clause?")
    # The query links Acme Corp; the Beta document's
    # absence is not Acme's answer and is gated out.
    assert [i["finding_id"] for i in _absence(out)] == [
        "nf-doc-a-termination-clause"]


def test_subject_mention_alone_matches_no_absence(
        tmp_path):
    svc = mcp_server.McpService(
        _fixture_store(tmp_path))
    out = svc.search("Acme Master Agreement")
    # Names alone are mentions, not answers — the
    # name_only doctrine, applied to findings.
    assert _absence(out) == []


def test_unrelated_query_returns_no_absence(tmp_path):
    svc = mcp_server.McpService(
        _fixture_store(tmp_path))
    out = svc.search("Acme Corp acquired Beta LLC")
    assert _absence(out) == []
    triples = [i for i in out["items"]
               if i["kind"] == "triple"]
    assert triples, "the attested claim still answers"


def test_absence_items_do_not_consume_k(tmp_path):
    svc = mcp_server.McpService(
        _fixture_store(tmp_path))
    # k bounds the retrieval arms; the absence section
    # is additional to it, in full, at any k.
    for k in (1, 8):
        out = svc.search("termination clause", k=k)
        assert len(_absence(out)) == 2


def test_absence_items_are_additional_never_displacing(
        tmp_path):
    base = mcp_server.McpService(_fixture_store(
        tmp_path / "base", findings=False)).search(
            "termination clause")
    with_f = mcp_server.McpService(_fixture_store(
        tmp_path / "withf", findings=True)).search(
            "termination clause")
    # Every claim/text item the findings-less store
    # returned is present, identical, and in place —
    # the absence items only extend the tail, and the
    # honest total grows by exactly their count.
    assert with_f["items"][:len(base["items"])] == \
        base["items"]
    assert len(with_f["items"]) == len(base["items"]) + 2
    assert with_f["total"] == base["total"] + 2
    assert all(i["kind"] == "negative_finding"
               for i in with_f["items"][len(base["items"]):])


def test_store_without_findings_behaves_as_before(
        tmp_path):
    svc = mcp_server.McpService(_fixture_store(
        tmp_path, findings=False))
    out = svc.search("termination clause")
    assert _absence(out) == []
    assert not any("absence records" in n
                   for n in out.get("notes", []))
    assert out["total"] == len(out["items"])
