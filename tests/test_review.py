"""Review-tool case identity (audit F8).

Two review-band cases from one document are distinct
triples. Resume and undo used to key on doc_id alone: a
document's second case was silently retired by the first
case's label (never shown, never recorded — against the
tool's own "never silently dropped" contract), and undo
jumped to the first case sharing the doc_id. These tests
pin the case-key behavior and the legacy-row fallback.
"""
import json

from pipeline import review


def _case(doc, s, r, o):
    return {"doc_id": doc, "subject": s, "relation": r,
            "object": o, "p_true": 0.4, "text": "x"}


CASE_A = _case("doc1", "Alice", "born_in", "Paris")
CASE_B = _case("doc1", "Bob", "born_in", "Lyon")
CASE_C = _case("doc2", "Carol", "born_in", "Nice")


def _write_labels(tmp_path, rows):
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_case_key_distinguishes_same_doc_triples():
    assert review.case_key(CASE_A) != review.case_key(CASE_B)
    assert review.case_key(CASE_A) == review.case_key(dict(CASE_A))


def test_resume_suppresses_only_the_labeled_case(tmp_path):
    out = _write_labels(tmp_path, [
        review._label_row(CASE_A, 1, 1)])
    keys, legacy, n = review.load_seen(out)
    assert n == 1 and not legacy
    assert review.is_labeled(CASE_A, keys, legacy)
    # the regression: the same document's other triple
    # must NOT be treated as labeled
    assert not review.is_labeled(CASE_B, keys, legacy)
    assert not review.is_labeled(CASE_C, keys, legacy)


def test_legacy_doc_id_row_suppresses_whole_document(tmp_path):
    out = _write_labels(tmp_path, [
        {"doc_id": "doc1", "label": 1, "p_true": 0.4, "pass": 1}])
    keys, legacy, n = review.load_seen(out)
    assert n == 1 and legacy == {"doc1"} and not keys
    assert review.is_labeled(CASE_A, keys, legacy)
    assert review.is_labeled(CASE_B, keys, legacy)
    assert not review.is_labeled(CASE_C, keys, legacy)


def test_load_seen_missing_file_is_empty(tmp_path):
    keys, legacy, n = review.load_seen(str(tmp_path / "nope.jsonl"))
    assert keys == set() and legacy == set() and n == 0


def test_find_case_index_matches_the_exact_triple():
    cases = [CASE_A, CASE_B, CASE_C]
    label_b = review._label_row(CASE_B, 0, 1)
    # not index 0 (the first case sharing doc1)
    assert review.find_case_index(cases, label_b) == 1
    assert review.find_case_index(cases,
                                  review._label_row(CASE_C, 1, 1)) == 2
    ghost = review._label_row(_case("doc9", "X", "r", "Y"), 1, 1)
    assert review.find_case_index(cases, ghost) is None


def test_label_row_carries_the_case_key():
    row = review._label_row(CASE_A, None, 2, abstained=True)
    assert row["subject"] == "Alice" and row["object"] == "Paris"
    assert row["label"] is None and row["abstained"] is True
    assert review.case_key(row) == review.case_key(CASE_A)
