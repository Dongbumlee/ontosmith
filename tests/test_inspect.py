"""Evidence/provenance viewer (pipeline/inspect.py).

Covered: normalization across the pipeline's record
families (judged candidates, supplementary emissions,
written-triple projections) — the same fact renders with
its evidence spans labeled by carrier field and its
provenance chain in pipeline order; honest gaps (a
projection record with no evidence span says so); batch
-directory resolution in the documented order; selection
by index and by subject/relation/object filters; the
report card and --json forms; and the refusal paths —
missing artefact, empty artefact, no triple records, no
filter match, out-of-range index, malformed JSONL.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import inspect as insp

JUDGED = {
    "index": 0, "subject": "Ali Hatami",
    "subject_type": "Person", "relation": "member of",
    "object": "Iranian New Wave", "object_type": "Movement",
    "doc_id": "sha256:001a886d",
    "sentence": "The movement includes pioneering "
                "directors such as Ali Hatami.",
    "p_true": 0.97, "cost": 0.0007, "tier": 1,
}
EMITTED = {
    "doc_id": "sha256:038b3650", "subject": "Chris Brown",
    "relation": "member of", "object": "The Hub",
    "quote": "consisting of", "attest": "chunk",
    "status": "guard-passed under issue-#68 phase-2 fixes",
}
PROJECTED = {
    "subject_id": "ent-0001", "relation": "member of",
    "surface_relation": "memberOf", "object_id": "ent-0042",
    "doc_id": "sha256:038b3650",
}


def _jsonl(tmp: Path, records, name="triples.jsonl") -> Path:
    p = tmp / name
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


class TestInspectReport(unittest.TestCase):
    def test_judged_record_full_report(self):
        rep = insp.build_report(JUDGED, "x.jsonl", 0, 1)
        self.assertEqual(rep["triple"]["subject"], "Ali Hatami")
        self.assertEqual(rep["triple"]["relation"], "member of")
        self.assertEqual(
            rep["evidence"],
            [{"kind": "sentence",
              "text": JUDGED["sentence"]}])
        self.assertEqual(rep["provenance"]["doc_id"],
                         "sha256:001a886d")
        self.assertEqual(rep["provenance"]["p_true"], 0.97)
        self.assertEqual(rep["gaps"], [])

    def test_emitted_record_quote_evidence(self):
        rep = insp.build_report(EMITTED, "x.jsonl", 0, 1)
        self.assertEqual(rep["evidence"][0]["kind"], "quote")
        self.assertEqual(rep["evidence"][0]["text"],
                         "consisting of")
        self.assertEqual(rep["provenance"]["attest"], "chunk")
        self.assertNotIn("no evidence span recorded",
                         rep["gaps"])

    def test_projected_record_names_its_gaps(self):
        rep = insp.build_report(PROJECTED, "x.jsonl", 0, 1)
        self.assertEqual(rep["triple"]["subject"], "ent-0001")
        self.assertEqual(rep["triple"]["surface_relation"],
                         "memberOf")
        self.assertIn("no evidence span recorded",
                      rep["gaps"])
        self.assertIn("no batch/status provenance recorded",
                      rep["gaps"])
        card = insp.render_card(rep)
        self.assertIn("Gaps", card)
        self.assertIn("ent-0001", card)


class TestInspectSelect(unittest.TestCase):
    def setUp(self):
        self.records = [JUDGED, EMITTED, PROJECTED]

    def test_default_first_and_index(self):
        rec, pos, n = insp.select(self.records)
        self.assertEqual((rec, pos, n), (JUDGED, 0, 3))
        rec, pos, _ = insp.select(self.records, index=2)
        self.assertEqual((rec, pos), (PROJECTED, 2))

    def test_filters(self):
        rec, pos, _ = insp.select(self.records,
                                  subject="chris")
        self.assertEqual((rec, pos), (EMITTED, 1))
        rec, _, _ = insp.select(self.records,
                                object_="ent-0042")
        self.assertEqual(rec, PROJECTED)
        rec, _, _ = insp.select(self.records,
                                relation="MEMBER")
        self.assertEqual(rec, JUDGED)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            insp.select([])
        with self.assertRaises(ValueError):
            insp.select([{"id": "A-001", "verdict": "ok"}])
        with self.assertRaises(ValueError):
            insp.select(self.records, subject="nobody")
        with self.assertRaises(ValueError):
            insp.select(self.records, index=3)
        with self.assertRaises(ValueError):
            insp.select(self.records, index=-1)


class TestInspectCli(unittest.TestCase):
    def test_card_and_json_forms(self):
        with tempfile.TemporaryDirectory() as td:
            p = _jsonl(Path(td), [JUDGED, EMITTED, PROJECTED])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = insp.main([str(p), "--subject", "Chris"])
            self.assertEqual(rc, 0)
            card = buf.getvalue()
            self.assertIn("Chris Brown", card)
            self.assertIn("[quote] consisting of", card)
            self.assertIn("Provenance", card)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = insp.main([str(p), "--json"])
            self.assertEqual(rc, 0)
            rep = json.loads(buf.getvalue())
            self.assertEqual(rep["triples"], 3)
            self.assertEqual(rep["triple"]["subject"],
                             "Ali Hatami")

    def test_batch_dir_resolution_order(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _jsonl(tmp, [PROJECTED], "written_triples.jsonl")
            _jsonl(tmp, [EMITTED], "routed.jsonl")
            _jsonl(tmp, [JUDGED], "judgments.jsonl")
            self.assertEqual(insp.resolve_artefact(tmp),
                             tmp / "judgments.jsonl")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = insp.main([str(tmp)])
            self.assertEqual(rc, 0)
            self.assertIn("Ali Hatami", buf.getvalue())

    def test_refusal_paths(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cases = [
                [str(tmp / "missing.jsonl")],          # absent
                [str(_jsonl(tmp, [], "empty.jsonl"))],  # empty
                [str(tmp)],                             # bare dir
            ]
            bad = tmp / "bad.jsonl"
            bad.write_text('{"subject": "x", "relation": '
                           '"r", "object": "y"}\nnot json\n')
            cases.append([str(bad)])                    # malformed
            good = _jsonl(tmp, [JUDGED], "good.jsonl")
            cases.append([str(good), "--index", "5"])   # range
            cases.append([str(good), "--subject", "zz"])  # nomatch
            for argv in cases:
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rc = insp.main(argv)
                self.assertEqual(rc, 1, argv)
                self.assertIn("refused", err.getvalue(), argv)


if __name__ == "__main__":
    unittest.main()
