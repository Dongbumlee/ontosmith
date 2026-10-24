"""Pack schema tests.

The pack format's field-level definition lives in
packs/pack.schema.json + packs/conformance.schema.json
(draft 2020-12), validated by pipeline.pack_schema and
enforced by the loader. These tests pin: the four shipped
packs and a builder-emitted pack validate clean; malformed
manifests fail closed naming the defect; the conformance
record validates too; and the field tables in
docs/PACK_DESIGN.md §1.6 carry exactly the schema's field
paths (docs cannot drift from the schema silently).
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import ontology_builder as ob
from pipeline import pack_schema, profile_config

REPO = Path(__file__).parent.parent
PACKS = ("core", "open-domain", "bio-medical", "legal-contract")


def _copy_pack(slug: str, root: Path) -> Path:
    import shutil
    dst = root / "packs" / slug
    shutil.copytree(REPO / "packs" / slug, dst)
    return dst


def _edit_manifest(pack_dir: Path, fn) -> dict:
    mpath = pack_dir / "pack.json"
    manifest = json.loads(mpath.read_text())
    fn(manifest)
    mpath.write_text(json.dumps(manifest, indent=1))
    return manifest


class TestShippedPacksValidate(unittest.TestCase):

    def test_all_four_packs_are_schema_clean(self):
        for slug in PACKS:
            with self.subTest(pack=slug):
                self.assertEqual(
                    pack_schema.validate_pack_dir(
                        REPO / "packs" / slug), [])

    def test_all_four_packs_load(self):
        for slug in PACKS:
            with self.subTest(pack=slug):
                manifest = profile_config.load_pack_manifest(
                    REPO / "packs" / slug)
                self.assertEqual(manifest["pack_id"], slug)

    def test_lab_provenance_fields_do_not_ship(self):
        for slug in PACKS:
            with self.subTest(pack=slug):
                manifest = json.loads(
                    (REPO / "packs" / slug
                     / "pack.json").read_text())
                self.assertNotIn("issue", manifest)
                self.assertNotIn("lab_codename", manifest)
                record = json.loads(
                    (REPO / "packs" / slug
                     / "conformance.json").read_text())
                self.assertNotIn("lab_codename", record)
                self.assertEqual(record["record_format"], 1)
                self.assertEqual(record["pack_id"], slug)


class TestFailClosed(unittest.TestCase):
    """Malformed manifests refuse at load, naming the
    defect (the loader's standing style)."""

    def _refusal(self, fn) -> str:
        with tempfile.TemporaryDirectory() as td:
            pack_dir = _copy_pack("core", Path(td))
            _edit_manifest(pack_dir, fn)
            with self.assertRaises(ValueError) as ctx:
                profile_config.load_pack_manifest(pack_dir)
            return str(ctx.exception)

    def test_missing_required_field(self):
        msg = self._refusal(lambda m: m.pop("status"))
        self.assertIn("status", msg)

    def test_bad_status_enum(self):
        msg = self._refusal(
            lambda m: m.update(status="active"))
        self.assertIn("not one of", msg)
        self.assertIn("in_force", msg)

    def test_undefined_field_refused(self):
        msg = self._refusal(lambda m: m.update(issue=71))
        self.assertIn("issue", msg)
        self.assertIn("not defined by the schema", msg)

    def test_bad_pin_shape(self):
        msg = self._refusal(
            lambda m: m["payload"].update(sha256="abc123"))
        self.assertIn("sha256", msg)

    def test_bad_pack_id_shape(self):
        msg = self._refusal(
            lambda m: m.update(pack_id="Bad Slug!"))
        self.assertIn("pack_id", msg)

    def test_nested_defect_names_its_path(self):
        msg = self._refusal(
            lambda m: m["convention_notes"][0].update(
                status="bogus"))
        self.assertIn("convention_notes[0].status", msg)

    def test_conformance_defects_named(self):
        with tempfile.TemporaryDirectory() as td:
            pack_dir = _copy_pack("core", Path(td))
            record = json.loads(
                (pack_dir / "conformance.json").read_text())
            record["entries"][0]["status"] = "sort of adopted"
            (pack_dir / "conformance.json").write_text(
                json.dumps(record))
            defects = pack_schema.validate_pack_dir(pack_dir)
            self.assertTrue(any(
                "entries[0].status" in d for d in defects),
                defects)

    def test_conformance_record_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            pack_dir = _copy_pack("core", Path(td))
            record = json.loads(
                (pack_dir / "conformance.json").read_text())
            record["pack_id"] = "someone-else"
            (pack_dir / "conformance.json").write_text(
                json.dumps(record))
            defects = pack_schema.validate_pack_dir(pack_dir)
            self.assertTrue(any(
                "does not match" in d for d in defects),
                defects)


class TestValidatorSemantics(unittest.TestCase):

    def test_bool_is_not_an_integer(self):
        manifest = json.loads(
            (REPO / "packs/core/pack.json").read_text())
        manifest["pack_format"] = True
        defects = pack_schema.validate_pack_manifest(manifest)
        self.assertTrue(any("pack_format" in d
                            for d in defects), defects)

    def test_contents_index_values_must_be_strings(self):
        manifest = json.loads(
            (REPO / "packs/core/pack.json").read_text())
        manifest["contents_index"]["triple_shape"] = 42
        defects = pack_schema.validate_pack_manifest(manifest)
        self.assertTrue(any("contents_index" in d
                            for d in defects), defects)


class TestEmittedPackValidates(unittest.TestCase):

    def test_emitted_pack_is_schema_clean_and_checks(self):
        spec = importlib.util.spec_from_file_location(
            "test_ontology_builder",
            Path(__file__).with_name("test_ontology_builder.py"))
        tob = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tob)
        draft = tob._good_draft()
        record = ob.approve_schema_draft(draft, approver="T")
        with tempfile.TemporaryDirectory() as td:
            ob.emit_pack(draft, record, td)
            pack_dir = Path(td) / "packs" / "field-notes"
            self.assertEqual(
                pack_schema.validate_pack_dir(pack_dir), [])
            proc = subprocess.run(
                [sys.executable, "-m", "pipeline", "ontology",
                 "check", str(pack_dir)],
                cwd=REPO, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout
                             + proc.stderr)
            self.assertIn("pack check: clean", proc.stdout)

    def test_check_command_refuses_a_broken_pack(self):
        with tempfile.TemporaryDirectory() as td:
            pack_dir = _copy_pack("core", Path(td))
            _edit_manifest(
                pack_dir, lambda m: m.update(status="active"))
            proc = subprocess.run(
                [sys.executable, "-m", "pipeline", "ontology",
                 "check", str(pack_dir)],
                cwd=REPO, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1, proc.stdout)
            self.assertIn("active", proc.stdout)


def _schema_paths(schema: dict, prefix: str = "",
                  root: dict | None = None) -> set:
    root = root if root is not None else schema
    out = set()
    for name, sub in (schema.get("properties") or {}).items():
        path = prefix + name
        out.add(path)
        ref = sub.get("$ref")
        if ref:
            sub = root["$defs"][ref.rsplit("/", 1)[-1]]
        if sub.get("type") == "object" and sub.get("properties"):
            out |= _schema_paths(sub, path + ".", root)
        elif sub.get("type") == "array":
            items = sub.get("items") or {}
            iref = items.get("$ref")
            if iref:
                items = root["$defs"][iref.rsplit("/", 1)[-1]]
            if items.get("properties"):
                out |= _schema_paths(items, path + "[].", root)
    return out


def _doc_table_paths(section: str, end_marker: str) -> set:
    doc = (REPO / "docs/PACK_DESIGN.md").read_text()
    start = doc.index(section)
    end = doc.index(end_marker, start)
    return set(re.findall(r"^\| `([^`]+)` \|", doc[start:end],
                          flags=re.M))


class TestDocsTableMatchesSchema(unittest.TestCase):
    """PACK_DESIGN §1.6 is the human view of the schemas;
    the test makes drift a failure, not a discovery."""

    def test_pack_table_matches_schema(self):
        schema = json.loads(
            (REPO / "packs/pack.schema.json").read_text())
        self.assertEqual(
            _doc_table_paths("#### pack.json fields",
                             "#### conformance.json fields"),
            _schema_paths(schema))

    def test_conformance_table_matches_schema(self):
        schema = json.loads(
            (REPO / "packs/conformance.schema.json").read_text())
        self.assertEqual(
            _doc_table_paths("#### conformance.json fields",
                             "\n## 2. Conformance measurement"),
            _schema_paths(schema))


if __name__ == "__main__":
    unittest.main()
