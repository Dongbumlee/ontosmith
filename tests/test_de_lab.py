"""De-lab tests: no lab identifiers in the tree, secrets only
via environment names, product identifiers in place.

Covers the run-2 de-lab edits: neo4j_store._password (env-only),
export_graph (imports without any lab file; env-derived endpoint
and auth), turtle_export (product base IRI + pure helpers),
promotion (ledger repo named by environment, never baked in).
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


class TestTreeHygiene(unittest.TestCase):
    FORBIDDEN = ("jev-kg-pilot", "Dongbumlee", "/home/hatch",
                 "~/workspace", "azureuser", "sdlc-testbox")

    def test_no_lab_identifiers_in_pipeline_sources(self):
        offenders = []
        for p in sorted((ROOT / "pipeline").glob("*.py")):
            text = p.read_text()
            for token in self.FORBIDDEN:
                if token in text:
                    offenders.append(f"{p.name}:{token}")
        self.assertEqual(offenders, [])

    def test_no_lab_identifiers_in_pack_manifests(self):
        offenders = []
        for p in sorted((ROOT / "packs").glob("*/pack.json")):
            text = p.read_text()
            for token in ("/home/hatch", "azureuser", "sdlc-testbox"):
                if token in text:
                    offenders.append(f"{p}:{token}")
        self.assertEqual(offenders, [])


class TestNeo4jPassword(unittest.TestCase):
    def test_password_comes_from_env_only(self):
        from pipeline import neo4j_store
        with mock.patch.dict(os.environ, {"NEO4J_PASSWORD": "s3cr3t"}):
            self.assertEqual(neo4j_store._password(), "s3cr3t")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(neo4j_store._password(), "")


class TestExportGraph(unittest.TestCase):
    def test_imports_without_lab_env_file(self):
        # The pre-tidying module read ~/workspace/neo4j/.env at
        # import time. With HOME pointed at an empty directory
        # the import must still succeed and touch nothing.
        with tempfile_home() as home:
            env = dict(os.environ, HOME=str(home),
                       PYTHONPATH=str(ROOT))
            out = subprocess.run(
                [sys.executable, "-c",
                 "import pipeline.export_graph as e;"
                 "print(e.tx_url())"],
                capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(),
                         "http://localhost:7474/db/neo4j/tx/commit")

    def test_tx_url_and_auth_from_env(self):
        from pipeline import export_graph as e
        with mock.patch.dict(
                os.environ,
                {"NEO4J_URL": "http://db.internal:7474/",
                 "NEO4J_PASSWORD": "pw123"}):
            self.assertEqual(
                e.tx_url(), "http://db.internal:7474/db/neo4j/tx/commit")
            expect = "Basic " + base64.b64encode(
                b"neo4j:pw123").decode()
            self.assertEqual(e._auth_header(), expect)

    def test_query_payload_shape(self):
        from pipeline import export_graph as e
        captured = {}

        class Resp:
            def read(self):
                return json.dumps({"results": []}).encode()

        def fake_urlopen(req):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            captured["body"] = json.loads(req.data.decode())
            return Resp()

        with mock.patch.object(urllib.request, "urlopen",
                               fake_urlopen):
            with mock.patch.dict(os.environ,
                                 {"NEO4J_PASSWORD": "pw"}):
                # json.load on Resp: read() path returns dict text
                out = e.q("MATCH (n) RETURN n", {"x": 1})
        self.assertEqual(captured["body"],
                         {"statements": [{
                             "statement": "MATCH (n) RETURN n",
                             "parameters": {"x": 1}}]})
        self.assertEqual(out, {"results": []})


class TestTurtleExport(unittest.TestCase):
    def test_base_iri_is_product(self):
        from pipeline import turtle_export as t
        self.assertEqual(t.BASE, "http://example.org/ontosmith/")

    def test_pure_helpers(self):
        from pipeline import turtle_export as t
        self.assertEqual(t.sanitize("  Born In (City)! "), "born_in_city")
        self.assertEqual(t.sanitize("!!!"), "unnamed")
        self.assertEqual(t.esc('a"b\\c'), 'a\\"b\\\\c')


class TestPromotionLedgerRepo(unittest.TestCase):
    def test_unset_repo_fails_closed(self):
        from pipeline import promotion
        with mock.patch.object(promotion, "LEDGER_REPO", ""):
            with self.assertRaises(RuntimeError):
                promotion.doc_check_issue("X.md")

    def test_repo_comes_from_setting(self):
        from pipeline import promotion

        class Completed:
            returncode = 0
            stdout = json.dumps([{
                "number": 5, "title": "[doc-check] X.md",
                "state": "CLOSED", "url": "https://example/5"}])
            stderr = ""

        captured = {}

        def fake_run(argv, **kw):
            captured["argv"] = argv
            return Completed()

        with mock.patch.object(promotion, "LEDGER_REPO",
                               "example/ledger"):
            with mock.patch.object(promotion.subprocess, "run",
                                   fake_run):
                issue = promotion.doc_check_issue("X.md")
        self.assertEqual(issue["number"], 5)
        self.assertIn("example/ledger", captured["argv"])


class tempfile_home:
    """Context manager: a fresh empty HOME directory."""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        return Path(self._td.name)

    def __exit__(self, *exc):
        self._td.cleanup()


if __name__ == "__main__":
    unittest.main()
