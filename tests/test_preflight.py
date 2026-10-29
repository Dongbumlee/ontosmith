"""Preflight binding checks.

The documented contract: a run whose pending stages need a seat
with an unset key variable refuses with the connection layer's
named BindingKeyError — before any stage runs — instead of
starting and dying mid-run. Stage CANDIDATES' own key check
raises the same named error (it used to raise a bare
RuntimeError).
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
from pipeline.connections import BindingKeyError

NO_KEY = {"OPENROUTER_API_KEY": ""}


def _batch(status=BatchStatus.CREATED):
    return Batch(batch_id="pf-test", run_id="run-1", source="corp",
                 collection="docs", doc_ids=["doc-1"],
                 config=BatchConfig(),
                 status=status)


class TmpRoot(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)


class TestPreflightHelper(TmpRoot):
    def test_llm_source_pending_candidates_refuses(self):
        with mock.patch.dict(os.environ, NO_KEY):
            with self.assertRaises(BindingKeyError) as cm:
                batch_run._preflight_bindings(_batch(), [], False)
        self.assertIn("proposer", str(cm.exception))
        self.assertIn("OPENROUTER_API_KEY", str(cm.exception))

    def test_completed_candidates_never_rechecks_proposer(self):
        with mock.patch.dict(os.environ, NO_KEY):
            batch_run._preflight_bindings(
                _batch(), ["INGEST", "NER", "CANDIDATES"], False)

    def test_key_present_passes(self):
        with mock.patch.dict(os.environ,
                             {"OPENROUTER_API_KEY": "sk-test"}):
            batch_run._preflight_bindings(_batch(), [], False)

    def test_live_decide_checks_judge(self):
        with mock.patch.dict(os.environ, NO_KEY):
            with self.assertRaises(BindingKeyError) as cm:
                batch_run._preflight_bindings(
                    _batch(), ["INGEST", "NER", "CANDIDATES"],
                    True)
        self.assertIn("judge", str(cm.exception))

    def test_stub_decide_never_checks_judge(self):
        with mock.patch.dict(os.environ, NO_KEY):
            batch_run._preflight_bindings(
                _batch(), ["INGEST", "NER", "CANDIDATES"],
                False)

    def test_gates_completed_skips_judge(self):
        with mock.patch.dict(os.environ, NO_KEY):
            batch_run._preflight_bindings(
                _batch(),
                ["INGEST", "NER", "CANDIDATES", "GATES"], True)


class TestRunBatchRefusesBeforeStages(TmpRoot):
    def test_retired_sources_refused_by_name(self):
        # The candidate_source setting was removed with the
        # GLiNER+spaCy pipeline source (2026-10-29). A stored
        # config still naming a retired value refuses by
        # name at load (BatchConfig.from_dict) — never
        # silently runs a different builder than the stored
        # config states. The llm-era value and its absence
        # load silently.
        for source in ("pipeline", "both"):
            with self.assertRaises(ValueError) as cm:
                BatchConfig.from_dict(
                    {"candidate_source": source})
            self.assertIn("retired", str(cm.exception))
        BatchConfig.from_dict({"candidate_source": "llm"})
        BatchConfig.from_dict({})

    def test_refusal_leaves_batch_unstarted(self):
        b = _batch()
        with mock.patch.dict(os.environ, NO_KEY):
            with self.assertRaises(BindingKeyError):
                batch_run.run_batch(b, self.root,
                                    {"doc-1": "Some text."})
        # The refusal happened before the stage try-block: the
        # batch was not marked FAILED and no stage completed.
        self.assertEqual(b.status, BatchStatus.CREATED)
        d = batch_dir(self.root, "pf-test")
        state = d / "state.json"
        if state.exists():
            self.assertEqual(
                json.loads(state.read_text()).get("stages_completed"),
                [])


class TestStageCandidatesNamedError(TmpRoot):
    def test_llm_branch_raises_binding_key_error(self):
        d = batch_dir(self.root, "pf-test")
        d.mkdir(parents=True)
        (d / "texts.jsonl").write_text(json.dumps({
            "doc_id": "doc-1",
            "text": "Acme hired Anna Brian in Paris.",
            "source_path": "doc-1.txt"}) + "\n")
        (d / "ner.jsonl").write_text(json.dumps({
            "doc_id": "doc-1",
            "entities": [{"text": "Acme", "label": "organization"},
                         {"text": "Anna Brian", "label": "person"},
                         {"text": "Paris", "label": "location"}]}) + "\n")
        b = _batch(status=BatchStatus.NER_DONE)
        with mock.patch.dict(os.environ, NO_KEY):
            with self.assertRaises(BindingKeyError) as cm:
                batch_run.stage_candidates(b, self.root,
                                           ["INGEST", "NER"])
        self.assertIs(type(cm.exception), BindingKeyError)


class TestCliRefusalIsClean(unittest.TestCase):
    """The unified CLI prints a named binding refusal as a
    one-line error (exit 2), never as a traceback (round-2
    fresh-eyes N1)."""

    def test_binding_key_error_exits_clean(self):
        import io
        from contextlib import redirect_stderr

        from pipeline import __main__ as cli
        fake = mock.Mock()
        fake.main.side_effect = BindingKeyError(
            "proposer seat: API key environment variable "
            "OPENROUTER_API_KEY is not set")
        err = io.StringIO()
        with mock.patch.object(cli, "COMMANDS",
                               {"batch": ("fake_mod", "g", "d", ())}), \
                mock.patch.dict("sys.modules", {"fake_mod": fake}), \
                mock.patch.object(cli.sys, "argv", ["ontosmith", "batch"]), \
                redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("OPENROUTER_API_KEY", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


class _DeadClient:
    """A Neo4j client double whose store never answers."""

    url = "http://127.0.0.1:9/db/neo4j/tx/commit"

    def run(self, statements, timeout=None):
        raise ConnectionError("Connection refused")


class _LiveClient:
    url = "http://127.0.0.1:7474/db/neo4j/tx/commit"

    def run(self, statements, timeout=None):
        return [[{"row": [1]}]]


class _GraphDouble:
    def __init__(self, client):
        self.c = client


class TestStorePreflight(TmpRoot):
    """a build requires a reachable Neo4j. run_batch
    refuses BY NAME before any stage runs — the BYO walk died
    at WRITE on a raw connection error after the paid stages
    because nothing checked the store up front."""

    def test_dead_store_refuses_by_name(self):
        from pipeline.neo4j_store import StoreUnreachableError
        with self.assertRaises(StoreUnreachableError) as cm:
            batch_run._preflight_store(
                _batch(), [],
                _GraphDouble(_DeadClient()))
        msg = str(cm.exception)
        self.assertIn("Neo4j", msg)
        self.assertIn("NEO4J_URL", msg)
        self.assertIn("docs/connections.md", msg)

    def test_live_store_passes(self):
        batch_run._preflight_store(
            _batch(), [], _GraphDouble(_LiveClient()))

    def test_completed_graph_stages_skip_probe(self):
        batch_run._preflight_store(
            _batch(),
            ["INGEST", "NER", "CANDIDATES", "GATES", "ROUTE",
             "WRITE", "INDEX", "VERIFY"],
            _GraphDouble(_DeadClient()))

    def test_stub_graph_without_client_untouched(self):
        batch_run._preflight_store(_batch(), [],
                                    object())

    def test_run_batch_refuses_before_stages(self):
        from pipeline.neo4j_store import StoreUnreachableError
        b = _batch()
        # A stub proposer key so the bindings preflight
        # passes and the STORE preflight is the refusal
        # under test (the pipeline source that used to skip
        # the proposer check was retired 2026-10-29).
        with mock.patch.dict(os.environ,
                             {"OPENROUTER_API_KEY": "sk-test"}), \
                self.assertRaises(StoreUnreachableError):
            batch_run.run_batch(b, self.root,
                                {"doc-1": "Some text."},
                                graph=_GraphDouble(_DeadClient()))
        self.assertEqual(b.status, BatchStatus.CREATED)
        d = batch_dir(self.root, "pf-test")
        state = d / "state.json"
        if state.exists():
            self.assertEqual(
                json.loads(state.read_text()).get("stages_completed"),
                [])

    def test_probe_reports_auth_rejection(self):
        import urllib.error

        from pipeline.neo4j_store import probe

        class AuthClient:
            url = "http://127.0.0.1:7474/db/neo4j/tx/commit"

            def run(self, statements, timeout=None):
                raise urllib.error.HTTPError(
                    self.url, 401, "Unauthorized", {}, None)

        problem = probe(AuthClient())
        self.assertIn("rejected the configured credentials", problem)
        self.assertIn("NEO4J_PASSWORD", problem)

    def test_store_error_exits_clean(self):
        import io
        from contextlib import redirect_stderr

        from pipeline import __main__ as cli
        from pipeline.neo4j_store import StoreUnreachableError
        fake = mock.Mock()
        fake.main.side_effect = StoreUnreachableError(
            "Neo4j at http://127.0.0.1:7474 is unreachable")
        err = io.StringIO()
        with mock.patch.object(cli, "COMMANDS",
                               {"batch": ("fake_mod", "g", "d", ())}), \
                mock.patch.dict("sys.modules", {"fake_mod": fake}), \
                mock.patch.object(cli.sys, "argv", ["ontosmith", "batch"]), \
                redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Neo4j", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
