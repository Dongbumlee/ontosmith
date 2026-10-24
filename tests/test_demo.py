"""Hands-on demo tests (Phase C run 2, R13).

The committed smoke set is a few hand-sized documents; the
real demo corpus is fetched, never bundled. These tests
pin the smoke set's honesty (every recorded evidence quote
is a verbatim substring of its document), the fetch
script's deterministic subset contract, and the replay
proof: the shipped appendix alone reproduces the flagship
record's headlines.
"""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "examples" / "demo"
SMOKE = DEMO / "smoke"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSmokeSet(unittest.TestCase):
    def test_documents_and_recorded_output_exist(self):
        docs = sorted(SMOKE.glob("*.txt"))
        self.assertEqual(len(docs), 3)
        for doc in docs:
            self.assertTrue(doc.read_text(encoding="utf-8").strip())
        self.assertTrue((SMOKE / "triples.jsonl").is_file())

    def test_recorded_quotes_are_verbatim(self):
        texts = {p.stem: p.read_text(encoding="utf-8")
                 for p in SMOKE.glob("*.txt")}
        rows = [json.loads(line) for line in
                (SMOKE / "triples.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertIn(row["quote"], texts[row["doc_id"]])


class TestFetchScript(unittest.TestCase):
    def test_subset_is_deterministic_prefix(self):
        import tempfile

        fetch = _load("fetch_demo", DEMO / "fetch_demo.py")
        payload = [{"title": f"Doc {i}", "sents": [[f"sentence {i}."]]}
                   for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            out_a, out_b = Path(tmp) / "a", Path(tmp) / "b"
            written_a = fetch.write_subset(payload, out_a, 2)
            written_b = fetch.write_subset(payload, out_b, 2)
            self.assertEqual(len(written_a), 2)
            self.assertEqual(
                [p.read_bytes() for p in written_a],
                [p.read_bytes() for p in written_b])
            self.assertIn("sentence 0.", written_a[0].read_text())
            self.assertIn("sentence 1.", written_a[1].read_text())

    def test_source_is_the_upstream_dataset(self):
        fetch = _load("fetch_demo", DEMO / "fetch_demo.py")
        self.assertTrue(fetch.SOURCE_URL.startswith(
            "https://raw.githubusercontent.com/tonytan48/Re-DocRED/"))


class TestReplayProof(unittest.TestCase):
    def test_appendix_reproduces_the_record(self):
        replay = _load("replay_scores", DEMO / "replay_scores.py")
        appendix = ROOT / "benchmarks" / "appendix"
        # main() raises SystemExit(1) on any drift from the
        # published figures; 0 means full reproduction.
        self.assertEqual(replay.main(["--appendix", str(appendix)]), 0)


if __name__ == "__main__":
    unittest.main()
