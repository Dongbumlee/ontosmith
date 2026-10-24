"""Golden tests for pipeline.gate_cli — the collapsed run_* drivers.

The goldens under tests/golden/ were captured from the ORIGINAL
lab runner scripts (pilot/run_er.py, run_conflict.py, run_onto.py,
run_llm.py, run_ollama.py, run_v0.py, run_cascade.py) executed
with the same deterministic stub judges used here. Any behavior
drift in the collapse — record shapes, summary math, error
records, output formats — fails against them.

Run 32 re-capture: the er/conflict/onto/llm_text/ollama goldens
were regenerated with the same stub judges after the deliberate
error-shape change (audit F17) — error records now carry an
explicit "error" field and every summary reports "errors" (the
llm summary also reports "parse_failures"). The diff against the
originals is exactly those additions; every judged value, cost,
and latency is unchanged.
"""
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import gate_cli as g

FIX = Path(__file__).parent / "fixtures"
GOLD = Path(__file__).parent / "golden"

USAGE = {"cost": 0.001, "latency_ms": 12.5, "provider": "FakeProv",
         "input_tokens": 10, "output_tokens": 3}


def p_of(*parts):
    return (sum(ord(c) for p in parts for c in str(p)) % 100) / 100.0


def boom_check(*parts):
    if any("BOOM" in str(p) for p in parts):
        raise RuntimeError("stubbed judge failure")


def load_golden(name):
    return json.loads((GOLD / name).read_text())


class GateCase(unittest.TestCase):
    def setUp(self):
        self._saved = {}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def patch(self, name, fn):
        self._saved[name] = getattr(g, name)
        setattr(g, name, fn)
        self.addCleanup(setattr, g, name, self._saved[name])

    def out(self, name):
        return str(Path(self.tmp.name) / name)

    def assert_golden(self, name, produced_path):
        self.assertEqual(load_golden(name),
                         json.loads(Path(produced_path).read_text()))


class TestEr(GateCase):
    def test_matches_original(self):
        def fake(context, a, b, model=None):
            boom_check(a, b)
            return p_of(a, b), dict(USAGE)
        self.patch("er_judge", fake)
        out = self.out("er.json")
        g.main(["er", "--candidates", str(FIX / "er.json"),
                "--out", out])
        self.assert_golden("er.json", out)


class TestConflict(GateCase):
    def test_matches_original(self):
        def fake(existing, s, r, o, model=None):
            boom_check(s, r, o)
            return ("supports", 0.75,
                    {"supports": 0.75, "contradicts": 0.25},
                    dict(USAGE))
        self.patch("conflict_judge", fake)
        out = self.out("conflict.json")
        g.main(["conflict", "--candidates", str(FIX / "conflict.json"),
                "--out", out])
        self.assert_golden("conflict.json", out)


class TestOnto(GateCase):
    def test_matches_original(self):
        def fake(ontology, axiom, a, b, model=None):
            boom_check(a, b)
            return p_of(a, b), dict(USAGE)
        self.patch("ontology_judge", fake)
        out = self.out("onto.json")
        g.main(["onto", "--candidates", str(FIX / "onto.json"),
                "--out", out])
        self.assert_golden("onto.json", out)


class TestLlm(GateCase):
    def setUp(self):
        super().setUp()

        def fake_text(text, s, r, o, model=None, max_tokens=300):
            boom_check(s, r, o)
            return (1 if p_of(s, r, o) >= 0.5 else 0), dict(USAGE)

        self.patch("llm_text_judge", fake_text)

    def test_text_mode_matches_original(self):
        out = self.out("llm_text.json")
        g.main(["llm", "--candidates", str(FIX / "llm.json"),
                "--model", "fake/model", "--out", out])
        self.assert_golden("llm_text.json", out)


class TestOllama(GateCase):
    def test_load_records_matches_original(self):
        self.assertEqual(g.load_records(str(FIX / "llm.json")),
                         load_golden("ollama_load_records.json"))

    def test_main_matches_original(self):
        def fake(prompt, model, timeout=120):
            if "BOOM" in prompt:
                raise RuntimeError("stubbed ollama failure")
            return (1 if "Yes" in prompt or "Alice" in prompt else 0), \
                {"latency_ms": 5.0}
        self.patch("ollama_judge", fake)
        out = self.out("ollama.json")
        g.main(["ollama", "--candidates", str(FIX / "llm.json"),
                "--model", "fake:1", "--out", out])
        self.assert_golden("ollama.json", out)


SUMMARIZE_RECORDS = [
    {"p_true": 0.9, "label": 1, "neg_kind": None,
     "latency_ms": 10.0, "cost": 0.01, "input_tokens": 100},
    {"p_true": 0.8, "label": 1, "neg_kind": None,
     "latency_ms": 20.0, "cost": 0.02, "input_tokens": 200},
    {"p_true": 0.3, "label": 0, "neg_kind": "random",
     "latency_ms": 30.0, "cost": 0.03, "input_tokens": 300},
    {"p_true": 0.6, "label": 0, "neg_kind": "hard_swap",
     "latency_ms": 40.0, "cost": 0.04, "input_tokens": 400},
    {"p_true": 0.1, "label": 0, "neg_kind": "random",
     "latency_ms": 50.0, "cost": 0.05, "input_tokens": 500},
]


class TestTriple(GateCase):
    def test_summarize_matches_original(self):
        self.assertEqual(
            json.dumps(g.summarize(SUMMARIZE_RECORDS), indent=1,
                       default=str),
            (GOLD / "v0_summarize.json").read_text())

    def test_calibration_bins_match_original(self):
        self.assertEqual(
            json.dumps(g.calibration_bins(
                [0.05, 0.15, 0.55, 0.95, 1.0], [0, 1, 1, 0, 1]),
                indent=1),
            (GOLD / "v0_bins.json").read_text())

    def test_main_all_gates_match_original(self):
        def fake_validate(text, s, r, o, model=None, variant="v1"):
            boom_check(s, r, o)
            return p_of(s, r, o), dict(USAGE)

        def fake_score(text, s, r, o, model=None):
            boom_check(s, r, o)
            return 4, 0.9, dict(USAGE)

        def fake_choice(text, s, o, options, model=None):
            boom_check(s, o)
            return options[0], 0.8, {options[0]: 0.8, "none": 0.2}, \
                dict(USAGE)

        self.patch("validate_triple", fake_validate)
        self.patch("score_gate", fake_score)
        self.patch("choice_gate", fake_choice)
        self.patch("load_dev", lambda: ([], {}))
        self.patch("RESULTS_DIR", Path(self.tmp.name))
        for gate in ("noul", "noul_v2", "score", "choice"):
            g.main(["triple", "--load-candidates",
                    str(FIX / "v0.json"), "--gate", gate,
                    "--out", f"v0_{gate}.json"])
            self.assert_golden(f"v0_{gate}.json",
                               Path(self.tmp.name) / f"v0_{gate}.json")


DOCS = [
    {"title": "DocA",
     "vertexSet": [[{"name": "Alice", "type": "PER", "sent_id": 0}],
                   [{"name": "Paris", "type": "LOC", "sent_id": 0}],
                   [{"name": "Dan", "type": "PER", "sent_id": 1}]],
     "labels": [{"h": 0, "t": 1, "r": "P69"}]},
    {"title": "DocB",
     "vertexSet": [[{"name": "Carol", "type": "PER", "sent_id": 0}],
                   [{"name": "Rome", "type": "LOC", "sent_id": 0}]],
     "labels": [{"h": 0, "t": 1, "r": "P69"}]},
]
REL = {"P69": "born_in"}


class TestCascade(GateCase):
    def setUp(self):
        super().setUp()
        self.patch("doc_text", lambda doc: " ".join(
            m["name"] for vs in doc["vertexSet"] for m in vs))
        self.patch("entity_name",
                   lambda doc, i: doc["vertexSet"][i][0]["name"])

    def test_signatures_match_original(self):
        sig = g.build_type_signatures(DOCS, REL)
        self.assertEqual(
            json.dumps({k: sorted(v) for k, v in sig.items()},
                       indent=1),
            (GOLD / "cascade_sig.json").read_text())

    def test_gen_pairs_match_original(self):
        sig = g.build_type_signatures(DOCS, REL)
        self.assertEqual(
            json.dumps(g.gen_pairs(DOCS, REL, sig, 2, 42), indent=1),
            (GOLD / "cascade_pairs.json").read_text())

    def test_main_matches_original(self):
        def fake_choice(text, s, o, options, model=None):
            if s == "Carol":
                return "none", 0.9, {"none": 0.9}, dict(USAGE)
            return "born_in", 0.8, {"born_in": 0.8, "none": 0.2}, \
                dict(USAGE)

        def fake_validate(text, s, r, o, model=None, variant="v1"):
            boom_check(s, r, o)
            return p_of(s, r, o), dict(USAGE)

        self.patch("choice_gate", fake_choice)
        self.patch("validate_triple", fake_validate)
        self.patch("load_dev", lambda: (DOCS, REL))
        out = self.out("cascade_main.json")
        g.main(["cascade", "--n-docs", "2", "--seed", "42",
                "--threshold", "0.5", "--out", out])
        self.assert_golden("cascade_main.json", out)


if __name__ == "__main__":
    unittest.main()
