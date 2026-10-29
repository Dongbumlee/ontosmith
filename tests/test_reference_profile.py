"""Reference-profile pinning tests.

The reference profile (pipeline/reference_profile.py) is the
single home of the tested model bindings. These tests pin
(a) the bindings themselves to the values the quality record
was measured with — the pre-tidying literals — and (b) every
module default that was rewired to flow from it, so a silent
default drift anywhere fails the suite.
"""
import dataclasses
import inspect
import unittest

from pipeline import (
    batch,
    batch_cli,
    batch_run,
    decisions,
    embedder,
    pair_proposer,
    proposer,
    query_stack,
    roster_pass,
    setup_service,
)
from pipeline import (
    reference_profile as rp,
)

PRE_CHANGE = {
    "PROPOSER_MODEL": "anthropic/claude-haiku-5.5",
    "JUDGE_MODEL": "typesafe/jev-1.13",
    "ESCALATION_MODEL": "anthropic/claude-sonnet-5.5",
    "EMBED_MODEL": "openai/text-embedding-3-large",
    "EMBED_DIMS": 3072,
    "NER_MODEL": "urchade/gliner_medium-v2.1",
    "CHAT_URL": "https://openrouter.ai/api/v1/chat/completions",
    "DECISIONS_URL": "https://openrouter.ai/api/alpha/decisions",
    "SYSTEMONE_URL": "https://openrouter.ai/api/v1/systemone",
}


class TestReferenceProfileValues(unittest.TestCase):
    def test_bindings_match_measured_values(self):
        for name, expected in PRE_CHANGE.items():
            self.assertEqual(getattr(rp, name), expected, name)

    def test_unmeasured_names_are_not_preset_defaults(self):
        # The cast is exactly the models tested in
        # their roles. The bake-off candidate that lost to
        # Jev and the one-off drafting experiment are not
        # presented as defaults anywhere in the preset.
        for gone in ("DECISION_MODEL", "DRAFT_MODEL"):
            self.assertFalse(hasattr(rp, gone), gone)


class TestRewiredDefaults(unittest.TestCase):
    def test_proposer_family_defaults(self):
        self.assertEqual(proposer.DEFAULT_MODEL, rp.PROPOSER_MODEL)
        self.assertEqual(pair_proposer.DEFAULT_MODEL, rp.PROPOSER_MODEL)
        self.assertEqual(roster_pass.DEFAULT_MODEL, rp.PROPOSER_MODEL)

    def test_batch_config_defaults(self):
        defaults = {f.name: f.default
                    for f in dataclasses.fields(batch.BatchConfig)}
        self.assertEqual(defaults["ner_model"], rp.NER_MODEL)
        self.assertEqual(defaults["gate_model"], rp.JUDGE_MODEL)
        self.assertEqual(defaults["embed_model"], rp.EMBED_MODEL)
        self.assertEqual(defaults["llm_model"], rp.PROPOSER_MODEL)

    def test_query_stack_constants(self):
        self.assertEqual(query_stack.EMBED_MODEL, rp.EMBED_MODEL)
        self.assertEqual(query_stack.CHAT_MODEL, rp.PROPOSER_MODEL)
        self.assertEqual(query_stack.CHAT_URL, rp.CHAT_URL)

    def test_embedder_default(self):
        sig = inspect.signature(embedder.OpenRouterEmbedder.__init__)
        self.assertEqual(sig.parameters["model"].default, rp.EMBED_MODEL)

    def test_setup_service_defaults(self):
        # The drafting seat ships unbound, so the
        # profile-setup draft model has no preset default —
        # it resolves from the caller's binding at run time.
        sig = inspect.signature(setup_service.run_setup)
        self.assertIsNone(sig.parameters["model"].default)
        self.assertFalse(hasattr(setup_service,
                                 "DEFAULT_DRAFT_MODEL"))
        self.assertEqual(setup_service.PLAYBOOK_DEFAULTS["embed_model"],
                         rp.EMBED_MODEL)
        self.assertEqual(setup_service.CHAT_URL, rp.CHAT_URL)

    def test_decisions_defaults(self):
        self.assertEqual(decisions.DEFAULT_MODEL, rp.ESCALATION_MODEL)
        self.assertEqual(decisions.DECISIONS_URL, rp.DECISIONS_URL)
        self.assertEqual(decisions.CHAT_URL, rp.CHAT_URL)
        sig = inspect.signature(decisions.er_judge)
        self.assertEqual(sig.parameters["model"].default, rp.JUDGE_MODEL)

    def test_batch_cli_gate_defaults(self):
        for fn in (batch_cli._decide_fn, batch_cli._er_decide_fn):
            sig = inspect.signature(fn)
            self.assertEqual(sig.parameters["model"].default,
                             rp.JUDGE_MODEL, fn.__name__)

    def test_batch_run_judge_constant(self):
        # batch_run no longer re-exports (and stage
        # code no longer reads) the frozen judge constant —
        # the ER identity pings take the batch's own
        # gate_model, resolved from the judge seat binding
        # at create. The constant's one legitimate home in
        # the batch layer is BatchConfig's default (the
        # no-binding fallback), which stays the reference
        # profile's model.
        self.assertFalse(hasattr(batch_run, "JUDGE_MODEL"))
        from pipeline.batch import BatchConfig
        self.assertEqual(BatchConfig().gate_model, rp.JUDGE_MODEL)

    def test_no_provider_literals_outside_reference_profile(self):
        # The provider's endpoints live in exactly one module
        # (the connection-layer preset). Any other pipeline
        # module naming the provider host is a layering leak.
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        offenders = []
        for p in sorted((root / "pipeline").glob("*.py")):
            if p.name == "reference_profile.py":
                continue
            if "openrouter.ai" in p.read_text():
                offenders.append(p.name)
        self.assertEqual(offenders, [])

    def test_provider_sort_single_home(self):
        # The provider-routing preference
        # is a connection-layer knob with exactly one literal
        # home — the reference profile. Transports reference
        # the preset; none restates the value.
        self.assertEqual(rp.PROVIDER_SORT, "throughput")
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        offenders = []
        for p in sorted((root / "pipeline").glob("*.py")):
            if p.name == "reference_profile.py":
                continue
            if '"sort": "throughput"' in p.read_text() \
                    or "'sort': 'throughput'" in p.read_text():
                offenders.append(p.name)
        self.assertEqual(offenders, [])
        # The setup tool's authored defaults carry the preset,
        # not a restated literal.
        self.assertEqual(setup_service.PLAYBOOK_DEFAULTS[
            "provider_sort"], rp.PROVIDER_SORT)


if __name__ == "__main__":
    unittest.main()
