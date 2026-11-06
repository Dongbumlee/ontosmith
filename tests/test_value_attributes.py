"""Value attributes (design note 2026-11-06): declaration,
processing, storage and lineage.

Covered: the declaration layer — a schema_front entity entry
may declare `attributes` (name + closed value_type +
required/multi), held to one shape by the ontology builder's
draft validation and by setup_config validation, and rendered
into proposer guidance ONLY when declared (a schema without
attributes renders byte-identically, so frozen guidance never
moves); the processing path — a candidate passes a
deterministic format gate before any judge is consulted, an
undeclared attribute does not exist, multi is enforced per
value, a missing required attribute is a warning that keeps
the entity; storage — accepted values land on the graph node
as typed properties (native date/datetime/duration) with the
original surface beside them, while an entity carrying no
attributes is written with the legacy statement untouched;
and lineage — attribute addition is safe, a value_type
change, a requirement strengthening or a removal is a
meaning/deletion change that needs its structured
seed-change record.
"""
from __future__ import annotations

import copy
import unittest

from pipeline import attribute_values as av
from pipeline import ontology_builder as ob
from pipeline import profile_config


def _schema(with_attributes: bool = True) -> dict:
    agreement = {
        "type": "Agreement", "role": "entity",
        "definition": "A contract between two parties.",
        "form_examples": ["a signed supply agreement"],
        "not_entities": {"generic": ["the agreement"]},
    }
    if with_attributes:
        agreement["attributes"] = [
            {"name": "effective_date", "value_type": "date",
             "required": True, "multi": False,
             "definition": "The day the agreement takes "
                           "effect."},
            {"name": "contract_value", "value_type": "money",
             "required": False, "multi": False},
            {"name": "renewal_notice_days",
             "value_type": "number", "required": False,
             "multi": True},
        ]
    return {
        "purpose": "Contract schema.",
        "entity_inventory": [
            agreement,
            {"type": "Party", "role": "entity",
             "definition": "A party to an agreement.",
             "form_examples": ["a company name"],
             "not_entities": {"generic": ["the party"]}},
        ],
        "relation_inventory": [
            {"type": "PartyTo",
             "definition": "The text states the party is a "
                           "party to the agreement.",
             "endpoint_signature": ["Party", "Agreement"],
             "boundary_examples": [
                 "'entered into by Acme' states partyhood; "
                 "a mere mention does not"]},
        ],
    }


def _draft(schema=None, **overrides) -> dict:
    draft = {"kind": ob.DRAFT_KIND, "family": "contracts",
             "pack_id": "contract-pack",
             "schema_front": schema or _schema()}
    draft.update(overrides)
    return draft


def _cfg(schema=None) -> dict:
    return {"schema_front": schema or _schema(),
            "entity_types": ["Agreement", "Party"],
            "relation_types": ["PartyTo"]}


def _codes(findings) -> set[str]:
    return {f["code"] for f in findings}


class TestDeclarationValidation(unittest.TestCase):

    def test_valid_declaration_is_clean(self):
        self.assertEqual(ob.validate_schema_draft(_draft()),
                         [])

    def test_no_attributes_is_clean(self):
        draft = _draft(_schema(with_attributes=False))
        self.assertEqual(ob.validate_schema_draft(draft), [])

    def test_unknown_value_type_is_a_finding(self):
        schema = _schema()
        schema["entity_inventory"][0]["attributes"][0][
            "value_type"] = "freetext"
        findings = ob.validate_schema_draft(_draft(schema))
        self.assertIn("BAD_ATTRIBUTE", _codes(findings))

    def test_duplicate_attribute_name_is_a_finding(self):
        schema = _schema()
        schema["entity_inventory"][0]["attributes"].append(
            {"name": "effective_date", "value_type": "date"})
        findings = ob.validate_schema_draft(_draft(schema))
        self.assertIn("BAD_ATTRIBUTE", _codes(findings))

    def test_non_slug_name_is_a_finding(self):
        schema = _schema()
        schema["entity_inventory"][0]["attributes"][0][
            "name"] = "Effective Date"
        findings = ob.validate_schema_draft(_draft(schema))
        self.assertIn("BAD_ATTRIBUTE", _codes(findings))

    def test_non_boolean_flags_are_a_finding(self):
        schema = _schema()
        schema["entity_inventory"][0]["attributes"][0][
            "required"] = "yes"
        findings = ob.validate_schema_draft(_draft(schema))
        self.assertIn("BAD_ATTRIBUTE", _codes(findings))

    def test_attributes_not_a_list_is_a_finding(self):
        schema = _schema()
        schema["entity_inventory"][0]["attributes"] = {
            "name": "effective_date"}
        findings = ob.validate_schema_draft(_draft(schema))
        self.assertIn("BAD_ATTRIBUTE", _codes(findings))

    def test_setup_config_holds_the_same_shape(self):
        errors = profile_config.setup_config_errors(_cfg())
        self.assertEqual(errors, [])
        schema = _schema()
        schema["entity_inventory"][0]["attributes"][1][
            "value_type"] = "blob"
        errors = profile_config.setup_config_errors(
            _cfg(schema))
        self.assertTrue(any("contract_value" in e or
                            "attributes[1]" in e
                            for e in errors), errors)

    def test_declaration_accessors(self):
        cfg = _cfg()
        decls = profile_config.entity_attributes(cfg,
                                                 "Agreement")
        self.assertEqual([d["name"] for d in decls],
                         ["effective_date", "contract_value",
                          "renewal_notice_days"])
        self.assertEqual(
            profile_config.entity_attributes(cfg, "Party"),
            [])
        self.assertIsNotNone(profile_config.declared_attribute(
            cfg, "Agreement", "contract_value"))
        self.assertIsNone(profile_config.declared_attribute(
            cfg, "Agreement", "signing_date"))
        self.assertTrue(
            profile_config.schema_front_has_attributes(cfg))
        self.assertFalse(
            profile_config.schema_front_has_attributes(
                _cfg(_schema(with_attributes=False))))
        self.assertFalse(
            profile_config.schema_front_has_attributes({}))


class TestGuidanceRendering(unittest.TestCase):

    def test_attributes_render_when_declared(self):
        text = profile_config.derive_guidance(_cfg())
        self.assertIn("Attribute effective_date (date, "
                      "required)", text)
        self.assertIn("Attribute contract_value (money)", text)
        self.assertIn("multiple values allowed", text)
        self.assertIn('"attribute": <exact attribute name>',
                      text)

    def test_no_attributes_renders_without_attribute_text(self):
        text = profile_config.derive_guidance(
            _cfg(_schema(with_attributes=False)))
        self.assertNotIn("Attribute ", text)
        self.assertNotIn('"attribute"', text)

    def test_no_attributes_rendering_is_byte_identical(self):
        # A schema_front that declares no attributes must
        # render exactly what the pre-attribute renderer
        # produced — pinned here by rendering the same front
        # with the attributes key stripped entirely vs set
        # to an empty list on one entry.
        plain = _schema(with_attributes=False)
        with_empty = copy.deepcopy(plain)
        with_empty["entity_inventory"][0]["attributes"] = []
        self.assertEqual(
            profile_config.derive_guidance(_cfg(plain)),
            profile_config.derive_guidance(_cfg(with_empty)))


def _candidate(**overrides) -> dict:
    candidate = {
        "entity": "Supply Agreement A",
        "entity_type": "Agreement",
        "attribute": "effective_date",
        "value": "March 1, 2024",
        "quote": "effective March 1, 2024",
        "doc_id": "doc-1",
    }
    candidate.update(overrides)
    return candidate


class TestProcessing(unittest.TestCase):

    def test_accepted_record_carries_value_and_surface(self):
        out = av.process_attribute_candidates(
            [_candidate()], _cfg())
        self.assertEqual(len(out["accepted"]), 1)
        record = out["accepted"][0]
        self.assertEqual(record["value"], "2024-03-01")
        self.assertEqual(record["surface"], "March 1, 2024")
        self.assertEqual(record["quote"],
                         "effective March 1, 2024")
        self.assertFalse(record["verdict"]["judged"])

    def test_format_failure_never_reaches_the_judge(self):
        calls = []

        def judge(candidate, normalized):
            calls.append(candidate)
            return True

        out = av.process_attribute_candidates(
            [_candidate(value="whenever it starts",
                        quote="whenever it starts")],
            _cfg(), judge=judge)
        self.assertEqual(calls, [])
        self.assertEqual(len(out["format_rejected"]), 1)
        self.assertEqual(out["accepted"], [])

    def test_missing_quote_fails_at_the_gate(self):
        out = av.process_attribute_candidates(
            [_candidate(quote="")], _cfg(),
            judge=lambda c, n: True)
        self.assertEqual(len(out["format_rejected"]), 1)
        self.assertIn("no quote", out["format_rejected"][0]
                      ["reason"])

    def test_undeclared_attribute_does_not_exist(self):
        out = av.process_attribute_candidates(
            [_candidate(attribute="signing_bonus")], _cfg(),
            judge=lambda c, n: True)
        self.assertEqual(len(out["undeclared"]), 1)
        self.assertEqual(out["accepted"], [])
        out = av.process_attribute_candidates(
            [_candidate(entity_type="Party")], _cfg())
        self.assertEqual(len(out["undeclared"]), 1)

    def test_judge_rejection_is_recorded(self):
        out = av.process_attribute_candidates(
            [_candidate()], _cfg(),
            judge=lambda c, n: {"accepted": False,
                                "p_true": 0.1})
        self.assertEqual(out["accepted"], [])
        self.assertEqual(len(out["judge_rejected"]), 1)
        self.assertEqual(out["judge_rejected"][0]["verdict"]
                         ["p_true"], 0.1)

    def test_judged_acceptance_records_the_verdict(self):
        out = av.process_attribute_candidates(
            [_candidate()], _cfg(),
            judge=lambda c, n: {"accepted": True,
                                "p_true": 0.97})
        record = out["accepted"][0]
        self.assertTrue(record["verdict"]["judged"])
        self.assertEqual(record["verdict"]["p_true"], 0.97)

    def test_proposed_conversion_rides_to_the_judge(self):
        seen = {}

        def judge(candidate, normalized):
            seen.update(normalized)
            return True

        out = av.process_attribute_candidates(
            [_candidate(value="next Friday",
                        quote="next Friday",
                        proposed="2024-03-08")],
            _cfg(), judge=judge)
        self.assertEqual(len(out["accepted"]), 1)
        self.assertFalse(seen["code_normalized"])
        self.assertFalse(out["accepted"][0]
                         ["code_normalized"])

    def test_single_value_attribute_rejects_second_value(self):
        out = av.process_attribute_candidates(
            [_candidate(),
             _candidate(value="April 2, 2024",
                        quote="revised to April 2, 2024")],
            _cfg())
        self.assertEqual(len(out["accepted"]), 1)
        self.assertEqual(len(out["multi_rejected"]), 1)

    def test_identical_repeat_is_a_duplicate(self):
        out = av.process_attribute_candidates(
            [_candidate(), _candidate()], _cfg())
        self.assertEqual(len(out["accepted"]), 1)
        self.assertEqual(len(out["duplicates"]), 1)

    def test_multi_attribute_accepts_distinct_values(self):
        out = av.process_attribute_candidates(
            [_candidate(attribute="renewal_notice_days",
                        value="30", quote="30 days' notice"),
             _candidate(attribute="renewal_notice_days",
                        value="60", quote="60 days for cause")],
            _cfg())
        self.assertEqual(len(out["accepted"]), 2)
        self.assertEqual(out["multi_rejected"], [])

    def test_money_incompleteness_rides_the_record(self):
        out = av.process_attribute_candidates(
            [_candidate(attribute="contract_value",
                        value="500,000",
                        quote="valued at 500,000")],
            _cfg())
        record = out["accepted"][0]
        self.assertEqual(record["value"], 500000)
        self.assertIsNone(record["currency"])
        self.assertFalse(record["complete"])


class TestRequiredWarnings(unittest.TestCase):

    def test_missing_required_warns_and_keeps_the_entity(self):
        entities = [{"entity": "Supply Agreement A",
                     "entity_type": "Agreement"}]
        warnings = av.required_attribute_warnings(
            entities, [], _cfg())
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["attribute"],
                         "effective_date")
        self.assertEqual(warnings[0]["code"],
                         "REQUIRED_ATTRIBUTE_MISSING")

    def test_covered_required_does_not_warn(self):
        out = av.process_attribute_candidates(
            [_candidate()], _cfg())
        entities = [{"entity": "Supply Agreement A",
                     "entity_type": "Agreement"}]
        self.assertEqual(av.required_attribute_warnings(
            entities, out["accepted"], _cfg()), [])

    def test_optional_attributes_never_warn(self):
        entities = [{"entity": "Acme Corp",
                     "entity_type": "Party"}]
        self.assertEqual(av.required_attribute_warnings(
            entities, [], _cfg()), [])


class TestGroups(unittest.TestCase):

    def test_groups_carry_typed_values_in_order(self):
        out = av.process_attribute_candidates(
            [_candidate(),
             _candidate(attribute="renewal_notice_days",
                        value="30", quote="30 days' notice"),
             _candidate(attribute="renewal_notice_days",
                        value="60", quote="60 days for cause")],
            _cfg())
        groups = av.entity_attribute_groups(out["accepted"])
        group = groups["Supply Agreement A"]
        self.assertEqual(
            group["attributes"]["effective_date"]["values"],
            [{"value": "2024-03-01",
              "surface": "March 1, 2024"}])
        self.assertEqual(
            [v["value"] for v in group["attributes"]
             ["renewal_notice_days"]["values"]], [30, 60])


class TestProposerEmission(unittest.TestCase):

    def test_attribute_items_are_split_from_triples(self):
        import json
        from unittest import mock

        from pipeline import proposer

        payload = json.dumps([
            {"subject": "Acme", "relation": "PartyTo",
             "object": "Supply Agreement A",
             "quote": "Acme is party to Supply Agreement A"},
            {"entity": "Supply Agreement A",
             "entity_type": "Agreement",
             "attribute": "effective_date",
             "value": "March 1, 2024",
             "quote": "effective March 1, 2024"},
        ])

        def chat(content, model, max_tokens, key):
            return payload, {"cost": 0.0}

        with mock.patch.object(proposer, "_chat", chat):
            res = proposer.propose_triples_for_doc(
                "doc", "short text", "key")
        self.assertEqual(len(res["items"]), 1)
        self.assertEqual(len(res["attribute_items"]), 1)
        item = res["attribute_items"][0]
        self.assertEqual(item["attribute"], "effective_date")
        self.assertEqual(item["value"], "March 1, 2024")

    def test_no_attribute_items_means_empty_list(self):
        import json
        from unittest import mock

        from pipeline import proposer

        payload = json.dumps([
            {"subject": "Acme", "relation": "PartyTo",
             "object": "Supply Agreement A",
             "quote": "Acme is party to Supply Agreement A"}])

        def chat(content, model, max_tokens, key):
            return payload, {"cost": 0.0}

        with mock.patch.object(proposer, "_chat", chat):
            res = proposer.propose_triples_for_doc(
                "doc", "short text", "key")
        self.assertEqual(res["attribute_items"], [])


if __name__ == "__main__":
    unittest.main()
