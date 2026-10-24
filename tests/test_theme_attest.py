"""BO-2 / BO-3 regressions (BYO ontology review, 2026-10-23).

BO-3: the theme-presence guard blocked a hand-authored pack's
verb+preposition relation wholesale (10/10 "works at"
proposals) with the reason visible only inside a metrics
JSON. The guard stays — it exists to stop fabricated
endpoints — but it is now VISIBLE (run output names the
blocked count by reason; see batch_run stage_candidates),
DOCUMENTED (hands-on Track 3, candidate guards), and
GOVERNABLE: the pack author declares `theme_attest_chunk`
on the schema draft, the declaration rides the emitted
payload, and the free-proposal guard then accepts the
object attested anywhere in the proposal's chunk. Default
(absent) behaviour is byte-identical to before.

BO-2: the Track 3 first command died with a raw
FileNotFoundError when the drafts/ directory did not exist
yet; the builder now creates its output directories like
the rest of the product.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline import ontology_builder as ob
from pipeline import profile_config
from pipeline.proposer import guard_triples, norm_ws

DOC = ("Marta Ellison is the head barista at Hearthline "
       "Coffee Roasters, where she has worked since 2019. "
       "A rival chain, Northbeam Coffee, opened nearby.")


def _item(**over):
    it = {"subject": "Marta Ellison",
          "relation": "works at",
          "object": "Hearthline Coffee Roasters",
          "quote": "Marta Ellison is the head barista",
          "chunk_norm": norm_ws(
              "Marta Ellison is the head barista at "
              "Hearthline Coffee Roasters, where she has "
              "worked since 2019.")}
    it.update(over)
    return it


class TestThemePresenceGuard(unittest.TestCase):
    """The quote-only default: object inside the chunk but
    outside the shortest quote blocks by name."""

    def test_default_blocks_object_outside_quote(self):
        kept, blocked = guard_triples([_item()], DOC)
        self.assertEqual(kept, [])
        self.assertEqual([b["reason"] for b in blocked],
                         ["theme_presence"])

    def test_lever_accepts_object_attested_in_chunk(self):
        kept, blocked = guard_triples(
            [_item()], DOC, theme_attest_chunk=True)
        self.assertEqual(blocked, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["attest"], "chunk")

    def test_lever_never_invents_attestation(self):
        # The object is in the document (L0 passes) but NOT
        # in this proposal's chunk: still blocked — the lever
        # widens the span to the chunk, not to the document.
        it = _item(object="Northbeam Coffee",
                   quote="Marta Ellison is the head barista")
        kept, blocked = guard_triples(
            [it], DOC, theme_attest_chunk=True)
        self.assertEqual(kept, [])
        self.assertEqual([b["reason"] for b in blocked],
                         ["theme_presence"])

    def test_quote_attested_needs_no_lever(self):
        it = _item(quote="Marta Ellison is the head barista "
                         "at Hearthline Coffee Roasters")
        kept, blocked = guard_triples([it], DOC)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["attest"], "quote")


class TestLeverDeclaration(unittest.TestCase):
    """The declaration surface: draft -> payload -> pack
    setup_config validation (the same channel
    merge_prohibited_type_pairs rides)."""

    def _draft(self, **over):
        draft = {"kind": ob.DRAFT_KIND, "family": "coffee",
                 "pack_id": "coffee-shops",
                 "schema_front": {
                     "purpose": "Coffee-shop schema.",
                     "entity_inventory": [
                         {"type": "Person", "role": "entity",
                          "definition": "A named individual.",
                          "form_examples": ["a full name"],
                          "not_entities": {"counts": ["3 staff"]}},
                         {"type": "Organization",
                          "role": "entity",
                          "definition": "A named business.",
                          "form_examples": ["a shop name"],
                          "not_entities":
                              {"generic": ["the shop"]}}],
                     "relation_inventory": [
                         {"type": "WorksAt",
                          "definition": "Employment stated.",
                          "endpoint_signature":
                              ["Person", "Organization"],
                          "boundary_examples":
                              ["'barista at Hearthline' states "
                               "it; co-mention does not"],
                          "direction": "directed"}]}}
        draft.update(over)
        return draft

    def test_payload_carries_true_declaration(self):
        payload = ob._payload_for(
            self._draft(theme_attest_chunk=True), "0.1.0")
        self.assertIs(payload["theme_attest_chunk"], True)

    def test_payload_omits_absent_declaration(self):
        payload = ob._payload_for(self._draft(), "0.1.0")
        self.assertNotIn("theme_attest_chunk", payload)

    def test_non_bool_declaration_is_a_named_finding(self):
        codes = {f["code"] for f in ob.validate_schema_draft(
            self._draft(theme_attest_chunk="yes"))}
        self.assertIn("BAD_THEME_ATTEST", codes)

    def test_bool_declaration_validates_clean(self):
        codes = {f["code"] for f in ob.validate_schema_draft(
            self._draft(theme_attest_chunk=True))}
        self.assertNotIn("BAD_THEME_ATTEST", codes)

    def test_lever_is_approval_bound(self):
        # The lever rides the payload, so it sits inside the
        # approval hash like merge_prohibited_type_pairs:
        # toggling it after approval stale-dates the record.
        base = ob.schema_hash(self._draft())
        levered = ob.schema_hash(
            self._draft(theme_attest_chunk=True))
        self.assertNotEqual(base, levered)
        self.assertEqual(
            base, ob.schema_hash(
                self._draft(theme_attest_chunk=False)))

    def test_setup_config_registration(self):
        self.assertIn("theme_attest_chunk",
                      profile_config.CONSUMED_KEYS)
        self.assertEqual(profile_config.setup_config_errors(
            {"theme_attest_chunk": True}), [])
        errs = profile_config.setup_config_errors(
            {"theme_attest_chunk": "yes"})
        self.assertTrue(any("theme_attest_chunk" in e
                            for e in errs))


class TestDraftCreatesOutputDirs(unittest.TestCase):
    """BO-2: the Track 3 first command writes into a
    drafts/ directory the walkthrough never creates."""

    def test_write_json_creates_parents(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drafts" / "schema.json"
            ob._write_json(p, {"a": 1})
            self.assertEqual(json.loads(p.read_text()),
                             {"a": 1})

    def test_draft_cli_into_missing_dir(self):
        profile = {
            "family": "fieldnotes",
            "profile_hash": "sha256:0123456789abcdef",
            "inventory": {"n_files": 40, "n_unique_content": 40,
                          "total_chars": 120000},
            "claims": [
                {"section": "inventory", "kind": "fact",
                 "grade": "FULL",
                 "statement": "Corpus holds 40 document files."},
                {"section": "labels", "kind": "fact",
                 "grade": "FULL",
                 "statement": "No gold annotations."}]}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            profile_path = td / "profile.json"
            profile_path.write_text(json.dumps(profile))
            out = td / "drafts" / "field-notes.json"
            rc = ob.main(["draft", "--profile",
                          str(profile_path),
                          "--pack-id", "field-notes",
                          "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
