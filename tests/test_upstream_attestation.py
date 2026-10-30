"""Upstream attestation program (2026-10-29): document
registry, tiered attestation, theme grading, duplicate
accounting, and the identity absorption fix.

Each test pins one measured failure from the provenance
audit / decision tests (launch notes: decision-tests.md).
"""
import unittest

from pipeline.proposer import guard_triples


class TestRegistry(unittest.TestCase):
    def test_defined_term_canonicalizes_and_skips_role(self):
        # The CUAD anatomy: 77 of 89 role-word blocks were
        # the contract's own defined parties.
        doc = ('Audible Inc. (the "Company") entered the '
               'agreement. The Company will pay ACSI the '
               'amount due.')
        kept, blocked = guard_triples(
            [{"subject": "Company", "relation": "will pay",
              "object": "ACSI",
              "quote": "The Company will pay ACSI the "
                       "amount due"}],
            doc, role_words={"company"})
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["subject"], "Audible Inc.")
        self.assertEqual(
            kept[0]["canonicalized"],
            {"subject": ["Company", "Audible Inc."]})

    def test_means_defined_term_is_identity_not_role(self):
        doc = ('"Seller" means the party identified on the '
               'cover page. The Seller shall deliver the '
               'goods to Buyer Corp.')
        kept, blocked = guard_triples(
            [{"subject": "Seller", "relation": "shall deliver",
              "object": "Buyer Corp",
              "quote": "The Seller shall deliver the goods "
                       "to Buyer Corp"}],
            doc, role_words={"seller"})
        self.assertEqual(blocked, [])
        # Kept (the defined term is an identity here);
        # the census canonical carries the article, as
        # with "The Hague".
        self.assertEqual(kept[0]["subject"], "The Seller")

    def test_descriptor_interrupted_definition(self):
        # The real preamble form: a descriptor clause
        # ('a Delaware corporation') sits between the
        # name and the parenthetical term.
        doc = ('Agreement between Amazon.com Commerce '
               'Services, Inc., a Delaware corporation '
               '("ACSI"), and Audible Inc. a Delaware '
               'corporation ("Company"). The Company will '
               'pay ACSI the amount due.')
        kept, blocked = guard_triples(
            [{"subject": "Company", "relation": "will pay",
              "object": "ACSI",
              "quote": "The Company will pay ACSI the "
                       "amount due"}],
            doc, role_words={"company"})
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["subject"], "Audible Inc.")

    def test_collective_and_bare_shall_mean_definitions(self):
        doc = ('ACSI and Company are sometimes referred '
               'to collectively herein as the "Parties". '
               'PARTY shall mean each of them. The Parties '
               'shall form the Joint Committee.')
        kept, blocked = guard_triples(
            [{"subject": "Parties", "relation": "shall form",
              "object": "Joint Committee",
              "quote": "The Parties shall form the Joint "
                       "Committee"}],
            doc, role_words={"parties", "party"})
        self.assertEqual(blocked, [])
        self.assertEqual(len(kept), 1)

    def test_lookback_never_binds_a_lone_suffix(self):
        # Regression (CUAD replay): ("Vyera") must bind
        # to 'Vyera Pharmaceuticals', never to the lone
        # suffix run 'LLC' that follows it.
        doc = ('CytoDyn Inc. entered a license with Vyera '
               'Pharmaceuticals, LLC ("Vyera"). Vyera will '
               'market the drug in the Territory.')
        kept, blocked = guard_triples(
            [{"subject": "CytoDyn Inc.", "relation": "entered",
              "object": "Vyera",
              "quote": "CytoDyn Inc. entered a license with "
                       "Vyera Pharmaceuticals, LLC"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["object"],
                         "Vyera Pharmaceuticals")

    def test_lookback_stops_at_sentence_boundary(self):
        # Regression (CUAD replay): a term whose
        # parenthesis follows a different sentence's name
        # is defined-only — never bound across the
        # boundary ('Development IP' is not the
        # 'Development Plan').
        doc = ('The Development Plan was approved. The '
               'assets ("Development IP") are listed in '
               'Schedule 2. Development IP includes the '
               'patents of Qualigen Therapeutics.')
        kept, blocked = guard_triples(
            [{"subject": "Development IP",
              "relation": "includes",
              "object": "the patents of Qualigen Therapeutics",
              "quote": "Development IP includes the patents "
                       "of Qualigen Therapeutics"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["subject"], "Development IP")
        self.assertNotIn("canonicalized", kept[0])

    def test_clause_defined_term_not_bound_to_shared_modifier(self):
        # Regression (CUAD replay, Qualigen): the term is
        # defined by the clause and shares only a modifier
        # with the preceding name — it is defined, but it
        # is NOT that name.
        doc = ('Qualigen shall own all Intellectual '
               'Property Rights resulting from Qualigen\'s '
               'activities under the Development Plan '
               '("Development IP"). Qualigen shall own '
               'the Development IP.')
        kept, blocked = guard_triples(
            [{"subject": "Qualigen", "relation": "shall own",
              "object": "Development IP",
              "quote": "Qualigen shall own the "
                       "Development IP"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["object"], "Development IP")
        self.assertNotIn("canonicalized", kept[0])

    def test_undefined_role_word_still_blocks(self):
        doc = "The Company will pay ACSI the amount due."
        kept, blocked = guard_triples(
            [{"subject": "Company", "relation": "will pay",
              "object": "ACSI",
              "quote": "The Company will pay ACSI the "
                       "amount due"}],
            doc, role_words={"company"})
        self.assertEqual(kept, [])
        self.assertEqual([b["reason"] for b in blocked],
                         ["l0_role_word"])

    def test_korean_defined_term_is_identity(self):
        doc = ('홍길동(이하 "갑")은 물품을 공급한다. '
               '갑은 대금을 주식회사 한빛에 지급한다.')
        kept, blocked = guard_triples(
            [{"subject": "갑", "relation": "지급한다",
              "object": "주식회사 한빛",
              "quote": "갑은 대금을 주식회사 한빛에 지급한다"}],
            doc, role_words={"갑"})
        self.assertEqual(blocked, [])
        self.assertEqual(len(kept), 1)

    def test_acronym_canonicalizes_to_long_form(self):
        doc = ('The Food and Drug Administration (FDA) '
               'announced the rule. The FDA published the '
               'guidance on Friday.')
        kept, blocked = guard_triples(
            [{"subject": "FDA", "relation": "published",
              "object": "the guidance",
              "quote": "The FDA published the guidance "
                       "on Friday"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["subject"],
                         "Food and Drug Administration")


class TestTieredAttestation(unittest.TestCase):
    def test_spacing_repair_is_the_normalized_tier(self):
        # Tokenized corpora space their hyphens:
        # 'Won - yong' in the text, 'Won-yong' proposed.
        doc = "Kang Won - yong won the race in 2019."
        kept, blocked = guard_triples(
            [{"subject": "Kang Won-yong", "relation": "won",
              "object": "the race",
              "quote": "Kang Won - yong won the race"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["attest_kind"],
                         {"subject": "normalized"})

    def test_declared_surface_is_the_declared_tier(self):
        doc = ("The German chancellor addressed the "
               "parliament on Tuesday.")
        item = {"subject": "Germany",
                "relation": "addressed",
                "object": "the parliament",
                "quote": "The German chancellor addressed "
                         "the parliament",
                "subject_attested": "German"}
        kept, blocked = guard_triples([item], doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["attest_kind"],
                         {"subject": "declared"})
        # Without the declaration the same item blocks:
        # the guard verifies the declared surface, it
        # does not invent one.
        item2 = {k: v for k, v in item.items()
                 if k != "subject_attested"}
        kept2, blocked2 = guard_triples([item2], doc)
        self.assertEqual(kept2, [])
        self.assertEqual([b["reason"] for b in blocked2],
                         ["l0_attest"])


class TestIdentityAbsorption(unittest.TestCase):
    def test_complete_name_not_absorbed_by_longer_name(self):
        # "Calgary" the city is not "Calgary City
        # Council" the organization.
        doc = ("Calgary is a city in Alberta. The Calgary "
               "City Council met on Tuesday. Calgary City "
               "Council approved the budget.")
        kept, blocked = guard_triples(
            [{"subject": "Calgary", "relation": "is a city in",
              "object": "Alberta",
              "quote": "Calgary is a city in Alberta"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["subject"], "Calgary")
        self.assertNotIn("canonicalized", kept[0])


    def test_fragment_of_coordination_not_absorbed(self):
        # "Cambodia" must not be rewritten into the
        # coordination "Vietnam and Cambodia": the pair
        # is not an entity.
        doc = ("Phnom Penh is the capital of Cambodia. "
               "Vietnam and Cambodia share a long border.")
        kept, blocked = guard_triples(
            [{"subject": "Phnom Penh",
              "relation": "is the capital of",
              "object": "Cambodia",
              "quote": "Phnom Penh is the capital of "
                       "Cambodia"}],
            doc)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["object"], "Cambodia")
        self.assertNotIn("canonicalized", kept[0])


class TestDuplicateAccounting(unittest.TestCase):
    def test_duplicate_is_recorded_not_silent(self):
        doc = "Kang Won - yong won the race in 2019."
        item = {"subject": "Kang Won-yong", "relation": "won",
                "object": "the race",
                "quote": "Kang Won - yong won the race"}
        kept, blocked = guard_triples([item, dict(item)], doc)
        self.assertEqual(len(kept), 1)
        self.assertEqual([b["reason"] for b in blocked],
                         ["duplicate"])


if __name__ == "__main__":
    unittest.main()
