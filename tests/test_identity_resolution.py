"""Identity resolution — fragment endpoints are mention
forms, not entities.

The measured anatomy (issue #58, sweep pilot cuts 1-2):
the proposer types endpoints in the surface the local
text uses — a first name alone (Wilfried), a nickname
(Willi), a surname alone (Alger), a name missing its
suffix (Paul Desmarais for Paul Desmarais Jr.). The
graph keys nodes by name string, so every fragment was
a separate node: one person as three nodes, surname
fragments wiring relations to the wrong bearer, a
father and son merged. guard_triples attested the
fragments (they occur in the text) and nothing resolved
them.

The fix is deterministic code in the shared funnel:
census_entities builds a roster whose surfaces are
canonical (no paragraph glue, no sentence-initial prose
words, quoted nicknames bridged and recorded as
aliases, alias forms folded), and guard_triples
resolves every endpoint against that roster — a unique
resolution rewrites and is recorded in `canonicalized`;
an ambiguous one is recorded in `identity_ambiguous`
and left as proposed. No model is consulted: identity
is a code question.
"""
import unittest

from pipeline import proposer

SCHNEIDER_DOC = (
    "Willi Schneider (skeleton racer)\n\n"
    "Wilfried \" Willi \" Schneider ( born 13 March 1963 ) "
    "is a German skeleton racer who competed from 1992 "
    "to 2002 . Schneider won a medal in 2002 ."
)
SCHNEIDER_QUOTE = (
    "Wilfried \" Willi \" Schneider ( born 13 March "
    "1963 ) is a German skeleton racer who competed "
    "from 1992 to 2002"
)

ALGER_DOC = (
    "Ross Alger\n\n"
    "Ross Patterson Alger ( 1920 ) was a politician . "
    "Harry Alger was a writer . "
    "Alger served in the assembly ."
)

DESMARAIS_DOC = (
    "Paul Desmarais Jr. studied at INSEAD . "
    "H\u00e9l\u00e8ne Desmarais is a businesswoman . "
    "Paul Desmarais founded the company ."
)


def _norms(census):
    return [e["norm"] for e in census["entities"]]


class TestCensusIdentity(unittest.TestCase):
    def test_nickname_bridged_and_alias_form_folded(self):
        c = proposer.census_entities(SCHNEIDER_DOC)
        norms = _norms(c)
        self.assertIn("wilfried schneider", norms)
        # The title form is an alias, not a second entity.
        self.assertNotIn("willi schneider", norms)
        ent = next(e for e in c["entities"]
                   if e["norm"] == "wilfried schneider")
        self.assertIn("willi", ent["aliases"])
        self.assertIn("willi schneider", ent["aliases"])
        # The bare first name never forms a surface here:
        # its only occurrence is inside the full name.
        self.assertNotIn("wilfried", norms)

    def test_no_paragraph_glue(self):
        c = proposer.census_entities(ALGER_DOC)
        norms = _norms(c)
        self.assertIn("ross alger", norms)
        self.assertIn("ross patterson alger", norms)
        self.assertNotIn("ross alger ross patterson alger",
                         norms)

    def test_sentence_initial_prose_word_dropped(self):
        c = proposer.census_entities(
            "During Alger 's term , the council built "
            "roads . The roads lasted .")
        norms = _norms(c)
        self.assertNotIn("during alger", norms)
        self.assertIn("alger", norms)


class TestGuardResolution(unittest.TestCase):
    def test_nickname_endpoint_resolves(self):
        kept, blocked = proposer.guard_triples(
            [{"subject": "Willi", "relation": "end time",
              "object": "2002", "quote": SCHNEIDER_QUOTE}],
            SCHNEIDER_DOC)
        self.assertEqual(blocked, [])
        self.assertEqual(kept[0]["subject"],
                         "Wilfried Schneider")
        self.assertEqual(
            kept[0]["canonicalized"],
            {"subject": ["Willi", "Wilfried Schneider"]})

    def test_surname_endpoint_resolves_uniquely(self):
        kept, _ = proposer.guard_triples(
            [{"subject": "Schneider", "relation": "won",
              "object": "a medal",
              "quote": "Schneider won a medal in 2002"}],
            SCHNEIDER_DOC)
        self.assertEqual(kept[0]["subject"],
                         "Wilfried Schneider")

    def test_fragment_and_full_duplicate_merge(self):
        items = [
            {"subject": "Willi", "relation": "end time",
             "object": "2002", "quote": SCHNEIDER_QUOTE},
            {"subject": "Wilfried Schneider",
             "relation": "end time", "object": "2002",
             "quote": SCHNEIDER_QUOTE},
        ]
        kept, _ = proposer.guard_triples(items, SCHNEIDER_DOC)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["subject"],
                         "Wilfried Schneider")

    def test_ambiguous_surname_marked_not_guessed(self):
        kept, _ = proposer.guard_triples(
            [{"subject": "Alger", "relation": "served in",
              "object": "the assembly",
              "quote": "Alger served in the assembly"}],
            ALGER_DOC)
        self.assertEqual(kept[0]["subject"], "Alger")
        self.assertNotIn("canonicalized", kept[0])
        self.assertEqual(
            kept[0]["identity_ambiguous"],
            {"subject": ["Harry Alger",
                         "Ross Patterson Alger"]})

    def test_anchor_disambiguates(self):
        kept, _ = proposer.guard_triples(
            [{"subject": "Alger", "relation": "served in",
              "object": "the assembly",
              "quote": "Alger served in the assembly",
              "anchor_norm": "ross alger"}],
            ALGER_DOC)
        self.assertEqual(kept[0]["subject"], "Ross Alger")

    def test_suffix_completed(self):
        kept, _ = proposer.guard_triples(
            [{"subject": "Paul Desmarais",
              "relation": "studied at", "object": "INSEAD",
              "quote": "Paul Desmarais Jr. studied at "
                       "INSEAD"}],
            DESMARAIS_DOC)
        self.assertEqual(kept[0]["subject"],
                         "Paul Desmarais Jr")

    def test_unresolvable_endpoint_untouched(self):
        kept, _ = proposer.guard_triples(
            [{"subject": "Wilfried Schneider",
              "relation": "end time", "object": "2002",
              "quote": SCHNEIDER_QUOTE}],
            SCHNEIDER_DOC)
        self.assertEqual(kept[0]["subject"],
                         "Wilfried Schneider")
        self.assertNotIn("canonicalized", kept[0])
        self.assertNotIn("identity_ambiguous", kept[0])


if __name__ == "__main__":
    unittest.main()


class TestHyphenCompound(unittest.TestCase):
    # The corpus writes compounds spaced: "Bernheimer
    # - Haus" is one name, and a bare "Haus" endpoint
    # names it under a part.
    DOC = ("A dealer built the Bernheimer - Haus in "
           "Munich . The Haus still stands in Munich .")

    def test_spaced_hyphen_compound_resolves(self):
        kept, _ = proposer.guard_triples(
            [{"subject": "Haus", "relation": "located in",
              "object": "Munich",
              "quote": "built the Bernheimer - Haus in "
                       "Munich"}],
            self.DOC)
        self.assertEqual(kept[0]["subject"],
                         "Bernheimer-Haus")
        self.assertEqual(
            kept[0]["canonicalized"],
            {"subject": ["Haus", "Bernheimer-Haus"]})
