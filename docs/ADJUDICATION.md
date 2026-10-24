# The adjudication contract

The standing specification for machine adjudication in
OntoSmith's quality records: how blind samples of predicted
and gold triples are judged, what counts as support, and —
since 2026-10-22 — the explicit rulings that complete the
contract at the six surface situations where judges
otherwise supply their own readings. Every corrected
number in the quality record (whitepaper §5) rests on
adjudication under this contract, and future corrections
inherit it as written: a correction that needs a different
reading changes this document first, by the same
pre-registered discipline, never mid-run.

## The base contract (as-typed)

- **Blind.** Judges see the claim (subject, relation,
  object, as typed) and the document text. They never see
  gold status, sampling stratum, or control flags.
- **As-typed.** The question is whether the document text
  supports the claim *as typed* — the typed subject, the
  typed relation, the typed object — not whether a nearby
  or normalized fact is true.
- **Quote verification.** A *supported* verdict counts
  only with a verbatim quote from the document.
  Quote-failed supported verdicts are recorded unknown and
  excluded from every denominator, as are *unclear*
  verdicts; the counts of both are published with the
  estimate.
- **Controls.** Known items are shuffled into the queue
  blind. Control results are reported beside the estimate,
  never inside it.
- **Verdict vocabulary.** `supported` / `not_supported` /
  `unclear`, each with its evidence quote (or its absence
  stated) and a note.

## The seam rulings (2026-10-22)

Six recurring surface situations the base contract left
silent. Each is settled as a definition; where a ruling
decides a verdict, the judge records the ruling's code
(`seam_applied`), so the seam population of any future
correction is measured, not guessed. The packet text
judges receive is the R1–R6 block of the appendix's
`RULING.md`; the rulings in brief:

1. **DEM — ACCEPT.** An adjectival/demonym form that
   unambiguously denotes the typed entity counts as the
   typed value. Surface-form variation for the same
   referent is not a mismatch.
2. **RELPREC — STRICT.** Support requires the typed
   relation itself; a neighbouring relation (proposing
   vs. founding, formalising vs. developing, co-parents
   vs. spouses, credited vs. cast member, bloc vs. member)
   does not support it.
3. **EVAL — ACCEPT, operationally.** For the evaluative
   relation "notable work": a work the document presents
   prominently about the subject — treated at length,
   named among the subject's works in the lead or a works
   list, itself a main topic — counts as supported. A
   passing mention does not.
4. **CONT — SPLIT.** (a) Transitive containment counts:
   A in B and B in C, both stated, supports A in C.
   (b) Mere co-occurrence, association, or a shared event
   does not support a containment claim. (c) The typed
   endpoint must be the container the document states: a
   macro-region named where the document states a
   different container (or an administrative entity where
   it states only a macro-region) is not supported as
   typed.
5. **IDENT — STRICT.** The typed subject must be the
   entity the document's statements are about. A subject
   truncated, conflated with another entity, or shifted
   in granularity (person vs. office, place vs. sub-area)
   does not carry the fact. *Unclear* remains available
   where the text genuinely does not allow the identity
   question to be decided.
6. **QUAL — STRICT.** A fact the document states only
   with a qualifier — it held initially or formerly, or
   holds for one market, territory, or period — does not
   support the unqualified claim as typed.

## Provenance

The rulings were adopted on 2026-10-22, after a
pre-registered cross-family blind recheck of the flagship
record's verdicts failed its registered bar (agreement
0.815 against 0.95) and localized the disagreement to
exactly these six situations. The flagship's 565 flagged
items were then re-judged blind under this contract and
the record re-priced; the full account, the ruling text
as the judges received it, and both verdict sets are in
the whitepaper (§5.4) and the benchmark appendix
(`benchmarks/appendix/RULING.md`, `judgments_ruled/`).
