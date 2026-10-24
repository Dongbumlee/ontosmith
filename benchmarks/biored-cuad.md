# Quality records — BioRED & CUAD supporting builds (pointer)

**The records' canonical text lives in the [whitepaper](../docs/whitepaper.md)**
— BioRED at §5.7, CUAD at §5.8 (placement revision,
2026-10-22). This file is the in-tree
pointer: what the records cover, and where their data
lives. The claim-level index of every published number
is [`benchmarks/README.md`](README.md).

## Scope

- **Bio Medical pack — BioRED pilot (Dev-100).** The
  schema-first pilot build over the BioRED dev split
  (100 documents; test split sealed), scored against
  criteria carried unchanged since round 3. The
  whitepaper record carries the build, the criteria
  table — including the frozen-precision criterion
  recorded as **FAIL**, unsoftened — the conversion
  census, the schema-first and judge-fitting deltas, the
  Stage D audit in full, and the relation-typing gate
  records (five measured candidates, none adopted; type
  assignment is the pilot's named open problem).
- **Legal Contract pack — CUAD.** The full-corpus build
  (510 documents) under the pack pinned from the
  2026-10-16 profile approval: the extraction-seat
  bake-off, and the certification that closed
  **PROMOTABLE** on its three legs (audit record, pinned
  QA, doc-check closure). The whitepaper record carries
  the bake-off table, the audit and pinned-QA figures
  with their scope (the graph's measured advantage is a
  long-document result), and both limitation sets.

> **Label.** Corrected numbers in the BioRED record rest
> on **machine adjudication** under the P2 pattern (blind
> samples, machine judges, quote verification, embedded
> controls) — machine judgments, not a human audit. The
> CUAD certification audit is stated exactly as its
> record states it (blind labels; the human round-2
> rulings sustained 43/43, 0 overruled). The two records'
> precision figures come from different instruments and
> are never averaged.

## Where the data lives

- The fitting-line measurements behind both records are
  registered as **pack conformance entries**: BioRED as
  CR-1…CR-7 in `packs/bio-medical/conformance.json`,
  CUAD as CC-1…CC-4 in
  `packs/legal-contract/conformance.json` — gate,
  instrument, population, and delta per entry. The
  BioRED pilot's build, criteria, and Stage D audit
  figures (§5.7) are carried in the whitepaper record
  itself; the conformance entries register the
  schema-first, judge-fitting, and relation-typing
  measurements around them.
- The flagship record's adjudication appendix (samples,
  verdicts, replay) ships at [`appendix/`](appendix/);
  the BioRED/CUAD instruments run on the same shipped
  verification cores (`harness/verification/`).
