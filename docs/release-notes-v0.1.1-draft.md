**Release notes for v0.1.1**

> **Status: published 2026-10-27** as the
> [v0.1.1 GitHub release](https://github.com/Dongbumlee/ontosmith/releases/tag/v0.1.1).
> The body below is the posted text. Wording follows
> the project's placement rule: scores live in the
> whitepaper, front surfaces carry the pointer, not
> the numbers.

---

# OntoSmith 0.1.1 — ontology import and the first-run serving fix

**An ontology-driven knowledge-graph builder.**
0.1.1 adds the third door into a schema — importing an
external ontology — and fixes the first-run serving
gap that a from-scratch verification walk found in
0.1.0. Everything measured in 0.1.0 stands as recorded;
the whitepaper's corrections in this release are
record-keeping, not re-measurement.

*Powered by Jev · MIT · Copyright (c) 2026 Dongbum Lee*

## What ships in 0.1.1

- **Ontology import — the BYO-schema door**
  (`ontosmith ontology import`). An external OWL/RDF
  ontology enters as a mechanically extracted
  inventory (every declared class and property with
  its IRI, label, definition, and parents) plus a
  human-authored scoping spec that accounts for every
  scoped concept exactly once — kept, merged, or
  flattened; role classes converted to relations;
  datatype properties demoted; drops only with stated
  reasons. Import emits an ordinary builder draft and
  a complete scoping record, and from there nothing
  differs: the draft validates, is approved, and
  emits through the same gates as a drafted schema.
- **Pack: `fibo-contracts` 0.1.0** — FIBO's FND
  Agreements module (EDM Council, MIT) imported
  through the builder: 88 scoped concepts → 15 entity
  types + 13 relation types, with its working papers
  in `examples/fibo-import/`. Its record is a
  coverage account over CUAD, not a quality
  measurement; it is emitted `emitted_unmeasured`,
  like every builder pack.
- **Hands-on Track 4** — the import walked end to
  end (fetch → inventory → scope → import → gates →
  build) over a 100-contract CUAD subset, verified
  from a vanilla environment.
- **First-run serving fix (issue #48).** In 0.1.0 a
  build's first `search` against a fresh store
  refused by name — the serving schema and the
  build's chunk projection existed only behind a
  module CLI that a store carried only if someone
  had run it by hand. The INDEX stage now ensures
  the serving schema and projects the batch's
  chunks as part of the stage, and `search` /
  `serve` heal a pre-fix store at open, with no
  provider spend. A store that is down still
  refuses by name.
- **Hands-on instruction detail.** The candidate-path
  choice is stated before any install command;
  Neo4j setup no longer implies Docker (a compact
  example block plus Neo4j's official installation
  guide, with the OntoSmith contract stated as
  ours); checkpoints and the predictable refusals
  are quoted where they occur.
- **Whitepaper corrections.** Reference citations
  corrected to the canonical records (Re-DocRED,
  BioRED); the cover status line updated to the
  shipped release line; §5.7 names the populations
  of its two corrected precisions and declares its
  Tier-1-wide figure record data; §3.3–§3.4
  describe the fourth pack and the import door.

## The quality record

Unchanged in substance from 0.1.0: the full record is
§5 of the project whitepaper (`docs/whitepaper.md`),
with its scope, method, intervals, and failures, and
`python examples/demo/replay_scores.py` recomputes
every headline from the shipped appendix files alone
— no API key, no model calls.

## Try it without a key

```console
$ pip install ontosmith
$ ontosmith profile examples/corpus --family demo
$ ontosmith inspect examples/triples.jsonl
$ python examples/demo/replay_scores.py
```

The hands-on guide (`docs/hands-on.md`) walks four
tracks: the keyless replay, a live build, designing
a schema with the builder, and — new in 0.1.1 — the
import track.

## Known limitations (from the whitepaper, §7)

- Recall is low and enumeration-bound; the shipped
  enumeration default is the measured mitigation, not a
  cure.
- One pre-registered criterion (NONE-loss share) failed
  its bar and stands as FAIL in the record.
- The graph's retrieval advantage is scope-bound: real
  on long documents, measured at exactly zero on short
  abstracts — stated that way wherever it is cited.
- Numbers are stated for the pinned reference profile;
  custom bindings are supported, and their numbers are
  the user's to re-measure.
- Serving in v1 is deliberately minimal; the fuller
  application is later work.

## Links

- Whitepaper: `docs/whitepaper.md`
- Benchmarks index (claim → harness → readout):
  `benchmarks/README.md`
- Connections (seats, reference profile, custom
  bindings): `docs/connections.md`
- Full changelog: `CHANGELOG.md`
