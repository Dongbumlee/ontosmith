**DRAFT — release notes for v0.1.0 (NOT POSTED)**

> **Status: draft.** This is the prepared body for the
> v0.1.0 GitHub release. It has not been posted anywhere.
> Publication — public repository creation, the release
> itself, and the PyPI upload — is the launch checkpoint's
> decision, taken separately. Wording follows the
> project's placement rule: scores live in the whitepaper,
> front surfaces carry the pointer, not the numbers.

---

# OntoSmith 0.1.0 — initial release

**An ontology-driven knowledge-graph builder.** Design
the ontology first — entity and relation types,
definitions, boundary examples — get it approved, and
build the graph *under* it: a pipeline that separates
proposing from judging at every step, where models
propose candidates and deterministic gates plus a fitted
decision model (**Jev**, a pinned component inside
OntoSmith) decide what is true enough to write. Every
written triple carries its evidence span and provenance,
and every build prices itself before it spends.

*Powered by Jev · MIT · Copyright (c) 2026 Dongbum Lee*

## What ships

- **The batch pipeline** — eight stages (ingest →
  roster → candidates → gates → route → write → index →
  verify) under a versioned pack, with per-stage state
  persisted under the batch: a failed run resumes where
  it stopped and never re-spends. A ledger-enforced
  stop-line caps every run.
- **Three domain packs** — Open Domain, Bio Medical,
  Legal Contract — over the **Core baseline** (the Core
  is the baseline every pack extends, not a domain pack
  itself) — each a pinned
  manifest plus payload (ontology, extraction framing,
  gate and entity-resolution defaults) and a conformance
  record of the measurements that adopted it, rejections
  included. Domain content lives only in packs, never in
  code branches.
- **The corpus profiler** (`ontosmith profile`) — a
  deterministic census of your corpus (no LLM) whose
  claims are scanner-verified and whose decision points
  are a human's to approve.
- **The ontology builder** (`ontosmith ontology`) —
  draft → validate → approve → emit, with fail-closed
  validation and human approval bound to the schema
  hash; an emitted pack is marked `emitted_unmeasured`
  until a registered measurement adopts it.
- **The verification tooling** (`ontosmith verify`) —
  raw *and* audit-corrected precision, ceiling-adjusted
  recall, a failure-anatomy census that assigns every
  gold item exactly one fate, judge-fitting cells with
  pre-registered gates, and counted cost projections.
- **The connection layer** — provider-neutral per-seat
  bindings (any OpenAI-compatible endpoint), declared
  with `ontosmith configure` (an interactive wizard, or
  its fill-in template twin); the measured
  reference profile ships as the default preset, and run
  records state which binding produced them. Keys are
  environment values, never settings.
- **Serving** — hybrid retrieval (vector + lexical +
  graph) via `ontosmith search` and a sample API
  (`ontosmith serve`: search / entity / neighborhood /
  triple-with-evidence) over the graph in Neo4j — the
  store holds one graph, the current build's, and
  serving answers from it with no scope to specify —
  or over the keyless file read-back of a build's
  recorded artifacts (lexical + graph arms; the demo
  path) — plus a minimal demo app served at `/`
  (search box, triple/evidence inspector, entity report,
  click-to-walk neighborhood view).

## The quality record

OntoSmith's construction quality is measured, and the
measurement is published — with its scope, its method,
its intervals, and its failures (one pre-registered
criterion failed and is recorded as failed). The full
record is §5 of the project whitepaper
(`docs/whitepaper.md`). Corrected numbers rest on
**machine adjudication** under a pre-registered protocol;
they are machine judgments, not a human audit, and not a
corrected gold dataset. The adjudication samples and
verdicts ship in `benchmarks/appendix/`, and
`python examples/demo/replay_scores.py` recomputes every
headline of the record from those files alone — no API
key, no model calls, exit non-zero on drift.

## Try it without a key

```console
$ pip install ontosmith
$ ontosmith profile examples/corpus --family demo
$ ontosmith inspect examples/triples.jsonl
$ python examples/demo/replay_scores.py
```

The hands-on guide (`docs/hands-on.md`) walks the whole
loop twice: the keyless replay track above, and a live build
track (fetch → profile → counted projection → capped
run → scoring) at roughly $1–2 for the demo subset.

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
