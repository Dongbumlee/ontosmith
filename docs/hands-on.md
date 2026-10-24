# Hands-on — three tracks over your own documents

This guide is organized the way a new user actually
arrives: *I have documents — what do I run to search
them?* Three tracks, all over **your** documents, each
changing one thing:

- **Track 1 · Your documents** — install → connect →
  profile → build → search. The quickstart: minutes to
  a first answer you can inspect.
- **Track 2 · Your documents + an ontology domain pack** — the
  same documents, with a domain pack bound. What the
  pack changes is the lesson.
- **Track 3 · Your documents → your own ontology** — no
  shipped pack fits, so you draft one with the ontology
  builder, approve it, and build under it. This track
  doubles as the pack-authoring guide.

Every track ends at the **same finish line**: the build
served, one question searched, and one fact opened to
its evidence with `inspect`. Stop after any track and
you hold a working graph; continue and you change
exactly one variable.

Worked examples below run on two small corpora whose
outputs were captured from the real tools: the shipped
**Alder & Finch** example (`examples/corpus`, four short
documents written for this project) and the fetched
**Re-DocRED** subset of the appendix track. Substitute
your own folder of documents anywhere a corpus path
appears.

## Prerequisites

**Requirements**

- **Linux** — developed and tested on Linux. Windows
  users: run under **WSL2** (native Windows is not a
  tested platform).
- **Python ≥ 3.11**.
- An **LLM connection** — only for the steps that spend
  (each track marks them **LIVE**). Profiling, batch
  creation, validation, and inspection all run with no
  key at all; connect when you are ready to build.
- **Neo4j** (5.x) running — for the **LIVE** builds
  only: it is the graph store the build writes into.
  Install and the connection variables are in
  [connections.md](connections.md) (Neo4j — the graph
  store); with a local install, setting
  `NEO4J_PASSWORD` is the whole configuration. A run
  checks the store before it spends and refuses by
  name if it is not serving; the keyless steps never
  touch it.

Install from a clone of this repository (at launch this
becomes `pip install ontosmith` on PyPI):

```console
$ pip install .
```

The extraction extra — GLiNER and spaCy — is required
by **one candidate path only**:

- `--candidate-source pipeline` (the product default
  outside these tracks): NER runs locally on GLiNER,
  so the extra is required.
- `--candidate-source llm` (the path every track below
  uses): proposals come from the proposer seat — GLiNER
  is never loaded (the build log says so: `NER: GLiNER
  skipped (candidate_source=llm …)`) and spaCy's
  structural pass is simply absent. The extra is not
  needed; installing it changes nothing on this path.

```console
$ pip install ".[nlp]"        # pipeline candidate path only
$ python -m spacy download en_core_web_sm
```

(The spaCy model is a separate download; without it a
pipeline-path build logs the miss and continues
without the structural pass. The GLiNER model
downloads itself on first load.)

Size warning: the extra pulls the CUDA torch stack —
several gigabytes of wheels, and slow on a constrained
connection. If the torch download is a problem, stay
on the tracks' `llm` path and skip the extra entirely.

## Two words first: profile, pack

You will meet these in the first three commands, and
they are not the same thing.

A **profile** is a description of **your corpus**: a
deterministic census — document inventory, conventions,
the decision points a human should settle — produced by
`ontosmith profile` with no model calls. The same
documents always yield the same profile hash, which is
what makes it a record the rest of the pipeline can pin
to.

A **pack** is a **domain's ontology as data**: the type
inventory, definitions, boundary examples, and operating
declarations a build runs under — versioned, pinned by
sha256, and carrying the conformance record of what its
content measured. Profiles describe a corpus; packs
carry a domain. A batch binds one of each. The full
pack contract is [PACK_DESIGN.md](PACK_DESIGN.md).

## Example document sets

The tracks work on any folder of plain-text documents —
yours. If you want the exact corpora the packs were
measured on, fetch them from their sources (this
repository bundles no dataset text):

| Order | Set | Pack | Source | License / terms |
|---|---|---|---|---|
| 1 | **Open Domain** — Re-DocRED dev subset (general documents) | `open-domain` (family `redocred`) | [Re-DocRED dataset repository](https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/dev_revised.json) — fetched by `examples/demo/fetch_demo.py`, which writes a `SOURCE.md` beside the documents | Dataset: MIT. Document text derives from Wikipedia and remains under upstream **CC BY-SA** — fetched for use, never bundled here |
| 2 | **Legal Contract** — CUAD v1 (510 commercial contracts, 41 clause categories) | `legal-contract` (family `cuad`) | [The Atticus Project — CUAD](https://github.com/TheAtticusProject/cuad) | **CC BY 4.0** (source contracts are SEC EDGAR filings, public domain). Download the v1 archive from the project and unpack it beside your work |
| 3 | **Bio Medical** — BioRED (600 PubMed abstracts, document-level relations) | `bio-medical` (family `biored`) | [NLM — BioRED download](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) | Freely available from NLM for research use; the abstracts are PubMed records. Fetch from NLM under its terms — do not re-host |

## Track 1 — Your documents

### 1. Profile the corpus (no key)

```console
$ ontosmith profile examples/corpus --family demo
Corpus profile
  family          demo
  status          COMPLETE
  documents       4
  claims          6
  decision points 0
  profile hash    sha256:7ae51d31fa9120ec
  spend           $0.0000
  written         profiles/demo (profile.json, setup_config.json, profile.md)
```

The `--family` name is yours to choose for your own
documents — it names the profile, and later binds the
pack whose family matches it. Read
`profiles/<family>/profile.md`: the claims are
scanner-verified statements about your corpus, and any
**decision points** it lists are the questions to settle
before a build spends anything.

### 2. Create the batch (no key)

```console
$ ontosmith batch create --source examples/corpus --collection demo \
    --docs "examples/corpus/*.txt" \
    --profile profiles/demo/profile.json \
    --candidate-source llm --max-cost-usd 2
profile: demo sha256:7ae51d31fa9120ec
binding: reference
created batch demo-4docs with 4 docs
run: python -m pipeline batch run demo-4docs
```

Three lines deserve a word:

- **No `--pack`** — an undifferentiated corpus runs the
  **Core baseline**, which applies by default. `--pack`
  names a *domain* pack (Track 2), and creation refuses
  a pack whose family does not match the profile rather
  than degrading silently.
- **`binding: reference`** — this is the good state,
  not a warning: it stamps that the run's connection
  resolved exactly to the shipped reference preset (the
  measured bindings). Re-point any seat (a seat is one
  capability slot of the connection — proposer, judge,
  escalation, drafting, embedder) and the stamp
  becomes `custom`, and the record says what ran
  ([connections.md](connections.md)).
- **`--max-cost-usd 2`** — the stop-line, frozen into
  the batch. Nothing has been spent; creation is
  keyless.

### 3. Build — LIVE (the one step that spends)

First, the key. The export names **the preset's** key
variable — because the reference preset is an OpenRouter
binding — not a product requirement:

```console
$ export OPENROUTER_API_KEY=...   # the reference preset's key variable, env only
$ ontosmith batch run demo-4docs
```

OntoSmith needs *a* connection, not *this* one.
Bindings are declared with **`ontosmith configure`**:
it asks per seat — base URL, model, key variable name,
dialect — starting from the preset's values, checks the
result with zero-spend probes, and writes
`ontosmith-bindings.env` — the full per-seat
declaration, as per-seat `ONTOSMITH_<SEAT>_…`
variables. Load the file before running:

```console
$ set -a; . ./ontosmith-bindings.env; set +a
```

and the same step runs under your custom binding.
Prefer files to prompts? `ontosmith configure
--emit-template` prints the shipped template
([bindings-template.env](bindings-template.env)); fill
a copy in and validate it with `ontosmith configure
--from-file <file>` — both doors write the identical
file. The file holds nothing but those variables:
setting them directly in your environment is the same
contract without the file — the file is what
`configure` writes, and you can write it by hand. The
full contract is in [connections.md](connections.md).

Second, the store. A build writes into **Neo4j** —
start yours before running (install, and the
connection variables `NEO4J_URL` / `NEO4J_USER` /
`NEO4J_PASSWORD`, are in
[connections.md](connections.md#neo4j--the-graph-store-required-for-builds)).
With a local default install and its password set,
exporting `NEO4J_PASSWORD` is all the configuration
there is. The run probes the store before the first
stage: unreachable, or rejecting the credentials, it
refuses with a named error and spends nothing — it
never runs the paid stages and dies at WRITE.

The run
executes the staged pipeline (ingest → NER/roster →
candidates → gates → route → write → index → verify)
under the batch's
frozen configuration; the ledger enforces the stop-line
— a unit that would cross the cap is never started, and
the run resumes from its ledger instead of
re-spending. Watch it with `ontosmith batch status
demo-4docs`. A corpus this size — four documents —
prices in cents of model spend (the README's $1–2
figure for the same build counts the fixed overhead
around it);
`ontosmith verify project` counts and prices a larger
run before it starts (appendix track).

### 4. The finish line: serve, search, inspect one fact

```console
$ ontosmith serve --batch demo-4docs
ontosmith serve: file store (… triples, 4 documents) at http://127.0.0.1:8765 — /search /entity /neighborhood /triple /health
ontosmith serve: demo app at http://127.0.0.1:8765/

$ ontosmith search --batch demo-4docs "Who founded Alder & Finch?"
$ ontosmith inspect batches/demo-4docs
```

`search` answers with ranked documents, linked entities,
and evidence-carrying triples — no generated prose: what
you see is what the graph holds. The demo app at `/` is
the same surface in a browser.

`inspect` is pointed at the batch **directory** — that
is the form with the full view (resolved names, quoted
evidence, provenance). Pointed at a bare records file
(`written_triples.jsonl`) it degrades: `ent:` IDs,
"(none recorded)" evidence, and a Gaps section, because
a file alone carries no registry context. One Gaps line
can appear even in the directory form —
`no batch/status provenance recorded`. Read it
precisely: it says the *triple record itself* carries
no batch/status fields (batch provenance lives in the
batch manifest and status, not on the triple); it is
not a missing-evidence warning, and the evidence and
judgment provenance printed above it are unaffected.
The full viewer contract is in
[cli-reference.md](cli-reference.md).

**Held pairs, settled.** A build can hold
entity-resolution pairs it would not merge on its own
— the WRITE metrics' `er_review_pairs` count names
how many. Held is a decision waiting, not a dead end:
list the pairs with the evidence the gate saw, settle
each one as a recorded human decision, and apply the
merges to the graph:

```console
$ ontosmith batch er-review demo-4docs
$ ontosmith batch er-decide demo-4docs --pair 1 --merge
$ ontosmith batch er-apply demo-4docs
```

`--keep-separate` is the other settlement: the pair
stays two entities, the decision is recorded, and
later runs never re-hold it.

## Track 2 — Your documents + an ontology domain pack

Same documents, one change: bind the pack whose family
matches the profile. With the Open Domain example set
(appendix track fetches it):

```console
$ ontosmith profile demo-data --family redocred
...
  profile hash    sha256:aceac6fb15f59e38
...

$ ontosmith batch create --source demo --collection redocred-dev-subset \
    --docs "demo-data/doc_*.txt" --profile profiles/redocred/profile.json \
    --pack open-domain --candidate-source llm --max-cost-usd 2
profile: redocred sha256:aceac6fb15f59e38
pack: open-domain 0.4.0 (payload sha256:ee9a1960f01d44ae…)
binding: reference
setup_config: resolved (core baseline + open-domain, 34 keys) -> setup_config.resolved.json
created batch redocred-dev-subset-40docs with 40 docs
```

What the pack bought, mechanically: its payload was
**pin-verified** against the manifest, **overlaid per
key** on the Core baseline (the pack governs every key
it declares; undeclared keys keep the core defaults —
the resolution order is the contract,
[PACK_DESIGN.md](PACK_DESIGN.md) §1.3), and the resolved
configuration was **frozen into the batch** as
`setup_config.resolved.json`. What it bought in content
is the domain's schema: the type inventory,
definitions, boundary examples, and fitted declarations
the proposer and gates now run under — itemized in the
pack's contents index (`packs/open-domain/pack.json`)
and priced in its conformance record.

Run it and take the same finish line as Track 1 (build
→ serve → search → inspect one fact). The honest
comparison is Track 1 vs Track 2 at that finish line:
same documents, same seats, one variable — the pack.

The shipped pairings:

| Your documents | Profile family | Pack |
|---|---|---|
| General documents (Re-DocRED set) | `redocred` | `open-domain` |
| Contracts (CUAD set) | `cuad` | `legal-contract` |
| Biomedical abstracts (BioRED set) | `biored` | `bio-medical` |
| Anything else | your name | Core baseline (Track 1), or your own pack (Track 3) |

## Track 3 — Your documents → your own ontology

No shipped pack fits your domain? Draft one. This is
the pack-authoring loop — `draft → validate → brief →
approve → emit` — walked on the Alder & Finch profile
from Track 1. Everything here is keyless until the
optional `--llm` drafting variant.

The loop consumes a **profile** of your corpus. If you
skipped Track 1, make one first — it is a single
keyless command, and every step below reads the file
it writes:

```console
$ ontosmith profile <your corpus dir> --family <family>
# writes profiles/<family>/profile.json
```

(What a profile is, and how it differs from a pack:
"Two words first", above; the full walk is Track 1 §1.)

### 1. Draft

A **brief** tells the builder what the domain is. It is
a small JSON file — purpose plus the type names you
expect:

```console
$ cat brief.json
{"text": "A small ceramics studio: its people, products, and workshops.",
 "purpose": "Track who founded and works at the studio, what it makes, and what it offers.",
 "entity_types": ["Organization", "Person", "Product", "Event"],
 "relation_types": ["founded by", "works at", "makes", "offers"]}

$ ontosmith ontology draft --profile profiles/demo/profile.json \
    --brief brief.json --pack-id demo-ceramics \
    --display-name "Demo Ceramics" --out drafts/demo-ceramics.json
draft written: drafts/demo-ceramics.json (status DRAFT_NOT_APPROVED, 4 entity types, 4 relation types, schema hash sha256:62ea9e9067a5…)
validation findings (20) — the decision-point list approval must settle:
  [MISSING_DEFINITION] entity_inventory[0].definition: entity type 'Organization' has no definition
  …
```

The brief's type names land as a skeleton — names
filled in, definitions deliberately empty, and the 20
findings are exactly the work of step 2. (Omit the
brief and the skeleton is empty instead: two
`EMPTY_…_INVENTORY` findings and no types at all. With
`--llm`, the drafting seat writes a fuller draft. The
drafting seat **ships unbound** — bind a model first
with `ontosmith configure` (the declaration it writes
is `ONTOSMITH_DRAFTING_MODEL`). Its output is still
only a draft: drafting never judges, and nothing emits
without the approval below.)

### 2. Validate — the findings are the work list

```console
$ ontosmith ontology validate drafts/demo-ceramics.json
validation findings (20):
  [MISSING_DEFINITION] entity type 'Organization' has no definition
  [MISSING_BOUNDARY_EXAMPLES] entity type 'Organization' has no boundary examples (form_examples / not_entities) — boundary material is what separates it from its neighbours
  [BAD_ENDPOINT_SIGNATURE] relation type 'founded by' has no endpoint signature
  …
```

Every type needs three things before the pack may
exist: a **definition**, **boundary material** (examples
of what it is, and near-misses that are *not* it), and —
for relations — an **endpoint signature** (which types
it runs between). Edit them into the draft JSON
(`schema_front`), and re-validate until:

(The draft document's full field list — every key,
its type, and which fields are yours to edit — is
[PACK_DESIGN.md](PACK_DESIGN.md) §4, including the
two operating declarations a draft may carry:
`merge_prohibited_type_pairs` and `theme_attest_chunk`
(the second one matters at build time — §5 below).)

```console
$ ontosmith ontology validate drafts/demo-ceramics.json
validation: clean
```

### 3. Approve — a human decision, bound to the bytes

```console
$ ontosmith ontology brief drafts/demo-ceramics.json   # the approval brief: inventory + decision points
$ ontosmith ontology approve drafts/demo-ceramics.json --approver "A. Reader"
approval recorded: drafts/approval_2026-10-22.json
  schema hash sha256:eb92888135464743…
```

Approval binds to the schema hash: edit the schema
afterwards and the approval stale-dates and emission
refuses. Approving a draft that fails validation is
refused outright — the findings from step 2 are the
decision points the approver is settling.

### 4. Emit — the pack exists

```console
$ ontosmith ontology emit drafts/demo-ceramics.json \
    --approval drafts/approval_2026-10-22.json --out-root .
pack emitted: ./packs/demo-ceramics
  payload pin sha256:0e521c59d46b0db6…
```

```text
packs/demo-ceramics/
  pack.json                 manifest: identity, pin, contents index
  conformance.json          the record (starts as one entry: emitted_unmeasured)
  profile/setup_config.json the payload the pin binds
```

Emission never overwrites: emitting into a tree that
already holds this pack is refused by name. A changed
draft — a fixed definition, a newly declared operating
lever (§5) — is a **new version**, not an edit: approve
the changed draft and emit it to a fresh `--out-root`
with the new `--pack-version` (e.g. `0.2.0`), then
bind the batch to that tree's pack directory.

The first conformance entry states the honest position:
`emitted_unmeasured` — the artifact exists; no
measurement has priced it. Content earns its place
later, by the conformance measurement of
[PACK_DESIGN.md](PACK_DESIGN.md) §2 — per item, held-out
gold, pre-registered gates — exactly the discipline the
shipped packs' records show.

### 5. Build under your pack — same finish line

```console
$ ontosmith batch create --source examples/corpus --collection demo-own \
    --docs "examples/corpus/*.txt" --profile profiles/demo/profile.json \
    --pack ./packs/demo-ceramics --candidate-source llm --max-cost-usd 2
profile: demo sha256:7ae51d31fa9120ec
pack: demo-ceramics 0.1.0 (payload sha256:0e521c59d46b0db6…)
binding: reference
setup_config: resolved (core baseline + demo-ceramics, 25 keys) -> setup_config.resolved.json
created batch demo-own-4docs with 4 docs
```

`--pack` also accepts a pack directory, as here. Build
(**LIVE**, same key step as Track 1 — and the same
Neo4j prerequisite: the store must be running, Track 1
§3), then serve, search, and inspect one fact — the
same finish line, now under an ontology that is yours.

**Candidate guards — read this before your first build
under a pack you authored.** Between proposal and
judgment, every candidate triple passes deterministic
guards, and a blocked candidate never reaches the
judge. The guards are: `g1_quote` (the evidence quote
must occur in the document), `l0_attest` (both
endpoints must occur in the document), `l0_role_word`
(an endpoint that is only a role noun — "the manager"
— is not an entity), `theme_missing` (the relation is
nothing but prepositions), and `theme_presence`. Blocks
are never silent: the CANDIDATES stage's completion
line names the blocked count by reason, the batch
metrics carry the same counts, and every blocked item
is written to `candidates_blocked.jsonl` in the batch
directory with its reason and its evidence quote.

`theme_presence` deserves a paragraph of its own,
because it can block a relation **wholesale**. For a
relation whose surface form ends in a preposition —
*works at*, *diagnosed with*, *located in* — the
guard requires the object to be attested inside the
proposal's shortest evidence quote, not merely
somewhere in the document. The reason is fabrication
control: without it, a model can pair any subject with
any object the document mentions, anchored to an
unrelated quote fragment. But natural text often names
the object across the passage rather than inside the
minimal quote — and then every proposal for that
relation blocks, and the relation is simply absent
from the graph.

If that is your corpus's shape, the pack author has a
lever: declare **`theme_attest_chunk`: true** on the
schema draft (before approval — the declaration rides
the emitted payload and is bound by the approval hash,
like `merge_prohibited_type_pairs`). The guard then
also accepts the object attested anywhere in the
proposal's chunk. It is deliberately an opt-in: chunk
attestation is weaker than quote attestation, so the
default stays quote-only, and a pack that never
declares it behaves exactly as before. The field is
defined in [PACK_DESIGN.md](PACK_DESIGN.md) §4.

**Held pairs, under your pack too.** A build under
your pack can hold entity-resolution pairs it would
not merge on its own — the WRITE output's
`er_review_pairs` count names how many, and the WRITE
log line names the remedy. The settlement flow is
Track 1 §4's *Held pairs, settled*
(`batch er-review` → `er-decide` → `er-apply`); it
works unchanged under a pack you authored.

## Appendix track — reproduce the reference build

The three tracks above are how you use OntoSmith. This
track is how you **check its published record**: the
flagship build's whole loop — profile → counted
projection → gold audit → capped live run → corrected
scoring — replayed and re-run against shipped artifacts
and a fetched dataset subset. Steps A1–A5 spend nothing
and need no key.

### A1. Profile the smoke corpus

Three hand-sized documents ship in `examples/demo/smoke/`:

```console
$ ontosmith profile examples/demo/smoke --family smoke
Corpus profile
  family          smoke
  status          COMPLETE
  documents       3
  claims          6
  decision points 0
  profile hash    sha256:560f304c6651e3a6
  spend           $0.0000
  written         profiles/smoke (profile.json, setup_config.json, profile.md)
```

Check yourself: the same corpus always yields the same
profile hash. That determinism is what makes a profile
a *record* rather than a log.

### A2. Inspect a triple

The smoke set's build output ships as a recorded
artifact, so inspection runs offline:

```console
$ ontosmith inspect examples/demo/smoke/triples.jsonl
Triple  (1 of 3 in examples/demo/smoke/triples.jsonl)
  subject   North Loop Cycles
  relation  founded by
  object    Dana Whitfield
Evidence
  [quote] North Loop Cycles is a community bicycle workshop founded in 2016 by Dana Whitfield and Rob Okafor.
Provenance
  doc_id               north-loop-history
  tier                 1
  p_true               0.97
```

This is the records-file form; the batch-directory form
of Track 1 §4 is the richer view. Try `--object
"Priya Raman"` and `--json`.

### A3. Price a build before spending anything

A counted projection prices counted units at
caller-supplied rates and refuses to bless a cell whose
governing scenario crosses its stop-line. This is the
flagship build's own counted input — 26,938 candidate
pairs at its realized per-pair rate — against the
stop-line it actually ran under:

```console
$ ontosmith verify project --counts candidate_pairs=26938 \
    --unit-cost candidate_pairs=0.0006683 --cap 21 --stop-line 21
{
 "scenarios": {
  "base": 18.002665
 },
 "governing_usd": 18.002665,
 "cap_usd": 21.0,
 "stop_line_usd": 21.0,
 "verdict": "within stop-line — the cell may start"
}
```

In a live batch the same discipline is enforced by the
run's spend ledger: a unit that would cross the cap is
never started, and the run resumes from its ledger
instead of re-spending.

### A4. Audit the gold; score corrected

Benchmarks' gold is incomplete, so a raw score grades
your build against a partially wrong answer key. The
audit measures how incomplete: blind samples of your
unmatched predictions (Direction A) and of the gold
itself (Direction B), judged without gold status, with
verbatim quote verification and embedded controls —
then a stratified correction with intervals. The
flagship record ([whitepaper](whitepaper.md) §5; in-tree
pointer [`benchmarks/redocred.md`](../benchmarks/redocred.md))
is the worked instance, and its samples + verdicts ship
in [`benchmarks/appendix/`](../benchmarks/appendix/).

### A5. Replay the corrected scoring

Recompute the flagship's headlines from the appendix
alone, through the shipped estimator cores:

```console
$ python examples/demo/replay_scores.py
Corrected-scoring reproduction — from benchmarks/appendix/
Ruled adjudication contract (the record's headlines, whitepaper §5):
  Direction A support estimate  p_A = 0.7919 [0.7472, 0.8305]
  Direction B gold support       g_B = 0.5244 [0.4754, 0.573]
  Blind controls                 37/40 supported
  Quote-verification failures    11
  Corrected precision  FULL-500  0.911 [0.8919, 0.9275]
  Corrected precision  CONF-440  0.9121 [0.8897, 0.931]
  Ceiling-adj. recall  FULL-500  0.2028 [0.1856, 0.2237]
  Ceiling-adj. recall  CONF-440  0.2036 [0.1852, 0.2262]
Original adjudication protocol (provenance — the estimate as first published):
  Direction A support estimate  p_A = 0.945 [0.9161, 0.9643]
  Direction B gold support       g_B = 0.841 [0.8008, 0.8743]
  Blind controls                 40/40 supported
  Quote-verification failures    32
  Corrected precision  FULL-500  0.9765 [0.9641, 0.9847]
  Corrected precision  CONF-440  0.9753 [0.9603, 0.9849]
  Ceiling-adj. recall  FULL-500  0.1265 [0.1216, 0.1328]
  Ceiling-adj. recall  CONF-440  0.1263 [0.1212, 0.1332]
All headlines reproduce the published record (whitepaper §5), both verdict sets.
```

The script exits non-zero if any reproduced value leaves
the published figure — this is the failure-anatomy lesson
in executable form. The figures themselves are the
whitepaper's §5 record (the original protocol's estimate
stands there as provenance, §5.4); the record's anatomy
(§Failure anatomy; chart shipped as the whitepaper asset
`benchmarks/assets/anatomy.png`) shows where the recall
went.

On your own build, the same tooling runs as
`ontosmith verify audit-sets --predicted <triples.jsonl>
--gold <gold.jsonl> --out-dir <dir>` (builds the blind
A/B/C judgment sets) and `ontosmith verify report
--predicted <triples.jsonl> --gold <gold.jsonl>` (raw
scores, corrected estimates, and the failure anatomy as
one report card).

### A6. Fetch the demo corpus

One command pulls a Re-DocRED dev subset from the
dataset's own repository and writes it as plain text
beside your work, with a `SOURCE.md` carrying the
attribution and terms (dataset: MIT; underlying text:
Wikipedia, CC BY-SA):

```console
$ python examples/demo/fetch_demo.py --out demo-data --docs 40
...
Wrote 40 documents (48,011 bytes) to demo-data/
```

`--dry-run` prints the source, terms, and the expected
cost basis without downloading. First 40 documents in
file order — deterministic, so your run and this
guide's agree. This is the corpus Track 2 profiles.

### A7. Profile it, create the batch

```console
$ ontosmith profile demo-data --family redocred
Corpus profile
  family          redocred
  status          COMPLETE
  documents       40
  claims          6
  decision points 0
  profile hash    sha256:aceac6fb15f59e38
  spend           $0.0000
  written         profiles/redocred (profile.json, setup_config.json, profile.md)
```

The create is Track 2's command, verbatim — same
command, same output (profile line, pin-verified pack
line, `created batch redocred-dev-subset-40docs with
40 docs`), closing with the run line:

```console
run: python -m pipeline batch run redocred-dev-subset-40docs
```

Creation froze the executed configuration into the
batch (core baseline + pack overlay, pin-verified) and
stamped the $2 stop-line into it. Nothing has been
spent.

> **Note — creating under a custom binding.** This
> appendix runs under the reference preset, whose
> provider reports cost, so the $2 stop-line can
> meter every seat. Under a custom binding whose
> seats cannot meter cost (cost basis
> `not-provided` — see
> [connections.md](connections.md)), the same create
> command **refuses by name**: a USD cap cannot guard
> spend it cannot meter. Declare per-seat rates for
> estimated costs, or create with `--no-cost-cap` to
> run uncapped explicitly. Bindings themselves are
> declared with `ontosmith configure`.

### A8. Build — LIVE (reference preset)

The export names the preset's key variable — the
reference preset is an OpenRouter binding, so its key
variable is the one below; the product needs a
connection, not this provider
([connections.md](connections.md)):

```console
$ export OPENROUTER_API_KEY=...        # the reference preset's key variable, env only
$ ontosmith batch run redocred-dev-subset-40docs
```

This is the one step that spends. It runs the staged
pipeline (ingest → NER/roster → candidates → gates →
route → write → index → verify) under the reference
preset. Bind a different
endpoint or model set — declared with `ontosmith
configure` and loaded from the `ontosmith-bindings.env`
it writes, as in Track 1 §3 — and the same step runs
under your custom binding instead. The flagship's realized rate
puts 40 documents at roughly $1.49, and the batch's $2
cap is ledger-enforced — the run stops at the cap rather
than crossing it, and resumes with `ontosmith batch run
redocred-dev-subset-40docs` after a top-up decision,
never by silently re-spending. Watch it with
`ontosmith batch status redocred-dev-subset-40docs`.

One recorded-but-unwired switch, stated plainly:
`--frontier-escalation` (and its band bounds) is written
into the batch config at creation, but **no v1 pipeline
stage consumes it** — review-band items route to the
Tier-2 review queue, and frontier adjudication +
promotion live in the verification harness, not the
product (whitepaper §3.2). Setting it changes nothing
in a v1 run; it is named here so the config record is
never mistaken for a working switch.

### A9. Score and inspect your build

```console
$ ontosmith verify score --predicted batches/redocred-dev-subset-40docs/written_triples.jsonl \
    --gold <your gold file>
$ ontosmith verify report --predicted batches/redocred-dev-subset-40docs/written_triples.jsonl \
    --gold <your gold file>
$ ontosmith inspect batches/redocred-dev-subset-40docs
```

`verify report` ends in the report card: raw scores,
corrected estimates with intervals (once you have run
the audit sets of step A4 over your own output), and
the failure anatomy of *your* build — the same readout
this project's own flagship record publishes, produced
by the same code. `inspect` here is the batch-directory
form — the rich view of Track 1 §4.

## Where to go next

- The method and the evidence: [`README.md`](../README.md)
  and the quality record ([whitepaper](whitepaper.md) §5; index:
  [`benchmarks/README.md`](../benchmarks/README.md)).
- Your own provider, your own models:
  [`connections.md`](connections.md) — seats, recommended
  models, and the custom-binding contract.
- The pack contract — manifest, pins, candidates,
  conformance: [`PACK_DESIGN.md`](PACK_DESIGN.md).
- The worked example, slowly: [`examples/`](../examples/).
- Serving in depth: [`serving.md`](serving.md).
