# Serving: search and the sample API

A built graph is only useful if you can ask it
questions. OntoSmith serves one through three faces of
the same surface — a `search` command (and its library
form), a small sample HTTP API, and an MCP server for
AI callers — plus a minimal demo app over that API,
the test-by-using surface.
A build's graph lives in **Neo4j** (the graph store —
[connections.md](connections.md)), and serving a build
reads it from there.

The same surface also runs over a **file store**, which
reads recorded artifacts directly — the shipped demo,
or a finished batch's on-disk records. That is the
keyless demo / read-back path, and the demo's records
form is **degraded by design**: it carries no
embeddings, so the **vector arm never runs** on it —
lexical and graph arms only, exactly as the `arms`
line of every result says. A batch read-back is not
degraded that way: it loads the build's shipped chunk
and sentence indexes, embeddings included. Either way
the file store is a way to inspect recordings, not a
second place a build can live.

Everything below was executed against the shipped
demo build — *Pride and Prejudice*, the full novel,
built by this pipeline (737 triples) — so you can
reproduce every line before building anything of
your own. The demo ships as frozen records (triples
and their evidence quotes, not the source text), so
its document arm has no novel text to rank; watch
the graph arm carry the answers, and see the batch
read-back below for the full surface over your own
build, text included.

## The 30-second version

```bash
python -m pipeline search "What is Mr. Darcy's income?"
python -m pipeline serve   # http://127.0.0.1:8765
```

No keys, no database, no build step — the example
ships with the package.

## `ontosmith search`

```console
$ python -m pipeline search "What is Mr. Darcy's income?"
Search  "What is Mr. Darcy's income?"
  store   file · …/examples (737 triples, 4 documents)
  arms    lexical · graph
  note    sentence serving unavailable (sentence serving needs an embedder: …); fell back to chunk-level hybrid search
Entities
  - Miss Darcy
  - Mr. Darcy
Documents
  (no chunk matched)
Triples
  - Mr. Darcy — estate_has_income_of → ten thousand per annum  (p 0.96, tier 1, ch-14)
    "A clear ten thousand per annum."
  - Mr. Darcy — annual_income_of → ten thousand a year  (p 0.89, tier 1, ch-56)
    "Ten thousand a year"
```

(The store line counts the worked example's four
corpus files, which share the examples directory, as
its documents; the demo's own text is not shipped —
see the note at the top. The triple list is trimmed
here; the command prints it in full. The note line
is the fallback being named: this keyless run binds
no embedder, so the default sentence-level serving
cannot run and the chunk-level form answers instead —
see below.)

By default the search serves **sentence-level
evidence**. The build's INDEX stage ships a sentence
index — every sentence of every chunk, embedded with
the build's embedder — and three channels draw from
the store: the **direct** channel (the index's closest
sentences to the question), the **graph pool** (the
sentences carrying the top triples' evidence quotes),
and a **beam channel** — a pruned two-hop traversal
from the question's linked entities that adds its
quote sentences as a protected side path. Candidates
are scored in two layers (sentence similarity, then
owning-chunk similarity), merged into one ranked
list, reordered by a local cross-encoder rerank when
the `rerank` extra is installed, and filled as whole
units under a 4,000-token budget. A build that
predates the shipped index gets the same sentence
universe built at query time, with a note saying so.

What a search returns, in order:

- **Entities** — the names in your question, linked
  to graph entities deterministically (exact,
  token-set, then substring match over canonical
  names and registry aliases — no model call).
- **Evidence units** — the budgeted fill, in order:
  each unit is one sentence (or one chunk) with the
  channel that brought it in (`direct` / `graph` /
  `beam` / `chunk`), its score, and its token cost.
  This is what a reader actually receives.
- **Triples** — the graph arm: accepted triples whose
  subject/relation/object overlap the question,
  ranked by overlap weighted with each triple's
  judged probability, each with its evidence quote
  and source document.

When sentence serving cannot run — no embedder can
be bound, or the store serves no sentence index —
the command falls back to chunk-level hybrid search
and the result names the fallback in its notes (the
card above). That form returns **Documents** instead
of evidence units: the hybrid ranking — a lexical
(full-text) arm and, when the store carries
embeddings, a vector arm, fused per document with
reciprocal-rank fusion, the same definitions the
evaluation harness (`harness/`) measures — plus, in
`--json`, **graph documents**: the documents behind
the ranked triples, ordered the same way.

The card names the arms that actually ran
(`arms    lexical · graph` above). An arm that
cannot run — no embeddings in the store, no triple
overlap — is absent from the list and explained in
the JSON's `notes`, never silently implied.

**Language.** Query tokenization is script-aware:
Latin, Hangul, CJK, kana, Cyrillic and other scripts
tokenize on whitespace and punctuation, so a
Korean-only question fires the lexical and graph
arms like an English one. It is whitespace
tokenization, not morphological analysis — a Korean
particle attached to a noun does not match the bare
noun, so questions phrased with bare entity names
match best. On the store side, the full-text
indexes take an analyzer chosen per deployment
(`ONTOSMITH_FULLTEXT_ANALYZER`,
[connections.md](connections.md)); stock Neo4j
bundles no Korean morphological analyzer, and `cjk`
(bigram) is the closest shipped option.

```bash
python -m pipeline search --json "Mr. Darcy danced with"
python -m pipeline search --k 5 --batch my-batch-id "…"
python -m pipeline search --source path/to/build "…"
```

### Which store answers?

One rule set, shared with the API:

| You did this | Store |
|---|---|
| `--store neo4j`, or `NEO4J_URL` is set | Neo4j — the store's one current graph (below) |
| `--store file` | the file store, whatever the environment says |
| `--batch <id>` | with the file store: that batch's artifacts under the state root (read-back). With Neo4j: the build you expect the store to hold — a name to check, not a scope |
| `--source <dir>` | with the file store: a batch directory, or a records directory like the example. With Neo4j: the same expectation check, by directory name |
| nothing | with Neo4j configured: the store's current graph. Otherwise: the shipped example build |

**The Neo4j store holds one graph** — the current
build's, stamped at WRITE with the identity of the
ontology that produced it ([connections.md](connections.md)).
Serving resolves that graph from the stamp: there is
no scope to specify, and no question of which build
you are searching — the store can only hold one.

A
`--batch` or `--source` beside `--store neo4j` names
the build you *expect*; if the store's current graph
is a different build's, serving refuses by name
(build the named one to make its graph the store's,
or drop the name and serve what is there). A store
with no stamped graph — empty, or holding content no
build stamped — refuses the same way, saying which.

Chunk retrieval is pinned to that one graph's build,
so no arm can range over a replaced graph's documents.

The file store reads a batch's routed attestations,
source texts and chunk index, or a records directory
(`triples.jsonl` + `corpus/*.txt`, the example's
shape — chunked deterministically at load).

Entity
identity comes from the build's registry when one is
present next to the source. A store that cannot
answer — a missing source, an unreachable Neo4j —
refuses with a named error; it never falls back to a
different store quietly.

Point it at your own finished build (file read-back;
with `NEO4J_URL` configured the same command checks
the name against the store's current graph):

```bash
python -m pipeline search --batch 2026-10-01T120000Z-ab12cd34 "your question"
```

### The library form

```python
from pipeline.search import hybrid_search, open_store

store = open_store()                    # same selection rules
result = hybrid_search(store, "What is Mr. Darcy's income?", k=10)
result["triples"][0]["relation"]        # 'estate_has_income_of'
result["graph_documents"][0]            # 'ch-14'
```

`hybrid_search` is the chunk-level form — the one the
sample API serves. The CLI's default path is the
library's `sentence_search` (same store contract; an
embedder is required, and a store that serves no
sentence index refuses by name —
`SentenceServingUnavailable` — rather than degrading
silently).

An embedder (the connections layer's embedding seat)
is used only when the store actually carries chunk
embeddings — the hybrid search never spends an
embedding call a store cannot answer from.

## The sample API: `ontosmith serve`

A small HTTP API over the same surface, so a demo
app — or your own frontend — can sit on top. It is
deliberately a *sample*: standard-library HTTP, no
framework dependency, localhost by default.

```console
$ python -m pipeline serve
ontosmith serve: file store (737 triples, 4 documents) at http://127.0.0.1:8765 — /search /entity /neighborhood /triple /health
ontosmith serve: this is the shipped demo graph — Pride and Prejudice, the full text, built by this pipeline. For your own document: ontosmith quickstart <your-file.txt>
ontosmith serve: demo app at http://127.0.0.1:8765/ (assets: …/app)
```

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /health` | — | status + the serving store's description |
| `GET /search` | `q` (required), `k` (1–100, default 10) | the search result above, as JSON |
| `GET /entity` | `id` or `name` | the entity, its aliases, and every triple it takes part in |
| `GET /neighborhood` | `id` or `name`, `hops` (1–4, default 1) | the entity's neighborhood: nodes + evidence-carrying edges |
| `GET /triple` | `index`, or any of `subject` / `relation` / `object` | one triple **with its evidence quote and provenance** |

```bash
$ curl -s http://127.0.0.1:8765/health
{"status": "ok", "store": {"kind": "file", "source": "…/examples", "batch_id": "examples", "entities": 280, "triples": 737, "documents": 4, "chunks": 4, "embeddings": false}}

$ curl -s "http://127.0.0.1:8765/triple?subject=Darcy"
{"source": "api", "position": 33, "triples": 737, "triple": {"subject": "Mr. Darcy", "relation": "danced_with", "object": "Mrs. Hurst"}, "evidence": [{"kind": "quote", "text": "danced only once with Mrs. Hurst"}], "provenance": {"doc_id": "ch-03", "batch_id": "examples", "tier": 1, "tag": "VERIFIED", "p_true": 0.99}, "gaps": []}

$ curl -s "http://127.0.0.1:8765/entity?name=Nobody"   # HTTP 404
{"error": {"code": "entity_not_found", "message": "no entity matches name 'Nobody'"}}
```

Errors are one shape everywhere —
`{"error": {"code", "message"}}` — with named codes:
`missing_parameter` and `invalid_parameter` (400),
`entity_not_found`, `triple_not_found`, `not_found`
(404). The store is chosen by the same flags as the
CLI (`serve --batch …`, `serve --store neo4j`,
`--host` / `--port` to move the bind).

## The demo app: test-drive the graph

`ontosmith serve` also serves a small demo app at
`http://127.0.0.1:8765/` — the test-by-using surface
next to the harness's test-by-data. It is four static
files (`app/` in the repository: plain HTML/CSS/JS,
no build step, no external assets, so it works
offline) over the same API — no second backend,
nothing to configure:

```console
$ curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:8765/
200 text/html; charset=utf-8
$ curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:8765/app/app.js
200 text/javascript; charset=utf-8
```

What it does, and only this:

- **Search box.** The sample API's hybrid search —
  the chunk-level form the CLI falls back to —
  organized the way the API's result is: linked
  entities, ranked documents, evidence-carrying
  triples — with the arms that ran named at the top
  and every skipped-arm note printed verbatim,
  exactly as the API reports them.
- **Triple inspector.** Click any triple and the app
  fetches its inspection report (`/triple`): the
  statement, the evidence quote, the full provenance
  chain, and the report's *gaps* — the accountability
  aspects the record does not carry — named, not
  hidden.
- **Entity report.** Click an entity — a search chip,
  a triple's subject or object, a neighbor — for
  every fact it takes part in, split as-subject /
  as-object.
- **Neighborhood walker.** From an entity report,
  walk the graph one entity at a time over
  `/neighborhood` (1–4 hops): each neighbor is a row
  naming the relation, the tier, and the judged
  probability of the edge that reaches it.

Graph *rendering* is deliberately not in the app —
there is no custom canvas. When the store is Neo4j,
the overview panel links to **Neo4j Browser** on the
same host as the Bolt endpoint (port 7474 next to
7687) with a starter Cypher query; no credentials
ever reach the page (the password is a separate
binding the API never exposes).

An embedded rendered
explorer belongs to the fuller app (v1.1); on the
file store, the walker above is the exploration
path. The app is read-only, like the API under it.

## The MCP surface: `ontosmith mcp`

The graph's consumer is often an AI agent, so the
same serving stack is also exposed over the Model
Context Protocol (stdio transport; the `mcp` extra —
`pip install "ontosmith[mcp]"`). The store binding is
the serve surface's (`--source` / `--batch` /
`--store`); with no source it serves the shipped
example build.

Five read-only tools, structured parameters only —
no query language crosses the boundary:

- `search` — hybrid retrieval: graph-arm triples
  interleaved by rank with document-arm passages.
- `entity` — resolve a name to the canonical entity
  (ambiguity returns candidates, never a silent pick).
- `neighborhood` — one entity's relations as triples.
- `connect` — shortest evidence-backed paths between
  two entities.
- `triple` — audit one triple: full evidence set and
  provenance chain.

Every tool returns one envelope: source documents
hoisted into a `docs` map and referenced by short
ids, items that bind each claim to its evidence
sentences with citation keys (`d1:s3`) the caller can
cite back, a hard token budget whose truncation is
always signalled with a resume cursor, and honest
totals. Conflicting triples are returned flagged on
both sides; derived facts are marked derived. There
is deliberately no `ask` tool — the caller's model
composes the answer from returned evidence, which
keeps the surface honest and model-agnostic.

## What serving does *not* do

- **No generative answers.** Search returns ranked
  documents, linked entities and evidence-carrying
  triples; composing a prose answer from them is the
  caller's job (the evaluation harness's answerer
  stays an evaluation tool, not a product surface).
- **No writes.** Serving is read-only; builds happen
  through the pipeline commands.
- **No auth, no TLS.** The sample API binds
  localhost for local demos. Put a real server in
  front of it before exposing it anywhere.
