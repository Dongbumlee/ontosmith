"""LLM proposer — first-class candidate source for CANDIDATES (E4).

Integration of the two proven standalone harnesses into the batch
pipeline (gap G-03): cuad_llm_triples.py (free triples) and
cuad_clause_extract.py (clause slots). The recipes are carried over
unchanged — 20K-char chunks (19.5K step), at most 10 triples per
chunk, shortest-verbatim-quote instruction, Haiku 5.5 as the
default proposer model — because Step 1A measured THIS recipe's
output through the Jev gates (Tier 1 yields: legal triples 97.4%,
clause slots 91.5%, novels triples 95.0%).

The proposer only PROPOSES. Every kept item becomes a candidate in
the existing queue -> GATES (Jev) -> ROUTE flow; no score is ever
assigned here (the standalone harnesses' hard-coded p_true 0.9 is
exactly what G-03 abolished). Between proposal and candidacy sit
the deterministic span/attestation guards (§5, re-scoped to LLM
proposals for G-04/G-05):

  g1_quote       the quote (whitespace-normalized) is not a
                 verbatim substring of the document text.
  l0_attest      L0 document-level attestation: subject or object
                 surface is not attested anywhere in the document
                 (the §5 attestation window is the document; the
                 level actually reached — quote / chunk / document —
                 is recorded on the candidate as `attest`).
  l0_role_word   an endpoint is a bare role word (the corpus
                 profile's rule: role nouns are not identities);
                 only when the profile declares role-word rules.
  theme_missing  the relation is a bare prepositional phrase —
                 no predicate, hence no theme.
  theme_presence a verb+preposition relation whose object is not
                 attested in the quote itself — the C2 substitution
                 pattern (a prepositional object standing in the
                 evidence span for an unstated theme).

Blocked items land in candidates_blocked.jsonl with the reason code,
the same sidecar the Arm B structural constraints use. Slot fills
pass the span gate (g1_quote) and per-(doc, category, value)
dedupe; their subject is the document's own Document Name fill
(fallback: source file stem), mirroring cuad_clause_tograph.py.

Profile wiring: the proposer takes an optional
`guidance` string rendered from the corpus profile's setup_config
(pipeline/profile_config.derive_guidance — the channel). No
guidance -> the prompt is byte-identical to the recipe above. With
guidance, the model is asked for typed endpoints (subject_type /
object_type); guard_triples validates the tags against
the profile's endpoint-type vocabulary.

Domain framing: the prompt templates are domain-neutral;
the framing line (and the slot rules block) that specialize them
to a family are profile data threaded through the `framing` /
`rules` parameters (pipeline/profile_config.proposer_framing /
slot_framing / slot_rules). The standalone harnesses' contract
text survives as the CUAD profile's declared framing — the
prompt CUAD builds is byte-identical to the recipe above.

Mention layer (issue #54, profile-gated by the
`mention_layer` setup_config key; absent = off and every
behavior below is inert): entities register and link by
noun surface forms, so a fact the text states through a
pronoun, a definite description, or a sentence other than
the one naming the entity has no citable path to its
endpoint — the proposer either resolves it silently or
skips it, and a dropped candidate records only a score.
With the layer on, the proposer additionally emits, per
endpoint, the mention span as it appears in the quote
(verbatim), its type (named / pronominal / descriptive),
and — when the mention is not the entity's name — a
resolution proposal: the antecedent span, verbatim from
the document, that names the entity. guard_triples
verifies every DECLARED span with the same verbatim
discipline as the quote itself: a mention that is
present but unverifiable blocks the item by name
(mention_missing / mention_span / mention_antecedent),
never a silent drop. An endpoint whose mention data is
absent — the proposer did not answer for it — is not
blocked: it falls back to the legacy evidence path and
is judged on its quote exactly as without the layer.
(Cut 2: the first cut blocked absence too, and the
measured Phase 2 A/B showed the proposer's unreliable
mention-field compliance culling a quarter of all
proposals before judging. The layer is additive by
policy — only a declared-but-invalid mention can
block — and every fallback is counted by endpoint in
the batch accounting, so the fallback rate is always
visible.) The quote is never rewritten — the resolution
travels as annotation, and GATES judges the resolution
as its own decision before the fact (stage_gates), so
the proposer proposes and Jev decides, as everywhere
else in the pipeline.

Entity sweep (issue #58, profile-gated by the
`entity_sweep` setup_config key; absent = off and every
behavior below is inert): the chunk pass proposes at
document density, and an entity the text names but the
pass never builds a triple around has no second chance —
the gold re-examination measured that population at
~30% of all verified labels (every one attested by name
in the text; reference-only presence was zero). The
sweep is an ADDITIVE second pass: a deterministic code
census extracts the document's candidate name surfaces
(maximal capitalized sequences, attested under norm_ws
containment or dropped and counted), the surfaces that
appear as no endpoint of any chunk-pass item become
anchors, and each anchor is re-asked in its own view —
its mention sentences ± 1, capped — under the chunk
prompt plus one rule (an endpoint of every triple must
be the anchor). Sweep items pass the same guards and
merge behind the chunk channel: a sweep item whose key
equals a chunk-pass item's is a counted duplicate, never
a replacement, and an item touching neither endpoint of
its anchor is counted off-anchor and dropped, never
blocked into the record. The chunk pass runs first,
untouched, with its own prompt — the pass can only add
proposal mass, which is the structural form of the
mention layer's measured lesson (a REPLACING variant
proposed less, and differently, and the draw fell).

Cut 2 (issue #58) makes the item's quote — its
evidence span — load-bearing in this channel only.
The first pilot's adjudicated precision was 0.676
against a 0.90 bar: assertions the text does not
state, proposed from a narrow anchor view, passed the
gates. The sweep prompt now states the span as a
verified term of the contract, and
propose_sweep_for_doc voids BY CODE any sweep
proposal whose span is missing (field absent, empty,
or non-string) or fails norm_ws containment in the
FULL document text — before the anchor test and
before the guards, never blocked into the record —
counted as `evidence_missing` in the stage metrics
and the finish record's restatement, so the void is
visible accounting, not the cut-1 parse filter's
silence (a span-less sweep item there vanished
uncounted). The accounting identity is
proposed = items + off_anchor + evidence_missing.
The void verifies presence, not sufficiency — a
verbatim span can still be over-read — which is why
the redesign is measured, not assumed. The chunk
channel is untouched: its items keep the legacy
discipline (a parse-required quote, guard-verified).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from . import connections
from .reference_profile import CHAT_URL as CHAT_URL  # preset route (re-export)
from .reference_profile import PROPOSER_MODEL, PROVIDER_SORT

DEFAULT_MODEL = PROPOSER_MODEL
CHUNK = 20000
STEP = 19500
MAX_PER_CHUNK = 10
WINDOW = 500  # judge context = quote ± 500 chars (audit standard)

# The base templates below are
# domain-neutral core text. A family's domain framing — the
# opening line of each prompt, and for slots the rules block —
# is profile data (pipeline/profile_config.py: proposer_framing /
# slot_framing / slot_rules), threaded in by stage_candidates.
# Before this fix, CUAD's contract framing was compiled into
# the core templates and every family ran under it (fiction,
# which declares no guidance, was told its documents were
# commercial contract excerpts). The defaults below are the
# neutral core framing a family gets when it declares none.
TRIPLE_FRAMING_DEFAULT = \
    "Extract factual relations from the excerpt below."

TRIPLE_PROMPT = """{framing}

Return ONLY a JSON array of at most {maxn} objects:
{{"subject": ..., "relation": ..., "object": ..., "quote": ...}}

Rules:
- subject / object: named entities only — parties, companies,
  people, defined terms, documents, dates, amounts, jurisdictions.
  Never pronouns, never bare role words alone ("the party").
- relation: the short verb phrase as stated in the text
  (e.g. "grants", "shall pay", "is located in", "terminates").
- quote: the SHORTEST verbatim phrase from the excerpt that states
  the fact (a clause, not a whole paragraph). Character-for-
  character; never paraphrase.
- Only facts the excerpt states. If none, return [].

Excerpt:
---
{text}
---"""

# Mention types the proposer may declare for an endpoint's
# mention span (issue #54). "named" = the mention is the
# entity's name or a form of it; "pronominal" = a pronoun;
# "descriptive" = a definite description that is not the
# name ("the company", "the German racer").
MENTION_TYPES = ("named", "pronominal", "descriptive")

# The mention-layer variant of the triple prompt (issue
# #54): identical rules for the triple itself, plus the
# per-endpoint mention fields. Used only when the profile
# declares the mention layer; the base template above is
# untouched so undeclared profiles build byte-identical
# prompts.
TRIPLE_PROMPT_MENTION = """{framing}

Return ONLY a JSON array of at most {maxn} objects:
{{"subject": ..., "relation": ..., "object": ..., "quote": ...,
  "subject_mention": ..., "subject_mention_type": ...,
  "subject_antecedent": ..., "object_mention": ...,
  "object_mention_type": ..., "object_antecedent": ...}}

Rules:
- subject / object: named entities only — parties, companies,
  people, defined terms, documents, dates, amounts, jurisdictions.
  Never pronouns, never bare role words alone ("the party").
- relation: the short verb phrase as stated in the text
  (e.g. "grants", "shall pay", "is located in", "terminates").
- quote: the SHORTEST verbatim phrase from the excerpt that states
  the fact (a clause, not a whole paragraph). Character-for-
  character; never paraphrase.
- subject_mention / object_mention: the exact words IN THE
  QUOTE that refer to the subject / object — the name itself,
  a pronoun ("He", "They", "it"), or a description
  ("the company"). Copy them character-for-character from
  the quote.
- subject_mention_type / object_mention_type: "named" when
  the mention is the entity's name or a form of it;
  "pronominal" for a pronoun; "descriptive" for a
  description that is not the name.
- subject_antecedent / object_antecedent: when the mention
  type is not "named", the SHORTEST verbatim phrase from
  the excerpt that names the entity the mention refers to
  (character-for-character; it may sit outside the quote).
  When the type is "named", use "".
- Only facts the excerpt states. If none, return [].

Excerpt:
---
{text}
---"""

SLOT_FRAMING_DEFAULT = \
    "You extract clause facts from the excerpt below."
# Neutral core rules: value/quote discipline only. Category
# semantics (what a category's value looks like, yes/no-style
# categories, party handling) are the family's domain content,
# declared as slot_rules in its profile — CUAD's rules
# block, carried verbatim in profiles/cuad/setup_config.json,
# names its clause categories and is not core text.
SLOT_RULES_DEFAULT = """- "value": the shortest complete value, as written in the
  excerpt. For dates, as written.
- "quote": copy the EXACT verbatim text from the excerpt that states
  the fact (a phrase or one sentence). It must match the excerpt
  character-for-character. Never paraphrase the quote."""

SLOT_PROMPT = """{framing}

For each category below, if the excerpt STATES the fact, return one
entry. If a category is not stated in this excerpt, omit it.

Categories: {cats}

Rules:
{rules}
- Return ONLY a JSON array of {{"category": ..., "value": ...,
  "quote": ...}}. No other text.

Excerpt:
---
{text}
---"""

# Prepositions for the theme-presence guard.
PREPS = {"in", "on", "of", "for", "to", "with", "by", "at", "from",
         "under", "over", "between", "into", "upon", "against",
         "within", "without", "through", "per", "via", "as", "about"}

# The role-word guard's vocabulary is NOT core data ():
# role nouns are one domain's content, so the set lives in the
# declaring pack (packs/legal-contract/role_words.json) and
# reaches guard_triples as a parameter, resolved by
# profile_config from the run's config / binding pack. Core
# carries the guard MECHANISM (an endpoint that is a bare role
# word is not an identity), never the word list.


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _is_haiku55(model: str) -> bool:
    # The Haiku behavior switch is a capability declaration
    # now: the rule table lives once, in the
    # connection layer (connections.derive_capabilities).
    return connections.CAP_NO_TEMPERATURE in \
        connections.derive_capabilities(model)


def _chunks(text: str) -> list[tuple[int, str]]:
    return [(i // STEP, text[i:i + CHUNK])
            for i in range(0, len(text), STEP)]


def _chat(content: str, model: str, max_tokens: int,
          key: str) -> tuple[str, dict]:
    binding = connections.seat("proposer")
    if not key:
        raise connections.BindingKeyError(
            f"seat 'proposer' cannot run: its key variable "
            f"{binding.key_env} is not set")
    import requests
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    # provider.sort is an OpenRouter-dialect field (issue
    # #41): emitted only when the seat's dialect declares it.
    if binding.is_openrouter_dialect:
        payload["provider"] = {"sort": PROVIDER_SORT}
    if not binding.has_capability(
            connections.CAP_NO_TEMPERATURE, model):
        payload["temperature"] = 0.0
    r = requests.post(
        binding.chat_url,
        headers={"Authorization": f"Bearer {key}"},
        json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```"
                         else lines[1:])
    return text, connections.normalize_usage(
        data.get("usage"), binding)


def _parse_array(content: str, salvage: bool) -> list:
    try:
        items = json.loads(content)
    except Exception:
        if not salvage:
            return []
        # Salvage a truncated array: keep complete objects only.
        cut = content.rfind("},")
        if cut <= 0:
            return []
        try:
            items = json.loads(content[:cut + 1] + "]")
        except Exception:
            return []
    return items if isinstance(items, list) else []


def _build_triple_prompt(chunk: str, guidance: str = "",
                         framing: str | None = None,
                         maxn: int | None = None,
                         mentions: bool = False) -> str:
    """The proposer prompt for one chunk .

    The base is the domain-neutral TRIPLE_PROMPT template filled
    with the family's declared framing (profile data;
    pipeline/profile_config.proposer_framing) or, absent a
    declaration, TRIPLE_FRAMING_DEFAULT. With the framing the
    CUAD profile declares, the construction is byte-identical to
    the pre-framing prompt — pinned by
    the profile-wiring tests against the golden string
    captured before the guidance channel existed, and per family
    by the domain-boundary tests. With guidance (rendered
    from the corpus profile by
    pipeline/profile_config.derive_guidance), the guidance block is
    appended after the base prompt.

    `mentions` (the profile's mention-layer declaration,
    issue #54) selects TRIPLE_PROMPT_MENTION instead of the
    base template; the default path is untouched.
    """
    template = TRIPLE_PROMPT_MENTION if mentions else TRIPLE_PROMPT
    base = template.format(
        framing=framing or TRIPLE_FRAMING_DEFAULT,
        maxn=MAX_PER_CHUNK if maxn is None else maxn, text=chunk)
    if not guidance:
        return base
    return base + "\n\n" + guidance


def _mention_from_item(it: dict, ep: str) -> dict:
    """One endpoint's mention record from a raw proposal
    item (issue #54): the flat subject_mention /
    subject_mention_type / subject_antecedent fields the
    mention prompt asks for, gathered into one dict.
    Missing or malformed fields stay missing — guard_triples
    decides what that means (an absent mention falls back
    to the legacy evidence path for that endpoint; a
    present-but-invalid one blocks by name), the parse
    layer invents nothing."""
    m: dict = {}
    span = it.get(f"{ep}_mention")
    if isinstance(span, str) and span.strip():
        m["span"] = span.strip()
    mtype = it.get(f"{ep}_mention_type")
    if isinstance(mtype, str) and mtype.strip():
        m["type"] = mtype.strip().lower()
    ant = it.get(f"{ep}_antecedent")
    if isinstance(ant, str) and ant.strip():
        m["antecedent"] = ant.strip()
    return m


def propose_triples_for_doc(doc_id: str, text: str, key: str,
                            model: str = DEFAULT_MODEL,
                            guidance: str = "",
                            framing: str | None = None,
                            max_per_chunk: int | None = None,
                            mentions: bool = False) -> dict:
    """Raw triple proposals for one document (unguarded).

    Returns {items, cost, calls, errors, chunks, chunks_at_cap};
    each item carries the chunk it came from (+ its normalized
    form) for the guards. When the profile's guidance asks for
    typed endpoints, items also carry subject_type / object_type
    as proposed (validated against the profile's type
    vocabulary in guard_triples).

    max_per_chunk is the prompt's per-chunk proposal ceiling:
    None selects the core default (MAX_PER_CHUNK — the CUAD
    standalone recipe's value); a profile may declare its own
    (profile_config.proposer_max_per_chunk). chunks_at_cap
    counts chunks whose parsed proposal count reached the
    ceiling — the saturation signal: a chunk at the cap is
    indistinguishable from a chunk that merely had that many
    facts, unless it is counted (issue #52).

    mentions (the profile's mention-layer declaration,
    issue #54) selects the mention prompt variant; items then
    also carry a `mentions` dict ({subject, object} ->
    {span, type, antecedent?}) as proposed — span
    verification is guard_triples' job, as for typed
    endpoints above.
    """
    cap = MAX_PER_CHUNK if max_per_chunk is None else max_per_chunk
    tasks = _chunks(text)

    def work(task):
        idx, chunk = task
        try:
            content, u = _chat(
                _build_triple_prompt(chunk, guidance, framing,
                                     maxn=cap, mentions=mentions),
                model, 12000 if _is_haiku55(model) else 6000, key)
            items = _parse_array(content, salvage=True)
            err = None
        except Exception as e:
            items, u, err = [], {}, str(e)[:200]
        out = []
        for it in items:
            if isinstance(it, dict) and all(
                    it.get(k) for k in
                    ("subject", "relation", "object", "quote")):
                rec = {
                    "subject": str(it["subject"]).strip(),
                    "relation": str(it["relation"]).strip(),
                    "object": str(it["object"]).strip(),
                    "quote": str(it["quote"]).strip(),
                    "chunk_idx": idx, "chunk_norm": norm_ws(chunk)}
                # Typed endpoints, when the proposer emits
                # them (guidance asks for them). Carried verbatim
                # here; guard_triples validates them against the
                # profile's endpoint-type vocabulary.
                for tk in ("subject_type", "object_type"):
                    tv = it.get(tk)
                    if isinstance(tv, str) and tv.strip():
                        rec[tk] = tv.strip()
                if mentions:
                    rec["mentions"] = {
                        ep: _mention_from_item(it, ep)
                        for ep in ("subject", "object")}
                out.append(rec)
        return out, u, err

    items, cost, calls, errors = [], 0.0, 0, 0
    chunks_at_cap = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for out, u, err in ex.map(work, tasks):
            items.extend(out)
            if len(out) >= cap:
                chunks_at_cap += 1
            cost = connections.add_cost(cost, u.get("cost"))
            calls += 1
            if err:
                errors += 1
    return {"items": items,
            "cost": None if cost is None else round(cost, 4),
            "calls": calls,
            "errors": errors, "proposed": len(items),
            "chunks": len(tasks), "chunks_at_cap": chunks_at_cap}


# ------------------------------------------------------------------
# Entity sweep (issue #58): deterministic census + per-anchor pass
# ------------------------------------------------------------------
# The per-anchor view's character ceiling: the mention
# sentences of one anchor, ± 1 neighbouring sentence,
# in document order. Windows beyond the ceiling are
# counted, never silently truncated.
SWEEP_CONTEXT_MAX = 3000

# Lowercase function words that may sit INSIDE a name
# surface, between two capitalized tokens ("Bank of
# America", "University of the Philippines", "Ludwig van
# Beethoven"). Language mechanics, not domain vocabulary:
# a connector never starts or ends a surface, and a
# surface containing one still has to attest verbatim
# (norm_ws containment) before it joins the roster.
_SWEEP_CONNECTORS = frozenset(
    {"of", "the", "and", "de", "del", "van", "von", "der"})

# Sentence-boundary discipline for the census and the
# anchor views: the pair census's abbreviation rule
# (pair_proposer.split_sentences_r3's guard, applied here
# unconditionally — a mention spanning a false boundary
# is exactly what the sweep exists to find). A token
# whose final period belongs to a dotted initialism
# ("U.S.") or a declared abbreviation word ("St.", "Dr.")
# does NOT end a sentence; any other token ending in
# [.!?] does, and a name sequence never crosses it.
_INITIALISM_RX = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.)+")
_ABBREV_WORD_RX = re.compile(
    r"(?i)(?<![A-Za-z])(?:st|mt|dr|mr|mrs|ms|no|fig|vs|jr|sr)\.")
_ABBREV_MASK = "\x00"


def _mask_abbrev_periods(text: str) -> str:
    chars = list(text)
    for rx in (_INITIALISM_RX, _ABBREV_WORD_RX):
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                if chars[i] == ".":
                    chars[i] = _ABBREV_MASK
    return "".join(chars)


# Trailing closers that may follow a sentence mark
# (quotes, parens, brackets) — stripped before the mark
# is read. Typographic quotes written as escapes, per
# the lint standard for ambiguous punctuation.
_TRAILING_CLOSERS = frozenset(
    "\"'\u201d\u2019\u00bb)]}")


def _sentence_final(raw_tok: str) -> bool:
    """True when a raw whitespace token ends a sentence:
    it closes with [.!?] (after trailing closers) and the
    mark is not an abbreviation period."""
    t = raw_tok
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1]
    if not t or t[-1] not in ".!?":
        return False
    if t[-1] != ".":
        return True
    return not any(m.end() == len(t)
                   for rx in (_INITIALISM_RX, _ABBREV_WORD_RX)
                   for m in rx.finditer(t))


def _token_core(tok: str) -> str:
    """A whitespace token stripped of its edge punctuation
    (internal apostrophes / hyphens / periods survive:
    "O'Brien", "Smith-Jones", "U.S"). Pure-punctuation
    tokens core to "" — they break name sequences, which
    is what separates "Paris. London" into two surfaces
    and keeps a comma-joined list from gluing."""
    i, j = 0, len(tok)
    while i < j and not tok[i].isalnum():
        i += 1
    while j > i and not tok[j - 1].isalnum():
        j -= 1
    return tok[i:j]


def _starts_capital(core: str) -> bool:
    """True when the token's first letter is uppercase."""
    for ch in core:
        if ch.isalpha():
            return ch.isupper()
    return False


_QUOTE_CHARS = frozenset("\"'\u201c\u201d\u2018\u2019\u00ab\u00bb")

# Sentence-initial words that are not name material: at a
# sentence or paragraph boundary a capitalised token from
# this set is prose syntax, not the start of a name
# ("During Alger 's term" censuses Alger, not a person
# called During Alger).
_INITIAL_DROP = frozenset(
    {"during", "after", "before", "when", "while", "as",
     "in", "on", "at", "by", "for", "from", "with"})


def _is_quote_token(raw: str) -> bool:
    """True for a whitespace token made only of quote
    characters (the corpus tokenises quotes standalone:
    'Wilfried " Willi " Schneider')."""
    return bool(raw) and all(ch in _QUOTE_CHARS
                             for ch in raw)


def census_entities(text: str) -> dict:
    """The deterministic census roster for one document.

    Extracts maximal capitalized-token sequences (a
    lowercase connector run may bridge two capitalized
    tokens; a sequence never crosses a sentence boundary
    — abbreviation periods excepted, per the boundary
    rule above — and never crosses a paragraph break: a
    title line and the body are different mention sites,
    and gluing them manufactures names that occur
    nowhere). Three surface disciplines keep the roster
    to names the document actually uses:

    - a leading article (The / A / An) is dropped when a
      two-or-more-token name follows it, so a
      sentence-initial "The Bank of America" censuses as
      "Bank of America", while a short name keeps its
      article: "The Hague"; a leading sentence-initial
      prose word (During / After / In / ...) at a
      sentence or paragraph boundary is dropped the
      same way.
    - a quote-wrapped token inside a sequence is a
      nickname insertion (Wilfried "Willi" Schneider):
      quote tokens bridge the sequence instead of
      breaking it, the wrapped token is excluded from
      the surface when two or more unwrapped tokens
      remain, and its normalized form is recorded as an
      alias of the surface. The entity is attested by
      its full bridged form, which the alias list also
      carries.
    - a surface that differs from another roster
      surface only by one token, where that token is a
      recorded nickname alias of the other surface
      (Willi Schneider vs Wilfried Schneider), folds
      into the other surface as an alias form.

    Each surface is kept only if attested in the
    document (norm_ws containment, or the bridged-form
    attestation above); a surface the normalization
    cannot find (a name whose internal punctuation the
    joining flattened, e.g. "U.S. Army" -> "u.s army")
    is dropped AND counted, never silently lost.
    Dedupes by normalized surface, first surface form
    wins. Pure function of the text: identical run to
    run, no API, so the roster's recall against any
    gold name set is measurable offline (the sweep's
    pre-flight gate).

    Returns {entities: [{surface, norm, aliases}],
    extracted, unattested}: `extracted` counts every
    sequence the scan produced, `unattested` the ones
    the containment test dropped. `aliases` are
    normalized alternate mention forms of the entity
    (nickname tokens, bridged full forms, folded alias
    forms) — mention forms, never separate entities.
    """
    doc_norm = norm_ws(text)
    doc_bridged = norm_ws(" ".join(
        t for t in text.split() if not _is_quote_token(t)))
    surfaces: list[tuple[str, list[str], bool]] = []
    for par in re.split(r"\n\s*\n", text):
        raw_tokens = par.split()
        merged: list[str] = []
        ti = 0
        while ti < len(raw_tokens):
            if ti + 2 < len(raw_tokens) + 1 \
                    and ti + 2 <= len(raw_tokens) - 1 \
                    and raw_tokens[ti + 1] in ("-", "\u2013") \
                    and _token_core(raw_tokens[ti]) \
                    and _starts_capital(
                        _token_core(raw_tokens[ti])) \
                    and _token_core(raw_tokens[ti + 2]) \
                    and _starts_capital(
                        _token_core(raw_tokens[ti + 2])):
                merged.append(
                    _token_core(raw_tokens[ti]) + "-"
                    + _token_core(raw_tokens[ti + 2]))
                ti += 3
                continue
            merged.append(raw_tokens[ti])
            ti += 1
        raw_tokens = merged
        tokens = [_token_core(t) for t in raw_tokens]
        quoted: list[bool] = []
        quote_open = False
        for rt in raw_tokens:
            if _is_quote_token(rt):
                quote_open = not quote_open
                quoted.append(False)
            else:
                quoted.append(
                    quote_open or rt[:1] in _QUOTE_CHARS)
        # Sentence-boundary map: a token starts at a
        # boundary at the paragraph start or right after
        # a sentence-final raw token (quote tokens are
        # transparent to the look-back).
        boundary: list[bool] = []
        last_raw: str | None = None
        for idx, rt in enumerate(raw_tokens):
            boundary.append(
                last_raw is None or _sentence_final(last_raw))
            if not _is_quote_token(rt):
                last_raw = rt
        i, n = 0, len(tokens)
        while i < n:
            if _is_quote_token(raw_tokens[i]):
                i += 1
                continue
            if tokens[i] and _starts_capital(tokens[i]):
                seq: list[int] = [i]
                j = i + 1
                while j < n:
                    if _is_quote_token(raw_tokens[j]):
                        j += 1
                        continue
                    # A name sequence never crosses a
                    # sentence boundary: the token before
                    # the next one must not be
                    # sentence-final.
                    prev = j - 1
                    while _is_quote_token(raw_tokens[prev]):
                        prev -= 1
                    if _sentence_final(raw_tokens[prev]):
                        break
                    t = tokens[j]
                    if t and _starts_capital(t):
                        seq.append(j)
                        j += 1
                        continue
                    k = j
                    while k < n and tokens[k] \
                            and tokens[k].lower() \
                            in _SWEEP_CONNECTORS:
                        k += 1
                    if k > j and k < n and tokens[k] \
                            and _starts_capital(tokens[k]):
                        seq.extend(range(j, k + 1))
                        j = k + 1
                        continue
                    break
                cores = [tokens[x] for x in seq]
                if len(cores) >= 3 \
                        and cores[0].lower() in ("the", "a", "an"):
                    seq = seq[1:]
                    cores = cores[1:]
                elif len(cores) >= 2 and boundary[i] \
                        and cores[0].lower() in _INITIAL_DROP:
                    seq = seq[1:]
                    cores = cores[1:]
                plain = [x for x in seq if not quoted[x]]
                aliases: list[str] = []
                attested = False
                if 2 <= len(plain) < len(seq):
                    aliases = [norm_ws(tokens[x])
                               for x in seq if quoted[x]]
                    full = " ".join(cores)
                    if norm_ws(full) in doc_bridged:
                        aliases.append(norm_ws(full))
                        seq = plain
                        cores = [tokens[x] for x in seq]
                        attested = True
                surface = " ".join(cores)
                if "-" in surface:
                    spaced = norm_ws(
                        surface.replace("-", " - "))
                    if spaced in doc_norm:
                        aliases.append(spaced)
                        attested = True
                surfaces.append(
                    (surface, aliases, attested))
                i = j
            else:
                i += 1
    entities, seen, unattested = [], set(), 0
    for s, aliases, attested in surfaces:
        sn = norm_ws(s)
        if not attested and sn not in doc_norm:
            unattested += 1
            continue
        if sn in seen:
            continue
        seen.add(sn)
        entities.append({"surface": s, "norm": sn,
                         "aliases": sorted(set(aliases))})
    # Fold alias forms: a surface differing from another
    # in exactly one token, where its token there is a
    # nickname alias of the other surface, is a mention
    # form of that entity, not an entity of its own.
    alias_of: dict[str, str] = {}
    for e in entities:
        for a in e["aliases"]:
            alias_of.setdefault(a, e["norm"])
    by_norm = {e["norm"]: e for e in entities}
    folded: set[str] = set()
    for e in entities:
        toks = e["norm"].split()
        for pos, t in enumerate(toks):
            canon = alias_of.get(t)
            if canon is None or canon == e["norm"] \
                    or canon not in by_norm:
                continue
            ctoks = canon.split()
            if len(ctoks) != len(toks):
                continue
            cand = toks[:pos] + [ctoks[pos]] + toks[pos + 1:]
            if " ".join(cand) == canon:
                by_norm[canon]["aliases"] = sorted(set(
                    by_norm[canon]["aliases"])
                    | set(e["aliases"]) | {e["norm"]})
                folded.add(e["norm"])
                break
    entities = [e for e in entities
                if e["norm"] not in folded]
    return {"entities": entities,
            "extracted": len(surfaces),
            "unattested": unattested}


def _identity_index(doc_text: str) -> tuple[dict, dict, set]:
    """The identity index guard_triples resolves against:
    (by_norm, alias_of, maximals) from the document's
    census. by_norm maps a census norm to its entity;
    alias_of maps an alias norm (nickname, bridged full
    form, folded alias form) to the canonical entity
    norm; maximals are the entity norms whose token set
    is not a proper subset of another entity's — the
    only surfaces a fragment may resolve INTO."""
    cen = census_entities(doc_text)
    by_norm = {e["norm"]: e for e in cen["entities"]}
    alias_of: dict[str, str] = {}
    for e in cen["entities"]:
        for a in e["aliases"]:
            alias_of.setdefault(a, e["norm"])
    toks = {n: _identity_tokens(n) for n in by_norm}
    maximals = {n for n in by_norm
                if not any(n != m and toks[n] < toks[m]
                           for m in by_norm)}
    return by_norm, alias_of, maximals


def _name_tokens(norm: str) -> set:
    """The token set identity resolution compares:
    whitespace- and hyphen-separated parts. A hyphenated
    compound (Bernheimer-Haus) is one census token but
    two name parts — a fragment may name either part."""
    return {t for t in re.split(r"[\s-]+", norm) if t}


def _identity_tokens(norm: str) -> set:
    """_name_tokens with a leading article removed:
    the / a / an at the head of a surface is syntax,
    not identity ("The Hague" is the entity an endpoint
    "Hague" names; "The Haus" must not outrank the
    compound it abbreviates)."""
    toks = _name_tokens(norm)
    first = norm.split(" ", 1)[0] if norm else ""
    if first in ("the", "a", "an"):
        toks.discard(first)
    return toks


def _resolve_endpoint(ep_norm: str, index: tuple,
                      anchor_norm: str | None = None
                      ) -> tuple[str | None, list[str]]:
    """Resolve one proposed endpoint (norm_ws form) to
    the document's canonical surface for the entity it
    names. Returns (surface, candidates): surface is the
    canonical surface to substitute, or None when no
    resolution applies; a non-empty candidates list
    means the fragment is ambiguous between those
    canonical surfaces — ambiguity is reported, never
    guessed.

    Resolution order (deterministic, code-only):
    1. an exact census entity that is maximal, or not
       contained in any other entity, stands as-is;
    2. a recorded alias (nickname / bridged form /
       folded alias form) resolves to its entity;
    3. a fragment of the item's own anchor (the sweep
       asked about that entity) resolves to the anchor;
    4. a proper token-subset of exactly ONE maximal
       surface resolves to it; of several, it is
       ambiguous; of none, it stands as proposed (the
       proposer may name entities the census missed —
       absence from the roster is not proof of error).
    """
    by_norm, alias_of, maximals = index
    if ep_norm in by_norm and ep_norm in maximals:
        return None, []
    if ep_norm in alias_of:
        return by_norm[alias_of[ep_norm]]["surface"], []
    etoks = _identity_tokens(ep_norm)
    if anchor_norm and anchor_norm in by_norm \
            and etoks < _identity_tokens(anchor_norm):
        return by_norm[anchor_norm]["surface"], []
    toks = {n: _identity_tokens(n) for n in by_norm}
    cands = sorted(by_norm[m]["surface"] for m in maximals
                   if etoks < toks[m]
                   or (etoks == toks[m] and ep_norm != m))
    if len(cands) == 1:
        return cands[0], []
    if len(cands) > 1:
        return None, cands
    return None, []


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence split for anchor views:
    boundaries after [.!?] followed by whitespace, with
    abbreviation periods masked first (the pair census's
    discipline, without its title-block special case and
    without any length filter — a short sentence can
    carry a mention)."""
    masked = _mask_abbrev_periods(text)
    return [p.strip().replace(_ABBREV_MASK, ".") for p in
            re.split(r"(?<=[.!?])\s+", masked) if p.strip()]


def anchor_contexts(text: str, anchor: str,
                    aliases: list[str] | None = None) -> dict:
    """One anchor's view: the sentences containing a
    census-attested mention of the anchor (normalized
    substring; any of the anchor's census aliases — a
    nickname or bridged form the canonical surface's
    own substring cannot find — also counts), each
    with ± 1 neighbouring sentence, in document order,
    total ≤ SWEEP_CONTEXT_MAX chars.

    Contiguous kept sentences merge into runs; a run that
    would push the view past the ceiling is left out and
    counted (`windows_dropped`) — never silently
    truncated mid-run. Returns {context, mention_sentences,
    windows, windows_dropped}."""
    sents = split_sentences(text)
    forms = [norm_ws(anchor)] + list(aliases or [])
    hits = [i for i, s in enumerate(sents)
            if any(f in norm_ws(s) for f in forms)]
    keep: set[int] = set()
    for i in hits:
        keep.update((i - 1, i, i + 1))
    runs: list[list[int]] = []
    for i in sorted(x for x in keep if 0 <= x < len(sents)):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    blocks, dropped, total = [], 0, 0
    for run in runs:
        block = " ".join(sents[x] for x in run)
        add = len(block) + (1 if blocks else 0)
        if total + add > SWEEP_CONTEXT_MAX:
            dropped += 1
            continue
        blocks.append(block)
        total += add
    return {"context": "\n".join(blocks),
            "mention_sentences": len(hits),
            "windows": len(blocks),
            "windows_dropped": dropped}


def _build_sweep_prompt(context: str, anchor: str,
                        guidance: str = "",
                        framing: str | None = None,
                        maxn: int | None = None,
                        mentions: bool = False) -> str:
    """The sweep prompt for one anchor: the chunk prompt
    built by _build_triple_prompt (the base template stays
    untouched), over the anchor's view instead of a chunk,
    plus two added rules: the anchor rule (an endpoint of
    every triple must be the anchor entity, named as it
    appears in the text) and — cut 2 — the evidence rule
    (every triple must carry its evidence span, and the
    span is verified; a triple without a verifying span
    is discarded)."""
    base = _build_triple_prompt(context, guidance, framing,
                                maxn=maxn, mentions=mentions)
    return base + (
        f"\n\nAdditional rule for this pass: one endpoint "
        f"of every triple — the subject or the object — "
        f"must be the entity \"{anchor}\", named exactly "
        f"as it appears in the excerpt."
        f"\n\nEvidence rule for this pass: every triple "
        f"must carry its evidence span in \"quote\" — the "
        f"verbatim phrase from the excerpt that states "
        f"the fact, copied character-for-character. The "
        f"span is verified against the document text: a "
        f"triple whose quote is missing, or whose text "
        f"does not appear in the document, is discarded.")


def propose_sweep_for_doc(doc_id: str, text: str,
                          anchors: list[dict], key: str,
                          model: str = DEFAULT_MODEL,
                          guidance: str = "",
                          framing: str | None = None,
                          max_per_chunk: int | None = None,
                          mentions: bool = False) -> dict:
    """Per-anchor proposals for one document (unguarded).

    One proposer-seat call per anchor over its mention
    view (anchor_contexts), same decoding and per-call
    ceiling as the chunk pass. Items are parsed with the
    chunk pass's discipline and carry the anchor they
    were asked under (`anchor`, `anchor_norm`) plus the
    view as their chunk fields, so guard_triples' chunk
    attestation rung reads the view.

    Cut 2 — the evidence void, checked FIRST, before
    the anchor test: an item whose evidence span (its
    `quote`) is missing — the field absent, empty, or
    not a string — or whose span fails verification
    (`norm_ws(quote)` not a substring of the document's
    `norm_ws` — the FULL document text, strict
    whitespace-normalized verbatim; no
    punctuation-insensitive rung softens it) is VOID:
    counted (`evidence_missing`) and dropped here,
    never returned, never blocked into the record. An
    item failing both the evidence test and the anchor
    test counts as evidence_missing — the order is
    fixed so the accounting is deterministic.

    An item whose NEITHER endpoint is its anchor is
    contract drift: it is counted (`off_anchor`) and
    dropped here — never returned, never blocked into
    the record. An anchor with no view (no mention
    sentence survived the ceiling) is skipped and counted
    (`anchors_skipped`), never called.

    Returns {items, cost, calls, errors, proposed,
    off_anchor, evidence_missing, anchors_asked,
    anchors_skipped, windows_dropped}; `proposed` counts
    every parsed item carrying the triple fields, so the
    pass's attrition is fully accounted by
    proposed = items + off_anchor + evidence_missing.
    (Items lacking a triple field entirely are not
    triples — malformed, invisible, exactly as on the
    chunk path.)
    """
    cap = MAX_PER_CHUNK if max_per_chunk is None \
        else max_per_chunk
    doc_norm = norm_ws(text)

    def work(task):
        _idx, anchor = task
        view = anchor_contexts(text, anchor["surface"],
                               anchor.get("aliases"))
        if not view["context"]:
            return [], {}, None, 0, 0, \
                view["windows_dropped"], True
        try:
            content, u = _chat(
                _build_sweep_prompt(view["context"],
                                    anchor["surface"],
                                    guidance, framing,
                                    maxn=cap, mentions=mentions),
                model, 12000 if _is_haiku55(model) else 6000, key)
            items = _parse_array(content, salvage=True)
            err = None
        except Exception as e:
            items, u, err = [], {}, str(e)[:200]
        out, off, ev = [], 0, 0
        for it in items:
            if not (isinstance(it, dict) and all(
                    it.get(k) for k in
                    ("subject", "relation", "object"))):
                continue
            # The evidence void (cut 2): the span must be
            # present and verify against the full document
            # text. A missing or unverifiable span voids
            # the proposal here — it never reaches the
            # anchor test or the guards.
            q = it.get("quote")
            qn = norm_ws(q) if isinstance(q, str) else ""
            if not qn or qn not in doc_norm:
                ev += 1
                continue
            rec = {
                "subject": str(it["subject"]).strip(),
                "relation": str(it["relation"]).strip(),
                "object": str(it["object"]).strip(),
                "quote": q.strip(),
                "chunk_idx": -1,
                "chunk_norm": norm_ws(view["context"]),
                "anchor": anchor["surface"],
                "anchor_norm": anchor["norm"]}
            for tk in ("subject_type", "object_type"):
                tv = it.get(tk)
                if isinstance(tv, str) and tv.strip():
                    rec[tk] = tv.strip()
            if mentions:
                rec["mentions"] = {
                    ep: _mention_from_item(it, ep)
                    for ep in ("subject", "object")}
            # On-anchor: an endpoint names the anchor —
            # equality or containment either way (the
            # mention layer's naming discipline: a
            # fuller or shorter form of the same name
            # still names it; an unrelated endpoint
            # does not).
            ends = (norm_ws(rec["subject"]),
                    norm_ws(rec["object"]))
            if not any(e == anchor["norm"]
                       or e in anchor["norm"]
                       or anchor["norm"] in e
                       for e in ends):
                off += 1
                continue
            out.append(rec)
        return out, u, err, off, ev, \
            view["windows_dropped"], False

    items, cost, calls, errors = [], 0.0, 0, 0
    off_anchor = evidence_missing = skipped = wdropped = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for out, u, err, off, ev, wd, skip in ex.map(
                work, list(enumerate(anchors))):
            items.extend(out)
            off_anchor += off
            evidence_missing += ev
            wdropped += wd
            if skip:
                skipped += 1
                continue
            cost = connections.add_cost(cost, u.get("cost"))
            calls += 1
            if err:
                errors += 1
    # `proposed` counts every parsed item carrying the
    # triple fields — the items that reach the guards,
    # plus both proposal-time drops, which also carried
    # them: proposed = items + off_anchor +
    # evidence_missing.
    return {"items": items,
            "cost": None if cost is None else round(cost, 4),
            "calls": calls,
            "errors": errors,
            "proposed": len(items) + off_anchor
            + evidence_missing,
            "off_anchor": off_anchor,
            "evidence_missing": evidence_missing,
            "anchors_asked": len(anchors) - skipped,
            "anchors_skipped": skipped,
            "windows_dropped": wdropped}


def propose_slots_for_doc(doc_id: str, text: str,
                          categories: list[str], key: str,
                          model: str = DEFAULT_MODEL,
                          framing: str | None = None,
                          rules: str | None = None) -> dict:
    """Raw clause-slot fills for one document (unguarded).

    The prompt is the domain-neutral SLOT_PROMPT template filled
    with the family's declared framing + rules block (profile data;
    pipeline/profile_config.slot_framing / slot_rules) or the
    neutral core defaults. With the framing and rules the CUAD
    profile declares, the construction is byte-identical to the
    pre-framing prompt (pinned by the domain-boundary tests)."""
    tasks = _chunks(text)
    cats = ", ".join(categories)

    def work(task):
        _idx, chunk = task
        try:
            content, u = _chat(
                SLOT_PROMPT.format(
                    framing=framing or SLOT_FRAMING_DEFAULT,
                    cats=cats, rules=rules or SLOT_RULES_DEFAULT,
                    text=chunk),
                model, 4000 if _is_haiku55(model) else 1500, key)
            items = _parse_array(content, salvage=False)
            err = None
        except Exception as e:
            items, u, err = [], {}, str(e)[:200]
        out = []
        for it in items:
            if isinstance(it, dict) and it.get("category") in categories \
                    and it.get("value") and it.get("quote"):
                out.append({"category": it["category"],
                            "value": str(it["value"]).strip(),
                            "quote": str(it["quote"]).strip()})
        return out, u, err

    fills, cost, calls, errors = [], 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for out, u, err in ex.map(work, tasks):
            fills.extend(out)
            cost = connections.add_cost(cost, u.get("cost"))
            calls += 1
            if err:
                errors += 1
    return {"fills": fills,
            "cost": None if cost is None else round(cost, 4),
            "calls": calls,
            "errors": errors, "proposed": len(fills)}


# ------------------------------------------------------------------
# Evidence windows
# ------------------------------------------------------------------
def quote_window(text: str, quote: str, half: int = WINDOW) -> str:
    """quote ± half chars of source text (the audit-standard judge
    context). Locates the quote raw, then via a whitespace-normalized
    index map; falls back to the quote itself."""
    pos = text.find(quote)
    if pos >= 0:
        return text[max(0, pos - half): pos + len(quote) + half]
    norm, idx = [], []
    prev_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_ws:
                norm.append(" ")
                idx.append(i)
            prev_ws = True
        else:
            norm.append(ch)
            idx.append(i)
            prev_ws = False
    nq = " ".join(quote.split())
    npos = "".join(norm).find(nq)
    if npos < 0 or not idx:
        return quote
    start = idx[npos]
    end = idx[min(npos + len(nq) - 1, len(idx) - 1)] + 1
    return text[max(0, start - half): end + half]


# ------------------------------------------------------------------
# Deterministic guards
# ------------------------------------------------------------------
def _verify_mentions(it: dict, qn: str, doc_norm: str,
                     sn: str, on: str) -> tuple[dict, list] | str:
    """Verify one item's mention records (issue #54).

    Returns (cleaned, fallback) — the verified mention
    records by endpoint, and the endpoints whose mention
    data was ABSENT — or the block reason string when a
    DECLARED mention is invalid. The policy (cut 2) is
    per-endpoint and additive:

    - Absent: the endpoint carries no mention record, or
      the record has no usable span (the proposer did not
      answer for that endpoint — the measured Phase 2
      shape: the mention fields were simply omitted). The
      endpoint falls back to the legacy evidence path:
      the item is judged on its quote exactly as without
      the layer, and the fallback is counted by endpoint
      in the batch accounting.
    - Present but invalid blocks the item by name, with
      the quote's own verbatim discipline:
      - mention_missing: a span was declared but its type
        is missing or outside MENTION_TYPES — a malformed
        answer, not an absent one.
      - mention_span: the mention span is not a verbatim
        (whitespace-normalized) substring of the item's
        quote — the mention must appear in the quote, that
        is where the judge will look for it.
      - mention_antecedent: a non-named mention
        (pronominal / descriptive) without a usable
        resolution: the antecedent is absent, is not a
        verbatim substring of the document, or does not
        name the endpoint (neither surface contains the
        other — "Paul Desmarais" for "Paul Desmarais Jr."
        passes; an unrelated phrase does not).

    A "named" mention needs no antecedent: the span in the
    quote IS the attestation, judged as before.
    """
    raw = it.get("mentions")
    cleaned: dict = {}
    fallback: list = []
    for ep, surf_n in (("subject", sn), ("object", on)):
        m = raw.get(ep) if isinstance(raw, dict) else None
        if not isinstance(m, dict):
            fallback.append(ep)
            continue
        span = m.get("span")
        if not isinstance(span, str) or not span.strip():
            fallback.append(ep)
            continue
        mtype = m.get("type")
        if mtype not in MENTION_TYPES:
            return "mention_missing"
        if norm_ws(span) not in qn:
            return "mention_span"
        rec = {"span": span.strip(), "type": mtype}
        if mtype != "named":
            ant = m.get("antecedent")
            if not isinstance(ant, str) or not ant.strip():
                return "mention_antecedent"
            ant_n = norm_ws(ant)
            if ant_n not in doc_norm:
                return "mention_antecedent"
            if not (ant_n == surf_n or ant_n in surf_n
                    or surf_n in ant_n):
                return "mention_antecedent"
            rec["antecedent"] = ant.strip()
        cleaned[ep] = rec
    return cleaned, fallback


_PUNCT_SPACE_RX = re.compile(r"\s*([^\w\s])\s*")


def punct_collapse(s: str) -> str:
    """Remove whitespace adjacent to punctuation. Tokenized
    corpus texts space their punctuation ('Mediaș ,
    Transylvania'); a quote transcribed with natural
    punctuation ('Mediaș, Transylvania') is the document's
    own text modulo that spacing convention. Used only by
    the opt-in G1 relaxation in guard_triples."""
    return _PUNCT_SPACE_RX.sub(r"\1", s)


def guard_triples(items: list[dict], doc_text: str,
                  role_words: Iterable[str] | None = None,
                  entity_types: set[str] | None = None,
                  theme_attest_key: str | None = None,
                  quote_punct_insensitive: bool = False,
                  theme_attest_chunk: bool = False,
                  mentions: bool = False
                  ) -> tuple[list[dict], list[dict]]:
    """Apply G1 + L0 attestation + role-word + theme guards.

    Returns (kept, blocked). Kept items gain `attest` (the level at
    which both endpoints are attested: quote | chunk | document).
    Blocked items gain `reason` (see module docstring).

    `role_words` is the role-word guard's vocabulary — the
    normalized role nouns, supplied as pack data and resolved
    by profile_config (). An endpoint whose normalized
    surface is in the set is blocked (l0_role_word): a bare
    role word is not an identity. None (the default) disables
    the guard; profiles that declare no role-word rules run
    with it off.

    typing: a proposed subject_type / object_type survives
    into the kept (and blocked) record only when it belongs to the
    profile's endpoint-type vocabulary (`entity_types`); an
    out-of-vocabulary tag is dropped — the triple itself is NOT
    blocked for a bad tag — and the endpoint continues untyped.
    With entity_types=None (no profile vocabulary), proposed tags
    pass through as-is.

    Two opt-in relaxations (both default to
    the legacy behavior, byte-identical):

    - `quote_punct_insensitive`: G1 also accepts a quote that
      is a verbatim substring of the document after
      punctuation-spacing collapse on both sides. The #66
      anatomy showed 15 of 19 g1_quote blocks were the
      document's own words differing only by the corpus's
      punctuation-spacing convention; genuinely edited or
      invented quotes fail at every normalization rung and
      stay blocked.
    - `theme_attest_key`: name of an item field carrying the
      span the item's endpoints were posed from (pair
      enumeration: the pair sentence). When set, the
      theme-presence check accepts endpoint attestation in
      that span as well as in the quote. The C2 pattern the
      guard targets — an endpoint fabricated from an
      unrelated quote fragment — cannot occur for posed
      endpoints: both are attested in the posed span by
      construction (measured 186/186 in the #66 run, where
      the quote-only test blocked pair items whose judge
      cited a subject-side fragment). Free proposals carry
      no such field and keep the quote-only test.
    - `theme_attest_chunk`: the theme-presence check also
      accepts the object attested anywhere in the chunk the
      proposal came from (the item's `chunk_norm`), not only
      inside the shortest quote. For packs whose relations
      are verb+preposition forms ("works at", "diagnosed
      with") whose objects the text typically names across
      the passage rather than inside the minimal quote, the
      quote-only test blocks the relation wholesale (the
      2026-10-23 BYO walk: 10/10 "works at" proposals from a
      hand-authored pack). The relaxation is weaker than
      the quote test — the object need only co-occur in the
      chunk — so it is a declared opt-in, never a default.
    Identity resolution (always on, no profile switch):
    after the quote gate, each endpoint is resolved
    against the document's census roster
    (_resolve_endpoint) — a fragment that names exactly
    one canonical surface (a nickname alias, an alias
    form, a fragment of the item's own anchor, or a
    proper token-subset of one maximal surface) is
    rewritten to that surface and the rewrite is
    recorded in the kept record's `canonicalized`; a
    fragment naming several surfaces is recorded in
    `identity_ambiguous` and left as proposed.
    Attestation reads the effective endpoint through
    the entity's recorded mention forms, so a canonical
    name that never occurs contiguously (a nickname
    intervenes) is not blocked for occurring nowhere.
    Blocked records always carry the item as proposed:
    the resolution lands on the record only when the
    item is kept.

    - `mentions`: the profile declared the mention layer
      (issue #54). Each item's mention records are verified
      by _verify_mentions after the quote gate and before
      attestation: declared spans verbatim, a non-named
      mention carrying a verbatim, endpoint-naming
      antecedent. The layer is additive (cut 2): an
      endpoint whose mention data is absent falls back to
      the legacy evidence path — kept items name those
      endpoints in `mention_fallback` and carry the
      cleaned `mentions` dict (verified endpoints only;
      the key is absent when every endpoint fell back, so
      the record is exactly the legacy shape). Only a
      declared-but-invalid mention blocks, and blocked
      items carry the records as proposed. Default
      False = the checks do not run and records are
      exactly the legacy shape.
    """
    doc_norm = norm_ws(doc_text)
    doc_punct = punct_collapse(doc_norm) \
        if quote_punct_insensitive else None
    identity = _identity_index(doc_text) if items \
        else None
    doc_bridged = norm_ws(" ".join(
        t for t in doc_text.split()
        if not _is_quote_token(t))) \
        if identity is not None else None
    kept, blocked, seen = [], [], set()
    for it in items:
        s, r, o, q = (it["subject"], it["relation"], it["object"],
                      it["quote"])
        base = {"subject": s, "relation": r, "object": o, "quote": q}
        for tk in ("subject_type", "object_type"):
            tv = it.get(tk)
            if isinstance(tv, str) and tv.strip():
                tv = tv.strip()
                if entity_types is None or tv in entity_types:
                    base[tk] = tv
        if mentions and isinstance(it.get("mentions"), dict):
            # As proposed, for the blocked sidecar's audit;
            # replaced by the verified record below.
            base["mentions"] = it["mentions"]
        qn = norm_ws(q)
        if not qn or (qn not in doc_norm
                      and not (doc_punct is not None
                               and punct_collapse(qn) in doc_punct)):
            blocked.append({**base, "reason": "g1_quote"})
            continue
        sn, on = norm_ws(s), norm_ws(o)
        if mentions:
            mres = _verify_mentions(it, qn, doc_norm, sn, on)
            if isinstance(mres, str):
                blocked.append({**base, "reason": mres})
                continue
            cleaned_m, fallback_eps = mres
            if cleaned_m:
                base["mentions"] = cleaned_m
            else:
                # Every endpoint fell back: the kept record
                # is exactly the legacy shape.
                base.pop("mentions", None)
            if fallback_eps:
                base["mention_fallback"] = fallback_eps
        # Identity resolution runs before attestation.
        # A fragment endpoint (Willi / Alger / Paul
        # Desmarais for Paul Desmarais Jr.) names its
        # entity under a mention form, and the graph
        # keys nodes by name string — an unresolved
        # fragment is a separate node and a broken
        # graph, not a stylistic variant. A unique
        # resolution rewrites the endpoint to the
        # canonical surface (recorded in
        # `canonicalized`); an ambiguous one is
        # recorded in `identity_ambiguous` and left as
        # proposed. Attestation then reads the
        # EFFECTIVE endpoint through every mention
        # form the census recorded for its entity: the
        # canonical name may never occur contiguously
        # (Wilfried "Willi" Schneider) while its
        # bridged form does, so a raw substring test
        # would block exactly the correct name.
        forms: dict[str, tuple[list[str], bool]] = {}
        eff = {"subject": base["subject"],
               "object": base["object"]}
        canon: dict[str, list[str]] = {}
        amb: dict[str, list[str]] = {}
        if identity is not None:
            by_norm = identity[0]
            for ep in ("subject", "object"):
                res, cands = _resolve_endpoint(
                    norm_ws(base[ep]), identity,
                    anchor_norm=it.get("anchor_norm"))
                if res is not None and res != base[ep]:
                    canon[ep] = [base[ep], res]
                    eff[ep] = res
                elif cands:
                    amb[ep] = cands
                effn = norm_ws(eff[ep])
                ent = by_norm.get(effn)
                forms[ep] = (
                    [effn] + (list(ent["aliases"])
                              if ent else []),
                    ent is not None or ep in canon)

        def _attested(ep: str, hay: str,
                      bridged: bool = False) -> bool:
            # The bridged-document fallback belongs to
            # L0 alone: it answers "does this entity
            # occur in the document at all". The quote
            # and chunk haystacks must NOT fall back to
            # it — that would widen a quote-level test
            # into a document-level one and invent the
            # attestation the theme guard exists to
            # refuse.
            fs, bridged_ok = forms.get(
                ep, ([norm_ws(base[ep])], False))
            return any(f in hay for f in fs) or (
                bridged and bridged_ok
                and doc_bridged is not None
                and any(f in doc_bridged for f in fs))

        prop_sn, prop_on = sn, on
        sn, on = norm_ws(eff["subject"]), \
            norm_ws(eff["object"])
        # L0: attested somewhere in the document at all?
        if not _attested("subject", doc_norm, bridged=True) \
                or not _attested("object", doc_norm,
                                 bridged=True):
            blocked.append({**base, "reason": "l0_attest"})
            continue
        chunk_n = it.get("chunk_norm", "")
        if _attested("subject", qn) \
                and _attested("object", qn):
            attest = "quote"
        elif _attested("subject", chunk_n) \
                and _attested("object", chunk_n):
            attest = "chunk"
        else:
            attest = "document"
        if role_words and (sn in role_words
                           or on in role_words
                           or prop_sn in role_words
                           or prop_on in role_words):
            blocked.append({**base, "reason": "l0_role_word"})
            continue
        toks = [t for t in re.split(r"[^a-z]+", norm_ws(r)) if t]
        if toks and all(t in PREPS for t in toks):
            blocked.append({**base, "reason": "theme_missing"})
            continue
        if toks and toks[-1] in PREPS \
                and not _attested("object", qn):
            span = it.get(theme_attest_key) \
                if theme_attest_key else None
            if not (isinstance(span, str)
                    and _attested("object",
                                  norm_ws(span))) \
                    and not (theme_attest_chunk
                             and _attested("object",
                                           chunk_n)):
                blocked.append({**base,
                                "reason": "theme_presence"})
                continue
        # The resolution lands on the record only
        # here, once the item is kept: blocked records
        # carry the item exactly as proposed.
        if canon:
            base["subject"] = eff["subject"]
            base["object"] = eff["object"]
            base["canonicalized"] = canon
        if amb:
            base["identity_ambiguous"] = amb
        k = (sn, norm_ws(r), on)
        if k in seen:
            continue
        seen.add(k)
        kept.append({**base, "attest": attest})
    return kept, blocked


def guard_slots(fills: list[dict], doc_text: str) -> tuple[list[dict],
                                                          list[dict]]:
    """Span gate + per-(category, value) dedupe for slot fills."""
    doc_norm = norm_ws(doc_text)
    kept, blocked, seen = [], [], set()
    for f in fills:
        if not norm_ws(f["quote"]) or \
                norm_ws(f["quote"]) not in doc_norm:
            blocked.append({**f, "reason": "g1_quote"})
            continue
        k = (f["category"], norm_ws(f["value"]))
        if k in seen:
            continue
        seen.add(k)
        kept.append(f)
    return kept, blocked


def api_key() -> str:
    """The proposer seat's key, or "" when its binding's
    key variable is unset (callers that require the seat
    refuse with :func:`key_env` named in the message)."""
    return connections.seat("proposer").key_or_none() or ""


def key_env() -> str:
    """The environment variable the proposer binding
    reads its key from (preset: OPENROUTER_API_KEY)."""
    return connections.seat("proposer").key_env
