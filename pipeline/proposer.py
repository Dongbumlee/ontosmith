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
verifies BOTH spans with the same verbatim discipline as
the quote itself; an unverifiable span blocks the item
with a reason code (mention_missing / mention_span /
mention_antecedent), never a silent drop. The quote is
never rewritten — the resolution travels as annotation,
and GATES judges the resolution as its own decision
before the fact (stage_gates), so the proposer proposes
and Jev decides, as everywhere else in the pipeline.
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
    decides what that means (mention_missing), the parse
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
                     sn: str, on: str) -> dict | str:
    """Verify one item's mention records (issue #54).

    Returns the cleaned mentions dict, or the block reason
    string. The discipline mirrors the quote's: every span
    is verbatim or the item is blocked by name —

    - mention_missing: an endpoint carries no usable
      mention record (no span, or a type outside
      MENTION_TYPES) although the profile declared the
      layer — the proposer was asked and did not answer.
    - mention_span: the mention span is not a verbatim
      (whitespace-normalized) substring of the item's
      quote — the mention must appear in the quote, that
      is where the judge will look for it.
    - mention_antecedent: a non-named mention (pronominal /
      descriptive) without a usable resolution: the
      antecedent is absent, is not a verbatim substring of
      the document, or does not name the endpoint (neither
      surface contains the other — "Paul Desmarais" for
      "Paul Desmarais Jr." passes; an unrelated phrase
      does not).

    A "named" mention needs no antecedent: the span in the
    quote IS the attestation, judged as before.
    """
    raw = it.get("mentions")
    if not isinstance(raw, dict):
        return "mention_missing"
    cleaned: dict = {}
    for ep, surf_n in (("subject", sn), ("object", on)):
        m = raw.get(ep)
        if not isinstance(m, dict):
            return "mention_missing"
        span, mtype = m.get("span"), m.get("type")
        if not isinstance(span, str) or not span.strip() \
                or mtype not in MENTION_TYPES:
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
    return cleaned


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
    - `mentions`: the profile declared the mention layer
      (issue #54). Each item's mention records are verified
      by _verify_mentions after the quote gate and before
      attestation: both spans verbatim, a non-named mention
      carrying a verbatim, endpoint-naming antecedent.
      Kept items carry the cleaned `mentions` dict into
      candidacy; blocked items carry the records as
      proposed. Default False = the checks do not run and
      records are exactly the legacy shape.
    """
    doc_norm = norm_ws(doc_text)
    doc_punct = punct_collapse(doc_norm) \
        if quote_punct_insensitive else None
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
            base["mentions"] = mres
        # L0: attested somewhere in the document at all?
        if sn not in doc_norm or on not in doc_norm:
            blocked.append({**base, "reason": "l0_attest"})
            continue
        chunk_n = it.get("chunk_norm", "")
        if sn in qn and on in qn:
            attest = "quote"
        elif sn in chunk_n and on in chunk_n:
            attest = "chunk"
        else:
            attest = "document"
        if role_words and (sn in role_words or on in role_words):
            blocked.append({**base, "reason": "l0_role_word"})
            continue
        toks = [t for t in re.split(r"[^a-z]+", norm_ws(r)) if t]
        if toks and all(t in PREPS for t in toks):
            blocked.append({**base, "reason": "theme_missing"})
            continue
        if toks and toks[-1] in PREPS and on not in qn:
            span = it.get(theme_attest_key) \
                if theme_attest_key else None
            if not (isinstance(span, str)
                    and on in norm_ws(span)) \
                    and not (theme_attest_chunk
                             and on in chunk_n):
                blocked.append({**base,
                                "reason": "theme_presence"})
                continue
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
