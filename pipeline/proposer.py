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
                 no predicate, hence no theme (G-05 theme-presence).
  theme_presence a verb+preposition relation whose object is not
                 attested in the quote itself — the C2 substitution
                 pattern (a prepositional object standing in the
                 evidence span for an unstated theme).

Blocked items land in candidates_blocked.jsonl with the reason code,
the same sidecar the Arm B structural constraints use. Slot fills
pass the span gate (g1_quote) and per-(doc, category, value)
dedupe; their subject is the document's own Document Name fill
(fallback: source file stem), mirroring cuad_clause_tograph.py.

Profile wiring (issue #51): the proposer takes an optional
`guidance` string rendered from the corpus profile's setup_config
(pilot/profile_config.derive_guidance — the EG-R2-1 channel). No
guidance -> the prompt is byte-identical to the recipe above. With
guidance, the model is asked for typed endpoints (subject_type /
object_type, EG-R2-2); guard_triples validates the tags against
the profile's endpoint-type vocabulary.

Domain framing (PK-01): the prompt templates are domain-neutral;
the framing line (and the slot rules block) that specialize them
to a family are profile data threaded through the `framing` /
`rules` parameters (pilot/profile_config.proposer_framing /
slot_framing / slot_rules). The standalone harnesses' contract
text survives as the CUAD profile's declared framing — the
prompt CUAD builds is byte-identical to the recipe above.
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

# PK-01 (docs/GAP_REGISTER.md): the base templates below are
# domain-neutral core text. A family's domain framing — the
# opening line of each prompt, and for slots the rules block —
# is profile data (pilot/profile_config.py: proposer_framing /
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

SLOT_FRAMING_DEFAULT = \
    "You extract clause facts from the excerpt below."
# Neutral core rules: value/quote discipline only. Category
# semantics (what a category's value looks like, yes/no-style
# categories, party handling) are the family's domain content,
# declared as slot_rules in its profile (PK-01) — CUAD's rules
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

# Prepositions for the theme-presence guard (G-05).
PREPS = {"in", "on", "of", "for", "to", "with", "by", "at", "from",
         "under", "over", "between", "into", "upon", "against",
         "within", "without", "through", "per", "via", "as", "about"}

# The role-word guard's vocabulary is NOT core data (A2-F2):
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
    # now (issue #41): the rule table lives once, in the
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
                         framing: str | None = None) -> str:
    """The proposer prompt for one chunk (EG-R2-1 guidance channel).

    The base is the domain-neutral TRIPLE_PROMPT template filled
    with the family's declared framing (PK-01;
    pilot/profile_config.proposer_framing) or, absent a
    declaration, TRIPLE_FRAMING_DEFAULT. With the framing the
    CUAD profile declares, the construction is byte-identical to
    the pre-PK-01 prompt — pinned by
    scripts/test_profile_wiring.py against the golden string
    captured before the guidance channel existed, and per family
    by scripts/test_domain_boundary.py. With guidance (rendered
    from the corpus profile by
    pilot/profile_config.derive_guidance), the guidance block is
    appended after the base prompt.
    """
    base = TRIPLE_PROMPT.format(
        framing=framing or TRIPLE_FRAMING_DEFAULT,
        maxn=MAX_PER_CHUNK, text=chunk)
    if not guidance:
        return base
    return base + "\n\n" + guidance


def propose_triples_for_doc(doc_id: str, text: str, key: str,
                            model: str = DEFAULT_MODEL,
                            guidance: str = "",
                            framing: str | None = None) -> dict:
    """Raw triple proposals for one document (unguarded).

    Returns {items, cost, calls, errors}; each item carries the
    chunk it came from (+ its normalized form) for the guards.
    When the profile's guidance asks for typed endpoints, items
    also carry subject_type / object_type as proposed (validated
    against the profile's type vocabulary in guard_triples).
    """
    tasks = _chunks(text)

    def work(task):
        idx, chunk = task
        try:
            content, u = _chat(
                _build_triple_prompt(chunk, guidance, framing),
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
                # EG-R2-2: typed endpoints, when the proposer emits
                # them (guidance asks for them). Carried verbatim
                # here; guard_triples validates them against the
                # profile's endpoint-type vocabulary.
                for tk in ("subject_type", "object_type"):
                    tv = it.get(tk)
                    if isinstance(tv, str) and tv.strip():
                        rec[tk] = tv.strip()
                out.append(rec)
        return out, u, err

    items, cost, calls, errors = [], 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for out, u, err in ex.map(work, tasks):
            items.extend(out)
            cost = connections.add_cost(cost, u.get("cost"))
            calls += 1
            if err:
                errors += 1
    return {"items": items,
            "cost": None if cost is None else round(cost, 4),
            "calls": calls,
            "errors": errors, "proposed": len(items)}


def propose_slots_for_doc(doc_id: str, text: str,
                          categories: list[str], key: str,
                          model: str = DEFAULT_MODEL,
                          framing: str | None = None,
                          rules: str | None = None) -> dict:
    """Raw clause-slot fills for one document (unguarded).

    The prompt is the domain-neutral SLOT_PROMPT template filled
    with the family's declared framing + rules block (PK-01;
    pilot/profile_config.slot_framing / slot_rules) or the
    neutral core defaults. With the framing and rules the CUAD
    profile declares, the construction is byte-identical to the
    pre-PK-01 prompt (scripts/test_domain_boundary.py)."""
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
# Deterministic guards (G-04 / G-05 re-scoped to LLM proposals)
# ------------------------------------------------------------------
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
                  theme_attest_chunk: bool = False
                  ) -> tuple[list[dict], list[dict]]:
    """Apply G1 + L0 attestation + role-word + theme guards.

    Returns (kept, blocked). Kept items gain `attest` (the level at
    which both endpoints are attested: quote | chunk | document).
    Blocked items gain `reason` (see module docstring).

    `role_words` is the role-word guard's vocabulary — the
    normalized role nouns, supplied as pack data and resolved
    by profile_config (A2-F2). An endpoint whose normalized
    surface is in the set is blocked (l0_role_word): a bare
    role word is not an identity. None (the default) disables
    the guard; profiles that declare no role-word rules run
    with it off.

    EG-R2-2 typing: a proposed subject_type / object_type survives
    into the kept (and blocked) record only when it belongs to the
    profile's endpoint-type vocabulary (`entity_types`); an
    out-of-vocabulary tag is dropped — the triple itself is NOT
    blocked for a bad tag — and the endpoint continues untyped.
    With entity_types=None (no profile vocabulary), proposed tags
    pass through as-is.

    Two opt-in relaxations (issue #68 phase 2; both default to
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
        qn = norm_ws(q)
        if not qn or (qn not in doc_norm
                      and not (doc_punct is not None
                               and punct_collapse(qn) in doc_punct)):
            blocked.append({**base, "reason": "g1_quote"})
            continue
        sn, on = norm_ws(s), norm_ws(o)
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
