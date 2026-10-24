"""Pair-enumeration proposer — BioRED round 3 (issue #54).

The single-variable alternative to the free generative proposer
(pilot/proposer.py): instead of asking the model to write down the
pairs it notices in a document, enumerate EVERY pair of roster
mentions that co-occurs in a sentence and ask about each pair once.

Design (docs/BIORED_ROUND3.md, pre-registered):

- The roster is an INPUT (round 2's extracted typed mentions,
  materialized as roster.jsonl) — this module never extracts
  mentions; mention quality is the control, not the variable.
- Enumeration is deterministic pure code: the sentence rule, the
  occurrence rule and the pair ordering below are the exact rules
  the ceiling script (scripts/measure_biored_r3_ceiling.py)
  measures with, so the measured ceiling and the run cannot drift
  apart.
- The pair question carries the profile guidance rendered by
  pilot/profile_config.derive_guidance (the #51 channel — the same
  schema text the free proposer sees) and demands one inventory
  relation or NONE, plus the shortest verbatim quote from the
  shared sentence. Answers are shaped exactly like free-proposer
  items so the existing guards (proposer.guard_triples) and every
  downstream stage consume them unchanged.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from . import connections as _connections
from .proposer import _chat, _is_haiku55
from .proposer import norm_ws as _norm_ws_lower
from .reference_profile import PROPOSER_MODEL

DEFAULT_MODEL = PROPOSER_MODEL


def norm_surface(s: str) -> str:
    """Whitespace-normalized surface, case PRESERVED (DP-B6: case
    is part of a gene/protein name). Same normalization the frozen
    scorer (biored_score.norm_ws) applies to gold surfaces."""
    return re.sub(r"\s+", " ", (s or "")).strip()


# ------------------------------------------------------------------
# Sentence rule (declared in docs/BIORED_ROUND3.md §3.2)
# ------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Abbreviation guard (issue #68 phase 2; opt-in, default off =
# byte-identical R3). The plain boundary rule cuts a false
# boundary after any period + space, including the final
# period of a dotted initialism ('U.S. Route 20' -> 'U.S.' +
# 'Route 20 ...') or of a title abbreviation ('St. Louis').
# A mention whose surface spans that false boundary is inert
# under the plain rule — measured at 158 pairs / 170
# endpoints on the #66 dev-500 census. The guard masks
# periods inside dotted initialisms and after a small
# declared set of abbreviation words before splitting, and
# restores them in the returned sentences. Accepted
# tradeoff, counted in the phase-2 census: a sentence that
# genuinely ENDS with an initialism or abbreviation merges
# with the next sentence (a longer judge span; a few
# cross-boundary pairs join the census).
_INITIALISM_RX = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.)+")
_ABBREV_WORD_RX = re.compile(
    r"(?i)(?<![A-Za-z])(?:st|mt|dr|mr|mrs|ms|no|fig|vs|jr|sr)\.")
_MASK = "\x00"


def _mask_abbrev_periods(text: str) -> str:
    chars = list(text)
    for rx in (_INITIALISM_RX, _ABBREV_WORD_RX):
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                if chars[i] == ".":
                    chars[i] = _MASK
    return "".join(chars)


def split_sentences_r3(text: str, *,
                       abbrev_guard: bool = False) -> list[str]:
    """Sentence 0 = the title block (text before the first blank
    line); the remainder is split on sentence boundaries."""
    if abbrev_guard:
        text = _mask_abbrev_periods(text)
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    title = parts[0].strip()
    body = parts[1] if len(parts) > 1 else ""
    out: list[str] = []
    if title:
        out.append(title)
    for seg in _SENT_SPLIT.split(body):
        seg = seg.strip()
        if seg:
            out.append(seg)
    if abbrev_guard:
        out = [s.replace(_MASK, ".") for s in out]
    return out


def mention_surfaces(m: dict) -> list[str]:
    """A mention's resolution surface set: the canonical surface
    first, then any registered aliases (round 4, §4.3 — aliases
    are additional surfaces recorded against the mention; each
    entry is a string or a {"surface": ...} record). Round-3
    mentions carry no aliases and resolve to [surface] exactly
    as before."""
    out = [norm_surface(m["surface"])]
    for a in m.get("aliases") or []:
        s = norm_surface(a["surface"] if isinstance(a, dict) else a)
        if s and s not in out:
            out.append(s)
    return out


def mention_sentence_sets(text: str,
                          mentions: list[dict],
                          abbrev_guard: bool = False
                          ) -> list[set[int]]:
    """For each roster mention (in roster order), the set of
    sentence indices where the mention occurs. A mention occurs
    where ANY of its resolution surfaces occurs (normalized,
    case-preserved substring); with no aliases this is exactly
    the round-3 canonical-surface rule."""
    sents = split_sentences_r3(text, abbrev_guard=abbrev_guard)
    norm_sents = [norm_surface(s) for s in sents]
    out = []
    for m in mentions:
        occ: set[int] = set()
        for ns in mention_surfaces(m):
            occ |= {i for i, s in enumerate(norm_sents)
                    if ns and ns in s}
        out.append(occ)
    return out


def enumerate_pairs(text: str, mentions: list[dict],
                    window: str = "sentence",
                    max_dist=None,
                    abbrev_guard: bool = False) -> list[dict]:
    """Deterministic pair census for one document.

    mentions: roster order (sorted by (normalized surface, type)).

    ONE mechanism, parameterized by the maximum sentence
    distance between an occurrence of one endpoint and an
    occurrence of the other (enumeration-coverage program,
    docs/REDOCRED_ENUMERATION_DESIGN.md):

    - window="sentence" (the default; max_dist=0): a pair is
      enumerated when both mentions occur in at least one
      common sentence; the question context is the FIRST
      shared sentence in document order.
    - window="adjacent" (max_dist=1; pre-declared widening):
      some occurrence of one endpoint is within sentence
      distance 1 of an occurrence of the other; the context
      is that sentence pair in document order (a shared
      sentence when distance is 0).
    - max_dist=k (int >= 2): the same rule with bound k; the
      context is the whole sentence span from the first to
      the second endpoint sentence of the closest occurrence
      pair, joined in document order. For k <= 1 the span
      join reduces to exactly the legacy context strings.
    - max_dist="document": no distance bound — every pair
      of distinct mentions in the document (the coverage
      upper bound).

    Passing max_dist together with a non-default window is
    an error (one parameterization at a time). With
    max_dist=None the legacy window behavior is reproduced
    byte-identically (pinned by
    scripts/test_pair_enumeration_window.py against the
    pre-parameterization census of the #66 main run).

    Returns pairs sorted by (normalized subject, normalized
    object); endpoint order = roster order. Each pair:
    {subject, subject_type, object, object_type, sentence,
     shared_sentences}.
    """
    if max_dist is not None and window != "sentence":
        raise ValueError(
            "pass either window or max_dist, not both: "
            f"window={window!r}, max_dist={max_dist!r}")
    if max_dist is None:
        if window not in ("sentence", "adjacent"):
            raise ValueError(f"unknown pair window: {window!r}")
        bound = 0 if window == "sentence" else 1
    elif max_dist == "document":
        bound = None  # resolved once the sentence count is known
    elif isinstance(max_dist, int) and max_dist >= 0:
        bound = max_dist
    else:
        raise ValueError(f"unknown max_dist: {max_dist!r}")
    sents = split_sentences_r3(text, abbrev_guard=abbrev_guard)
    if bound is None:
        bound = len(sents)
    occ = mention_sentence_sets(text, mentions,
                                abbrev_guard=abbrev_guard)
    pairs: list[dict] = []
    for i in range(len(mentions)):
        for j in range(i + 1, len(mentions)):
            best = None  # (distance, first_idx, second_idx)
            shared = 0
            for a in occ[i]:
                for b in occ[j]:
                    d = abs(a - b)
                    if d == 0:
                        shared += 1
                    if d <= bound:
                        cand = (d, min(a, b), max(a, b))
                        if best is None or cand < best:
                            best = cand
            if best is None:
                continue
            _, lo, hi = best
            context = " ".join(sents[lo:hi + 1])
            pair = {
                "subject": mentions[i]["surface"],
                "subject_type": mentions[i].get("type", ""),
                "object": mentions[j]["surface"],
                "object_type": mentions[j].get("type", ""),
                "sentence": context,
                "shared_sentences": shared,
            }
            # Round 4: when a mention carries registered aliases,
            # its resolution surface set rides along so the pair
            # stage can emit the form attested in the question
            # sentence (design §4.3 emission rule). Absent for
            # round-3 rosters — pair records are unchanged there.
            s_surfs = mention_surfaces(mentions[i])
            o_surfs = mention_surfaces(mentions[j])
            if len(s_surfs) > 1:
                pair["subject_surfaces"] = s_surfs
            if len(o_surfs) > 1:
                pair["object_surfaces"] = o_surfs
            pairs.append(pair)
    pairs.sort(key=lambda p: (norm_surface(p["subject"]),
                              norm_surface(p["object"])))
    return pairs


# ------------------------------------------------------------------
# Pair question
# ------------------------------------------------------------------
PAIR_QUESTION = """

---

Pair question (round 3 pair enumeration — answer for THIS pair only):

Sentence from the document:
---
{sentence}
---
Entity A: {subject} (type: {subject_type})
Entity B: {object} (type: {object_type})

Does the sentence state a relation between Entity A and Entity B?
Answer with ONLY one JSON object, no other text:
- If the sentence states one of the inventory relations between
  A and B: {{"relation": "<exact inventory name>", "quote":
  "<the shortest verbatim phrase from the sentence that states
  the relation>"}}
- Otherwise: {{"relation": "NONE"}}
"""


PAIR_QUESTION_JF1 = """

---

Pair question (round 3 pair enumeration — answer for THIS pair only):

Sentence from the document:
---
{sentence}
---
Entity A: {subject} (type: {subject_type})
Entity B: {object} (type: {object_type})

Does the sentence state a relation between Entity A and Entity B?
Answer with ONLY one JSON object, no other text:
- If the sentence states one of the inventory relations between
  A and B: {{"relation": "<exact inventory name>", "direction":
  "<AB if the relation holds from Entity A to Entity B, BA if it
  holds from Entity B to Entity A>", "quote": "<the shortest
  verbatim phrase from the sentence that states the relation>"}}
- Otherwise: {{"relation": "NONE"}}
"""


def build_pair_prompt(sentence: str, subject: str, subject_type: str,
                      obj: str, object_type: str,
                      guidance: str = "", jf1: bool = False,
                      jf2: bool = False) -> str:
    """Guidance channel first (the #51 discipline: the profile's
    schema text is the pair judge's working instructions, exactly
    as it is the free proposer's), then the pair question.

    jf1=True selects the JF-F1 answer format (issue #66 stage 7
    design, docs/REDOCRED_JF_F1_DESIGN.md section 3): the
    positive branch gains a "direction" field. jf1=False is
    byte-identical to the rounds-3/4 prompt.

    jf2 is accepted for signature symmetry with
    build_pair_prompt_judge's restatement switch (design
    section 14) and changes nothing here: the restatement
    block belongs to the judge templates, and this sentence
    prompt delegates to build_pair_prompt_judge's refusal
    when a caller actually requests it there."""
    tpl = PAIR_QUESTION_JF1 if jf1 else PAIR_QUESTION
    q = tpl.format(sentence=sentence, subject=subject,
                   subject_type=subject_type or "unknown",
                   object=obj,
                   object_type=object_type or "unknown")
    if guidance:
        return guidance + q
    return q.lstrip("\n")


# ------------------------------------------------------------------
# Pair-judge variants (round 5, issue #56; docs/BIORED_ROUND5.md)
#
# The round-5 factorial varied the pair judge on two factors —
# CONTEXT (sentence | abstract) and INSTRUCTION (standard |
# evidence) — and J3 (abstract x evidence) won on the frozen
# slice census (eta 0.5412 -> 0.7412; guard audit upheld 0.70).
# The templates below are the experiment's registered texts,
# transcribed verbatim from scripts/run_biored_r5_judge_exp.py;
# scripts/test_r5_wiring.py byte-compares this module's output
# against the experiment runner's and against the recorded
# template hashes, so the production path cannot drift from the
# measured instrument. With both switches at their defaults
# (sentence / standard) the prompt is build_pair_prompt's —
# byte-identical to rounds 3/4.
# ------------------------------------------------------------------
_HEAD = """

---

Pair question (round 3 pair enumeration — answer for THIS pair only):

"""

_CTX_SENTENCE = """Sentence from the document:
---
{sentence}
---
"""

_CTX_ABSTRACT = """Full text of the document:
---
{text}
---
The sentence in which both entities co-occur:
---
{sentence}
---
"""

_ENTITIES = """Entity A: {subject} (type: {subject_type})
Entity B: {object} (type: {object_type})

"""

# PK-02 (docs/GAP_REGISTER.md): the evidence template names the
# inventory's generic-member relation through the {generic}
# placeholder — the name is profile data
# (pilot/profile_config.pair_judge_generic_relation; BioRED's
# profile declares "Association"), never a core literal. A
# profile that declares none gets the neutral fallback wording.
GENERIC_RELATION_DEFAULT = "generic relation"

_JUDGE_EVIDENCE = """How to judge:
- A relation is stated when the {scope} predicates something of
  the pair — including indirect forms: one entity found in,
  involved in, associated with, linked to, or playing a role in
  the other. Where the {scope} states the pair is related but
  no more specific inventory relation fits, that is what the
  inventory's "{generic}" names: {generic} is a positive
  verdict, not a weaker NONE.
- NONE is itself a claim: it asserts the {scope} does NOT state
  a relation between A and B. Answer NONE only when you can name
  the evidence that is missing.

"""

_ANSWER_STANDARD = """Answer with ONLY one JSON object, no other text:
- If the {scope} states one of the inventory relations between
  A and B: {{"relation": "<exact inventory name>", "quote":
  "<the shortest verbatim phrase from the {scope} that states
  the relation>"}}
- Otherwise: {{"relation": "NONE"}}
"""

_ANSWER_EVIDENCE = """Answer with ONLY one JSON object, no other text:
- If the {scope} states one of the inventory relations between
  A and B: {{"relation": "<exact inventory name>", "quote":
  "<the shortest verbatim phrase from the {scope} that states
  the relation>"}}
- Otherwise: {{"relation": "NONE", "missing": "<what evidence
  the {scope} lacks — e.g. 'the names co-occur but no predicate
  connects them'>"}}
"""

# JF-F1 answer templates (issue #66 stage 7; design section 3.1):
# the positive branch gains the direction field; the NONE branch
# is unchanged (under evidence it keeps its `missing` field).
_ANSWER_STANDARD_JF1 = """Answer with ONLY one JSON object, no other text:
- If the {scope} states one of the inventory relations between
  A and B: {{"relation": "<exact inventory name>", "direction":
  "<AB if the relation holds from Entity A to Entity B, BA if it
  holds from Entity B to Entity A>", "quote": "<the shortest
  verbatim phrase from the {scope} that states the relation>"}}
- Otherwise: {{"relation": "NONE"}}
"""

_ANSWER_EVIDENCE_JF1 = """Answer with ONLY one JSON object, no other text:
- If the {scope} states one of the inventory relations between
  A and B: {{"relation": "<exact inventory name>", "direction":
  "<AB if the relation holds from Entity A to Entity B, BA if it
  holds from Entity B to Entity A>", "quote": "<the shortest
  verbatim phrase from the {scope} that states the relation>"}}
- Otherwise: {{"relation": "NONE", "missing": "<what evidence
  the {scope} lacks — e.g. 'the names co-occur but no predicate
  connects them'>"}}
"""

# Stage-12 (J3F2) inventory restatement (design section
# 14): the relation inventory, names only, restated
# immediately before the answer format. The names are
# data ({names} = the inventory names in payload order,
# joined ", "); the rule text is registered verbatim
# in section 14.
_RESTATEMENT_JF2 = """The relation inventory — the ONLY names you may answer with:
{names}

Answer with EXACTLY one of these names, spelled exactly as
listed. If none of these names fits the pair exactly, answer
NONE. Do not invent, rename, or approximate a relation name.

"""


def _inventory_restatement(inventory_names) -> str:
    """The section-14 restatement block for the given
    inventory names (payload order)."""
    if not inventory_names:
        raise ValueError("jf2 restatement requires the "
                         "inventory names (data)")
    return _RESTATEMENT_JF2.format(
        names=", ".join(inventory_names))


JUDGE_CONTEXTS = ("sentence", "abstract")
JUDGE_INSTRUCTIONS = ("standard", "evidence")


def build_pair_prompt_judge(sentence: str, subject: str,
                            subject_type: str, obj: str,
                            object_type: str, guidance: str = "",
                            context_mode: str = "sentence",
                            instruction: str = "standard",
                            text: str = "",
                            generic_relation: str | None = None,
                            jf1: bool = False,
                            restat_inventory: list[str] | None = None
                            ) -> str:
    """The pair question under a round-5 judge variant.

    Defaults (sentence x standard) delegate to build_pair_prompt
    unchanged. Any other combination assembles the experiment's
    registered template: the factors change ONLY their own block
    (the context block for CONTEXT; the judging block + the NONE
    branch for INSTRUCTION). `text` is the full document text,
    required by the abstract context. `generic_relation` is the
    inventory's generic-member name for the evidence block
    (PK-02 — profile data; with the name BioRED's profile
    declares, the assembled prompt is byte-identical to the
    round-5 experiment's registered template).

    jf1=True selects the JF-F1 answer format (design section 3)
    in whichever template the variant assembles; jf1=False is
    byte-identical to the pre-JF-F1 prompts.

    restat_inventory (design section 14, cell J3F2): when a
    list of inventory names is passed, the section-14
    restatement block is inserted immediately before the
    answer format; None (the default) leaves every variant
    byte-identical to before. The block belongs to the
    assembled judge templates — the sentence x standard
    delegation has no place for it and refuses it loudly."""
    if context_mode not in JUDGE_CONTEXTS:
        raise ValueError(f"unknown pair judge context: "
                         f"{context_mode!r}")
    if instruction not in JUDGE_INSTRUCTIONS:
        raise ValueError(f"unknown pair judge instruction: "
                         f"{instruction!r}")
    if context_mode == "sentence" and instruction == "standard":
        if restat_inventory is not None:
            raise ValueError("restat_inventory applies to the "
                             "assembled judge templates, not "
                             "sentence x standard")
        return build_pair_prompt(sentence, subject, subject_type,
                                 obj, object_type, guidance, jf1=jf1)
    scope = "sentence" if context_mode == "sentence" else "text"
    ctx = _CTX_SENTENCE if context_mode == "sentence" \
        else _CTX_ABSTRACT
    judge = _JUDGE_EVIDENCE if instruction == "evidence" else ""
    if jf1:
        answer = _ANSWER_EVIDENCE_JF1 if instruction == "evidence" \
            else _ANSWER_STANDARD_JF1
    else:
        answer = _ANSWER_EVIDENCE if instruction == "evidence" \
            else _ANSWER_STANDARD
    question = "Does the sentence state" if scope == "sentence" \
        else "Does the document state"
    judge_block = ("\n" + judge.strip("\n") + "\n\n") if judge \
        else ""
    restat = _inventory_restatement(restat_inventory) \
        if restat_inventory is not None else ""
    body = (_HEAD + ctx + _ENTITIES
            + question + " a relation between Entity A and "
              "Entity B?\n"
            + judge_block + restat + answer)
    body = body.format(sentence=sentence, text=text,
                       subject=subject,
                       subject_type=subject_type or "unknown",
                       object=obj,
                       object_type=object_type or "unknown",
                       scope=scope,
                       generic=generic_relation
                       or GENERIC_RELATION_DEFAULT)
    if guidance:
        return guidance + body
    return body.lstrip("\n")


def _answer_obj(content: str) -> dict | None:
    """The pair judge's answer as a dict, or None when the raw
    content is not a JSON object (code-fence stripped first)."""
    txt = (content or "").strip()
    if txt.startswith("```"):
        lines = txt.split("\n")
        txt = "\n".join(lines[1:-1] if lines[-1].strip() == "```"
                        else lines[1:])
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_pair_answer(content: str,
                       relation_inventory: set[str]) -> tuple[str, str, str]:
    """-> (relation_or_NONE, quote, outcome). Outcome is one of
    ok / none / bad_json / out_of_inventory — recorded, never
    silently dropped."""
    obj = _answer_obj(content)
    if obj is None:
        return "NONE", "", "bad_json"
    rel = str(obj.get("relation") or "").strip()
    if not rel or rel.upper() == "NONE":
        return "NONE", "", "none"
    if rel not in relation_inventory:
        return "NONE", "", "out_of_inventory"
    quote = str(obj.get("quote") or "").strip()
    return rel, quote, "ok"


DIRECTIONS = ("AB", "BA")


def _parse_pair_answer_jf1(content: str,
                           relation_inventory: set[str]
                           ) -> tuple[str, str, str, str | None]:
    """JF-F1 successor to _parse_pair_answer (issue #66 stage 7;
    docs/REDOCRED_JF_F1_DESIGN.md section 3, rules R1/R4).

    -> (relation_or_NONE, quote, outcome, direction|None).
    Outcomes extend the legacy set with bad_direction. The
    inventory check runs FIRST (an unknown name is
    out_of_inventory regardless of direction); a positive
    answer whose direction is missing or not in {"AB", "BA"}
    fails closed as bad_direction — no candidate, counted,
    exactly as bad_json is treated. A direction field on a
    NONE answer is ignored: NONE asserts no fact, so there
    is nothing to orient, and the NONE branch stays
    parse-compatible with the legacy format."""
    obj = _answer_obj(content)
    if obj is None:
        return "NONE", "", "bad_json", None
    rel = str(obj.get("relation") or "").strip()
    if not rel or rel.upper() == "NONE":
        return "NONE", "", "none", None
    if rel not in relation_inventory:
        return "NONE", "", "out_of_inventory", None
    direction = obj.get("direction")
    if direction not in DIRECTIONS:
        return "NONE", "", "bad_direction", None
    quote = str(obj.get("quote") or "").strip()
    return rel, quote, "ok", direction


def jf1_swaps(direction: str | None, relation: str,
              symmetric_relations: set[str] | None) -> bool:
    """The single JF-F1 orientation mechanism (design rule R2):
    emission swaps the posed endpoints iff direction is "BA"
    and the relation is not declared symmetric (orientation
    is a no-op there). Declared inverse pairs are never
    consulted (rule R3): (name, direction) is a complete,
    non-redundant encoding, so no second mechanism can
    double-encode a contradictory direction."""
    return direction == "BA" \
        and relation not in (symmetric_relations or set())


def _parse_pair_missing(content: str) -> str:
    """The judge's stated evidence gap on a NONE answer — the
    evidence instruction (round 5) requires every NONE to carry
    one; round 8's FIX-P persists it (design §4.4). "" for
    positive answers, unparseable content, and NONE answers
    carrying no missing text (the standard instruction never
    asks for one)."""
    obj = _answer_obj(content)
    if obj is None:
        return ""
    rel = str(obj.get("relation") or "").strip()
    if rel and rel.upper() != "NONE":
        return ""
    return str(obj.get("missing") or "").strip()


def _occurrence_spans(haystack: str, needle: str) -> list[tuple[int, int]]:
    out = []
    i = haystack.find(needle)
    while i >= 0:
        out.append((i, i + len(needle)))
        i = haystack.find(needle, i + 1)
    return out


def attested_form(surfaces: list[str], context: str) -> str:
    """Round-4 emission rule (design §4.3): the surface form a
    candidate carries for an endpoint is the form attested in
    the question sentence when one is, else the canonical
    surface. A form is attested when it has a STANDALONE
    occurrence in the sentence — an occurrence not strictly
    inside an occurrence of a longer form in the set (the
    design's A3 attestation notion: "NIS" inside the text's
    "NIS gene" attests "NIS gene", not "NIS"). Priority: the
    canonical surface when it is attested standalone; else the
    earliest-occurring attested alias (ties: longer form, then
    lexicographic); else the canonical surface. surfaces[0] is
    the canonical surface."""
    canon = surfaces[0]
    nctx = norm_surface(context)
    spans = {s: _occurrence_spans(nctx, s) for s in surfaces}
    attested: list[tuple[int, str]] = []
    for s in surfaces:
        longer = [o for o in surfaces if len(o) > len(s)]
        for st, en in spans[s]:
            if not any(ls <= st and en <= le
                       for o in longer for ls, le in spans[o]):
                attested.append((st, s))
                break
    if not attested:
        return canon
    for _pos, s in attested:
        if s == canon:
            return canon
    attested.sort(key=lambda ps: (ps[0], -len(ps[1]), ps[1]))
    return attested[0][1]


def propose_pairs_for_doc(doc_id: str, text: str,
                          mentions: list[dict], key: str,
                          model: str = DEFAULT_MODEL,
                          guidance: str = "",
                          relation_inventory: set[str] | None = None,
                          window: str = "sentence",
                          judge_context: str = "sentence",
                          judge_instruction: str = "standard",
                          generic_relation: str | None = None,
                          jf1: bool = False,
                          symmetric_relations: set[str] | None = None,
                          restat_inventory: list[str] | None = None,
                          abbrev_guard: bool = False,
                          evidence_sentence_in_items: bool = False
                          ) -> dict:
    """Enumerate pairs for one doc and ask about each (one call).

    judge_context / judge_instruction select the round-5 pair
    judge variant (issue #56); at their defaults the prompts are
    byte-identical to rounds 3/4.

    jf1=True runs the JF-F1 answer format (issue #66 stage 7):
    positive answers carry a direction field, emission swaps
    endpoints iff jf1_swaps says so against
    `symmetric_relations` (the profile payload's declared
    symmetric set, read as data — design section 3.3), and the
    FIX-P verdict record gains the answered `direction`. With
    jf1=False the prompts, parse, emission and verdict records
    are exactly the pre-JF-F1 behavior.

    restat_inventory forwards the section-14 (J3F2)
    inventory names to the judge builder; None leaves every
    prompt byte-identical.

    abbrev_guard forwards the issue-#68 sentence-splitter
    guard to enumeration (default off = the frozen R3 split).
    evidence_sentence_in_items (default off) adds the pair's
    sentence to each ok item under `evidence_sentence`, for
    the guard_triples theme-attestation span (issue #68
    phase 2); with it off, items are byte-identical.

    Returns {items, cost, calls, errors, enumerated, answered,
    outcomes, verdicts} — items shaped like proposer items
    (subject / relation / object / quote + subject_type /
    object_type) so guard_triples consumes them unchanged. An
    answered item's quote is the model's quote; if the model
    affirms a relation without a usable quote, the shared
    sentence itself is the quote (it is verbatim by
    construction) — the guards still apply verbatim/attestation
    checks downstream. verdicts carries one FIX-P record per
    enumerated pair (round 8, issue #59; design §4.4): pair
    identity, the parse-level outcome and its coarse verdict
    (ok / none / error), relation, the model's quote, and — for
    NONE verdicts under the evidence instruction — the judge's
    stated `missing` evidence gap. Verdicts are computed from
    the raw answers before any guard runs; they never feed the
    items.
    """
    inventory = set(relation_inventory or set())
    pairs = enumerate_pairs(text, mentions, window=window,
                            abbrev_guard=abbrev_guard)

    def work(pair):
        direction = None
        try:
            content, u = _chat(
                build_pair_prompt_judge(
                    pair["sentence"], pair["subject"],
                    pair["subject_type"], pair["object"],
                    pair["object_type"], guidance,
                    context_mode=judge_context,
                    instruction=judge_instruction, text=text,
                    generic_relation=generic_relation, jf1=jf1,
                    restat_inventory=restat_inventory),
                model, 4000 if _is_haiku55(model) else 1000, key)
            if jf1:
                rel, quote, outcome, direction = \
                    _parse_pair_answer_jf1(content, inventory)
            else:
                rel, quote, outcome = _parse_pair_answer(
                    content, inventory)
            missing = _parse_pair_missing(content)
            err = None
        except Exception as e:
            rel, quote, outcome, u, err = "NONE", "", "error", {}, str(e)[:200]
            missing = ""
            direction = None
        item = None
        if outcome == "ok":
            s_form = attested_form(
                pair.get("subject_surfaces") or [pair["subject"]],
                pair["sentence"])
            o_form = attested_form(
                pair.get("object_surfaces") or [pair["object"]],
                pair["sentence"])
            s_type, o_type = pair["subject_type"], pair["object_type"]
            s_canon, o_canon = pair["subject"], pair["object"]
            if jf1 and jf1_swaps(direction, rel, symmetric_relations):
                s_form, o_form = o_form, s_form
                s_type, o_type = o_type, s_type
                s_canon, o_canon = o_canon, s_canon
            item = {
                "subject": s_form,
                "relation": rel,
                "object": o_form,
                "quote": quote or pair["sentence"],
                "subject_type": s_type,
                "object_type": o_type,
                "subject_canonical": s_canon,
                "object_canonical": o_canon,
                "chunk_idx": 0, "chunk_norm": _norm_ws_lower(text)}
            if evidence_sentence_in_items:
                item["evidence_sentence"] = pair["sentence"]
        # FIX-P verdict record (round 8, issue #59; design §4.4):
        # the pair's identity is the roster (canonical) endpoint
        # pair the question was asked about; the quote is the
        # model's own (an ok item may substitute the sentence
        # downstream — the verdict keeps what the judge said).
        verdict = {
            "doc_id": doc_id,
            "subject": pair["subject"],
            "subject_type": pair["subject_type"],
            "object": pair["object"],
            "object_type": pair["object_type"],
            "outcome": outcome,
            "verdict": ("ok" if outcome == "ok" else
                        "error" if outcome == "error" else "none"),
            "relation": rel,
            "quote": quote,
            "missing": missing,
            "error": err,
        }
        if jf1:
            # Rule R5: the answered direction rides verbatim
            # (AB / BA on ok answers; None otherwise) so
            # decompositions can split flips by judged
            # orientation without re-running.
            verdict["direction"] = direction
        return item, u, outcome, err, verdict

    items, verdicts, cost, calls, errors = [], [], 0.0, 0, 0
    outcomes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for item, u, outcome, err, verdict in ex.map(work, pairs):
            calls += 1
            cost = _connections.add_cost(cost, u.get("cost"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if err:
                errors += 1
            verdicts.append(verdict)
            if item:
                items.append(item)
    # Deterministic candidate order (roster-pair order), regardless
    # of call completion order.
    items.sort(key=lambda it: (norm_surface(it["subject"]),
                               norm_surface(it["relation"]),
                               norm_surface(it["object"])))
    return {"items": items,
            "cost": None if cost is None else round(cost, 4),
            "calls": calls,
            "errors": errors, "enumerated": len(pairs),
            "answered": len(items), "outcomes": outcomes,
            "verdicts": verdicts}
