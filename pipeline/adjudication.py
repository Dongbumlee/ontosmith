"""Pass D adjudication and the ship-gate audit (0.2.1).

The two-pass build separates proposal from judgment about the
proposal. Pass C (pipeline/proposer.py, driven by
stage_candidates) proposes candidates; this module is Pass D:
an adjudicator that did NOT write the candidates decides each
one under the eight proposal-discipline clauses, in the order
keep -> retype -> veto, against the full excerpt the candidate
was drawn from (R2), with quote completeness (R5) as a shipping
condition and mandatory retype logging on every veto (R4).

Two consumers:

- stage_adjudicate (between CANDIDATES and GATES) calls
  adjudicate_candidates() over the whole candidate set and
  persists every decision — vetoes included — as the batch's
  adjudication.jsonl.
- stage_ship_gate (between ROUTE and WRITE) calls
  audit_store_row() on every store-bound triple: an
  INDEPENDENT audit (a differently-worded prompt, no Pass D
  context) of quote completeness only. A failure gets one
  retype and one re-audit; a second failure excludes the row.

The judge is reached only through a chat transport callable,
``chat_fn(prompt, model) -> (text, cost_usd | None)`` — the same
abstraction shape the gates use for decide_fn — so tests drive
both stages with fake transports and no live model is needed
to exercise any of the deterministic machinery here.

The deterministic layer enforces the R4/R5 invariants on judge
output before anything ships: a kept or retyped decision must
carry a quote that is a verbatim substring of the excerpt,
names both endpoints (the document-subject exception for
clause-slot candidates, whose subject is the document itself),
and attests the relation surface. A mechanical repair —
extending the quote to the attesting span in the excerpt — is
applied when one exists; an unrepairable decision becomes a
veto with the full R4 record. The judge decides semantics;
the gate is deterministic about the invariant.
"""
from __future__ import annotations

import json
import re

from .proposer import norm_ws, quote_window

# The adjudication excerpt: the candidate's quote with a wide
# window of the document around it. R2 makes the excerpt the
# truth boundary; the proposer's own judge context is ±500
# chars (proposer.WINDOW) — adjudication reads wider, ±2000,
# because retype decisions legitimately draw on the statement
# around the quote (the Phase-1B compliance lesson: never veto
# from a truncated tail alone).
EXCERPT_HALF = 2000

# Relation surfaces that state no connection (R4's garbled
# list, from the adjudication procedure: "is the", "is a",
# "is an", a bare "is" where the text states a fuller verb
# phrase). A keep carrying one of these is invalid.
_GARBLED_RELATIONS = {
    "is", "is the", "is a", "is an", "are", "are the",
    "was", "were", "has", "have", "had", "does", "do", "did",
}

# Tokens that carry no relational content for the attestation
# check: auxiliaries/modals (mirroring relation_map's set) and
# articles/prepositions that the quote may phrase differently.
_REL_STOPWORDS = {
    "shall", "will", "must", "may", "might", "is", "are", "was",
    "were", "has", "have", "had", "does", "do", "did", "can",
    "could", "should", "would", "be", "been", "the", "a", "an",
}

ADJUDICATION_INSTRUCTIONS = """\
You are Pass D. You adjudicate extraction candidates you did NOT write.
For each candidate you are given its id, subject, relation, object,
evidence quote, and the FULL excerpt text the candidate was extracted
from. The excerpt as a whole is the truth boundary (R2): judge against
all of it, never from the quote alone.

Decision order — keep, then RETYPE, then veto (R1):
1. keep — the claim as typed is true against the full excerpt AND its
   quote passes R5 below (if the claim is true but the quote alone
   does not state it fully, apply remedy extend_quote and still keep).
2. retype — the candidate is loosely or wrongly typed, BUT a truthful
   typing exists using what the excerpt states. Retype to the truthful
   form: use the verb phrase the text itself uses, re-attach a dropped
   qualifier into the claim, resolve an endpoint to the entity the text
   points to, correct a demonym inference to the stated form. A
   loosely-typed-but-true candidate MUST be retyped, never vetoed.
   R5 narrowing (retype_to_quote) is also a retype.
3. veto — LAST RESORT, only when NO truthful typing of the candidate
   exists: the stated fact itself is fabricated, an endpoint is
   manufactured from content the excerpt does not state, a truncated
   word was completed by invention. If you can write any truthful
   version of the candidate from the excerpt, it is a retype.

R5 — quote completeness. A candidate may be kept or retyped only if
its evidence quote, read alone, states the FULL typed claim —
subject, relation, and object complete, including the object's bounds
and qualifiers. The two failure patterns: a truncated quote (it cuts
off before the typed claim's object completes) and object overreach
(the object or subject carries bounds, qualifiers, values, or
endpoints the quote does not state — truth elsewhere in the excerpt
does not count). When a candidate fails R5, apply remedies in order:
1. extend_quote — extend the quote to the complete statement span in
   the excerpt, so the quote alone states the claim. The extended
   quote must be verbatim excerpt text (an exact contiguous
   substring). Record the extended text as final_quote.
2. retype_to_quote — narrow the claim down to exactly what the
   (original or extended) quote states, no more. Decision = retype.
3. veto — only when neither extension nor narrowing yields a
   truthful, quote-complete claim.
Record the R5 remedy for EVERY decision as r5_remedy: "none" (the
original quote already states the full claim), "extend_quote",
"retype_to_quote", or "veto" (decision is veto). If a candidate is
retyped for clause reasons AND its quote needed extension, decision =
retype and r5_remedy = "extend_quote". final_quote is REQUIRED for
every keep/retype: the original quote when r5_remedy is "none" or
"retype_to_quote" (unless you also extended), the extended verbatim
text when r5_remedy is "extend_quote". For veto, final_quote = the
original quote.

R4 — mandatory log fields. Every veto MUST include:
- retype_attempt: the best truthful typing you could construct from
  the excerpt, as {"subject", "relation", "object"} — OR, if no
  typing can be constructed at all, {"unconstructible": "<the
  endpoint or relation that cannot be formed, and from what>"}.
- retype_failure: {"clause": <1-8>, "missing": "<what the text lacks
  that makes even the attempted typing untruthful>"}.
A veto without both fields is procedurally invalid and will be
converted to the recorded retype — so if your recorded attempt is in
fact truthful, decide retype instead. For an R5-driven veto, the
failure records why neither quote extension nor narrowing could
produce a quote-complete truthful claim; use clause 1 unless another
clause is the real driver.
A keep with a garbled relation is invalid: if the relation string is
a fragment that states no connection — "is the", "is a", "is an", or
a bare "is" where the text states a fuller verb phrase ("is titled",
"is defined as", "is known as", "is headquartered in") — you may NOT
keep it; retype to the relational phrase the text states, or veto.

The 8 clauses (adjudication standards):
1. No relation upgrade. A relation is supported only when the excerpt
   states that relation between these endpoints. A neighbouring verb,
   role, or association is not the relation it resembles. BUT (R2):
   loose relation phrasing over a connection the excerpt genuinely
   states = retype to the stated verb phrase, not veto.
2. No transfer across entities. A fact stated about one entity never
   transfers to a related one. Origin/base phrases state origin or
   base only.
3. No inference from a demonym. Never derive an entity or value from
   a nationality adjective. But where the text itself expresses a
   value in demonym form, the stated value IS the adjective — report
   it in that form.
4. Containment needs its own statement. Emit located in / contains /
   part of / country / continent only when the excerpt states that
   containment itself. Co-occurrence, proximity, topic never states
   it. Direction as stated.
5. Keep the full name form. Endpoints keep distinguishing parts
   (Jr./Sr., regnal numerals, fuller name forms). Never a bare
   surname or cut fragment when the text carries the fuller form.
6. Same name is not same entity. Attach the fact only to the referent
   the statement is about; if the text does not resolve which, the
   candidate cannot stand as typed (retype to the resolved referent
   if the excerpt resolves it elsewhere; veto only if no truthful
   typing exists).
7. Keep the qualifier — a bounded fact is a different fact. If the
   text states a fact only within a scope (formerly, originally, ex-,
   visiting, a period, one edition, one market/territory), the claim
   must carry that bound; if the typed form drops it, retype to
   attach the bound. The qualifier sitting in the quote does not
   count — the bound belongs in the claim.
8. Fill each slot from its own statement. Never fill a slot by
   copying another slot's filler. Where the text reaches a slot's
   filler through a reference (pronoun, appositive, by-phrase,
   relative clause, attribution), resolve the reference and fill the
   slot with the entity the text points to. Where the sentence states
   no second participant, do not manufacture one. Reflexive criterion
   (R3): where the text's statement is itself reflexive — a term
   defined by its own name, a title naming the document, a naming
   statement — the repetition IS the content of the statement, not an
   echo error. Such candidates are keep (or retype to normalise an
   endpoint), never veto for echo alone.

Output: a JSON array with one object per candidate, deciding EVERY
candidate — no skips:
{"candidate_id": "<id>", "decision": "keep"|"retype"|"veto",
 "final_subject": "...", "final_relation": "...",
 "final_object": "...", "final_quote": "...",
 "r5_remedy": "none"|"extend_quote"|"retype_to_quote"|"veto",
 "clause": <1-8 or null>, "reason": "<one line>",
 "retype_attempt": <object or null>,
 "retype_failure": <object or null>}
For keep, the finals are the candidate as typed. For retype, the
finals are your truthful retyping. For veto, the finals are the
candidate as typed, r5_remedy = "veto", and retype_attempt +
retype_failure are MANDATORY. "clause" is the primary clause that
drove a retype/veto, else null. Respond with the JSON array only."""

AUDIT_INSTRUCTIONS = """\
You are an independent verifier. You did not extract this triple and
you have no stake in it shipping. You are given one triple, its
evidence quote, and the excerpt the quote was drawn from.

The triple may ship only if its evidence quote, read alone, states
the FULL typed claim — subject, relation, and object complete,
including the object's bounds and qualifiers. Check exactly three
things:
1. The quote is a verbatim substring of the excerpt (not cut
   mid-phrase, not paraphrased). A quote that ends before the
   claim's object completes is a truncated quote.
2. The quote names the subject and the object, completely — no
   bounds, qualifiers, values, or endpoints in the typed claim
   that the quote does not state (they may be true elsewhere in
   the excerpt; that does not count).
3. The quote states the relation between the endpoints as typed —
   not a neighbouring association, not the reverse direction.

Respond with one JSON object only:
{"verdict": "pass"|"fail",
 "failure_class": null | "truncated_quote" | "subject_not_named" |
   "object_overreach" | "relation_unattested" | "other",
 "reason": "<one line>",
 "corrected": null | {"subject": "...", "relation": "...",
   "object": "...", "quote": "<verbatim excerpt text that states
   the corrected claim completely>"}}
Propose a corrected triple only when a truthful, quote-complete
typing of the same fact exists in the excerpt; otherwise null."""


# ------------------------------------------------------------------
# Evidence machinery (deterministic)
# ------------------------------------------------------------------

def excerpt_for(text: str, quote: str) -> str:
    """The adjudication excerpt for a candidate: its quote with
    a wide window of the document around it (R2's truth
    boundary for the decision)."""
    if not quote:
        return text[: 2 * EXCERPT_HALF]
    return quote_window(text, quote, half=EXCERPT_HALF)


def _span_of(text: str, quote: str) -> tuple[int, int] | None:
    """The (start, end) char span of quote in text, or None.

    Raw find first; then a whitespace-normalized find (the
    quote may differ from the text only in whitespace runs —
    the same tolerance quote_window applies)."""
    if not quote:
        return None
    pos = text.find(quote)
    if pos >= 0:
        return pos, pos + len(quote)
    norm_chars: list[str] = []
    idx: list[int] = []
    prev_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_ws:
                norm_chars.append(" ")
                idx.append(i)
            prev_ws = True
        else:
            norm_chars.append(ch)
            idx.append(i)
            prev_ws = False
    norm_text = "".join(norm_chars)
    norm_quote = re.sub(r"\s+", " ", quote)
    pos = norm_text.find(norm_quote)
    if pos < 0:
        return None
    end = pos + len(norm_quote)
    return idx[pos], idx[end - 1] + 1


def quote_is_verbatim(excerpt: str, quote: str) -> bool:
    """True when the quote is an exact substring of the excerpt
    (up to whitespace-run differences)."""
    return _span_of(excerpt, quote) is not None


def _sentence_expand(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand a char span to the boundaries of the sentence(s)
    it touches: left to just after the previous sentence-ending
    punctuation, right to the next sentence-ending punctuation
    (inclusive)."""
    left = start
    while left > 0 and text[left - 1] not in ".!?\n":
        left -= 1
    while left < start and text[left] in " \t":
        left += 1
    right = end
    while right < len(text) and text[right] not in ".!?\n":
        right += 1
    if right < len(text):
        right += 1
    return left, right


def relation_garbled(relation: str) -> bool:
    """R4's garbled-relation test: the surface states no
    connection — a bare auxiliary/article fragment, or nothing
    left once auxiliaries are stripped."""
    norm = norm_ws(relation)
    if norm in _GARBLED_RELATIONS:
        return True
    content = [t for t in re.findall(r"[a-z0-9']+", norm)
               if t not in _REL_STOPWORDS]
    return not content


def relation_attested(relation: str, quote: str) -> bool:
    """True when the relation's content words all appear in the
    quote — the surface the quote actually states. A declared
    slot category (a label, not a verb phrase) is attested by
    construction and checked by the caller's method test, not
    here."""
    if relation_garbled(relation):
        return False
    quote_tokens = set(re.findall(r"[a-z0-9']+", norm_ws(quote)))
    content = [t for t in re.findall(r"[a-z0-9']+", norm_ws(relation))
               if t not in _REL_STOPWORDS]
    return all(t in quote_tokens for t in content)


def _endpoint_named(endpoint: str, quote: str) -> bool:
    return bool(endpoint) and norm_ws(endpoint) in norm_ws(quote)


def quote_failure_class(quote: str, subject: str, relation: str,
                        object_: str, excerpt: str,
                        *, doc_subject: bool = False) -> str | None:
    """The deterministic R5 audit of one (triple, quote) pair
    against its excerpt. Returns None when the quote is
    complete, else the failure class:

    - "truncated_quote": the quote is not a verbatim substring
      of the excerpt (cut or paraphrased — it cannot be read
      against the text at all);
    - "subject_not_named": the quote does not name the subject
      (waived when doc_subject — a clause-slot candidate whose
      subject is the document itself, attested by the document
      rather than the quote);
    - "object_overreach": the quote does not state the object
      completely;
    - "relation_unattested": the relation surface is garbled
      or its content words are absent from the quote (waived
      for clause-slot candidates, whose relation is a declared
      category label, not a stated verb phrase).
    """
    if not quote or not quote_is_verbatim(excerpt, quote):
        return "truncated_quote"
    if not doc_subject and not _endpoint_named(subject, quote):
        return "subject_not_named"
    if not _endpoint_named(object_, quote):
        return "object_overreach"
    if not doc_subject and not relation_attested(relation, quote):
        return "relation_unattested"
    return None


def find_extending_span(excerpt: str, quote: str, subject: str,
                        object_: str,
                        *, doc_subject: bool = False) -> str | None:
    """The extend_quote remedy, mechanically: widen the quote
    within the excerpt — sentence by sentence — until the span
    states the full claim (both endpoints named), and return
    the verbatim span. None when no span in the excerpt states
    it. The widened span is always an exact substring of the
    excerpt."""
    span = _span_of(excerpt, quote)
    if span is None:
        return None
    start, end = _sentence_expand(excerpt, span[0], span[1])
    for _ in range(8):
        text_span = excerpt[start:end]
        if (doc_subject or _endpoint_named(subject, text_span)) \
                and _endpoint_named(object_, text_span):
            return text_span
        grew = False
        # Grow one sentence to the left.
        if start > 0:
            new_start, _ = _sentence_expand(
                excerpt, max(0, start - 2), start)
            if new_start < start:
                start = new_start
                grew = True
        # Grow one sentence to the right.
        if end < len(excerpt):
            _, new_end = _sentence_expand(excerpt, end,
                                          min(len(excerpt), end + 2))
            if new_end > end:
                end = new_end
                grew = True
        if not grew:
            break
    text_span = excerpt[start:end]
    if (doc_subject or _endpoint_named(subject, text_span)) \
            and _endpoint_named(object_, text_span):
        return text_span
    return None


# ------------------------------------------------------------------
# Pass D: prompt, parsing, enforcement
# ------------------------------------------------------------------

def build_adjudication_prompt(items: list[dict]) -> str:
    """One adjudication call's prompt: the procedure, then one
    block per candidate (id, triple, quote, full excerpt).
    ``items`` are dicts with candidate_id / subject / relation /
    object / quote / excerpt."""
    blocks = []
    for it in items:
        blocks.append(
            f"Candidate {it['candidate_id']}:\n"
            f"Triple: ({it['subject']} | {it['relation']} | "
            f"{it['object']})\n"
            f"Evidence quote: {it['quote']}\n"
            f"Excerpt:\n{it['excerpt']}")
    return (ADJUDICATION_INSTRUCTIONS + "\n\n"
            + "\n\n".join(blocks))


def parse_decision_array(text: str) -> list[dict]:
    """Parse the adjudicator's JSON array. Strict: a response
    that does not parse is a stage failure (raised), never a
    silent pass — an unjudged candidate must not slip through
    as if decided."""
    try:
        items = json.loads(text)
    except Exception as e:
        raise RuntimeError(
            f"adjudication response is not valid JSON: {e}") from e
    if not isinstance(items, list) or not all(
            isinstance(x, dict) for x in items):
        raise RuntimeError(
            "adjudication response is not a JSON array of objects")
    return items


def add_cost(total, cost):
    """None-aware cost accumulation (any unmetered call makes
    the total unknown), mirroring the connection layer's rule."""
    if total is None or cost is None:
        return None
    return total + cost


def _is_doc_subject(candidate: dict) -> bool:
    """Clause-slot candidates carry the document itself as
    subject (stage_candidates fills subject = document name):
    the quote attests the slot VALUE, and the relation is the
    declared category label. Their R5 form is therefore the
    doc-subject form."""
    return candidate.get("method") == "clause-llm"


def _veto_record(candidate: dict, candidate_id: str, *,
                 clause, reason: str,
                 retype_attempt, retype_failure,
                 enforcement: str | None) -> dict:
    return {
        "candidate_id": candidate_id,
        "doc_id": candidate.get("doc_id"),
        "method": candidate.get("method"),
        "decision": "veto",
        "final_subject": candidate.get("subject"),
        "final_relation": candidate.get("relation"),
        "final_object": candidate.get("object"),
        "final_quote": candidate.get("quote"),
        "r5_remedy": "veto",
        "clause": clause,
        "reason": reason,
        "retype_attempt": retype_attempt,
        "retype_failure": retype_failure,
        "enforcement": enforcement,
        "original": {
            "subject": candidate.get("subject"),
            "relation": candidate.get("relation"),
            "object": candidate.get("object"),
            "quote": candidate.get("quote")},
    }


def enforce_decision(raw: dict, candidate: dict, candidate_id: str,
                     excerpt: str) -> dict:
    """Validate one raw adjudicator decision against the
    deterministic R4/R5 invariants and produce the recorded
    decision row.

    The judge's semantics stand unless an invariant fails:
    - keep/retype: the final quote must be verbatim in the
      excerpt, name both endpoints, and attest the relation.
      A quote that fails only on completeness is repaired
      mechanically (extend_quote to the attesting span); a
      decision that cannot be repaired becomes a veto with
      the full R4 record.
    - a garbled-relation keep is invalid (R4): it is treated
      as an unrepairable decision unless the same output
      retyped it — a keep never ships a garbled relation.
    - veto: R4's mandatory fields are enforced. A veto whose
      recorded retype attempt is itself truthful and
      quote-complete is converted to that retype (the
      procedure's own rule); a veto missing the fields stands,
      with the layer recording the attempt as unconstructible
      and the failure naming the procedural defect — the
      record is complete and the defect is visible, never
      silently filled with an invented typing.
    """
    doc_subject = _is_doc_subject(candidate)
    decision = str(raw.get("decision") or "").strip().lower()
    clause = raw.get("clause")
    if clause is not None and not (isinstance(clause, int)
                                   and 1 <= clause <= 8):
        clause = None
    reason = str(raw.get("reason") or "")

    if decision == "veto":
        attempt = raw.get("retype_attempt")
        failure = raw.get("retype_failure")
        # R4 conversion: a recorded attempt that is itself a
        # truthful, quote-complete typing converts the veto to
        # that retype.
        if isinstance(attempt, dict) and attempt.get("subject") \
                and attempt.get("relation") and attempt.get("object"):
            a_quote = candidate.get("quote") or ""
            cls = quote_failure_class(
                a_quote, attempt["subject"], attempt["relation"],
                attempt["object"], excerpt, doc_subject=doc_subject)
            if cls == "object_overreach" or cls == "subject_not_named":
                span = find_extending_span(
                    excerpt, a_quote, attempt["subject"],
                    attempt["object"], doc_subject=doc_subject)
                if span is not None:
                    a_quote, cls = span, None
            if cls is None and not relation_garbled(
                    attempt["relation"]):
                return {
                    "candidate_id": candidate_id,
                    "doc_id": candidate.get("doc_id"),
                    "method": candidate.get("method"),
                    "decision": "retype",
                    "final_subject": attempt["subject"],
                    "final_relation": attempt["relation"],
                    "final_object": attempt["object"],
                    "final_quote": a_quote,
                    "r5_remedy": ("extend_quote" if a_quote
                                  != candidate.get("quote")
                                  else "none"),
                    "clause": clause,
                    "reason": (reason + " [R4: veto converted to "
                               "its recorded retype — the attempt "
                               "is truthful and quote-complete]"
                               ).strip(),
                    "retype_attempt": attempt,
                    "retype_failure": failure,
                    "enforcement": "veto_converted_to_retype",
                    "original": {
                        "subject": candidate.get("subject"),
                        "relation": candidate.get("relation"),
                        "object": candidate.get("object"),
                        "quote": candidate.get("quote")},
                }
        if not isinstance(attempt, dict) or not attempt:
            attempt = {"unconstructible":
                       "the adjudicator recorded no retype attempt"}
            failure = {"clause": clause or 1,
                       "missing": "veto returned without the "
                                  "mandatory R4 retype record"}
            reason = (reason + " [R4: veto lacked the mandatory "
                      "retype_attempt/retype_failure record]"
                      ).strip()
        elif not isinstance(failure, dict) or not failure:
            failure = {"clause": clause or 1,
                       "missing": "veto returned without a "
                                  "recorded retype_failure"}
            reason = (reason + " [R4: veto lacked a recorded "
                      "retype_failure]").strip()
        return _veto_record(
            candidate, candidate_id, clause=clause, reason=reason,
            retype_attempt=attempt, retype_failure=failure,
            enforcement=None)

    if decision not in ("keep", "retype"):
        return _veto_record(
            candidate, candidate_id, clause=clause,
            reason=(f"adjudicator returned an unrecognised "
                    f"decision {decision!r} (fail closed)"
                    + (f": {reason}" if reason else "")),
            retype_attempt={"unconstructible":
                            "no usable decision was returned"},
            retype_failure={"clause": clause or 1,
                            "missing": "no keep/retype/veto "
                                       "decision was recorded"},
            enforcement="unrecognised_decision")

    final_subject = raw.get("final_subject") or candidate.get("subject")
    final_relation = raw.get("final_relation") \
        or candidate.get("relation")
    final_object = raw.get("final_object") or candidate.get("object")
    final_quote = raw.get("final_quote") or candidate.get("quote") or ""
    remedy = str(raw.get("r5_remedy") or "none").strip().lower()
    if remedy not in ("none", "extend_quote", "retype_to_quote"):
        remedy = "none"
    enforcement = None

    if decision == "keep" and relation_garbled(final_relation):
        return _veto_record(
            candidate, candidate_id, clause=clause,
            reason=("keep with a garbled relation "
                    f"({final_relation!r}) is invalid (R4); no "
                    "retype was offered" + (f": {reason}"
                                             if reason else "")),
            retype_attempt={"subject": final_subject,
                            "relation": final_relation,
                            "object": final_object},
            retype_failure={"clause": clause or 1,
                            "missing": "the relation surface "
                                       "states no connection and "
                                       "no truthful retyping was "
                                       "recorded"},
            enforcement="garbled_relation_keep")

    cls = quote_failure_class(
        final_quote, final_subject, final_relation, final_object,
        excerpt, doc_subject=doc_subject)
    if cls in ("subject_not_named", "object_overreach",
               "truncated_quote"):
        span = find_extending_span(
            excerpt, candidate.get("quote") or "", final_subject,
            final_object, doc_subject=doc_subject)
        if span is not None:
            final_quote = span
            remedy = "extend_quote"
            enforcement = "quote_extended"
            cls = quote_failure_class(
                final_quote, final_subject, final_relation,
                final_object, excerpt, doc_subject=doc_subject)
    if cls is not None:
        return _veto_record(
            candidate, candidate_id, clause=clause or 1,
            reason=(f"decision {decision} failed quote "
                    f"completeness ({cls}) and no attesting span "
                    f"exists in the excerpt"
                    + (f": {reason}" if reason else "")),
            retype_attempt={"subject": final_subject,
                            "relation": final_relation,
                            "object": final_object},
            retype_failure={"clause": clause or 1,
                            "missing": f"quote completeness: {cls}"},
            enforcement="quote_incomplete")
    if decision == "retype" and clause is None \
            and enforcement is None:
        # A retype with no driving clause recorded: the record
        # stays complete and the gap is named in the census.
        enforcement = "retype_without_clause"
    return {
        "candidate_id": candidate_id,
        "doc_id": candidate.get("doc_id"),
        "method": candidate.get("method"),
        "decision": decision,
        "final_subject": final_subject,
        "final_relation": final_relation,
        "final_object": final_object,
        "final_quote": final_quote,
        "r5_remedy": remedy,
        "clause": clause,
        "reason": reason,
        "retype_attempt": raw.get("retype_attempt"),
        "retype_failure": raw.get("retype_failure"),
        "enforcement": enforcement,
        "original": {
            "subject": candidate.get("subject"),
            "relation": candidate.get("relation"),
            "object": candidate.get("object"),
            "quote": candidate.get("quote")},
    }


def adjudicate_candidates(chat_fn, candidates: list[dict],
                          texts: dict[str, str], model=None):
    """Adjudicate every candidate, one chat call per document.

    Returns (decision_rows, survivors, total_cost): the full
    decision record (one row per candidate, vetoes included),
    the surviving candidates with their final fields applied
    (input order preserved), and the summed chat cost (None
    when any call was unmetered). A candidate the adjudicator
    did not decide is recorded as a veto (fail closed, named)
    — never silently passed."""
    by_doc: dict[str, list[tuple[int, dict]]] = {}
    for i, c in enumerate(candidates):
        by_doc.setdefault(c.get("doc_id"), []).append((i, c))
    decisions: dict[int, dict] = {}
    total_cost: float | None = 0.0
    for doc_id, rows in by_doc.items():
        text = texts.get(doc_id) or ""
        items = []
        for i, c in rows:
            items.append({
                "candidate_id": f"cand-{i}",
                "subject": c.get("subject"),
                "relation": c.get("relation"),
                "object": c.get("object"),
                "quote": c.get("quote"),
                "excerpt": excerpt_for(text, c.get("quote") or ""),
            })
        prompt = build_adjudication_prompt(items)
        text_out, cost = chat_fn(prompt, model)
        total_cost = add_cost(total_cost, cost)
        raw_by_id = {}
        for raw in parse_decision_array(text_out):
            cid = raw.get("candidate_id")
            if cid is not None and cid not in raw_by_id:
                raw_by_id[cid] = raw
        for (i, c), item in zip(rows, items, strict=True):
            cid = f"cand-{i}"
            raw = raw_by_id.get(cid)
            if raw is None:
                decisions[i] = _veto_record(
                    c, cid, clause=None,
                    reason="adjudicator returned no decision "
                           "for this candidate (fail closed)",
                    retype_attempt={"unconstructible":
                                    "no decision returned"},
                    retype_failure={"clause": 1,
                                    "missing": "no adjudication "
                                               "decision was "
                                               "returned"},
                    enforcement="no_decision")
            else:
                decisions[i] = enforce_decision(
                    raw, c, cid, item["excerpt"])
    decision_rows = [decisions[i] for i in range(len(candidates))]
    survivors = []
    for i, c in enumerate(candidates):
        d = decisions[i]
        if d["decision"] == "veto":
            continue
        row = dict(c)
        row["subject"] = d["final_subject"]
        row["relation"] = d["final_relation"]
        row["object"] = d["final_object"]
        row["quote"] = d["final_quote"]
        if d["final_object"] != c.get("object"):
            # A retyped object is a new value: any canonical
            # key recorded for the original value no longer
            # applies (the canonical pass re-derives it).
            row.pop("canonical_object", None)
        row["adjudication"] = {
            "candidate_id": d["candidate_id"],
            "decision": d["decision"],
            "clause": d["clause"],
            "r5_remedy": d["r5_remedy"],
        }
        survivors.append(row)
    return decision_rows, survivors, total_cost


# ------------------------------------------------------------------
# Ship-gate audit: independent R5 verification of store-bound rows
# ------------------------------------------------------------------

def build_audit_prompt(row: dict, excerpt: str) -> str:
    return (
        AUDIT_INSTRUCTIONS + "\n\n"
        f"Triple: ({row.get('subject')} | {row.get('relation')} | "
        f"{row.get('object')})\n"
        f"Evidence quote: {row.get('quote')}\n"
        f"Excerpt:\n{excerpt}")


def parse_audit(text: str) -> dict:
    """Parse the auditor's single JSON object. Strict, like
    the adjudication parse: an unparseable audit is a stage
    failure, never a silent pass."""
    try:
        obj = json.loads(text)
    except Exception as e:
        raise RuntimeError(
            f"audit response is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or obj.get("verdict") not in (
            "pass", "fail"):
        raise RuntimeError(
            "audit response is not an object with verdict "
            "pass/fail")
    return obj


def _audit_once(chat_fn, row: dict, excerpt: str,
                model=None):
    """One audit round over one row: deterministic R5 checks
    first (verbatim / naming / attestation — classified
    mechanically), then the independent judge audit for the
    semantic residue. Returns (ok, failure_class, verdict,
    cost); verdict is None when the deterministic checks
    already failed the row."""
    doc_subject = _is_doc_subject(row)
    cls = quote_failure_class(
        row.get("quote") or "", row.get("subject") or "",
        row.get("relation") or "", row.get("object") or "",
        excerpt, doc_subject=doc_subject)
    if cls is not None:
        return False, cls, None, 0.0
    text_out, cost = chat_fn(build_audit_prompt(row, excerpt), model)
    verdict = parse_audit(text_out)
    if verdict["verdict"] == "pass":
        return True, None, verdict, cost
    return False, verdict.get("failure_class") or "other", \
        verdict, cost


def _validated_correction(verdict: dict | None, row: dict,
                          excerpt: str) -> dict | None:
    """The auditor's proposed correction, validated against
    the deterministic invariants before it may be re-audited:
    a correction that is itself quote-incomplete is not a
    retype, it is a second failure waiting to happen."""
    if not verdict:
        return None
    corr = verdict.get("corrected")
    if not isinstance(corr, dict):
        return None
    if not (corr.get("subject") and corr.get("relation")
            and corr.get("object") and corr.get("quote")):
        return None
    cls = quote_failure_class(
        corr["quote"], corr["subject"], corr["relation"],
        corr["object"], excerpt,
        doc_subject=_is_doc_subject(row))
    if cls is not None:
        return None
    out = dict(row)
    out["subject"] = corr["subject"]
    out["relation"] = corr["relation"]
    out["object"] = corr["object"]
    out["quote"] = corr["quote"]
    if corr["object"] != row.get("object"):
        out.pop("canonical_object", None)
    return out


def audit_store_row(chat_fn, row: dict, text: str,
                    model=None):
    """The ship gate for one store-bound row: audit, one
    retype on failure, one re-audit; a second failure
    excludes.

    The retype comes from the auditor's validated correction
    when one exists, else from the deterministic extend_quote
    remedy. Returns (outcome, shipped_row | None, record,
    total_cost): outcome is "shipped" (as audited),
    "shipped_retyped" (the retype passed re-audit), or
    "excluded". The record carries the full trail — both
    rounds' failure classes and the remedy applied — so the
    batch's accounting is reconstructible from
    verification_gate.jsonl alone."""
    excerpt = excerpt_for(text, row.get("quote") or "")
    record = {
        "doc_id": row.get("doc_id"),
        "subject": row.get("subject"),
        "relation": row.get("relation"),
        "object": row.get("object"),
        "quote": row.get("quote"),
        "rounds": [],
    }
    ok, cls, verdict, cost = _audit_once(chat_fn, row, excerpt, model)
    total_cost: float | None = cost
    record["rounds"].append(
        {"round": 1, "result": "pass" if ok else "fail",
         "failure_class": cls})
    if ok:
        record["outcome"] = "shipped"
        return "shipped", row, record, total_cost
    # One retype: the auditor's validated correction first,
    # else deterministic quote extension.
    repaired = _validated_correction(verdict, row, excerpt)
    remedy = "retype_to_quote" if repaired is not None else None
    if repaired is None and verdict is None:
        # The deterministic check failed the row before the
        # auditor was consulted: the failure still earns its
        # one retype attempt, so ask the auditor for a
        # correction now. (If the auditor passes a row the
        # deterministic check failed, the invariant governs —
        # the pass is ignored and named in the record; only a
        # validated correction counts.)
        text_out, cost = chat_fn(build_audit_prompt(row, excerpt),
                                 model)
        total_cost = add_cost(total_cost, cost)
        verdict = parse_audit(text_out)
        if verdict["verdict"] == "pass":
            record["judge_pass_overridden"] = True
        repaired = _validated_correction(verdict, row, excerpt)
        if repaired is not None:
            remedy = "retype_to_quote"
    if repaired is None:
        span = find_extending_span(
            excerpt, row.get("quote") or "", row.get("subject") or "",
            row.get("object") or "",
            doc_subject=_is_doc_subject(row))
        if span is not None and span != row.get("quote"):
            repaired = dict(row)
            repaired["quote"] = span
            remedy = "extend_quote"
    record["remedy"] = remedy
    if repaired is None:
        record["outcome"] = "excluded"
        record["excluded_class"] = cls
        return "excluded", None, record, total_cost
    ok2, cls2, _verdict2, cost2 = _audit_once(
        chat_fn, repaired, excerpt, model)
    total_cost = add_cost(total_cost, cost2)
    record["rounds"].append(
        {"round": 2, "result": "pass" if ok2 else "fail",
         "failure_class": cls2})
    if ok2:
        record["outcome"] = "shipped_retyped"
        record["shipped"] = {
            "subject": repaired.get("subject"),
            "relation": repaired.get("relation"),
            "object": repaired.get("object"),
            "quote": repaired.get("quote")}
        return "shipped_retyped", repaired, record, total_cost
    record["outcome"] = "excluded"
    record["excluded_class"] = cls2 or cls
    return "excluded", None, record, total_cost
