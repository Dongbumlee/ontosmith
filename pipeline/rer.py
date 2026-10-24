"""Relation Extraction (RER) via spaCy dependency parsing.

Design (2026-10-11, Dongbum):
- RER is as necessary as NER. First-class pipeline stage.
- Hybrid: spaCy (structure) → LLM (fallback) → Jev (validation).
- spaCy proposes candidates; Jev judges. Role separation maintained.

This module implements spaCy extraction and LLM fallback.
Jev validation is orchestrated by the caller.
"""
from __future__ import annotations

import json
import threading

from . import connections
from .reference_profile import CHAT_URL as CHAT_URL  # preset route (re-export)
from .reference_profile import PROPOSER_MODEL, PROVIDER_SORT


def extract_triples(sentence: str, nlp=None) -> list[tuple[str, str, str]]:
    """Extract (subject, verb, object) triples from a sentence.

    Handles:
    - Active voice: "Oswald assassinated Kennedy"
    - Passive voice: "Kennedy was assassinated by Oswald" → (Oswald, assassinate, Kennedy)
    - Prepositional objects: "worked at the Depository" → (Oswald, work, Depository)

    Returns empty list if no complete triple found (caller routes to LLM).
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(sentence)
    triples = []

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subj = token.text
            verb = token.head.lemma_

            # Passive: agent becomes subject
            if token.dep_ == "nsubjpass":
                for child in token.head.children:
                    if child.dep_ == "agent":
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                triples.append((gc.text, verb, subj))

            # Active: find objects
            objs = []
            for child in token.head.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    objs.append(child.text)
                elif child.dep_ == "prep":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            # Full noun phrase for entity matching
                            objs.append(gc.text)

            for obj in objs:
                # Avoid duplicates and self-relations
                if obj.lower() != subj.lower():
                    triples.append((subj, verb, obj))

    # Deduplicate
    seen = set()
    unique = []
    for t in triples:
        k = (t[0].lower(), t[1].lower(), t[2].lower())
        if k not in seen:
            seen.add(k)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# Structure-constrained extraction (2026-10-13, EXTRACTION_STRUCTURE_REVIEW)
#
# Arm-B extractor for the A/B test against extract_triples (Arm A, unchanged
# above). Adds:
#   C7  syntactic-path provenance per triple (subject source, object binding)
#   C3  passive clauses never emit patient-as-subject triples: with an
#       agent phrase, (agent, verb, patient) is kept; without one, nothing
#       is emitted. Arm A's would-be patient-as-subject triples are
#       returned as blocked records for the killed-set audit
#   C4  negation / modal marking: triples are kept but flagged
#       asserted=False so routing can keep them out of Tier 1
# C1 (preposition whitelist) and C2 (theme-presence guard) consume the C7
# obj_binding field and are applied in a later step.
# ---------------------------------------------------------------------------

_MODALS = {"will", "would", "shall", "should", "can", "could",
           "may", "might", "must", "ought"}


def extract_triples_detailed(sentence: str, nlp=None) -> list[dict]:
    """Extract triples with syntactic-path provenance (C7).

    Each result: {subject, verb, object, subj_source, obj_binding,
    negated, modal, asserted}. subj_source is "nsubj" or
    "nsubjpass_agent"; obj_binding is "dobj" | "attr" | "oprd" |
    "prep:<lemma>" | "nsubjpass" (the patient of an agent passive).
    Passive clauses without an agent phrase produce no triples here
    (see extract_triples_constrained for the blocked list).
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(sentence)
    out = []

    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass") or token.head.pos_ != "VERB":
            continue
        verb_tok = token.head
        verb = verb_tok.lemma_
        negated = any(c.dep_ == "neg" for c in verb_tok.children)
        modal = any(c.dep_ == "aux" and c.lemma_.lower() in _MODALS
                    for c in verb_tok.children)

        # Object bindings, tagged by how the object attaches to the verb.
        bindings = []
        for child in verb_tok.children:
            if child.dep_ in ("dobj", "obj", "attr", "oprd"):
                bindings.append((child.text, child.dep_))
            elif child.dep_ == "prep":
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        bindings.append((gc.text, f"prep:{child.text.lower()}"))

        if token.dep_ == "nsubjpass":
            # One subject per agent phrase: Arm A emits a triple for
            # EVERY agent pobj ("bordered by X, ... by Y, ... by Z"),
            # so Arm B must too — keeping only the last agent silently
            # dropped 6 Arm-A candidates in A/B round 1 (hard-gate find).
            agents = []
            for child in verb_tok.children:
                if child.dep_ == "agent":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            agents.append(gc.text)
            if not agents:
                continue  # C3: no fall-through with the patient as subject
            # The patient (the nsubjpass token itself) is the object.
            # The verb's other object bindings belong to Arm A's
            # patient-as-subject fall-through; they are never kept here
            # (extract_triples_constrained records them as blocked).
            for agent in agents:
                if agent.lower() == token.text.lower():
                    continue
                out.append({
                    "subject": agent, "verb": verb, "object": token.text,
                    "subj_source": "nsubjpass_agent",
                    "obj_binding": "nsubjpass",
                    "negated": negated, "modal": modal,
                    "asserted": not (negated or modal),
                })
            continue

        subj, subj_source = token.text, "nsubj"
        for obj, binding in bindings:
            if obj.lower() == subj.lower():
                continue
            out.append({
                "subject": subj, "verb": verb, "object": obj,
                "subj_source": subj_source, "obj_binding": binding,
                "negated": negated, "modal": modal,
                "asserted": not (negated or modal),
            })

    # Deduplicate on (subject, verb, object), keeping the first path seen.
    seen = set()
    unique = []
    for t in out:
        k = (t["subject"].lower(), t["verb"].lower(), t["object"].lower())
        if k not in seen:
            seen.add(k)
            unique.append(t)
    return unique


def extract_triples_constrained(sentence: str, nlp=None):
    """Arm-B extraction: (kept, blocked).

    kept: detailed triples from extract_triples_detailed (C4-flagged ones
    included, marked asserted=False). blocked: records for the
    patient-as-subject triples Arm A would have emitted from a passive
    clause but C3 suppresses — reason "C3_passive_no_agent" when the
    clause has no agent phrase, "C3_passive_patient_subject" when it has
    one (Arm B keeps (agent, verb, patient) instead). Kept for the A/B
    killed-set audit.
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    kept = extract_triples_detailed(sentence, nlp)

    # Reconstruct Arm A's patient-as-subject triples from passives.
    blocked = []
    kept_keys = {(t["subject"].lower(), t["verb"].lower(), t["object"].lower())
                 for t in kept}
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ != "nsubjpass" or token.head.pos_ != "VERB":
            continue
        verb_tok = token.head
        has_agent = any(c.dep_ == "agent" for c in verb_tok.children)
        reason = ("C3_passive_patient_subject" if has_agent
                  else "C3_passive_no_agent")
        for child in verb_tok.children:
            objs = []
            if child.dep_ in ("dobj", "obj", "attr", "oprd"):
                objs.append(child.text)
            elif child.dep_ == "prep":
                objs.extend(gc.text for gc in child.children
                            if gc.dep_ == "pobj")
            for obj in objs:
                k = (token.text.lower(), verb_tok.lemma_.lower(), obj.lower())
                if obj.lower() != token.text.lower() and k not in kept_keys:
                    blocked.append({
                        "subject": token.text, "verb": verb_tok.lemma_,
                        "object": obj, "reason": reason,
                    })
    return kept, blocked


# LLM fallback (2026-10-11): DeepSeek V4.1 Flash for noisy text where spaCy fails.
# Cost: ~$0.15/M input, $0.60/M output. 272-doc batch ≈ $0.68.

_LLM_PROMPT = """Extract subject-verb-object triples from the sentence below.
The sentence may contain OCR errors. Do your best to understand it.
Return ONLY a JSON list of triples, each with "subject", "verb", "object".
Use the exact entity names provided when possible.
If no clear relation exists, return [].

Sentence: {sentence}

Entities: {entities}

Triples:"""


_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

# LLM-fallback call accounting (run 32, audit F17): the fallback
# returns [] on ANY exception — an unreachable provider and a
# sentence with no triples were indistinguishable, so a dead
# fallback silently degraded extraction to the pairing phase.
# Attempts/failures are counted here; batch stages snapshot the
# delta into their metrics (batch_run stage_candidates).
_LLM_STATS_LOCK = threading.Lock()
LLM_CALL_STATS = {"attempts": 0, "failures": 0}


def _llm_stat(field: str) -> None:
    with _LLM_STATS_LOCK:
        LLM_CALL_STATS[field] += 1


def llm_call_stats() -> dict:
    """A snapshot copy of the fallback counters."""
    with _LLM_STATS_LOCK:
        return dict(LLM_CALL_STATS)


def extract_triples_llm_with_usage(
        sentence: str, entities: list[str],
        model: str = PROPOSER_MODEL,
        api_key: str | None = None,
        timeout: int = 30) -> tuple[list[tuple[str, str, str]], dict]:
    """LLM fallback with API usage (G-27): like extract_triples_llm,
    but also returns the OpenRouter usage dict — same shape as the
    Jev decision calls (decisions.py): input_tokens / output_tokens
    / cost — so callers can record actual spend in the cost ledger.
    Failed or keyless calls return zero usage.
    """
    import requests

    resolved = connections.resolve_bindings()
    binding = resolved.seat("proposer")
    key = api_key or binding.key_or_none()
    if not key:
        # Availability-gated fallback (measured contract):
        # with no binding declared and no key, the LLM pass
        # simply does not run. A *declared* proposer binding
        # whose key cannot be resolved refuses loudly
        # instead (connection-layer refusal rule).
        if "proposer" in resolved.declared:
            binding.require_key()
        return [], dict(_ZERO_USAGE)

    prompt = _LLM_PROMPT.format(
        sentence=sentence[:500],
        entities=", ".join(entities[:10]),
    )

    # Extraction default switched DeepSeek -> Haiku 5.5 (Dongbum,
    # 2026-10-14) on the clause bake-off: same gold recall, far
    # fewer false fills (FP 34 vs 114), zero span-gate drops, lower
    # cost. Haiku 5.5 rejects non-default temperature (HTTP 400)
    # and counts thinking tokens toward max_tokens, so omit the
    # temperature and give the output headroom on that model.
    # The behavior switch is the seat's capability declaration
    # now (issue #41; derived from the model id unless the
    # binding declares capabilities explicitly).
    is_haiku55 = binding.has_capability(
        connections.CAP_NO_TEMPERATURE, model)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000 if is_haiku55 else 300,
    }
    if binding.is_openrouter_dialect:
        # Route to the fastest provider (2026-10-12: default
        # price routing landed on DeepInfra at 0.8-30 tok/s with
        # 499 stalls; throughput sort picks StreamLake-class).
        # provider.sort is an OpenRouter-dialect field (issue
        # #41): emitted only under that dialect.
        payload["provider"] = {"sort": PROVIDER_SORT}
    if not is_haiku55:
        payload["temperature"] = 0.1
    _llm_stat("attempts")
    try:
        r = requests.post(
            binding.chat_url,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = connections.normalize_usage(
            data.get("usage"), binding)
        content = data["choices"][0]["message"]["content"]

        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        triples_data = json.loads(content)
        result = []
        for t in triples_data:
            if isinstance(t, dict) and all(k in t for k in ("subject", "verb", "object")):
                s, v, o = t["subject"].strip(), t["verb"].strip(), t["object"].strip()
                if s and v and o and s.lower() != o.lower():
                    result.append((s, v, o))
        return result, usage
    except Exception:
        _llm_stat("failures")
        return [], dict(_ZERO_USAGE)


def extract_triples_llm(sentence: str, entities: list[str],
                        model: str = PROPOSER_MODEL,
                        api_key: str | None = None,
                        timeout: int = 30) -> list[tuple[str, str, str]]:
    """LLM fallback: extract triples when spaCy cannot parse the sentence.

    Args:
        sentence: The sentence to analyze (may have OCR noise).
        entities: NER entities found in the sentence (for grounding).
        model: OpenRouter model ID.
        api_key: OpenRouter API key (defaults to env var).
        timeout: Request timeout in seconds.

    Returns:
        List of (subject, verb, object) triples. Empty if none found.
    """
    triples, _usage = extract_triples_llm_with_usage(
        sentence, entities, model=model, api_key=api_key,
        timeout=timeout)
    return triples
