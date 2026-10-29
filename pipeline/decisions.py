"""Thin client for the decisions API (the Jev gate routes).

Transports resolve through the connection layer
(``pipeline.connections``): each function runs under a
seat — ``judge`` for the pinned Jev gates, ``escalation``
for the gate-evaluation judges (the decisions-route
gates' default model and the chat judge) — and the
seat's binding supplies the endpoint and the key (read
from the binding's key environment variable only; never
logged or written to disk). Under the shipped reference
preset the endpoints and key variable are exactly the
pre-generalization literals.
"""
from __future__ import annotations

import json
import re
import time

import requests

from . import connections
from .reference_profile import CHAT_URL as CHAT_URL  # preset route (re-export)
from .reference_profile import DECISIONS_URL as DECISIONS_URL  # preset route
from .reference_profile import (
    ESCALATION_MODEL,
    JUDGE_MODEL,
    PROVIDER_SORT,
)

#: The gate-evaluation default: the escalation seat's
#: measured occupant. A per-run model argument always
#: wins — bake-off reproductions name their model
#: explicitly and stamp it in the run's record.
DEFAULT_MODEL = ESCALATION_MODEL


def er_judge(
    context: str,
    mention_a: str,
    mention_b: str,
    model: str = JUDGE_MODEL,
    timeout: int = 60,
) -> tuple[float, dict]:
    """noul gate: do two mentions refer to the same real-world entity?"""
    questions = {
        "same_entity": {
            "type": "noul",
            "instructions": (
                f"Do the mentions '{mention_a}' and '{mention_b}' in the text "
                "refer to the same real-world entity (the same person, "
                "organization, place, work, or event)? Answer true only if the "
                "text gives clear evidence they are the same entity. Two "
                "different entities that merely share a word (e.g. 'Skai TV' "
                "vs 'Skai Group') are NOT the same entity."
            ),
        }
    }
    answers, usage = decide(context, questions, model=model,
                            timeout=timeout, seat="judge")
    return float(answers["same_entity"]["noul"]), usage


def llm_text_judge(
    document: str,
    subject: str,
    relation: str,
    obj: str,
    model: str,
    timeout: int = 120,
    max_tokens: int = 300,
) -> tuple[int, dict]:
    """Text-based LLM judge: Yes/No answer parsed from content.

    Provider-agnostic (no logprobs needed). Reasoning models get enough
    tokens to finish thinking; the final Yes/No is read from content,
    falling back to the last yes/no in the reasoning trace.
    Returns (pred_label, usage) where pred_label is 1 for Yes, 0 for No.
    When the structured answer does not parse and the fallback
    produces the label, usage["parse_failed"] is True.
    """
    binding = connections.seat("escalation")
    headers = {
        "Authorization": "Bearer " + binding.require_key(),
        "Content-Type": "application/json",
        "X-Title": "ontosmith",
    }
    prompt = (
        "Document:\n" + document + "\n\n"
        f"Candidate triple: ({subject} | {relation} | {obj})\n\n"
        f"Does the document support that the relation '{relation}' holds "
        f"from '{subject}' to '{obj}' (in this direction)? "
        "Pay close attention to direction: if the document only supports "
        "the reverse direction, answer No. "
        'Respond with JSON only: {"answer": "Yes"} or {"answer": "No"}.'
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    # Reasoning models (gpt-5*, o1/o3, muse-spark): max_tokens is rejected or
    # misinterpreted; max_completion_tokens covers reasoning+output together.
    # Minimal reasoning effort: this is a Yes/No judgment, not a puzzle.
    # The switch is the seat's capability declaration now (issue
    # #41; derived from the model id unless declared), and the
    # "reasoning" field itself is OpenRouter dialect — the
    # portable subset carries only max_completion_tokens.
    if binding.has_capability(
            connections.CAP_REASONING_TOKENS, model):
        # reasoning eats budget too: even minimal effort needs ~300 tokens
        payload["max_completion_tokens"] = max(max_tokens, 500)
        if binding.is_openrouter_dialect:
            payload["reasoning"] = {"effort": "minimal"}
            payload["provider"] = {"sort": PROVIDER_SORT}
    else:
        payload["max_tokens"] = max_tokens
    t0 = time.perf_counter()
    resp = requests.post(binding.chat_url, headers=headers, json=payload, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    choice = body["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    label = -1
    if text:
        try:
            ans = json.loads(text).get("answer", "")
            label = 1 if ans.strip().lower() == "yes" else 0
        except Exception:
            pass
    parse_failed = label == -1
    if parse_failed:
        # Fallback: reasoning trace or plain text. This
        # recovery invents the label from the LAST yes/no
        # token in the raw output — it used to be invisible, so a
        # parse failure read exactly like a judged No. The usage
        # dict now carries parse_failed=True; the fallback label
        # may only be consumed with the failure visible in the
        # caller's record (gate_cli stamps it), never silently.
        trace = choice["message"].get("reasoning") or ""
        hits = re.findall(r"\b(yes|no)\b", (text + " " + trace).lower())
        label = 1 if (hits and hits[-1] == "yes") else 0
    usage = connections.normalize_usage(body.get("usage"), binding)
    usage["latency_ms"] = latency_ms
    usage["provider"] = body.get("provider")
    usage["parse_failed"] = parse_failed
    return label, usage


def decide(
    state: str | dict,
    questions: dict,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
    max_retries: int = 3,
    seat: str = "escalation",
) -> tuple[dict, dict]:
    """POST a decision request. Returns (answers, usage).

    usage contains input_tokens, output_tokens, cost, plus client-measured
    latency_ms.

    The request goes to the ``seat`` binding's decisions
    endpoint with that binding's key (connection layer);
    judge-seat callers pass ``seat="judge"``.
    """
    binding = connections.seat(seat)
    headers = {
        "Authorization": "Bearer " + binding.require_key(),
        "Content-Type": "application/json",
        "X-Title": "ontosmith",
    }
    payload = {"model": model, "state": state, "questions": questions}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = requests.post(binding.decisions_endpoint,
                                 headers=headers, json=payload,
                                 timeout=timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            body = resp.json()
            usage = connections.normalize_usage(
                body.get("usage"), binding)
            usage["latency_ms"] = latency_ms
            usage["provider"] = body.get("provider")
            return body.get("answers", {}), usage
        except Exception as exc:
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Decisions API failed after {max_retries} attempts: {last_err}")


def validate_triple(
    document: str,
    subject: str,
    relation: str,
    obj: str,
    model: str = DEFAULT_MODEL,
    variant: str = "v1",
) -> tuple[float, dict]:
    """noul gate: does relation R hold between A and B according to the document?

    variant v1: plain support question.
    variant v2: adds explicit direction emphasis.

    Returns (p_true, usage).
    """
    state = (
        "Document:\n" + document + "\n\n"
        f"Candidate triple: ({subject} | {relation} | {obj})"
    )
    instructions = (
        "Does the document support that the relation "
        f"'{relation}' holds between '{subject}' and '{obj}'? "
        "Answer true only if the document states or clearly implies it."
    )
    if variant == "v2":
        instructions += (
            f" Pay close attention to direction: '{subject}' must be the one "
            f"from which the relation points to '{obj}', not the reverse. "
            f"If the document only supports the reverse direction "
            f"('{obj}' -> '{subject}'), answer false."
        )
    questions = {"holds": {"type": "noul", "instructions": instructions}}
    answers, usage = decide(state, questions, model=model)
    return float(answers["holds"]["noul"]), usage


def choice_gate(
    document: str,
    subject: str,
    obj: str,
    options: list[str],
    model: str = DEFAULT_MODEL,
) -> tuple[str, float, dict, dict]:
    """choice gate: which relation holds between A and B (in this direction)?

    options: candidate relation labels; a "none" option is appended by the caller.
    Returns (choice, confidence, probabilities, usage).
    """
    state = (
        "Document:\n" + document + "\n\n"
        f"Subject: {subject}\nObject: {obj}\n"
        "Question: which relation below holds from the subject to the object, "
        "in exactly this direction, according to the document?"
    )
    criteria = {}
    for r in options:
        if r == "none":
            criteria[r] = (
                f"None of the listed relations holds from '{subject}' to '{obj}'."
            )
        else:
            criteria[r] = f"The relation '{r}' holds from '{subject}' to '{obj}'."
    questions = {
        "relation": {
            "type": "choice",
            "instructions": (
                f"Which relation holds from '{subject}' to '{obj}' according to "
                "the document? Consider direction carefully: 'A father B' is "
                "different from 'A child B'. If none of the listed relations "
                "holds in this direction, choose 'none'."
            ),
            "criteria": criteria,
        }
    }
    answers, usage = decide(state, questions, model=model)
    ans = answers["relation"]
    return ans["choice"], float(ans["confidence"]), dict(ans.get("probabilities", {})), usage


def ontology_judge(
    ontology: str,
    axiom: str,
    a: str,
    b: str,
    model: str = JUDGE_MODEL,
    timeout: int = 60,
) -> tuple[float, dict]:
    """noul gate: should this axiom be admitted to the ontology?

    axiom: subclass | disjoint | domain | range.
    Returns (p_true, usage).
    """
    if axiom == "subclass":
        stmt = f"({a} is a subclass of {b})"
        q = (f"In the '{ontology}' ontology, is every '{a}' also a '{b}'? "
             f"Should the axiom '{a} is a subclass of {b}' be admitted?")
    elif axiom == "disjoint":
        stmt = f"({a} is disjoint with {b})"
        q = (f"In the '{ontology}' ontology, can anything be both a '{a}' "
             f"and a '{b}'? If not, the axiom '{a} is disjoint with {b}' "
             f"should be admitted.")
    elif axiom == "domain":
        stmt = f"(domain of property '{a}' is {b})"
        q = (f"In the '{ontology}' ontology, does the property '{a}' apply "
             f"to '{b}'? Should the axiom 'domain of {a} is {b}' be admitted?")
    elif axiom == "range":
        stmt = f"(range of property '{a}' is {b})"
        q = (f"In the '{ontology}' ontology, are values of the property '{a}' "
             f"of type '{b}'? Should the axiom 'range of {a} is {b}' be admitted?")
    else:
        raise ValueError(axiom)
    state = f"Ontology: {ontology}\nProposed axiom: {stmt}"
    questions = {
        "admit": {
            "type": "noul",
            "instructions": q + " Answer yes only if the axiom is correct.",
        }
    }
    answers, usage = decide(state, questions, model=model,
                            timeout=timeout, seat="judge")
    return float(answers["admit"]["noul"]), usage


def conflict_judge(
    existing_facts: list[str],
    subject: str,
    relation: str,
    obj: str,
    model: str = JUDGE_MODEL,
    timeout: int = 60,
) -> tuple[str, float, dict, dict]:
    """choice gate: how does a new claim relate to existing facts?

    Classes: duplicate / contradiction / update / novel.
    Returns (choice, confidence, probabilities, usage).
    """
    facts = "\n".join(f"- ({subject} | {r} | {o})" for r, o in existing_facts)
    state = (
        f"Existing facts about '{subject}':\n{facts}\n\n"
        f"New claim: ({subject} | {relation} | {obj})"
    )
    criteria = {
        "duplicate": (
            "The new claim is already recorded in the existing facts "
            "(same subject, relation, and object)."
        ),
        "contradiction": (
            "The new claim conflicts with existing facts: the relation can "
            "hold only one value (e.g. place of birth, date of birth, father, "
            "mother), but a different value is already recorded."
        ),
        "update": (
            "The new claim gives a new value for a relation that can change "
            "over time (e.g. employer, residence, position held, country of "
            "citizenship); the recorded value is outdated, not contradictory."
        ),
        "novel": (
            "The new claim adds information that is neither recorded in nor "
            "conflicting with the existing facts."
        ),
    }
    questions = {
        "verdict": {
            "type": "choice",
            "instructions": (
                f"How does the new claim about '{subject}' relate to the "
                "existing facts? Choose exactly one."
            ),
            "criteria": criteria,
        }
    }
    answers, usage = decide(state, questions, model=model,
                            timeout=timeout, seat="judge")
    ans = answers["verdict"]
    return ans["choice"], float(ans["confidence"]), dict(ans.get("probabilities", {})), usage


def score_gate(
    document: str,
    subject: str,
    relation: str,
    obj: str,
    model: str = DEFAULT_MODEL,
) -> tuple[float, float, dict]:
    """score gate: how strongly does the document support the triple?

    5-point scale. Returns (score in 1..5, confidence, usage).
    """
    state = (
        "Document:\n" + document + "\n\n"
        f"Candidate triple: ({subject} | {relation} | {obj})"
    )
    questions = {
        "support": {
            "type": "score",
            "instructions": (
                f"How strongly does the document support that the relation "
                f"'{relation}' holds from '{subject}' to '{obj}' (in this "
                "direction)? Rate the strength of documentary support."
            ),
            "criteria": [
                "No support: the document says nothing relevant, or contradicts it",
                "Weak: ambiguous or indirect hints only",
                "Moderate: some supporting evidence but not decisive",
                "Strong: clear supporting statements",
                "Conclusive: the document states it explicitly",
            ],
        }
    }
    answers, usage = decide(state, questions, model=model)
    ans = answers["support"]
    return float(ans["score"]), float(ans["confidence"]), usage
