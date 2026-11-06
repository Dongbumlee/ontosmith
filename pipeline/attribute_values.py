"""Value attribute candidates: the propose -> gate ->
judge -> record path (value-attributes design note,
2026-11-06).

The proposer reports a declared attribute's value as its own
item ({"entity", "entity_type", "attribute", "value",
"quote"} — the shape the guidance rendering names). This
module is what the item travels:

1. DECLARATION. A candidate naming an attribute the
   schema_front does not declare for that entity type does
   not exist — it is recorded as undeclared and dropped.
   (Declaration first: an undeclared attribute is never
   extracted, judged or stored.)
2. FORMAT GATE (code). The surface is normalized by
   pipeline.value_normalize — deterministically, no model.
   A candidate with no quote has nothing for a judge to
   judge against and fails here too. A format failure is
   recorded as a format violation; the judge is NEVER
   consulted for it (the split the design fixes: form is
   code's, fidelity is the judge's).
3. JUDGE. A candidate that passes the gate goes to the
   judge callable with the normalized value: does the
   document's evidence state this value for this entity —
   and where the value rode on the proposer's own
   conversion (code_normalized False), is that conversion
   right? The judge is a callable the caller binds (the
   production binding is make_attribute_judge, over the
   decisions seat); passing judge=None accepts gate-passing
   candidates marked unjudged, for callers that run their
   own judgment stage.
4. RECORD. An accepted value becomes a record carrying the
   normalized typed value AND the original surface beside
   it (the audit convention triples keep with their quote),
   its verdict, and — when the caller supplies it — the
   pack version, so later expiry-candidate accounting can
   name the declaration the value was judged under.

Multi is per value (§11-2): a multi attribute accepts any
number of distinct values, each judged and recorded on its
own; a single-value attribute accepts one distinct value
per entity — a second distinct value is recorded as a
multi violation, an identical repeat is a duplicate.

A required attribute's ABSENCE is not processed here as an
error at all: required_attribute_warnings names it as a
warning record and the entity stands (§11-3).
"""
from __future__ import annotations

import json
from datetime import date

from . import profile_config, value_normalize


def _value_fingerprint(normalized: dict) -> str:
    return json.dumps(
        {"value": normalized.get("value"),
         "unit": normalized.get("unit"),
         "currency": normalized.get("currency")},
        sort_keys=True, ensure_ascii=False, default=str)


def _entity_key(candidate: dict) -> str:
    return str(candidate.get("entity_id")
               or candidate.get("entity") or "")


def process_attribute_candidates(
        candidates, cfg: dict | None, judge=None, *,
        reference_date: date | None = None,
        pack_version: str = "") -> dict:
    """Run attribute candidates through declaration, the
    format gate, and the judge.

    ``candidates`` are proposer-shaped dicts: entity,
    entity_type, attribute, value (the surface as written),
    quote; optionally entity_id, doc_id, and proposed (the
    proposer's own normalized value, used only when the
    surface does not parse — its correctness is then the
    judge's question). ``judge`` is a callable
    (candidate, normalized) -> bool or {"accepted": bool,
    ...}; None accepts gate-passing candidates marked
    unjudged. A judge exception propagates — failure policy
    belongs to the calling stage.

    Returns {"accepted", "format_rejected",
    "judge_rejected", "undeclared", "multi_rejected",
    "duplicates"}: accepted values as storage-ready records
    (see the module docstring), every rejection as the
    candidate plus its reason — nothing is dropped silently.
    """
    out = {"accepted": [], "format_rejected": [],
           "judge_rejected": [], "undeclared": [],
           "multi_rejected": [], "duplicates": []}
    accepted_values: dict[tuple, set] = {}
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            out["undeclared"].append(
                {"candidate": candidate,
                 "reason": "candidate is not an object"})
            continue
        entity_type = str(candidate.get("entity_type") or "")
        attr_name = str(candidate.get("attribute") or "")
        decl = profile_config.declared_attribute(
            cfg, entity_type, attr_name)
        if decl is None:
            out["undeclared"].append(
                {**candidate,
                 "reason": f"attribute {attr_name!r} is not "
                           f"declared for entity type "
                           f"{entity_type!r} — an undeclared "
                           f"attribute does not exist"})
            continue
        quote = candidate.get("quote")
        if not (isinstance(quote, str) and quote.strip()):
            out["format_rejected"].append(
                {**candidate,
                 "reason": "no quote — the judge judges a "
                           "value against its evidence, and "
                           "this candidate carries none"})
            continue
        normalized = value_normalize.normalize_value(
            decl["value_type"], candidate.get("value"),
            proposed=candidate.get("proposed"),
            reference_date=reference_date)
        if not normalized["ok"]:
            out["format_rejected"].append(
                {**candidate, "reason": normalized["reason"]})
            continue
        fingerprint = _value_fingerprint(normalized)
        key = (_entity_key(candidate), attr_name)
        seen = accepted_values.setdefault(key, set())
        if fingerprint in seen:
            out["duplicates"].append(
                {**candidate,
                 "reason": "the same normalized value is "
                           "already accepted for this entity "
                           "and attribute"})
            continue
        if seen and not decl.get("multi"):
            out["multi_rejected"].append(
                {**candidate,
                 "reason": f"attribute {attr_name!r} is not "
                           f"declared multi, and a distinct "
                           f"value is already accepted for "
                           f"this entity"})
            continue
        verdict = {"judged": False, "accepted": True}
        if judge is not None:
            answer = judge(candidate, normalized)
            if isinstance(answer, dict):
                verdict = {"judged": True, **answer}
                verdict["accepted"] = bool(
                    answer.get("accepted"))
            else:
                verdict = {"judged": True,
                           "accepted": bool(answer)}
            if not verdict["accepted"]:
                out["judge_rejected"].append(
                    {**candidate, "verdict": verdict,
                     "reason": "the judge did not accept the "
                               "value against its evidence"})
                continue
        seen.add(fingerprint)
        record = {
            "entity": str(candidate.get("entity") or ""),
            "entity_id": candidate.get("entity_id"),
            "entity_type": entity_type,
            "attribute": attr_name,
            "value_type": decl["value_type"],
            "normalized_type": normalized["normalized_type"],
            "value": normalized["value"],
            "complete": normalized["complete"],
            "surface": normalized["surface"],
            "quote": quote,
            "doc_id": str(candidate.get("doc_id") or ""),
            "code_normalized": normalized["code_normalized"],
            "verdict": verdict,
        }
        for extra in ("unit", "currency"):
            if extra in normalized:
                record[extra] = normalized[extra]
        if pack_version:
            record["pack_version"] = pack_version
        out["accepted"].append(record)
    return out


def required_attribute_warnings(entities, accepted,
                                cfg: dict | None) -> list[dict]:
    """Warning records for required attributes no accepted
    value covers. ``entities`` are dicts naming the entities
    a run produced ({"entity" and/or "entity_id",
    "entity_type"}); ``accepted`` are records from
    process_attribute_candidates. A warning is not a
    failure and deletes nothing (§11-3): required means
    "must be captured when a document states it", and an
    absence may be the document's own state — the warning
    leaves it for a human to see."""
    covered = {(_entity_key(r), r["attribute"])
               for r in accepted or []}
    warnings = []
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type") or "")
        key = _entity_key(entity)
        for decl in profile_config.entity_attributes(
                cfg, entity_type):
            if not decl.get("required"):
                continue
            if (key, decl["name"]) in covered:
                continue
            warnings.append({
                "entity": entity.get("entity"),
                "entity_id": entity.get("entity_id"),
                "entity_type": entity_type,
                "attribute": decl["name"],
                "code": "REQUIRED_ATTRIBUTE_MISSING",
                "message": f"required attribute "
                           f"{decl['name']!r} of entity type "
                           f"{entity_type!r} has no accepted "
                           f"value for this entity — a "
                           f"warning, not a failure: the "
                           f"entity stands",
            })
    return warnings


def entity_attribute_groups(accepted) -> dict:
    """Accepted records grouped for the graph stores:
    {entity key: {"entity_type", "attributes": {name:
    {"value_type", "values": [{value, surface, unit?,
    currency?}]}}}} — values in acceptance order, one entry
    per accepted value (multi is per value end to end)."""
    groups: dict = {}
    for record in accepted or []:
        key = _entity_key(record)
        group = groups.setdefault(
            key, {"entity_type": record["entity_type"],
                  "attributes": {}})
        slot = group["attributes"].setdefault(
            record["attribute"],
            {"value_type": record["normalized_type"],
             "values": []})
        value = {"value": record["value"],
                 "surface": record["surface"]}
        for extra in ("unit", "currency"):
            if extra in record:
                value[extra] = record[extra]
        slot["values"].append(value)
    return groups


_TEMPORAL_CONVERTERS = {"date": "date", "datetime": "datetime",
                        "duration": "duration"}


def attribute_write_statements(attr_rows) -> list:
    """Cypher write plans for grouped attribute values:
    [(statement, rows)] for a store to run AFTER its base
    entity MERGE (the nodes must exist; the base statement is
    never modified, so an entity with no attributes is
    written exactly as before).

    ``attr_rows`` are {"id": entity_id, "attributes":
    {name: {"value_type", "values": [...]}}} — the
    entity_attribute_groups shape per entity. Rows are
    grouped by attribute signature (the (name, value_type,
    multi) set), one UNWIND statement per signature, because
    Cypher property names cannot be parameters. Values land
    as TYPED node properties — date/datetime/duration
    convert through the matching Cypher temporal function,
    so range queries work — under `attr_<name>`, with the
    original surface beside them (`attr_<name>_surface`)
    and, where the type carries them, `attr_<name>_unit` /
    `attr_<name>_currency`. Names that are not declaration
    slugs are skipped: declaration validation owns that
    finding, the store never writes an unsanitized property
    name into a statement."""
    by_signature: dict[tuple, list] = {}
    for row in attr_rows or []:
        attributes = row.get("attributes") or {}
        clauses = []
        params = {"id": row["id"]}
        for name in sorted(attributes):
            if not value_normalize.ATTRIBUTE_NAME_RE.match(
                    name):
                continue
            slot = attributes[name] or {}
            values = slot.get("values") or []
            if not values:
                continue
            value_type = slot.get("value_type") or ""
            multi = len(values) > 1
            converter = _TEMPORAL_CONVERTERS.get(value_type)
            raw = [v.get("value") for v in values]
            params[f"a_{name}"] = raw if multi else raw[0]
            surfaces = [v.get("surface") for v in values]
            params[f"a_{name}_surface"] = \
                surfaces if multi else surfaces[0]
            if converter:
                expr = (f"[{converter}(v) FOR v IN "
                        f"row.a_{name}]") if multi else \
                    f"{converter}(row.a_{name})"
            else:
                expr = f"row.a_{name}"
            clause = (f"e.attr_{name} = {expr}, "
                      f"e.attr_{name}_surface = "
                      f"row.a_{name}_surface")
            if value_type == "number":
                units = [v.get("unit") for v in values]
                params[f"a_{name}_unit"] = \
                    units if multi else units[0]
                clause += (f", e.attr_{name}_unit = "
                           f"row.a_{name}_unit")
            if value_type == "money":
                codes = [v.get("currency") for v in values]
                params[f"a_{name}_currency"] = \
                    codes if multi else codes[0]
                clause += (f", e.attr_{name}_currency = "
                           f"row.a_{name}_currency")
            clauses.append((name, value_type, multi, clause))
        if not clauses:
            continue
        signature = tuple((n, t, m) for n, t, m, _ in clauses)
        by_signature.setdefault(signature, []).append(
            (", ".join(c for _, _, _, c in clauses), params))
    plans = []
    for _signature, items in by_signature.items():
        statement = (
            "UNWIND $rows AS row "
            "MATCH (e:Entity {entity_id: row.id}) SET "
            + items[0][0] + " RETURN count(e)")
        plans.append((statement, [p for _, p in items]))
    return plans


def _normalized_display(normalized: dict) -> str:
    parts = [str(normalized.get("value"))]
    if normalized.get("currency"):
        parts.append(str(normalized["currency"]))
    elif normalized.get("unit"):
        parts.append(str(normalized["unit"]))
    return " ".join(parts)


def make_attribute_judge(document_text: str, model: str | None = None):
    """The production judge binding: a decisions-seat call
    asking whether the document states the candidate's value
    for its entity (the validate_triple pattern, one noul
    question). The normalized value and its written surface
    are both put to the judge — the surface is what the
    document must support, the normalized form is what will
    be stored, and when the two diverge (a relative date the
    code could not resolve) the conversion itself is the
    question. Accepts at p_true > 0.5, the midpoint the
    triple gates use."""
    from . import decisions

    def judge(candidate: dict, normalized: dict) -> dict:
        entity = candidate.get("entity") or ""
        attribute = candidate.get("attribute") or ""
        state = (
            "Document:\n" + document_text + "\n\n"
            f"Candidate attribute value: the {attribute} of "
            f"'{entity}' is written as "
            f"{candidate.get('value')!r} and normalized to "
            f"{_normalized_display(normalized)!r}.")
        instructions = (
            f"Does the document state that the {attribute} "
            f"of '{entity}' is "
            f"{candidate.get('value')!r}? Answer true only "
            f"if the document states this value for this "
            f"entity. If the written value needed conversion "
            f"to reach the normalized form (for example a "
            f"relative date resolved against the document's "
            f"date), answer true only if that conversion is "
            f"also correct.")
        questions = {"holds": {"type": "noul",
                               "instructions": instructions}}
        answers, usage = decisions.decide(
            state, questions,
            model=model or decisions.DEFAULT_MODEL)
        p_true = float(answers["holds"]["noul"])
        return {"accepted": p_true > 0.5, "p_true": p_true,
                "model": model or decisions.DEFAULT_MODEL,
                "usage": usage}

    return judge
