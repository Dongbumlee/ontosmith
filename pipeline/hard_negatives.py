"""Realistic hard negatives for v0-hard.

Instead of corrupting a gold triple with a random relation, mimic the errors
a real candidate generator makes:

  entity_swap        - head/tail replaced by another entity of the same NER
                       type from the same document (entity-linking error)
  relation_confusion - relation replaced by a semantically similar one
                       (place of birth -> residence, employer -> member of, ...)
  direction_flip     - head/tail swapped for asymmetric relations
                       (child reversed, founder founded-by, ...)
"""
from __future__ import annotations

import hashlib
import random

# Semantically similar relation labels (DocRED rel_info values). A corrupted
# triple drawn from the same set is much harder than a random relation.
CONFUSION_SETS: list[list[str]] = [
    ["place of birth", "place of death", "residence", "work location", "location"],
    ["country", "country of citizenship", "country of origin", "continent"],
    ["employer", "member of", "member of political party", "member of sports team",
     "participant", "participant of"],
    ["father", "mother", "child", "sibling", "spouse"],
    ["capital", "capital of"],
    ["author", "creator", "composer", "director", "producer", "screenwriter",
     "lyrics by", "performer"],
    ["located in the administrative territorial entity", "location",
     "contains administrative territorial entity", "part of",
     "located in or next to body of water", "located on terrain feature"],
    ["inception", "publication date", "start time", "end time", "point in time",
     "date of birth", "date of death"],
    ["headquarters location", "location of formation"],
    ["followed by", "follows", "replaced by", "replaces"],
    ["subsidiary", "parent organization", "owned by", "operator"],
    ["head of state", "head of government", "chairperson", "position held"],
    ["original language of work", "languages spoken, written or signed",
     "official language"],
    ["award received", "notable work"],
    ["cast member", "characters", "present in work"],
    ["developer", "manufacturer", "production company", "publisher", "record label"],
    ["educated at", "employer"],
    ["founder", "founded by"],  # DocRED uses "founded by"; kept for completeness
]

# relation label -> set index, for O(1) lookup
_REL_TO_SET: dict[str, int] = {}
for _i, _s in enumerate(CONFUSION_SETS):
    for _r in _s:
        _REL_TO_SET.setdefault(_r, _i)

# Symmetric relations: flipping head/tail still yields a true statement.
SYMMETRIC = {"spouse", "sibling"}


def confusion_choices(relation: str) -> list[str]:
    idx = _REL_TO_SET.get(relation)
    if idx is None:
        return []
    return [r for r in CONFUSION_SETS[idx] if r != relation]


def build_hard_negatives(
    doc: dict,
    rel_info: dict[str, str],
    seed: int = 42,
    max_per_doc: int | None = None,
) -> list[dict]:
    """Build hard negatives for one DocRED document.

    Each negative: {subject, relation, object, neg_kind}
    """
    rng = random.Random(
        seed + int(hashlib.md5(doc["title"].encode()).hexdigest(), 16) % (2**31)
    )
    # entity id -> (name, type)
    entities: list[tuple[str, str]] = []
    for _eid, mentions in enumerate(doc["vertexSet"]):
        m0 = mentions[0]
        entities.append((m0["name"], m0.get("type", "")))
    by_type: dict[str, list[int]] = {}
    for eid, (_, t) in enumerate(entities):
        by_type.setdefault(t, []).append(eid)

    negatives: list[dict] = []
    for label in doc.get("labels", []):
        h_eid, t_eid = label["h"], label["t"]
        h_name, h_type = entities[h_eid]
        t_name, t_type = entities[t_eid]
        relation = rel_info.get(label["r"], label["r"])

        # 1. entity swap on head
        for eid in by_type.get(h_type, []):
            if eid != h_eid:
                negatives.append({
                    "subject": entities[eid][0], "relation": relation,
                    "object": t_name, "neg_kind": "entity_swap_head",
                })
                break
        # 2. entity swap on tail
        for eid in by_type.get(t_type, []):
            if eid != t_eid:
                negatives.append({
                    "subject": h_name, "relation": relation,
                    "object": entities[eid][0], "neg_kind": "entity_swap_tail",
                })
                break
        # 3. relation confusion
        choices = confusion_choices(relation)
        if choices:
            negatives.append({
                "subject": h_name, "relation": rng.choice(choices),
                "object": t_name, "neg_kind": "relation_confusion",
            })
        # 4. direction flip (asymmetric only)
        if relation not in SYMMETRIC and h_name != t_name:
            negatives.append({
                "subject": t_name, "relation": relation,
                "object": h_name, "neg_kind": "direction_flip",
            })

    if max_per_doc is not None:
        negatives = rng.sample(negatives, min(max_per_doc, len(negatives)))
    return negatives
