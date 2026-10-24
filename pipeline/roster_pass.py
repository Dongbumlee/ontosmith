"""Mention-first roster pass (BioRED design §4.1).

One LLM call per document
asking for MENTIONS ONLY — every mention of the profile's entity
types, including mentions that participate in no relation. The
operative instruction sentence ("...including mentions that
participate in no relation") is the operationalization of the
H-A1 test (relation-coupled extraction): the earlier roster
was the endpoints of kept triples, so entities the free proposer
built no triple around never entered it; this pass decouples
mention extraction from relation proposal.

The prompt is built through pipeline/profile_config.derive_guidance
(the #51 channel — the same schema text the proposers see) ahead
of the mention instruction. Output passes deterministic guards
mirroring the pipeline's verbatim discipline:

- an emitted surface must occur in the doc text under the
  roster's occurrence semantics (whitespace-normalized,
  case-preserved substring — pipeline.pair_proposer.norm_surface);
  non-verbatim surfaces are dropped AND counted;
- an emitted type must be in the profile's closed endpoint-type
  vocabulary (endpoint_type_names); others dropped and counted;
- duplicates by (normalized surface, type) collapse; a surface
  emitted with two types keeps BOTH rows and is counted as a
  roster type conflict.

The pass is an artifact-producing stage: its output is
materialized into roster_v2.jsonl and is thereafter a
frozen input. It is not a pipeline stage.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from .pair_proposer import norm_surface
from .proposer import _chat, _is_haiku55
from .reference_profile import PROPOSER_MODEL

DEFAULT_MODEL = PROPOSER_MODEL

MENTION_INSTRUCTION = """

---

Mention pass (round 4 roster — mentions only, NO relations):

From the document text below, list every mention of the profile's entity types in this text, including mentions that participate in no relation.
A mention is a verbatim span of the text that names
an entity of one of the profile's entity types (see the schema
above). List each distinct mention once, with the exact text of
the mention as it appears in the document and its entity type
(the exact type name from the schema). Do NOT list relations,
and do NOT invent or normalize mention text.

Document text:
---
{text}
---

Answer with ONLY a JSON array, no other text:
[{{"surface": "<verbatim mention text>", "type": "<exact entity type name>"}}, ...]
"""


def build_mention_prompt(text: str, guidance: str = "") -> str:
    """Guidance channel first (the #51 discipline), then the
    mention instruction with the document text."""
    q = MENTION_INSTRUCTION.format(text=text)
    if guidance:
        return guidance + q
    return q.lstrip("\n")


def prompt_template_sha256() -> str:
    """The pinned prompt template's hash (design §5.4: the prompt
    is hashed and committed before the stage runs; the run record
    carries this hash plus the guidance hash)."""
    return hashlib.sha256(MENTION_INSTRUCTION.encode()).hexdigest()


def _parse_mentions(content: str) -> list[dict]:
    """Parse the model's JSON array reply. Unparseable -> []."""
    txt = (content or "").strip()
    if txt.startswith("```"):
        lines = txt.split("\n")
        txt = "\n".join(lines[1:-1] if lines[-1].strip() == "```"
                        else lines[1:])
    try:
        arr = json.loads(txt)
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    out = []
    for it in arr:
        if isinstance(it, dict) and it.get("surface") \
                and it.get("type"):
            out.append({"surface": str(it["surface"]),
                        "type": str(it["type"])})
    return out


def guard_mentions(raw: list[dict], text: str,
                   valid_types: set[str]) -> dict:
    """Deterministic guards of design §4.1.

    Returns {mentions, raw_n, dropped_nonverbatim, dropped_type,
    type_conflicts}. Mentions are deduplicated by (normalized
    surface, type) and sorted by (normalized surface, type) —
    the roster's canonical order. The emitted surface kept is
    the whitespace-normalized form (the roster's own surface
    semantics; the verbatim test is on that same normalized
    form, case preserved).
    """
    norm_text = norm_surface(text)
    seen: dict[tuple[str, str], dict] = {}
    dropped_nonverbatim = 0
    dropped_type = 0
    for it in raw:
        surf = norm_surface(it.get("surface", ""))
        typ = (it.get("type") or "").strip()
        if not surf or surf not in norm_text:
            dropped_nonverbatim += 1
            continue
        if typ not in valid_types:
            dropped_type += 1
            continue
        seen.setdefault((surf, typ), {"surface": surf, "type": typ})
    mentions = sorted(seen.values(),
                      key=lambda m: (norm_surface(m["surface"]),
                                     m["type"]))
    by_surf: dict[str, set[str]] = {}
    for m in mentions:
        by_surf.setdefault(m["surface"], set()).add(m["type"])
    conflicts = sum(1 for ts in by_surf.values() if len(ts) > 1)
    return {"mentions": mentions, "raw_n": len(raw),
            "dropped_nonverbatim": dropped_nonverbatim,
            "dropped_type": dropped_type,
            "type_conflicts": conflicts}


def propose_mentions_for_doc(doc_id: str, text: str, key: str,
                             model: str = DEFAULT_MODEL,
                             guidance: str = "",
                             valid_types: set[str] | None = None,
                             chat_fn=None) -> dict:
    """One mention-first call for one document, guards applied.

    Returns {doc_id, mentions, raw_n, dropped_nonverbatim,
    dropped_type, type_conflicts, cost, error}. On call error the
    mention list is empty and error is set — the caller records
    it; nothing is silently retried beyond the pipeline-standard
    error handling of the runner script.
    """
    chat = chat_fn or _chat
    try:
        content, u = chat(build_mention_prompt(text, guidance),
                          model, 12000 if _is_haiku55(model) else 6000,
                          key)
        raw = _parse_mentions(content)
        err = None
        cost = u.get("cost")
    except Exception as e:
        raw, err, cost = [], str(e)[:200], 0.0
    g = guard_mentions(raw, text, set(valid_types or set()))
    return {"doc_id": doc_id, "mentions": g["mentions"],
            "raw_n": g["raw_n"],
            "dropped_nonverbatim": g["dropped_nonverbatim"],
            "dropped_type": g["dropped_type"],
            "type_conflicts": g["type_conflicts"],
            "cost": None if cost is None else round(cost, 4),
            "error": err}


def propose_mentions_for_docs(docs: list[tuple[str, str]], key: str,
                              model: str = DEFAULT_MODEL,
                              guidance: str = "",
                              valid_types: set[str] | None = None,
                              workers: int = 8) -> list[dict]:
    """The pass over many docs (order preserved)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(
            lambda dt: propose_mentions_for_doc(
                dt[0], dt[1], key, model=model, guidance=guidance,
                valid_types=valid_types), docs))
