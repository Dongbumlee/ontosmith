"""Evidence/provenance viewer (plan §7 R9): ``ontosmith
inspect``.

Every shipped triple is inspectable down to its evidence
spans and provenance chain — the accountability claim made
tangible at the level of a single fact. This is the v1 form:
a command/report, not a web UI.

Accepted artefacts:

- a JSONL file of triple records — any of the pipeline's
  record families (judged candidates, routed/emitted
  records, ``written_triples`` projections, supplementary
  emissions). Records are normalized across families: the
  same fact reads the same way regardless of which stage
  wrote it.
- a batch directory — resolved to its richest triple
  artefact, in order: ``judgments.jsonl``,
  ``routed.jsonl``, ``written_triples.jsonl``.

The viewer is honest about absence: a triple with no
recorded evidence span says so, in the report, under Gaps —
an uninspectable fact is a finding, not a rendering error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BATCH_ARTEFACTS = ("judgments.jsonl", "routed.jsonl",
                   "written_triples.jsonl")
SUBJECT_KEYS = ("subject", "subject_id")
OBJECT_KEYS = ("object", "object_id")
RELATION_KEYS = ("relation", "surface_relation")
EVIDENCE_KEYS = ("quote", "evidence_sentence", "evidence",
                 "sentence", "context")
PROVENANCE_KEYS = ("doc_id", "batch_id", "run_id", "corpus",
                   "tier", "tag", "p_true", "cost", "model",
                   "gate_model", "attest", "status",
                   "frozen_block_reason", "error")


def resolve_artefact(path) -> Path:
    """Resolve a CLI artefact argument to a records file.

    A directory is treated as a batch directory and resolved
    to the first known triple artefact inside it. Raises
    ValueError (naming what was looked for) when nothing
    resolves.
    """
    p = Path(path)
    if p.is_dir():
        for name in BATCH_ARTEFACTS:
            cand = p / name
            if cand.is_file():
                return cand
        raise ValueError(
            f"{p} is a directory with no triple artefact "
            f"(looked for {', '.join(BATCH_ARTEFACTS)})")
    if p.is_file():
        return p
    raise ValueError(f"artefact not found: {p}")


def load_records(path) -> list[dict]:
    """Parse a records artefact: JSONL (one object per line)
    or a .json file holding an object / array of objects.

    A malformed line refuses the whole read, naming the
    line — a partially-read evidence artefact would render
    a partially-true report.
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: invalid JSON: {e}") from e
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError(f"{p}: JSON artefact is neither an "
                         f"object nor an array of objects")
    records: list[dict] = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{p}:{i}: malformed JSONL line: {e}") from e
        if not isinstance(rec, dict):
            raise ValueError(
                f"{p}:{i}: record is not a JSON object")
        records.append(rec)
    return records


def _first(rec: dict, keys) -> tuple[str | None, object]:
    """First present, non-None value among ``keys``:
    (key, value), or (None, None)."""
    for k in keys:
        if rec.get(k) is not None:
            return k, rec[k]
    return None, None


def is_triple_record(rec: dict) -> bool:
    """A record is triple-shaped when it carries a relation
    and at least one endpoint (any family's key spelling)."""
    _, rel = _first(rec, RELATION_KEYS)
    _, subj = _first(rec, SUBJECT_KEYS)
    _, obj = _first(rec, OBJECT_KEYS)
    return rel is not None and (subj is not None
                                or obj is not None)


def build_report(rec: dict, source, position: int,
                 total_triples: int) -> dict:
    """Normalize one triple record into the inspection
    report: the triple, its evidence spans (labeled by the
    field that carried them), its provenance chain (the
    known keys, in pipeline order), and its gaps — the
    accountability aspects the record does *not* carry.
    """
    _, subj = _first(rec, SUBJECT_KEYS)
    _, obj = _first(rec, OBJECT_KEYS)
    _, rel = _first(rec, RELATION_KEYS)
    triple = {"subject": subj, "relation": rel, "object": obj}
    _, surface = _first(rec, ("surface_relation",))
    if surface is not None and surface != rel:
        triple["surface_relation"] = surface

    evidence: list[dict] = []
    seen: set[str] = set()
    for key in EVIDENCE_KEYS:
        val = rec.get(key)
        if isinstance(val, str) and val.strip() \
                and val not in seen:
            seen.add(val)
            evidence.append({"kind": key, "text": val})

    provenance = {k: rec[k] for k in PROVENANCE_KEYS
                  if rec.get(k) is not None}

    gaps: list[str] = []
    if not evidence:
        gaps.append("no evidence span recorded")
    if "doc_id" not in provenance:
        gaps.append("no source document (doc_id) recorded")
    if not any(k in provenance for k in
               ("batch_id", "status", "tier", "tag")):
        gaps.append("no batch/status provenance recorded")

    return {
        "source": str(source),
        "position": position,
        "triples": total_triples,
        "triple": triple,
        "evidence": evidence,
        "provenance": provenance,
        "gaps": gaps,
    }


def select(records: list[dict], index: int | None = None,
           subject: str | None = None,
           relation: str | None = None,
           object_: str | None = None) -> tuple[dict, int, int]:
    """Select one triple record: case-insensitive substring
    filters over the family spellings, then ``index`` among
    the matches (default: the first). Returns (record,
    position among triple records, triple-record count).
    Refuses — ValueError — on an empty artefact, no
    triple-shaped records, no filter match, or an
    out-of-range index.
    """
    triples = [r for r in records if is_triple_record(r)]
    if not records:
        raise ValueError("artefact holds no records")
    if not triples:
        raise ValueError("artefact holds no triple records")

    def _matches(rec: dict, keys, needle: str) -> bool:
        _, val = _first(rec, keys)
        return val is not None and needle.lower() \
            in str(val).lower()

    matched = triples
    for keys, needle in ((SUBJECT_KEYS, subject),
                         (RELATION_KEYS, relation),
                         (OBJECT_KEYS, object_)):
        if needle is not None:
            matched = [r for r in matched
                       if _matches(r, keys, needle)]
    if not matched:
        raise ValueError("no triple matches the given "
                         "filter(s)")
    i = 0 if index is None else index
    if i < 0 or i >= len(matched):
        raise ValueError(
            f"index {i} out of range: {len(matched)} "
            f"matching triple(s)")
    rec = matched[i]
    return rec, triples.index(rec), len(triples)


def render_card(report: dict) -> str:
    """The human report card (R15 style): the fact, its
    evidence, its provenance chain, its gaps."""
    t = report["triple"]
    lines = [
        f"Triple  ({report['position'] + 1} of "
        f"{report['triples']} in {report['source']})",
        f"  subject   {t['subject']}",
        f"  relation  {t['relation']}",
        f"  object    {t['object']}",
    ]
    if "surface_relation" in t:
        lines.append(
            f"  (surface relation: {t['surface_relation']})")
    lines.append("Evidence")
    if report["evidence"]:
        for ev in report["evidence"]:
            lines.append(f"  [{ev['kind']}] {ev['text']}")
    else:
        lines.append("  (none recorded)")
    lines.append("Provenance")
    if report["provenance"]:
        for k, v in report["provenance"].items():
            lines.append(f"  {k:<20} {v}")
    else:
        lines.append("  (none recorded)")
    if report["gaps"]:
        lines.append("Gaps")
        for g in report["gaps"]:
            lines.append(f"  - {g}")
    return "\n".join(lines)


def main(argv=None) -> int:
    """``ontosmith inspect <artefact> [--index N]
    [--subject S] [--relation R] [--object O] [--json]``."""
    ap = argparse.ArgumentParser(
        prog="ontosmith inspect",
        description="Evidence/provenance viewer (plan §7 "
                    "R9): print one triple with its "
                    "evidence spans and provenance chain")
    ap.add_argument("artefact",
                    help="triple-records JSONL/.json file, "
                         "or a batch directory")
    ap.add_argument("--index", type=int, default=None,
                    help="index among matching triples "
                         "(default: first match)")
    ap.add_argument("--subject", default=None,
                    help="subject substring filter")
    ap.add_argument("--relation", default=None,
                    help="relation substring filter")
    ap.add_argument("--object", default=None,
                    help="object substring filter")
    ap.add_argument("--json", action="store_true",
                    help="print the inspection report as "
                         "JSON instead of the report card")
    args = ap.parse_args(argv)

    try:
        path = resolve_artefact(args.artefact)
        records = load_records(path)
        rec, position, n_triples = select(
            records, index=args.index, subject=args.subject,
            relation=args.relation, object_=args.object)
    except ValueError as e:
        print(f"inspect refused: {e}", file=sys.stderr)
        return 1
    report = build_report(rec, path, position, n_triples)
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        print(render_card(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
