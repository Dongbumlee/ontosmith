"""Drift / spot-audit check (G-23, BATCH_DESIGN §4.8).

Compares the live graph store against the source rows a projection
was built from, at ATTESTATION level: the key is
(subject_id, type, object_id, doc_id). A deterministic stride sample
(sorted keys, stride from the sample size, start offset from a
SHA-256 of the seed string) is drawn from BOTH sides with the same
rule; the report counts, within the sampled key sets:

  missing        sampled source keys with no store edge
  extra          sampled store keys with no source row
  property_drift keys present on both sides whose stored p_true or
                 tier differs from the source row (tolerance 1e-9)

This module is the single implementation: the batch VERIFY stage
(stage 6, the store drift spot audit) imports it, and the CLI below
runs the same core standalone:

  python -m pipeline.drift_check --batch <batch_id> [--root <root>]
      expected rows from the batch's written_triples.jsonl;
      p_true/tier are joined from routed.jsonl through the
      registry (resolve the routed surface in the row's corpus)
      and the manifest's own surface_relation. Rows that cannot
      be joined unambiguously are existence-only (counted, never
      guessed). KNOWN LIMIT: the manifest carries IDs, routed
      rows carry display strings; where distinct entities share
      a display string the join can pick a namesake row, so
      manifest-mode property_drift is an UPPER bound. Missing /
      extra are ID-exact.

Workspace discipline: the batch tree and the registry live under
the operator's state root (batch_cli.state_root), never beside
this module — an installed copy has no batches/ or data/ tree
next to its code. Callers pass ``root`` (the batch stages do);
the default resolves the state root lazily. The lab's
projection-source mode (expected rows from the p3b-v6 projection
record) stays in the lab harness: it is pinned to that
measurement's artifacts, not a product surface.

Exit code: 0 = no drift in the sample AND the property join
compared at least one attestation, 1 = drift found or nothing
was property-compared, 2 = usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

TOL = 1e-9


def _sample(keys: list, n: int, seed: str) -> list:
    keys = sorted(keys)
    if n <= 0 or len(keys) <= n:
        return keys
    stride = max(1, len(keys) // n)
    start = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % stride
    return keys[start::stride][:n]


def audit(expected: dict, actual: dict, sample_n: int = 400,
          seed: str = "drift-v1") -> dict:
    """expected/actual: {key: {"p_true": float|None, "tier": int|None}}
    with key = (subject_id, type, object_id, doc_id)."""
    exp_keys = _sample(list(expected), sample_n, seed + ":exp")
    act_keys = _sample(list(actual), sample_n, seed + ":act")
    missing = [k for k in exp_keys if k not in actual]
    extra = [k for k in act_keys if k not in expected]
    prop = []
    for k in exp_keys:
        if k not in actual:
            continue
        e, a = expected[k], actual[k]
        for field in ("p_true", "tier"):
            ev, av = e.get(field), a.get(field)
            if ev is None or av is None:
                continue
            if field == "p_true":
                bad = abs(float(ev) - float(av)) > TOL
            else:
                bad = int(ev) != int(av)
            if bad:
                prop.append({"key": k, "field": field,
                             "expected": ev, "actual": av})
    return {
        "expected_total": len(expected), "actual_total": len(actual),
        "sampled_expected": len(exp_keys), "sampled_actual": len(act_keys),
        "missing": len(missing), "extra": len(extra),
        "property_drift": len(prop),
        "missing_sample": [list(k) for k in missing[:10]],
        "extra_sample": [list(k) for k in extra[:10]],
        "property_drift_sample": [
            {"key": list(p["key"]), "field": p["field"],
             "expected": p["expected"], "actual": p["actual"]}
            for p in prop[:10]],
        "drift_free": not missing and not extra and not prop,
    }


def _state_root() -> Path:
    """The batch/state root rule (batch_cli.state_root),
    imported lazily to keep this module light."""
    from .batch_cli import state_root
    return state_root()


def _default_registry(root: Path):
    """The batch CLI's registry default: $JEV_REGISTRY_DB when
    set, else the state root's data/registry.db."""
    from .file_registry import SqliteEntityRegistry
    db = os.environ.get("JEV_REGISTRY_DB") or str(
        Path(root) / "data" / "registry.db")
    return SqliteEntityRegistry(db)


def _resolve_registry(root: Path, batch_id: str):
    """The registry the audit reads when no handle is passed.

    A per-batch registry file living in the batch's own directory
    (registry*.db — the convention for batch-scoped registries)
    wins over the state-root default. Run 32 (audit F2): resolving
    the default for a batch whose entities live in its own file
    joins nothing, so the audit compares zero properties while
    reporting drift_free on key existence alone — that is how the
    BioRED Stage D VERIFY recorded VERIFIED on an empty join.
    A plain registry.db in the batch dir wins over sibling
    registry_*.db files; a single sibling is unambiguous; with
    several non-plain siblings the choice would be a guess, so
    the state-root default applies and the join stats say so.
    """
    from .batch import batch_dir
    from .file_registry import SqliteEntityRegistry
    d = batch_dir(root, batch_id)
    cands = sorted(d.glob("registry*.db")) if d.exists() else []
    plain = d / "registry.db"
    pick = None
    if plain in cands:
        pick = plain
    elif len(cands) == 1:
        pick = cands[0]
    if pick is not None:
        return SqliteEntityRegistry(pick)
    return _default_registry(root)


def expected_from_manifest(batch_id: str, root: Path | None = None,
                           registry=None) -> tuple[dict, dict]:
    """Manifest rows give the attestation keys; p_true/tier join from
    routed.jsonl via (doc_id, resolved subject id, resolved object
    id, canonical relation). Ambiguous or unresolvable joins stay
    existence-only. Returns (expected, join_stats)."""
    from .batch import batch_dir
    root = Path(root) if root is not None else _state_root()
    d = batch_dir(root, batch_id)
    if registry is None:
        registry = _resolve_registry(root, batch_id)
    # Corpus (run 32, audit F2): batch.json's `source` is the
    # authoritative field — it is written by save_batch from the
    # Batch record itself. state.json's `source` is only a legacy
    # fallback: save_state did not write it before run 32, so the
    # state read silently yielded "" and every resolve missed the
    # corpus namespace. (save_state now writes it too; batch.json
    # stays authoritative.)
    corpus = ""
    batch_p = d / "batch.json"
    if batch_p.exists():
        try:
            corpus = json.loads(batch_p.read_text()).get(
                "source") or ""
        except Exception:
            corpus = ""
    meta = {}
    state_p = d / "state.json"
    if state_p.exists():
        meta = json.loads(state_p.read_text())
    if not corpus:
        corpus = (meta.get("source") or "")
    manifest = [json.loads(line)
                for line in open(d / "written_triples.jsonl")]
    man_keys = {(m["subject_id"], m["relation"], m["object_id"],
                 m["doc_id"]) for m in manifest}
    surf_rel = {(m["subject_id"], m["relation"], m["object_id"],
                 m["doc_id"]):
                m.get("surface_relation", m["relation"]) for m in manifest}
    routed_props: dict[tuple, list] = {}
    for line in open(d / "routed.jsonl"):
        r = json.loads(line)
        if r.get("tier", 0) not in (1, 2):
            continue
        s = registry.resolve(r["subject"], corpus=corpus or None)
        o = registry.resolve(r["object"], corpus=corpus or None)
        if s is None or o is None:
            continue
        key = (s.entity_id, o.entity_id, r["doc_id"])
        routed_props.setdefault(key, []).append(r)
    expected, joined, ambiguous = {}, 0, 0
    for key4 in man_keys:
        s_id, rel, o_id, doc_id = key4
        cands = [r for r in routed_props.get((s_id, o_id, doc_id), [])
                 if r["relation"] == surf_rel.get((s_id, rel, o_id,
                                                  doc_id))]
        if len(cands) == 1:
            expected[key4] = {"p_true": float(cands[0]["p_true"]),
                              "tier": int(cands[0]["tier"])}
            joined += 1
        else:
            expected[key4] = {"p_true": None, "tier": None}
            if len(cands) > 1:
                ambiguous += 1
    return expected, {"manifest_rows": len(manifest),
                      "attestation_keys": len(man_keys),
                      "joined": joined, "ambiguous": ambiguous,
                      "existence_only": len(man_keys) - joined,
                      "corpus": corpus,
                      "registry": str(getattr(registry, "db_path",
                                              None) or "handle")}


def actual_from_store(batch_id: str, run_id: str | None = None,
                      client=None, attestation_keys: bool = False) -> dict:
    from .neo4j_store import Neo4jClient
    c = client or Neo4jClient()
    cond = "{batch_id: $b}" if run_id is None else \
        "{batch_id: $b, run_id: $r}"
    params = {"b": batch_id}
    if run_id is not None:
        params["r"] = run_id
    rows = c.run([{
        "statement": f"MATCH (a:Entity)-[r:REL {cond}]->(b:Entity) "
                     "RETURN a.entity_id, r.type, b.entity_id, "
                     "r.doc_id, r.p_true, r.tier, r.att_id",
        "parameters": params}])[0]
    out = {}
    for r in rows:
        v = r["row"]
        key = (v[0], v[1], v[2], v[3] or "")
        if attestation_keys:
            key = (*key, v[6] or "")
        out[key] = {
            "p_true": None if v[4] is None else float(v[4]),
            "tier": None if v[5] is None else int(v[5])}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["manifest"],
                    default="manifest")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--root", default=None,
                    help="state root holding batches/ and data/ "
                         "(default: batch_cli.state_root())")
    ap.add_argument("--run", default=None)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", default="drift-v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else _state_root()
    expected, join_stats = expected_from_manifest(
        args.batch, root=root)
    actual = actual_from_store(args.batch, args.run)
    rep = audit(expected, actual, args.sample, args.seed)
    rep.update({"source": args.source, "batch_id": args.batch,
                "run_id": args.run, "seed": args.seed,
                "join": join_stats})
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
    # A sample with zero property-compared attestations proves
    # nothing about drift (run 32 guard, audit F2): it is not a
    # pass, whatever the key-existence comparison says.
    return 0 if rep["drift_free"] and join_stats["joined"] > 0 \
        else 1


if __name__ == "__main__":
    sys.exit(main())
