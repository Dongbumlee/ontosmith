"""Promotion gate.

A graph version — identified by (batch_id, run_id, profile_hash) —
is promotable to served default only when ALL of these hold:

  (a0) the supplied profile_hash actually equals the one the
      batch recorded (batch.json / manifest pin) — the identity
      is compared, not echoed;
  (a) an audit record is attached: a JSON record carrying sample
      size, decided count, precision + Wilson CI, sampler seed,
      and label provenance (who/what labeled, under which rubric);
  (b) pinned-QA eval results exist from the PRODUCTION path (the
      query-stack measurement artifact, not an offline
      reimplementation — §7's contract);
  (c) the doc-check issues of every document the version touched
      are CLOSED in the Task Ledger (GitHub issues labelled
      doc-check, one per document).

The gate never fakes a pass: a missing artifact is a FAIL with the
missing condition named. `check` prints PASS/FAIL per condition
with evidence paths and writes the promotion record to
batches/<batch>/promotion/<run_id>.json (the record is data in the
repo, per §10's "audit numbers are data, visible per version").

CLI:
  python -m pipeline.promotion check --batch legal-full \
      --run run-1 --profile-hash sha256:f46cd0efb7b4b85c \
      --docs gap_register.md

The doc-check lookup talks to the project's own Task Ledger
(GitHub issues); the repository is deployment-specific, so it
is named by the ONTOSMITH_LEDGER_REPO environment variable
("owner/name") — no deployment identifier is baked into the
tree.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Task Ledger repository ("owner/name") for doc-check lookups,
#: from the environment (see module docstring).
LEDGER_REPO = os.environ.get("ONTOSMITH_LEDGER_REPO", "")

AUDIT_FIELDS = ("sample_size", "decided", "precision", "ci_low",
                "ci_high", "sampler_seed", "label_provenance")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def find_audit_record(batch_id: str, run_id: str) -> Path | None:
    """An audit record attaches to a version by living at
    batches/<batch>/audit/audit_record.json and naming the run,
    or at batches/<batch>/audit/audit_record_<run>.json."""
    d = ROOT / "batches" / batch_id / "audit"
    for cand in (d / f"audit_record_{run_id}.json",
                 d / "audit_record.json"):
        if cand.exists():
            try:
                rec = json.loads(cand.read_text())
            except Exception:
                continue
            if rec.get("run_id") in (None, run_id):
                return cand
    return None


def check_audit(batch_id: str, run_id: str) -> dict:
    path = find_audit_record(batch_id, run_id)
    if path is None:
        return {"pass": False, "evidence": None,
                "missing": ["no audit record attached to this "
                            "version (expected batches/<batch>/audit/"
                            "audit_record.json naming the run)"]}
    rec = json.loads(path.read_text())
    missing = [f"audit record lacks field: {f}" for f in AUDIT_FIELDS
               if rec.get(f) is None]
    if not missing and rec.get("decided", 0) <= 0:
        missing.append("audit record decided count is 0")
    return {"pass": not missing, "evidence": str(path.relative_to(ROOT)),
            "record": {f: rec.get(f) for f in AUDIT_FIELDS},
            "missing": missing}


def check_eval(batch_id: str, run_id: str) -> dict:
    """Pinned-QA eval from the production path. The production
    measurement artifact for the CUAD flagship is
    eval/p3b_ab.json (pipeline.query_stack.measure output: per-arm
    recall over the pinned QA, computed by the production
    retrievers). Generic batches: eval/eval_results.json with the
    same shape."""
    d = ROOT / "batches" / batch_id / "eval"
    for name in ("p3b_ab.json", "eval_results.json"):
        p = d / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        overall = data.get("overall_recall@20") or {}
        n = data.get("n_questions") or len(data.get("rows", []))
        if n and ("graph_current" in overall or "graph" in overall):
            return {"pass": True,
                    "evidence": str(p.relative_to(ROOT)),
                    "n_questions": n,
                    "overall_recall@20": {
                        k: overall[k] for k in
                        ("dense", "hybrid", "graph_current",
                         "graph_hybrid_current", "graph", "graph_hybrid")
                        if k in overall},
                    "missing": []}
    return {"pass": False, "evidence": None,
            "missing": ["no production-path pinned-QA eval artifact "
                        "(eval/p3b_ab.json or eval/eval_results.json "
                        "with overall recall + question rows)"]}


def doc_check_issue(doc: str) -> dict | None:
    """The Task Ledger issue for one document, via the gh CLI.
    Returns {number, state, url} or None when not found."""
    if not LEDGER_REPO:
        raise RuntimeError(
            "ONTOSMITH_LEDGER_REPO not set (Task Ledger repository, "
            "owner/name) — required for the doc-check lookup")
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", LEDGER_REPO,
         "--label", "doc-check", "--state", "all", "--limit", "100",
         "--json", "number,title,state,url"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"gh issue list failed: {out.stderr.strip()}")
    for issue in json.loads(out.stdout):
        if issue["title"].strip() == f"[doc-check] {doc}":
            return {"number": issue["number"], "state": issue["state"],
                    "url": issue["url"]}
    return None


def check_docs(docs: list[str]) -> dict:
    per_doc, missing = {}, []
    for doc in docs:
        try:
            issue = doc_check_issue(doc)
        except Exception as e:
            return {"pass": False, "evidence": "Task Ledger (gh)",
                    "missing": [f"Task Ledger unreachable: {e}"]}
        if issue is None:
            per_doc[doc] = None
            missing.append(f"{doc}: no [doc-check] issue in the "
                           "Task Ledger")
        else:
            per_doc[doc] = issue
            if issue["state"] != "CLOSED":
                missing.append(
                    f"{doc}: doc-check issue #{issue['number']} is "
                    f"{issue['state']} (must be CLOSED)")
    return {"pass": not missing, "evidence": "Task Ledger (gh)",
            "issues": per_doc, "missing": missing}


def check_profile_hash(batch_id: str, profile_hash: str) -> dict:
    """Compare the supplied profile hash against the one the
    batch itself recorded (batch.json; the manifest pin as
    fallback). The gate used to echo the
    caller's --profile-hash into the promotion record's version
    block without comparing it to anything, so a mistyped or
    stale hash 'passed' by construction — the version identity
    was asserted, never checked."""
    d = ROOT / "batches" / batch_id
    recorded = None
    src = None
    for name in ("batch.json", "manifest.json"):
        p = d / name
        if not p.exists():
            continue
        try:
            recorded = json.loads(p.read_text()).get("profile_hash")
        except Exception:
            recorded = None
        if recorded is not None:
            src = name
            break
    if recorded is None:
        return {"pass": False, "evidence": None,
                "missing": ["no profile hash recorded for this "
                            "batch (batch.json / manifest.json) — "
                            "the supplied hash cannot be confirmed"]}
    ok = recorded == profile_hash
    return {"pass": ok,
            "evidence": f"batches/{batch_id}/{src}",
            "recorded_profile_hash": recorded,
            "supplied_profile_hash": profile_hash,
            "missing": [] if ok else [
                f"profile hash mismatch: supplied "
                f"{profile_hash!r} != recorded {recorded!r}"]}


def check_promotion(batch_id: str, run_id: str, profile_hash: str,
                    docs: list[str]) -> dict:
    conditions = {
        "profile_hash_matches_record": check_profile_hash(
            batch_id, profile_hash),
        "audit_record": check_audit(batch_id, run_id),
        "pinned_qa_eval_production_path": check_eval(batch_id, run_id),
        "doc_check_issues_closed": check_docs(docs),
    }
    promotable = all(c["pass"] for c in conditions.values())
    return {
        "version": {"batch_id": batch_id, "run_id": run_id,
                    "profile_hash": profile_hash},
        "checked_at": _now(),
        "promotable": promotable,
        "conditions": conditions,
        "missing_conditions": [
            m for c in conditions.values() for m in c.get("missing", [])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="pipeline.promotion")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="run the promotion gate")
    ck.add_argument("--batch", required=True)
    ck.add_argument("--run", required=True)
    ck.add_argument("--profile-hash", default="")
    ck.add_argument("--docs", default="",
                    help="comma-separated docs/ filenames the "
                         "version touched")
    ck.add_argument("--write-record", action="store_true",
                    help="write batches/<batch>/promotion/<run>.json")
    args = ap.parse_args()
    docs = [d.strip() for d in args.docs.split(",") if d.strip()]
    rec = check_promotion(args.batch, args.run, args.profile_hash, docs)
    for name, cond in rec["conditions"].items():
        verdict = "PASS" if cond["pass"] else "FAIL"
        print(f"[{verdict}] {name}  (evidence: {cond.get('evidence')})")
        for m in cond.get("missing", []):
            print(f"       missing: {m}")
    print(f"OVERALL: {'PROMOTABLE' if rec['promotable'] else 'NOT PROMOTABLE'}"
          f" — {len(rec['missing_conditions'])} missing condition(s)")
    if args.write_record:
        out_dir = ROOT / "batches" / args.batch / "promotion"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.run}.json"
        out.write_text(json.dumps(rec, indent=1))
        print(f"record written: {out.relative_to(ROOT)}")
    return 0 if rec["promotable"] else 1


if __name__ == "__main__":
    sys.exit(main())
