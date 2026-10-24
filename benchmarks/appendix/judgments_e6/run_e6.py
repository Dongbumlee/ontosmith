#!/usr/bin/env python3
"""E-6 prep (repair program W-31): post-ruling cross-family
agreement measurement — CORRECTED DESIGN (round 3).

DESIGN NOTE (read before running). Whitepaper S5.4's scoped
sentence says cross-family agreement under the RULED contract
has not been re-measured. Getting the families right matters:

- The shipped RULED verdicts were produced by the recheck
  family, GPT-5.5 (`openai/gpt-5.5`) — the stricter judge of
  the 2026-10-22 stress test.
- The ORIGINAL adjudication panel was of the Anthropic line.

So a cross-family re-measurement judges a fresh sample with
an **Anthropic-line** judge and compares against the shipped
ruled (GPT-5.5) verdicts. Judging with GPT-5.5 again would
measure that family's self-consistency under the contract —
a different, weaker instrument. (The first staging of this
prep made exactly that mistake; the round-2 expert review
caught it. The --live path therefore REQUIRES --model and
refuses any model whose id does not start with `anthropic/`.)

  OPENROUTER_API_KEY=... python3 run_e6.py --live \
      --model anthropic/<model-id>

The model id is chosen and named at run time from the
record; it is deliberately not baked in here. Without
--live the script runs DRY: builds the sample, assembles
the judging packets, writes them out — no network, no
spend.

DISCIPLINE:
- Key from the environment only; never stored or printed.
- Hard spend cap $6: the runner tracks the provider's
  reported per-call cost and stops when cumulative spend
  reaches the cap (overshoot is bounded by one call's cost;
  per-call estimates are printed in the dry run header).
- The result is published in S5.4 as a dated addendum
  REGARDLESS of outcome.
- Agreement is reported with Cohen's kappa under BOTH
  conventions, labelled: (a) three-category over all
  sampled pairs (the basis of S5.4's published kappa ~0.45),
  (b) decisive-only.
- Overlap disclosure: the shipped appendix does not carry
  the first recheck's item list, so overlap between this
  fresh-seeded sample and the original 200 cannot be
  excluded from shipped data alone; the addendum must state
  the seed and this limitation.

Sample: direction-A items (sample_a), non-control,
stratified by the key file's stratum (pair_matched /
no-gold-pair per S5.4), proportional allocation, seed
20261024, n=150 by default (>= 120 required).

Packets carry the contract's own text: the base-contract
and seam-rulings sections of docs/ADJUDICATION.md, extracted
by heading at runtime — never a paraphrase.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

CAP_USD = 6.00
SEED = 20261024
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
RULED_FAMILY_PREFIX = "openai/"   # the ruled verdicts' family
JUDGE_FAMILY_PREFIX = "anthropic/"  # the original panel's family


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def contract_sections(text: str) -> str:
    """The base contract + the seam rulings, by heading."""
    out, keep = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line.startswith("## The base contract") or \
                line.startswith("## The seam rulings")
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def build_sample(appendix: Path, n: int, seed: int) -> list[dict]:
    items = {r["item_id"]: r for r in load_jsonl(appendix / "sample_a_items.jsonl")}
    keys = {r["item_id"]: r for r in load_jsonl(appendix / "sample_a_key.jsonl")}
    pool = [i for i, k in keys.items() if not k["control"]]
    by_stratum: dict[str, list[str]] = {}
    for i in pool:
        by_stratum.setdefault(keys[i]["stratum"], []).append(i)
    rng = random.Random(seed)
    sample: list[str] = []
    for stratum, ids in sorted(by_stratum.items()):
        take = round(n * len(ids) / len(pool))
        sample += rng.sample(sorted(ids), min(take, len(ids)))
    rng.shuffle(sample)
    return [{"item": items[i], "key": keys[i]} for i in sample]


def ruled_verdicts(appendix: Path) -> dict[str, str]:
    out = {}
    for shard in sorted((appendix / "judgments_ruled").glob("*.jsonl")):
        for row in load_jsonl(shard):
            out[row["id"]] = row["verdict"]
    return out


def packet(contract_text: str, item: dict) -> str:
    return (
        "Judge the claim below under the adjudication contract.\n\n"
        f"{contract_text}\n\n"
        "=== CLAIM ===\n"
        f"Subject: {item['subject']}\nRelation: {item['relation']}\n"
        f"Object: {item['object']}\n\n"
        "=== DOCUMENT TEXT ===\n"
        f"{item['context']}\n\n"
        "Answer with one JSON object only: "
        '{"verdict": "supported" | "not_supported" | "unclear", '
        '"evidence": "<verbatim quote from the document text, or empty>", '
        '"seam_applied": "<ruling code or none>", "note": "<one line>"}'
    )


def kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[v] / n) * (cb[v] / n) for v in set(ca) | set(cb))
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def _call_once(key: str, model: str, text: str) -> dict:
    # Transport via curl --http1.1: urllib reads are truncated
    # (IncompleteRead) by this VM's egress proxy — the same failure
    # and the same fix as the 2026-10-22 recheck run 1 -> run 2.
    # The key travels only in curl's stdin config (-K -): never
    # argv, never a file. The body (prompt text, not secret) goes
    # through a temp file that is deleted afterwards.
    body = {"model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "usage": {"include": True}}
    cfg = (f'header = "Authorization: Bearer {key}"\n'
           'header = "Content-Type: application/json"\n')
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as tf:
        tf.write(json.dumps(body))
        tpath = tf.name
    try:
        p = subprocess.run(
            ["curl", "-sS", "--http1.1", "--max-time", "300",
             "-K", "-", "--data", f"@{tpath}", CHAT_URL],
            input=cfg, capture_output=True, text=True, timeout=330)
        if p.returncode != 0:
            raise RuntimeError(
                f"curl exit {p.returncode}: {p.stderr.strip()[:200]}")
        data = json.loads(p.stdout)
        if "choices" not in data:
            raise RuntimeError(f"API error payload: {p.stdout[:300]}")
        return data
    finally:
        os.unlink(tpath)


def judge_live(key: str, model: str, text: str) -> tuple[dict, float]:
    # Retry with backoff on any transport/API failure (added
    # 2026-10-23 with Dongbum's approval). Packets, model and
    # verdict handling are unchanged.
    data = None
    for attempt in range(7):
        try:
            data = _call_once(key, model, text)
            break
        except Exception as e:
            if attempt == 6:
                raise
            wait = min(15 * 2 ** attempt, 120)
            print(f"  call failed ({type(e).__name__}: {str(e)[:120]}); "
                  f"retry in {wait:.0f}s (attempt {attempt + 2}/7)",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    assert data is not None
    cost = float((data.get("usage") or {}).get("cost") or 0.0)
    content = data["choices"][0]["message"]["content"]
    start = content.find("{")
    # Parse the FIRST complete JSON object and ignore any trailing
    # text/objects (a strict loads() crashed on a two-object reply).
    obj, _ = json.JSONDecoder().raw_decode(content[start:])
    return obj, cost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appendix", type=Path,
                    default=Path.home() / "workspace/ontosmith-staging/benchmarks/appendix")
    ap.add_argument("--contract", type=Path,
                    default=Path.home() / "workspace/ontosmith-staging/docs/ADJUDICATION.md")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--model", default="",
                    help="REQUIRED for --live: an anthropic/ model id "
                         "(the original panel's family). Refused otherwise.")
    args = ap.parse_args()

    sample = build_sample(args.appendix, args.n, args.seed)
    ruled = ruled_verdicts(args.appendix)
    contract = contract_sections(args.contract.read_text(encoding="utf-8"))
    packets = [{"item_id": s["item"]["item_id"], "stratum": s["key"]["stratum"],
                "packet": packet(contract, s["item"])} for s in sample]
    (args.out / "e6_sample.json").write_text(json.dumps(
        {"seed": args.seed, "n": len(sample),
         "strata": Counter(s["key"]["stratum"] for s in sample),
         "item_ids": [s["item"]["item_id"] for s in sample]}, indent=1, default=str))
    (args.out / "e6_packets.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in packets))
    print(f"sample: {len(sample)} items, strata "
          f"{dict(Counter(s['key']['stratum'] for s in sample))}; "
          f"contract sections: {len(contract)} chars")

    if not args.live:
        print("DRY RUN — packets written, no network calls, no spend.")
        return 0

    if not args.model.startswith(JUDGE_FAMILY_PREFIX):
        print(f"REFUSED: --model must be an {JUDGE_FAMILY_PREFIX}* id "
              f"(cross-family means the original panel's family; the "
              f"ruled verdicts are {RULED_FAMILY_PREFIX}*). Got: "
              f"{args.model or '(none)'}", file=sys.stderr)
        return 2
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        print("BLOCKED-BY-KEY: OPENROUTER_API_KEY is not set in the "
              "environment; the live run was not attempted.", file=sys.stderr)
        return 2

    items = {s["item"]["item_id"]: s["item"] for s in sample}
    # Incremental persistence + resume (bookkeeping only): every
    # verdict is appended to e6_verdicts.jsonl as it lands, and a
    # restarted run skips items already judged in that file. The
    # sample, packets and verdict handling are unchanged.
    vfile = args.out / "e6_verdicts.jsonl"
    spent, verdicts = 0.0, {}
    if vfile.exists():
        for line in vfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                verdicts[r["item_id"]] = r["verdict"]
                spent += float(r["cost"])
        if verdicts:
            print(f"RESUME: {len(verdicts)} verdicts already in "
                  f"{vfile.name} (${spent:.4f} spent); skipping those.")
    for p in packets:
        if p["item_id"] in verdicts:
            continue
        if spent >= CAP_USD:
            print(f"CAP: cumulative spend ${spent:.4f} reached the "
                  f"${CAP_USD} cap; stopping.")
            break
        v, cost = judge_live(key, args.model, p["packet"])
        spent += cost
        verdict = v.get("verdict", "unclear")
        ev = v.get("evidence") or ""
        if verdict == "supported" and ev not in items[p["item_id"]]["context"]:
            verdict = "unlabeled"  # quote verification, per contract
        verdicts[p["item_id"]] = verdict
        with vfile.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"item_id": p["item_id"],
                                 "verdict": verdict, "cost": cost}) + "\n")
        print(f"{p['item_id']}: {verdict} (${spent:.4f} cumulative)", flush=True)
        time.sleep(1.0)  # pacing between calls (transport only)

    all_pairs = [(ruled[i], verdicts[i]) for i in verdicts if i in ruled]
    dec_pairs = [(a, b) for a, b in all_pairs
                 if a in ("supported", "not_supported")
                 and b in ("supported", "not_supported")]
    po_all, k_all = kappa(all_pairs)
    po_dec, k_dec = kappa(dec_pairs)
    result = {"judge_model": args.model,
              "judge_family": "Anthropic line (original panel family)",
              "reference_family": "GPT-5.5 (ruled verdicts)",
              "seed": args.seed, "n_judged": len(verdicts),
              "agreement_all_categories": round(po_all, 4),
              "kappa_all_categories": round(k_all, 4),
              "n_decisive_pairs": len(dec_pairs),
              "agreement_decisive_only": round(po_dec, 4),
              "kappa_decisive_only": round(k_dec, 4),
              "spend_usd": round(spent, 4),
              "confusion_decisive": {f"{a}|{b}": n for (a, b), n
                                     in Counter(dec_pairs).items()}}
    (args.out / "e6_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
