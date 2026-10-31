"""The `seat` command: judge identity decision envelopes
with a model, producing a frozen decisions file.

The flow the identity-seat program measured (2026-10-30):

1. a batch run with `identity_seat_collect` declared
   harvests rule-4 decision points as envelopes
   (pipeline/seat_envelope.py) — that run's guard
   outcomes are the rule chain's alone;
2. this command renders each envelope through the
   frozen seat prompt and asks the declared model for
   complete / stand / abstain;
3. a later batch run with `identity_seat_decisions`
   declared consumes the verdicts; points the file does
   not cover — including every abstention and every
   failed call here — defer to the rule chain.

The judge can only choose between the census surfaces
inside the envelope; it cannot invent a name. Abstain
and error outcomes are recorded in the run record and
omitted from the decisions file (no opinion), never
silently converted to a verdict.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import connections

# The frozen seat prompt (identity-surgery D5,
# 2026-10-30; sha256 of SYSTEM + "\n" + USER below).
# Changing either string is a new seat contract: the
# measured behavior belongs to this exact text.
SYSTEM_PROMPT = """You are the identity-resolution seat for a knowledge-graph builder. A proposer wrote triples using a fragment name; the document census contains a longer name that shares the fragment's words. Decide whether the fragment, IN THIS DOCUMENT, refers to the same entity as the longer name, so the mention should be rewritten to it (complete), or the fragment must stand as written (stand). If the material given genuinely cannot decide the question, answer abstain.

Complete only when the longer form is the same referent's genuine fuller name: a surname or full name the document itself gives for this referent, a defined term's full form, an attested possessive or expanded form of the same name. Stand when the longer form is a different entity (an institution, work, place, or event that merely contains the name), a coordination of two names, a title or descriptive phrase wrapped around the name, words glued from neighboring text (headers, footers, bylines), a narrower derived entity (a division, edition, or sub-unit of the referent), or a corrupted variant of the name. A candidate that merely prefixes an article or determiner (The, the, A, An) to the fragment is never a fuller name — an article is not part of an entity's name — so stand in that case even when the document uses the article form in running text. Stand also when the fragment is an established short name with its own independent usage.

Base the decision only on the material below. Answer with JSON only: {"decision": "complete" or "stand" or "abstain", "reason": "<one line>"}"""

USER_TEMPLATE = """Fragment: {fragment!r}
Candidate longer name: {candidate!r}

Triple(s) using the fragment:
{items}

Document headline: {headline}
Document text:
{document_text}

Every occurrence of the fragment in the document:
{fragment_occurrences}

Every occurrence of the candidate in the document:
{candidate_occurrences}

Corpus registry for this document set:
- Other candidates paired with this fragment: {fragment_pairs}
- Other fragments paired with this candidate: {candidate_pairs}"""

PROMPT_SHA256 = hashlib.sha256(
    (SYSTEM_PROMPT + "\n" + USER_TEMPLATE).encode()
).hexdigest()


def render(envelope: dict) -> str:
    p = envelope["point"]
    items = "\n".join(
        f"  - ({i['s']!r}, {i['r']!r}, {i['o']!r})"
        for i in envelope["items"]) or "  (none)"
    fo = "\n".join(
        f"  | {o['sentence']}"
        for o in envelope["occurrences"]["fragment"]
    ) or "  (none)"
    co = "\n".join(
        f"  | {o['sentence']}"
        for o in envelope["occurrences"]["candidate"]
    ) or "  (none)"
    reg = envelope["registry"]
    fp = ", ".join(
        f"{x['candidate']} x{x['points']}"
        for x in reg["fragment_pairs"]) or "(none)"
    cp = ", ".join(
        f"{x['fragment']} x{x['points']}"
        for x in reg["candidate_pairs"]) or "(none)"
    return USER_TEMPLATE.format(
        fragment=p["fragment"], candidate=p["candidate"],
        items=items,
        headline=envelope["document"].get("headline", ""),
        document_text=envelope["document"]["text"],
        fragment_occurrences=fo, candidate_occurrences=co,
        fragment_pairs=fp, candidate_pairs=cp)


def _post_chat(binding, model: str, user: str,
               timeout: int):
    """One chat-completions call through a seat binding.
    Returns (content, cost_usd | None). The binding
    supplies the endpoint and key; usage is normalized
    through the connection layer, so cost is the
    provider-reported figure or None when the route does
    not meter."""
    payload = {"model": model, "temperature": 0,
               "max_tokens": 2000,
               "messages": [
                   {"role": "system",
                    "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user}]}
    if binding.is_openrouter_dialect:
        payload["usage"] = {"include": True}
    req = urllib.request.Request(
        binding.chat_url,
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer "
                 + binding.require_key(),
                 "X-Title": "ontosmith-seat"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    usage = connections.normalize_usage(
        resp.get("usage"), binding)
    return (resp["choices"][0]["message"]["content"],
            usage.get("cost"))


def judge_envelopes(envelopes: list[dict], post,
                    max_cost_usd: float | None = None,
                    concurrency: int = 6):
    """Judge every envelope through `post(user) ->
    (content, cost | None)`. Returns (decision rows,
    record). A row is written only for complete / stand
    verdicts; abstentions and failures are listed in the
    record and omitted (no opinion — the rule chain
    decides those points downstream). When the cost cap
    is reached, unjudged points are omitted the same way
    and the record carries cap_reached."""
    record = {"points": len(envelopes), "complete": 0,
              "stand": 0, "abstain": [], "errors": [],
              "unjudged": [], "total_cost_usd": 0.0,
              "cap_reached": False}
    spent = 0.0
    metered = True

    def one(env):
        cost = None
        try:
            content, cost = post(render(env))
            verdict = json.loads(content)
            decision = verdict["decision"]
            if decision not in ("complete", "stand",
                                "abstain"):
                raise ValueError(
                    f"unknown decision {decision!r}")
            return env, decision, cost, None
        except Exception as e:
            # A call that happened still cost what it
            # cost, even when its answer is unusable.
            return env, None, cost, \
                f"{type(e).__name__}: {e}"[:200]

    rows = []
    pending = list(envelopes)
    with ThreadPoolExecutor(
            max_workers=concurrency) as ex:
        while pending:
            if max_cost_usd is not None \
                    and spent >= max_cost_usd:
                record["cap_reached"] = True
                record["unjudged"] = [
                    e["point"] for e in pending]
                break
            batch, pending = pending[:concurrency], \
                pending[concurrency:]
            for env, decision, cost, err in ex.map(
                    one, batch):
                if cost is None:
                    metered = False
                else:
                    spent += cost
                pt = env["point"]
                if err is not None:
                    record["errors"].append(
                        {**pt, "error": err})
                elif decision == "abstain":
                    record["abstain"].append(pt)
                else:
                    record[decision] += 1
                    rows.append({
                        "doc": pt["doc"], "ep": pt["ep"],
                        "cand": pt["cand"],
                        "decision": decision})
    record["total_cost_usd"] = \
        round(spent, 6) if metered else None
    rows.sort(key=lambda r: (r["doc"], r["ep"], r["cand"]))
    return rows, record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ontosmith seat",
        description="Judge identity decision envelopes "
                    "with a model; write a frozen "
                    "decisions file.")
    ap.add_argument("--envelopes", required=True,
                    help="envelopes JSONL from a batch run "
                         "that declared identity_seat_collect")
    ap.add_argument("--out", required=True,
                    help="decisions JSONL to write "
                         "(consumed via identity_seat_decisions)")
    ap.add_argument("--model", required=True,
                    help="the seat model to judge with "
                         "(declared, never defaulted: the "
                         "seat's measured behavior belongs "
                         "to a named model)")
    ap.add_argument("--binding", default="judge",
                    help="seat binding supplying the "
                         "endpoint and key (default: judge)")
    ap.add_argument("--max-cost-usd", type=float,
                    default=None,
                    help="stop judging when the metered "
                         "spend reaches this cap; requires "
                         "a cost-reporting route")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)

    bindings = connections.resolve_bindings()
    binding = bindings.seat(args.binding)
    if args.max_cost_usd is not None \
            and not binding.reports_cost:
        raise connections.BindingCostError(
            f"seat binding '{args.binding}' does not "
            "report cost; a USD cap cannot guard spend it "
            "cannot meter — use a cost-reporting route or "
            "drop --max-cost-usd")
    envelopes = [json.loads(line) for line in
                 open(args.envelopes, encoding="utf-8")
                 if line.strip()]
    rows, record = judge_envelopes(
        envelopes,
        lambda user: _post_chat(
            binding, args.model, user, args.timeout),
        max_cost_usd=args.max_cost_usd,
        concurrency=args.concurrency)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)
                    + "\n")
    record = {"model": args.model,
              "binding": args.binding,
              "prompt_sha256": PROMPT_SHA256,
              "envelopes": str(args.envelopes),
              **record}
    rec_path = out.with_suffix(out.suffix + ".record.json")
    rec_path.write_text(json.dumps(
        record, ensure_ascii=False, indent=1),
        encoding="utf-8")
    cost_s = (f"${record['total_cost_usd']:.4f}"
              if record["total_cost_usd"] is not None
              else "not metered")
    print(f"seat: {record['points']} points -> "
          f"{record['complete']} complete, "
          f"{record['stand']} stand, "
          f"{len(record['abstain'])} abstain, "
          f"{len(record['errors'])} errors, "
          f"{len(record['unjudged'])} unjudged "
          f"| cost={cost_s} | wrote {out} + {rec_path.name}")
    if record["cap_reached"]:
        print("seat: cost cap reached; unjudged points "
              "are omitted (no opinion)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
