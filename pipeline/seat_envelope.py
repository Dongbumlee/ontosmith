"""Identity-seat decision envelopes.

The material an offline identity judge rules from, per
rule-4 decision point: the document (headline + full
text), the triples that use the fragment, every sentence
attesting the fragment and the candidate, and the
document registry — the other candidates this fragment
was paired with in the same document, and the other
fragments this candidate was paired with there. The
envelope is
the seat's whole world: it carries no labels and no rule
outcomes, and the judge can only choose between census
surfaces already inside it.

Built by the CANDIDATES stage when a profile declares
`identity_seat_collect`; consumed by the `seat` command
(pipeline/seat_cli.py), whose verdicts return as a frozen
decisions file for a later run.
"""
from collections import Counter, defaultdict

from .proposer import norm_ws, split_sentences


def _occurrences(text: str, surface: str) -> list[dict]:
    """Sentences of the document that attest the surface
    (normalized containment), in document order."""
    target = norm_ws(surface)
    if not target:
        return []
    return [{"sentence": s} for s in split_sentences(text)
            if target in norm_ws(s)]


def _headline(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def build_envelopes(points: list[dict],
                    texts_by_doc_key: dict[str, str],
                    items_by_doc_key: dict[str, list[dict]],
                    ) -> list[dict]:
    """One envelope per unique decision point.

    `points` are the collecting seat's raw records
    ({"doc", "ep", "cand", "fragment", "candidate"}; the
    same point may be consulted more than once and is
    deduplicated here). `texts_by_doc_key` maps the
    seat's document key to the document text;
    `items_by_doc_key` maps it to the document's proposed
    triples ({"s", "r", "o"} records — kept candidates and
    guard-blocked proposals alike)."""
    unique: dict[tuple[str, str, str], dict] = {}
    for p in points:
        unique.setdefault((p["doc"], p["ep"], p["cand"]), p)
    # The registry is per-document, matching the measured
    # seat contract: the other candidates this fragment
    # was paired with in the same document, and the other
    # fragments this candidate was paired with there.
    frag_pairs: dict[tuple[str, str], Counter] = \
        defaultdict(Counter)
    cand_pairs: dict[tuple[str, str], Counter] = \
        defaultdict(Counter)
    for p in unique.values():
        frag_pairs[(p["doc"], p["fragment"])][
            p["candidate"]] += 1
        cand_pairs[(p["doc"], p["candidate"])][
            p["fragment"]] += 1
    envelopes = []
    for idx, key in enumerate(sorted(unique)):
        p = unique[key]
        text = texts_by_doc_key.get(p["doc"], "")
        frag, cand = p["fragment"], p["candidate"]
        items = [it for it in items_by_doc_key.get(p["doc"], [])
                 if norm_ws(it.get("s", "")) == p["ep"]
                 or norm_ws(it.get("o", "")) == p["ep"]]
        envelopes.append({
            "idx": idx,
            "point": {"doc": p["doc"], "ep": p["ep"],
                      "cand": p["cand"], "fragment": frag,
                      "candidate": cand},
            "document": {"headline": _headline(text),
                         "text": text},
            "items": items,
            "occurrences": {
                "fragment": _occurrences(text, frag),
                "candidate": _occurrences(text, cand)},
            "registry": {
                "fragment_pairs": [
                    {"candidate": c, "points": n}
                    for c, n in sorted(frag_pairs[
                        (p["doc"], frag)].items())
                    if c != cand],
                "candidate_pairs": [
                    {"fragment": f, "points": n}
                    for f, n in sorted(cand_pairs[
                        (p["doc"], cand)].items())
                    if f != frag]},
        })
    return envelopes
