"""Ontology-gate candidates from the OntoAxiom ground truth.

Axiom types: subclass / disjoint / domain / range.
Positives = true axioms; negatives = reversed / sibling / unrelated corruptions,
verified against the full true set of the same ontology.

Split unit for eval: ontology (doc_id = ontology name).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path("data/ontoaxiom")
ONTOLOGIES = ["era", "foaf", "goodrelations", "gufo", "music",
              "nordstream", "pizza", "saref", "time"]


def load(onto: str, kind: str) -> list:
    f = DATA / kind / f"{onto}_{kind}.json"
    if not f.exists():
        return []
    return json.loads(f.read_text())


def build(rng: random.Random) -> list[dict]:
    cands: list[dict] = []

    def add(onto, ax, a, b, gold):
        cands.append({"doc_id": onto, "ontology": onto, "axiom": ax,
                      "a": a, "b": b, "gold": gold})

    for onto in ONTOLOGIES:
        sub = {(c, p) for c, p in load(onto, "subclassof")}
        dis = {(a, b) for a, b in load(onto, "disjoint")}
        dom = {(c, p) for c, p in load(onto, "domain")}
        ran = {(p, c) for p, c in load(onto, "range")}
        classes = list({c for c, _ in sub} | {p for _, p in sub}
                       | {a for a, _ in dis} | {b for _, b in dis}
                       | {c for c, _ in dom} | {c for _, c in ran})
        if not classes:
            continue
        children_of: dict[str, set] = {}
        for c, p in sub:
            children_of.setdefault(p, set()).add(c)

        # --- subclass ---
        sub_l = list(sub)
        n_pos = min(170, len(sub_l))
        for c, p in rng.sample(sub_l, n_pos):
            add(onto, "subclass", c, p, 1)
        neg = 0
        target_neg = n_pos
        # reversed
        for c, p in rng.sample(sub_l, len(sub_l)):
            if neg >= target_neg:
                break
            if (p, c) not in sub:
                add(onto, "subclass", p, c, 0)
                neg += 1
        # siblings
        sib_pairs = [(x, y) for p_, ch in children_of.items()
                     for x in ch for y in ch if x < y]
        rng.shuffle(sib_pairs)
        for x, y in sib_pairs:
            if neg >= target_neg:
                break
            if (x, y) not in sub and (y, x) not in sub:
                add(onto, "subclass", x, y, 0)
                neg += 1
        # random unrelated
        while neg < target_neg:
            x, y = rng.sample(classes, 2)
            if (x, y) not in sub and (y, x) not in sub:
                add(onto, "subclass", x, y, 0)
                neg += 1

        # --- disjoint ---
        dis_l = [(a, b) for a, b in dis if a < b]  # dedupe symmetric
        n_pos = min(140, len(dis_l))
        for a, b in rng.sample(dis_l, n_pos):
            add(onto, "disjoint", a, b, 1)
        neg = 0
        # subclass pairs are NOT disjoint
        rel = rng.sample(sub_l, min(len(sub_l), n_pos))
        for c, p in rel:
            if neg >= n_pos:
                break
            add(onto, "disjoint", c, p, 0)
            neg += 1
        while neg < n_pos:
            x, y = rng.sample(classes, 2)
            if (x, y) not in dis and (y, x) not in dis \
                    and (x, y) not in sub and (y, x) not in sub:
                add(onto, "disjoint", x, y, 0)
                neg += 1

        # --- domain ---
        dom_l = list(dom)
        n_pos = min(125, len(dom_l))
        for c, p in rng.sample(dom_l, n_pos):
            add(onto, "domain", p, c, 1)  # a=property, b=class
        neg = 0
        while neg < n_pos:
            c, p = rng.choice(dom_l)
            w = rng.choice(classes)
            if (w, p) not in dom:
                add(onto, "domain", p, w, 0)
                neg += 1

        # --- range ---
        ran_l = list(ran)
        n_pos = min(125, len(ran_l))
        for p, c in rng.sample(ran_l, n_pos):
            add(onto, "range", p, c, 1)
        neg = 0
        while neg < n_pos:
            p, c = rng.choice(ran_l)
            w = rng.choice(classes)
            if (p, w) not in ran:
                add(onto, "range", p, w, 0)
                neg += 1

    rng.shuffle(cands)
    return cands


def main() -> None:
    rng = random.Random(123)
    cands = build(rng)
    out = Path("results/candidates_onto1000.json")
    # cap at 1000 balanced-ish
    out.write_text(json.dumps(cands[:1000], ensure_ascii=False))
    from collections import Counter
    print(len(cands[:1000]), Counter(
        (c["axiom"], c["gold"]) for c in cands[:1000]), "->", out)


if __name__ == "__main__":
    main()
