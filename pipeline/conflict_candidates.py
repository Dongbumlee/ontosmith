"""Conflict-classification candidates from the Neo4j graph.

Classes (250 each):
  duplicate     - new triple identical to an existing edge
  contradiction - same subject+functional-relation, different object
                  (place/date of birth/death, father, mother)
  update        - same subject+time-varying-relation, different object
                  (citizenship, employer, position, team, party)
  novel         - subject in graph, relation new for that subject

Needs NEO4J_PASSWORD in env.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

FUNCTIONAL = {
    "place of birth", "place of death", "date of birth", "date of death",
    "father", "mother",
}
TIME_VARYING = {
    "country of citizenship", "employer", "position held",
    "member of sports team", "member of political party", "residence",
}


def main() -> None:
    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        sys.exit("set NEO4J_PASSWORD")
    rng = random.Random(123)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pw))
    with driver.session() as s:
        edges = [(r["s"], r["rel"], r["o"]) for r in s.run(
            "MATCH (a:Entity)-[e]->(b:Entity) "
            "RETURN a.name AS s, e.relation AS rel, b.name AS o")]
    driver.close()

    by_rel = {}
    subj_rels: dict[str, set] = {}
    subj_facts: dict[str, list] = {}
    for sn, rel, ob in edges:
        by_rel.setdefault(rel, []).append((sn, ob))
        subj_rels.setdefault(sn, set()).add(rel)
        subj_facts.setdefault(sn, []).append((rel, ob))

    cands = []
    def add(subj, rel, obj, gold):
        cands.append({
            "doc_id": subj,  # split unit: no subject in both dev/held-out
            "subject": subj, "relation": rel, "object": obj,
            "existing_facts": subj_facts[subj], "gold": gold,
        })

    # duplicate
    for sn, rel, ob in rng.sample(edges, 250):
        add(sn, rel, ob, "duplicate")
    # contradiction
    func_edges = [(sn, rel, ob) for sn, rel, ob in edges if rel in FUNCTIONAL]
    i = 0
    while sum(1 for c in cands if c["gold"] == "contradiction") < 250:
        sn, rel, ob = func_edges[i % len(func_edges)]
        i += 1
        alts = [o for _, o in by_rel[rel] if o != ob]
        if not alts:
            continue
        add(sn, rel, rng.choice(alts), "contradiction")
    # update
    tv_edges = [(sn, rel, ob) for sn, rel, ob in edges if rel in TIME_VARYING]
    i = 0
    while sum(1 for c in cands if c["gold"] == "update") < 250:
        sn, rel, ob = tv_edges[i % len(tv_edges)]
        i += 1
        alts = [o for _, o in by_rel[rel] if o != ob]
        if not alts:
            continue
        add(sn, rel, rng.choice(alts), "update")
    # novel
    rels = list(by_rel)
    while sum(1 for c in cands if c["gold"] == "novel") < 250:
        sn = rng.choice(list(subj_rels))
        choices = [r for r in rels if r not in subj_rels[sn]]
        if not choices:
            continue
        rel = rng.choice(choices)
        ob = rng.choice(by_rel[rel])[1]
        add(sn, rel, ob, "novel")

    rng.shuffle(cands)
    out = Path("results/candidates_conflict1000.json")
    out.write_text(json.dumps(cands, ensure_ascii=False))
    from collections import Counter
    print(len(cands), Counter(c["gold"] for c in cands), "->", out)


if __name__ == "__main__":
    main()
