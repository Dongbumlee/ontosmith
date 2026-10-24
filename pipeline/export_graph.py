"""Export the Neo4j graph to JSON for visualization.

Reads the graph over the Neo4j HTTP transactional endpoint and
writes ``results/graph_export.json`` (nodes + edges). Connection
comes from the environment: NEO4J_URL names the
server base (default http://localhost:7474) and NEO4J_PASSWORD
the password — secrets are never read from files in the tree.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path


def tx_url() -> str:
    """Transactional-commit endpoint for the configured server."""
    base = os.environ.get("NEO4J_URL", "http://localhost:7474")
    return base.rstrip("/") + "/db/neo4j/tx/commit"


def _auth_header() -> str:
    pw = os.environ.get("NEO4J_PASSWORD", "")
    return "Basic " + base64.b64encode(f"neo4j:{pw}".encode()).decode()


def q(stmt, params=None):
    """Run one Cypher statement against the tx endpoint."""
    payload = {"statements": [{"statement": stmt, "parameters": params or {}}]}
    req = urllib.request.Request(
        tx_url(),
        data=json.dumps(payload).encode(),
        headers={"Authorization": _auth_header(),
                 "Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))


def main():
    """Export all nodes and edges (10k caps) to
    results/graph_export.json."""
    # nodes with labels and names
    r = q("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, n.name AS name LIMIT 10000")
    nodes = [
        {"id": row["row"][0], "labels": row["row"][1], "name": row["row"][2]}
        for row in r["results"][0]["data"]
    ]
    # relationships with types
    r = q(
        "MATCH (a)-[rel]->(b) RETURN id(a) AS src, id(b) AS dst, "
        "type(rel) AS type LIMIT 10000"
    )
    edges = [
        {"src": row["row"][0], "dst": row["row"][1], "type": row["row"][2]}
        for row in r["results"][0]["data"]
    ]
    out = {"nodes": nodes, "edges": edges}
    out_path = Path("results/graph_export.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"nodes={len(nodes)} edges={len(edges)}")


if __name__ == "__main__":
    main()
