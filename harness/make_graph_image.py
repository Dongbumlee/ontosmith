"""Render the flagship build's full Tier 1 graph as a promotional image.

Real data only: the flagship batch's routed.jsonl (Tier 1 triples,
the same content projected into the graph store). The web look comes from
the corpus itself: each force-layout cluster is roughly one book's
world (Valjean's Paris, Quixote's La Mancha, Pierre's Russia...).

Output: docs/assets/graph-full.png (dark, high-res).

Palette is the brand stage mode (visual-unification design):
deep-navy ground, cream labels, and community colors derived
from the brand hues (amber / clay / teal / steel) instead of
the stock tab20 family.
"""
import colorsys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Brand stage tokens (design §3: deep ground, cream edges).
GROUND = "#0a1620"
EDGE = "#f6f0e4"
LABEL = "#f6f0e4"
CAPTION = "#cfe3ee"
FAINT = "#5f8296"

# Community palette: 12 muted colors from the brand hues,
# cycled (design §3) — deterministic lightness/saturation
# variants of amber / clay / teal / steel.
_BRAND_HUES = (0.09, 0.04, 0.45, 0.57)  # amber, clay, teal, steel


def _brand_palette(n):
    variants = ((0.62, 0.50), (0.50, 0.58), (0.70, 0.42))
    out = []
    for light, sat in variants:
        for hue in _BRAND_HUES:
            r, g, b = colorsys.hls_to_rgb(hue, light, sat)
            out.append((r, g, b))
    return out[:max(n, 1)]

ROOT = Path(__file__).parent
BATCH = ROOT / "batches" / "clean-main-v1"

G = nx.Graph()
assoc = []
for line in open(BATCH / "routed.jsonl"):
    r = json.loads(line)
    if r.get("tier") != 1:
        continue
    s, o = r["subject"].strip(), r["object"].strip()
    if not s or not o or s == o:
        continue
    if r["relation"] == "associated_with":
        assoc.append((s, o))
    else:
        G.add_edge(s, o)
for s, o in assoc:
    if s in G and o in G and not G.has_edge(s, o):
        G.add_edge(s, o)
# largest connected component only (the web); specks dropped
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
n, m = G.number_of_nodes(), G.number_of_edges()
print(f"graph: {n} entities, {m} relations")

# communities for coloring (roughly = book worlds)
# v2: Louvain instead of greedy modularity (sharper boundaries).
import community as community_louvain

partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
color_of = partition
n_comms = len(set(partition.values()))
print(f"communities: {n_comms}")

pos = nx.spring_layout(G, seed=42, k=0.55 / np.sqrt(n), iterations=120)
deg = dict(G.degree())
sizes = np.array([22 + 260 * np.sqrt(deg[v]) / np.sqrt(max(deg.values()))
                  for v in G.nodes()])

# v2: every community gets its OWN color (v1 reused 20 colors for
# 26 communities, so distant clusters falsely shared a color).
# v3: the colors come from the brand hues, not tab20.
palette = _brand_palette(n_comms)
colors = [palette[color_of[v] % len(palette)] for v in G.nodes()]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Noto Sans", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(16, 12), dpi=200)
fig.patch.set_facecolor(GROUND)
ax.set_facecolor(GROUND)
ax.axis("off")

edges = list(G.edges())
ex = [pos[u][0] for u, v in edges] + [pos[v][0] for u, v in edges]
nx.draw_networkx_edges(G, pos, ax=ax, width=0.35, edge_color=EDGE,
                       alpha=0.16)
# glow pass, then core pass
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes * 3.2,
                       node_color=colors, alpha=0.10, linewidths=0)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                       node_color=colors, alpha=0.92, linewidths=0)

# label only clean, recognizable hubs (no pronoun artifacts)
LABEL = {"Marius", "Don Quixote", "Pierre", "Cosette", "Jean Valjean",
         "Prince Andrew", "Monte Cristo", "Sancho", "Mr. Pickwick",
         "Natásha", "Nicholas", "Stephen", "Sherlock Holmes",
         "Elizabeth", "Jane Eyre", "Emma", "Odysseus", "Achilles"}
labels = {v: v for v in G.nodes() if v in LABEL}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10.5,
                        font_color=LABEL, font_weight="bold")

ax.text(0.5, 0.035,
        "jev Knowledge Graph — 391 documents, one web of verified facts",
        transform=ax.transAxes, ha="center", color=CAPTION,
        fontsize=12.5)
ax.text(0.5, 0.008,
        f"{n:,} entities · {m:,} Tier 1 relations · each cluster a book's world",
        transform=ax.transAxes, ha="center", color=FAINT,
        fontsize=10)
out = ROOT / "docs" / "assets"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "graph-full-v2.png", facecolor=fig.get_facecolor(),
            bbox_inches="tight", pad_inches=0.25)
print("saved:", out / "graph-full-v2.png")
