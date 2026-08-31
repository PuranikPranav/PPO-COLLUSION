"""Figure 0 — the market, drawn.

One 5-node Liu & Hobbs topology, two ownership structures. Everything on the
diagram (line limits, demand scale, plant costs, capacities, siting) is READ
from iso_market.node_network under each MARKET_CONFIG, so the picture cannot
drift from the model it describes.

Node numbering follows the papers (1..5); the code indexes them 0..4, so the
"node 2" generation hub is index 1.

    python -m qlearning_collusion.fig0_network
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlearning_collusion.figures import _save, SERIES, INK, INK2, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))

# Node positions: triangle 1-2-3 on the left (the loop), radial tail 3-4-5.
POS = {
    0: (0.00, 0.95),    # paper's node 1
    1: (0.00, -0.05),   # paper's node 2  <- the generation hub
    2: (1.20, 0.45),    # paper's node 3
    3: (2.40, 0.45),    # paper's node 4
    4: (3.60, 0.45),    # paper's node 5
}
LINES = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
HUB = 1
R = 0.17


def _read_config(tag: str) -> dict:
    """Load node_network under a given MARKET_CONFIG."""
    old = os.environ.get("MARKET_CONFIG")
    os.environ["MARKET_CONFIG"] = tag
    import iso_market.node_network as nn
    importlib.reload(nn)
    cfg = dict(
        P0=nn.P0.copy(), Q0=nn.Q0.copy(),
        LINE_LIMITS=nn.LINE_LIMITS.copy(),
        MC=dict(nn.MC), QC=dict(nn.QC), CAP=dict(nn.CAP),
        SPECS=list(nn.PLANT_SPECS),
    )
    if old is None:
        os.environ.pop("MARKET_CONFIG", None)
    else:
        os.environ["MARKET_CONFIG"] = old
    return cfg


def _delta(results_dir: str) -> float | None:
    p = os.path.join(HERE, results_dir, "imperfect_stochastic.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)["result"]["delta"]


def _draw(ax, cfg, title, subtitle, delta):
    n_firms = len(set(f for f, _, _ in cfg["SPECS"]))
    # centroid of the 1-2-3 loop, so loop-edge labels can be pushed OUTSIDE it
    tri = np.array([POS[0], POS[1], POS[2]]).mean(axis=0)

    # ---- transmission lines ------------------------------------------------
    for li, (u, v) in enumerate(LINES):
        x0, y0 = POS[u]; x1, y1 = POS[v]
        ax.plot([x0, x1], [y0, y1], color="#b9b8b2", lw=2.6, zorder=1,
                solid_capstyle="round")
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        px, py = -dy / L, dx / L                      # unit normal
        if li <= 2:                                   # loop edge: point outward
            if (px * (mx - tri[0]) + py * (my - tri[1])) < 0:
                px, py = -px, -py
        elif py < 0:                                  # radial edge: label above
            px, py = -px, -py
        ang = np.degrees(np.arctan2(dy, dx))          # keep text upright
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        ax.text(mx + px * 0.17, my + py * 0.17, f"{cfg['LINE_LIMITS'][li]:g} MW",
                color=MUTED, fontsize=7.6, ha="center", va="center",
                rotation=ang, rotation_mode="anchor", zorder=2)

    # ---- nodes -------------------------------------------------------------
    gen_nodes = {n for _, n, _ in cfg["SPECS"]}
    for i, (x, y) in POS.items():
        # highlight every node that hosts generation, whatever the structure
        is_hub = i in gen_nodes
        ax.add_patch(Circle((x, y), R, facecolor="#eef3fb" if is_hub else "white",
                            edgecolor=INK if is_hub else "#9a9993",
                            lw=2.2 if is_hub else 1.3, zorder=3))
        ax.text(x, y, str(i + 1), color=INK, fontsize=11, ha="center",
                va="center", fontweight="semibold", zorder=4)
        # demand scale: above node 1 (nothing there), below everywhere else
        if i == 0:
            ax.text(x, y + R + 0.10, f"D {cfg['Q0'][i]:g}", color=MUTED,
                    fontsize=7.4, ha="center", va="bottom", zorder=4)
        else:
            ax.text(x, y - R - 0.11, f"D {cfg['Q0'][i]:g}", color=MUTED,
                    fontsize=7.4, ha="center", va="top", zorder=4)
    # "generation hub" only means something where generation is CONCENTRATED:
    # label the node hosting more than one plant, and say so plainly when none is.
    counts = {}
    for _, n, _ in cfg["SPECS"]:
        counts[n] = counts.get(n, 0) + 1
    hub_node = max(counts, key=counts.get)
    if counts[hub_node] > 1:
        hx, hy = POS[hub_node]
        ax.text(hx, hy - R - 0.36, "generation hub", color=INK2, fontsize=8.4,
                ha="center", va="top", fontweight="semibold", zorder=4)
    else:
        ax.text(1.35, -1.22, "generation at every loop node", color=INK2,
                fontsize=8.6, ha="left", va="bottom", fontweight="semibold",
                zorder=4)

    # ---- plants, grouped by the node they sit on ---------------------------
    by_node: dict[int, list] = {}
    for firm, node, key in cfg["SPECS"]:
        by_node.setdefault(node, []).append((firm, key))

    multi = {f for f, _, _ in cfg["SPECS"]
             if sum(1 for g, _, _ in cfg["SPECS"] if g == f) > 1}

    BW, BH, GAP = 1.28, 0.40, 0.50
    # Where a node's plant boxes sit, per node, so they never land on the loop.
    # Nodes 1 and 2 hang their boxes off to the left; node 3 (which is inside the
    # picture, between the loop and the radial tail) hangs its box BELOW.
    BOX_AT = {0: (-1.05, 0.30), 1: (-1.05, -0.10), 2: (0.05, -1.20)}
    for node, plants in by_node.items():
        nx, ny = POS[node]
        dx, dy = BOX_AT.get(node, (-1.05, 0.0))
        top = ny + dy + (len(plants) - 1) * GAP / 2
        for j, (firm, key) in enumerate(plants):
            cy = top - j * GAP
            cx = nx + dx - BW / 2 if dx < 0 else nx + dx
            col = SERIES[firm % len(SERIES)]
            # leader from the edge of the box that faces the node
            lx = cx + BW / 2 if cx < nx else cx - BW / 2
            ax.plot([lx, nx], [cy, ny - R if cy < ny else ny + R],
                    color=col, lw=1.1, alpha=0.55, zorder=2)
            ax.add_patch(FancyBboxPatch(
                (cx - BW / 2, cy - BH / 2), BW, BH,
                boxstyle="round,pad=0.015,rounding_size=0.07",
                facecolor="white", edgecolor=col, lw=1.7, zorder=5))
            role = ""
            if firm in multi:
                mcs = [cfg["MC"][k] for f, _, k in cfg["SPECS"] if f == firm]
                role = "  base" if cfg["MC"][key] == min(mcs) else "  peaker"
            ax.text(cx, cy + 0.075, f"Firm {firm}{role}", color=col,
                    fontsize=9.2, ha="center", va="center",
                    fontweight="semibold", zorder=6)
            ax.text(cx, cy - 0.088,
                    f"MC ${cfg['MC'][key]:g} · cap {cfg['CAP'][key]:g} MW",
                    color=INK2, fontsize=7.6, ha="center", va="center", zorder=6)

    # ---- titles ------------------------------------------------------------
    ax.text(-2.45, 2.10, title, color=INK, fontsize=12.5, ha="left",
            va="top", fontweight="semibold")
    ax.text(-2.45, 1.80, subtitle, color=INK2, fontsize=9.4, ha="left", va="top")
    if delta is not None:
        ax.text(-2.45, -1.24, f"Δ = {100*delta:.1f}%", color=SERIES[0],
                fontsize=15, ha="left", va="bottom", fontweight="semibold")
        ax.text(-1.05, -1.22,
                f"{n_firms} firms · {len(cfg['SPECS'])} plants",
                color=MUTED, fontsize=8.6, ha="left", va="bottom")

    ax.set_xlim(-2.55, 4.05)
    ax.set_ylim(-1.40, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")


def make():
    panels = [
        ("two_firm", "results_two_firm", "Structure 1 — two asymmetric firms",
         "Firm 0: cheap base plant at node 1 + peaker at the hub.  "
         "Firm 1: one mid-cost plant at the hub.\n"
         "→ matches the paper's duopoly baseline"),
        ("three_firm", "results", "Structure 2 (old) — three firms at the hub",
         "One plant each, all three sited at the node-2 generation hub.\n"
         "→ the paper's §5.2 three-firm robustness run"),
        ("three_firm_dist", "results_three_firm_dist",
         "Structure 2 (new) — one plant per node",
         "One plant each at nodes 1, 2 and 3, the most expensive at node 3.\n"
         "→ node 5 is an inelastic load pocket behind the 3–4–5 tail"),
    ]
    cfgs = [(_read_config(tag), res, t, s) for tag, res, t, s in panels]

    fig, axes = plt.subplots(1, len(cfgs), figsize=(7.7 * len(cfgs), 5.3))
    axes = np.atleast_1d(axes)
    for ax, (cfg, res, t, s) in zip(axes, cfgs):
        _draw(ax, cfg, t, s, _delta(res))

    fig.suptitle("The market — one 5-node network, three ownership structures",
                 color=INK, fontsize=14.5, x=0.007, ha="left", y=0.995,
                 fontweight="semibold")
    fig.text(0.007, 0.012,
             "Nodes numbered as in the papers (1–5). Grey edges are transmission "
             "lines, labelled with their thermal limits; 1–2–3 is a loop, 3–4–5 a "
             "radial tail. “D” is each node's demand scale Q⁰.\nOnly the scalars "
             "change between panels — the topology is identical. Δ is the learned "
             "profit gain under imperfect monitoring with stochastic demand "
             "(0 = Cournot–Nash, 1 = joint monopoly).",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    return _save(fig, "fig0_network.png")


if __name__ == "__main__":
    make()
