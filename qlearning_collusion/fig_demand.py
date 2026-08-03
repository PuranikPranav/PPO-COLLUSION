"""The nodal demand curves — is node 5 actually a load pocket?

Andrew, last meeting: *"we want node 5 to be a load pocket. Load pocket means the
demand is inelastic … you can somehow use a flatter, more flattened demand curve
to mimic that."*

Both halves of that sentence are drawn here, because they are the same fact seen
in two orientations and it is easy to talk past each other:

  left  panel  quantity against price (the orientation Andrew was describing).
               INELASTIC demand is FLAT here: the quantity taken barely moves as
               the price moves.
  right panel  the textbook inverse-demand picture, price against quantity.
               The same inelastic demand is STEEP here.

The number that settles it is the point price-elasticity, printed per node. With
linear demand d_i = Q0_i (1 - p / P0_i) it is

    eps_i = (dd_i/dp)(p/d_i) = -(Q0_i/P0_i) * p / d_i

so a load pocket needs a LARGE P0_i relative to Q0_i — a near-vertical
inverse-demand curve whose intercept is effectively the value of lost load.

    MARKET_CONFIG=three_firm_dist python -m qlearning_collusion.fig_demand
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.node_network import P0, Q0, MARKET
from qlearning_collusion.figures import _style, _despread, INK, INK2, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_COL = ["#2a78d6", "#008300", "#e87ba4", "#8c8b85", "#d1495b"]


def elasticity(i, p):
    d = Q0[i] * (1.0 - p / P0[i])
    return (-(Q0[i] / P0[i]) * p / d) if d > 1e-9 else float("nan")


def make(price_lo=15.0, price_hi=75.0, mark=(30.0, 45.0, 60.0), figdir=None):
    figdir = figdir or os.path.join(
        HERE, "figures" if MARKET == "three_firm" else f"figures_{MARKET}")
    os.makedirs(figdir, exist_ok=True)
    p = np.linspace(price_lo, price_hi, 400)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    ends = []
    for i in range(len(P0)):
        d = np.clip(Q0[i] * (1.0 - p / P0[i]), 0, None)
        ax.plot(p, d, color=NODE_COL[i], lw=2.4 if i == 4 else 1.7,
                solid_capstyle="round", zorder=5 if i == 4 else 3)
        ends.append(float(d[-1]))
    # nodes 1, 4 and 5 all end up near the axis, so push the labels apart
    span = max(ends) - min(ends)
    for i, y in _despread(ends, min_gap=span * 0.10):
        ax.annotate(f"node {i+1}", xy=(p[-1], ends[i]),
                    xytext=(p[-1] + (p[-1] - p[0]) * 0.02, y), textcoords="data",
                    color=NODE_COL[i], fontsize=9.5, va="center",
                    fontweight="semibold" if i == 4 else "normal",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=NODE_COL[i], lw=0.7,
                                    alpha=0.5, shrinkA=0, shrinkB=2))
    _style(ax, "Price (USD/MWh)", "Quantity demanded (MW)", "")
    ax.set_title("Quantity against price — an inelastic node is FLAT",
                 color=INK, fontsize=11.5, loc="left", pad=8, fontweight="semibold")
    ax.set_xlim(price_lo, price_hi + (price_hi - price_lo) * 0.10)

    ax = axes[1]
    for i in range(len(P0)):
        d = np.clip(Q0[i] * (1.0 - p / P0[i]), 0, None)
        ax.plot(d, p, color=NODE_COL[i], lw=2.4 if i == 4 else 1.7,
                solid_capstyle="round", zorder=5 if i == 4 else 3)
    ax.set_ylim(price_lo, price_hi)
    _style(ax, "Quantity (MW)", "Price (USD/MWh)", "")
    ax.set_title("Inverse demand — the same node is near-VERTICAL",
                 color=INK, fontsize=11.5, loc="left", pad=8, fontweight="semibold")

    # NB: no "$" anywhere in this block — matplotlib reads a dollar sign as the
    # start of a mathtext expression and silently reflows the whole line.
    rows = ["node      P0      Q0    slope USD/MWh per MW    elasticity at "
            + " / ".join(f"{m:.0f}" for m in mark) + " USD/MWh"]
    for i in range(len(P0)):
        es = "  ".join(f"{elasticity(i, m):+7.4f}" for m in mark)
        rows.append(f"  {i+1}   {P0[i]:8.1f}{Q0[i]:8.1f}{P0[i]/Q0[i]:16.3f}"
                    f"          {es}")
    txt = "\n".join(rows)
    fig.suptitle(f"Nodal demand — node 5 as a load pocket  ({MARKET})",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.02,
                 fontweight="semibold")
    fig.text(0.005, -0.30, txt, color=INK2, fontsize=8.6, ha="left",
             family="monospace")
    fig.text(0.005, -0.40,
             "Node 5 carries a must-serve load: its quantity is essentially fixed, so its inverse-demand intercept is large (the value of lost load) and its\n"
             "elasticity is an order of magnitude below every other node. All the demand response in this market therefore has to come from nodes 1-4.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout()
    out = os.path.join(figdir, "fig_demand_curves.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(txt)
    print(f"\n  wrote {out}")
    return out


if __name__ == "__main__":
    make()
