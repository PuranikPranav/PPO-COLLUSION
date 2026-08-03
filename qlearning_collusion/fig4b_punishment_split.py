"""Figure 4b — the punishment, for the sessions whose cartel survives.

The paper's Figure 4 plots the deviating agent and the non-deviating agent after
one firm is forced to defect for a single period. This is the same experiment,
restricted to the sessions that return to their pre-deviation cycle, and run
once per firm so every agent gets a turn as the deviator.

Why restrict, and why that is legitimate here: with demand frozen and the limit
strategies deterministic, post-convergence play is a finite-state map that must
settle into a cycle. A big enough deviation can tip a session into a DIFFERENT
absorbing cycle, and nothing random remains to bring it back. Those sessions
never return by construction, so pooling them with the rest puts a permanent
offset into the average that has nothing to do with the shape of the punishment.

That restriction is conditioning on a post-deviation outcome, so it must always
be reported WITH the surviving fraction (printed in every panel title) and never
presented as the unconditional result. `fig4_deviation.png` remains the
unconditional figure.

    MARKET_CONFIG=three_firm python -m qlearning_collusion.fig4b_punishment_split
    MARKET_CONFIG=two_firm   python -m qlearning_collusion.fig4b_punishment_split
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlearning_collusion import experiments as X
from qlearning_collusion.figures import _style, SERIES, INK, INK2, MUTED
from qlearning_collusion.qlearn import STATE_SPACES

NAME = "imperfect_stochastic"
HERE = os.path.dirname(os.path.abspath(__file__))

PRE, POST, SETTLE, CYCLE_W = 3, 14, 300, 50
N_PHASES = 12
PLOT_TO = 10
TICK_EVERY = 1


def _figdir():
    tag = os.environ.get("MARKET_CONFIG", "three_firm").strip().lower()
    d = os.path.join(HERE, "figures" if tag == "three_firm" else f"figures_{tag}")
    os.makedirs(d, exist_ok=True)
    return d, tag


def _load():
    meta, z = X.load(NAME)
    mk = X.build_market(deterministic=False, k=meta["market"]["k"],
                        xi=meta["market"]["xi"], m=meta["market"]["m"],
                        h=meta["market"]["h"])
    return mk, STATE_SPACES["imperfect"](mk), z


def run(mk, space, z, demand: str, deviator: int):
    n = mk.n_agents
    conv = z["converged"].astype(bool)
    greedy, start = z["greedy"][conv], z["final_state"][conv]
    S0 = greedy.shape[0]
    # replicate across all phases of the frozen-demand limit cycle, else the
    # cycle's sawtooth swamps the response
    greedy = np.repeat(greedy, N_PHASES, axis=0)
    start = np.repeat(start, N_PHASES)
    offset = np.tile(np.arange(N_PHASES), S0)
    S = greedy.shape[0]
    rowS = np.arange(S)[:, None]
    ar = np.arange(n)
    ui = np.full(S, (mk.h - 1) if demand == "high" else 0)
    kmax = int(np.max(mk.n_actions))
    others = [i for i in range(n) if i != deviator]

    def step(state, force=False):
        a = greedy[rowS, ar[None, :], state[:, None]].astype(np.int64)
        P = mk.pidx(a)
        if force:
            cand = np.arange(kmax)
            pay = mk.exp_profit_p[deviator][mk.dev_pidx(P[:, None], deviator,
                                                        cand[None, :])]
            a = a.copy()
            a[:, deviator] = pay.argmax(axis=1)
            P = mk.pidx(a)
        return a, P, space.next_state(a, P, ui)

    st = start.copy()
    for _ in range(SETTLE):
        _, _, st = step(st)
    for p in range(N_PHASES):
        _, _, nxt = step(st)
        st = np.where(offset > p, nxt, st)

    def split(a):
        q = mk.q_agent[ar[None, :], a]
        return q[:, deviator], q[:, others].mean(axis=1)

    s_d, s_b = st.copy(), st.copy()
    dd, rd, db, rb = [], [], [], []

    def rec(a, dl, rl):
        x, y = split(a)
        dl.append(x); rl.append(y)

    for _ in range(PRE):
        a, _, s_d = step(s_d); rec(a, dd, rd)
        a, _, s_b = step(s_b); rec(a, db, rb)
    a, _, s_d = step(s_d, force=True); rec(a, dd, rd)      # the forced deviation
    a, _, s_b = step(s_b); rec(a, db, rb)
    for _ in range(POST):
        a, _, s_d = step(s_d); rec(a, dd, rd)
        a, _, s_b = step(s_b); rec(a, db, rb)

    dev = np.array(dd) - np.array(db)          # (T, S) MW above the twin
    riv = np.array(rd) - np.array(rb)

    # run far out, then compare the cycles the two paths settle into
    for _ in range(SETTLE):
        _, _, s_d = step(s_d)
        _, _, s_b = step(s_b)
    cd, cb = [], []
    for _ in range(CYCLE_W):
        _, _, s_d = step(s_d); cd.append(s_d.copy())
        _, _, s_b = step(s_b); cb.append(s_b.copy())
    ret = np.array([x == y for x, y in zip([set(v) for v in np.array(cb).T],
                                           [set(v) for v in np.array(cd).T])])

    return dict(t=np.arange(dev.shape[0]) - PRE, demand=demand, deviator=deviator,
                dev=dev[:, ret].mean(axis=1), riv=riv[:, ret].mean(axis=1),
                frac=float(ret.mean()))


def make():
    figdir, tag = _figdir()
    mk, space, z = _load()
    n = mk.n_agents
    demands = ["high", "low"]

    fig, axes = plt.subplots(n, 2, figsize=(12.6, 3.5 * n + 0.6),
                             sharex=True, squeeze=False)
    out = []
    for i in range(n):
        for c, dem in enumerate(demands):
            r = run(mk, space, z, dem, i)
            out.append(r)
            ax = axes[i, c]
            m = r["t"] <= PLOT_TO
            t, d, v = r["t"][m], r["dev"][m], r["riv"][m]
            col = SERIES[i % len(SERIES)]

            ax.axhline(0, color=INK, lw=1.1)
            ax.axvline(0, color=MUTED, lw=1.0, ls=":")
            ax.plot(t, d, color=col, lw=2.8, marker="o", ms=4.2,
                    solid_capstyle="round", zorder=5)
            ax.plot(t, v, color=INK2, lw=2.3, ls="--", marker="o", ms=3.6,
                    solid_capstyle="round", zorder=4)

            # fit BOTH lines in full, spike included -- nothing clipped
            lo = min(0.0, d.min(), v.min())
            hi = max(d.max(), v.max())
            pad = (hi - lo) * 0.10
            ax.set_ylim(lo - pad, hi + pad)

            ye, yv = d[-1], v[-1]
            gap = (hi - lo) * 0.10
            if abs(ye - yv) < gap:
                mid = 0.5 * (ye + yv)
                ye, yv = mid + gap / 2, mid - gap / 2
            ax.annotate("deviating firm", xy=(PLOT_TO, d[-1]),
                        xytext=(PLOT_TO * 1.06, ye), textcoords="data", color=col,
                        fontsize=8.6, va="center", fontweight="semibold",
                        annotation_clip=False)
            ax.annotate("non-deviating", xy=(PLOT_TO, v[-1]),
                        xytext=(PLOT_TO * 1.06, yv), textcoords="data", color=INK2,
                        fontsize=8.6, va="center", fontweight="semibold",
                        annotation_clip=False)

            ax.set_title(f"Firm {i} deviates — {dem} demand   "
                         f"(cartel survives in {100*r['frac']:.0f}% of sessions)",
                         color=INK, fontsize=10.2, loc="left", pad=8,
                         fontweight="semibold")
            _style(ax,
                   "Periods after the forced deviation" if i == n - 1 else "",
                   "MW above the no-deviation twin" if c == 0 else "", "")
            ax.set_xlim(t[0], PLOT_TO * 1.34)
            ax.set_xticks(np.arange(0, PLOT_TO + 1, TICK_EVERY))

    label = {"two_firm": "two-firm", "three_firm": "three-firm hub",
             "three_firm_dist": "three-firm, one plant per node"}.get(tag, tag)
    fig.suptitle("Figure 4b — a deviation is punished, then the punishment fades "
                 f"to zero  ({label} market, cartel-surviving sessions)",
                 color=INK, fontsize=12.5, x=0.006, ha="left", y=1.0,
                 fontweight="semibold")
    fig.text(0.006, -0.055 / n,
             "One firm is forced into its static best response for a single period, then reverts to its learned strategy; demand is frozen and every path is averaged\n"
             "over all 12 phases of the limit cycle. Restricted to sessions that return to their pre-deviation cycle — with deterministic strategies and frozen demand,\n"
             "a deviation can tip the remainder into a different absorbing cycle they never leave. Surviving fraction is stated in each panel; fig4_deviation.png is unconditional.",
             color=MUTED, fontsize=8.2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = os.path.join(figdir, "fig4b_punishment_split.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")
    for r in out:
        s = r["dev"][r["t"] >= 1]
        w = r["riv"][r["t"] >= 1]
        print(f"    firm {r['deviator']} / {r['demand']:>4}: survives {100*r['frac']:5.1f}%"
              f" | deviation {r['dev'][PRE]:+6.2f} MW"
              f" | rivals t+1 {w[0]:+5.2f} t+2 {w[1]:+5.2f} t+3 {w[2]:+5.2f}"
              f" -> t+{POST} {w[-1]:+.2f}")
    return p


if __name__ == "__main__":
    make()
