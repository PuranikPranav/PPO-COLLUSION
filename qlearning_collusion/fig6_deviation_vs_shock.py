"""Figure 6 (this replication's own result, not in the paper).

The algorithms punish a DEVIATION but not an adverse DEMAND SHOCK, even though
the only thing they observe is the price. Both experiments are paired: the same
converged sessions are run twice from the identical state, so the difference is
the pure causal effect of the perturbation.

  deviation  : one firm is forced into its static best response for one period,
               demand frozen. Price falls ~$2.4.
  shock      : the demand draw is forced adverse for one period instead of
               favourable; from the next period on both copies get the IDENTICAL
               shock sequence. Price falls ~$5.8 -- a BIGGER price drop.

If the algorithms simply mapped "low price -> expand", the shock (larger drop)
would trigger the larger price war. It does not: the response to a shock is a
coin flip with zero median, while the response to a deviation is a sharp,
decaying expansion. The learned strategy is a trigger, not a smooth reaction
function.
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
from qlearning_collusion.figures import _style, _save, SERIES, INK, INK2, MUTED
from qlearning_collusion.qlearn import STATE_SPACES

NAME = "imperfect_stochastic"
H = 12


def _load():
    meta, z = X.load(NAME)
    mk = X.build_market(deterministic=False, k=meta["market"]["k"],
                        xi=meta["market"]["xi"], m=meta["market"]["m"],
                        h=meta["market"]["h"])
    space = STATE_SPACES["imperfect"](mk)
    greedy = z["greedy"][z["converged"].astype(bool)]
    start = z["final_state"][z["converged"].astype(bool)]
    return mk, space, greedy, start


def deviation_irf(mk, space, greedy, start, demand="high", n_phases=12):
    """Total-generation impulse response to a forced deviation (demand frozen)."""
    r = X.deviation_experiment(NAME, deviator=0, demand=demand, pre=6, post=H + 2,
                               mode="best_response", n_phases=n_phases)
    dp = r["dev_period"]
    tot = r["q"].sum(axis=1) - r["q_nodev"].sum(axis=1)
    dpr = r["price"] - r["price_nodev"]
    return tot[dp + 1:dp + 1 + H], dpr[dp + 1:dp + 1 + H], tot[dp], dpr[dp]


def shock_irf(mk, space, greedy, start, reps=40, seed=11):
    """Paired impulse response to one adverse demand draw."""
    rng = np.random.default_rng(seed)
    S, n = greedy.shape[0], mk.n_agents
    rowS = np.arange(S)[:, None]; ar_n = np.arange(n)

    def act(state):
        a = greedy[rowS, ar_n[None, :], state[:, None]].astype(np.int64)
        return a, mk.pidx(a)

    st = start.copy()
    for _ in range(500):
        a, J = act(st)
        st = space.next_state(a, J, rng.integers(0, mk.h, size=S))

    dg = np.zeros((reps, H)); dp = np.zeros((reps, H))
    imp_g = np.zeros(reps); imp_p = np.zeros(reps)
    for r in range(reps):
        for _ in range(20):
            a, J = act(st)
            st = space.next_state(a, J, rng.integers(0, mk.h, size=S))
        a, J = act(st)
        imp_g[r] = 0.0
        imp_p[r] = (mk.price_p[J, 0] - mk.price_p[J, mk.h - 1]).mean()
        s_a = space.next_state(a, J, np.zeros(S, dtype=np.int64))
        s_f = space.next_state(a, J, np.full(S, mk.h - 1, dtype=np.int64))
        for t in range(H):
            u = rng.integers(0, mk.h, size=S)
            a1, J1 = act(s_a); a2, J2 = act(s_f)
            dg[r, t] = (mk.total_gen_p[J1] - mk.total_gen_p[J2]).mean()
            dp[r, t] = (mk.price_p[J1, u] - mk.price_p[J2, u]).mean()
            s_a = space.next_state(a1, J1, u)
            s_f = space.next_state(a2, J2, u)
    return dg.mean(axis=0), dp.mean(axis=0), 0.0, imp_p.mean()


def _shock_response_split(mk, space, greedy, start, seed=11):
    """Per-session sign of the output response to one adverse demand draw.

    Paired: the same state gets the adverse draw in one copy and the favourable
    draw in the other, so the difference isolates the response to the shock.
    Reported in the figure caption rather than hard-coded, since it differs
    between markets.
    """
    S, n = greedy.shape[0], mk.n_agents
    rowS = np.arange(S)[:, None]; ar = np.arange(n)
    rng = np.random.default_rng(seed)
    st = start.copy()
    for _ in range(500):
        a = greedy[rowS, ar[None, :], st[:, None]].astype(np.int64)
        st = space.next_state(a, mk.pidx(a), rng.integers(0, mk.h, size=S))
    a = greedy[rowS, ar[None, :], st[:, None]].astype(np.int64)
    P = mk.pidx(a)
    sa = space.next_state(a, P, np.zeros(S, dtype=np.int64))
    sf = space.next_state(a, P, np.full(S, mk.h - 1, dtype=np.int64))
    a1 = greedy[rowS, ar[None, :], sa[:, None]].astype(np.int64)
    a2 = greedy[rowS, ar[None, :], sf[:, None]].astype(np.int64)
    d = mk.total_gen_p[mk.pidx(a1)] - mk.total_gen_p[mk.pidx(a2)]
    tol = 1e-9
    return (float((d > tol).mean()), float((d < -tol).mean()),
            float((np.abs(d) <= tol).mean()), float(np.median(d)))


def make(demand="high"):
    mk, space, greedy, start = _load()
    dev_g, dev_p, dev_g0, dev_p0 = deviation_irf(mk, space, greedy, start, demand)
    shk_g, shk_p, shk_g0, shk_p0 = shock_irf(mk, space, greedy, start)

    t = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- panel 1: the impulse itself -------------------------------------
    ax = axes[0]
    labels = ["forced deviation\n(one firm, one period)", "adverse demand shock\n(one period)"]
    vals = [dev_p0, shk_p0]
    cols = [SERIES[0], SERIES[2]]
    bars = ax.barh(labels, vals, color=cols, height=0.5)
    for b, v in zip(bars, vals):
        ax.annotate(f"${v:.2f}", xy=(v, b.get_y() + b.get_height() / 2),
                    xytext=(-8 if v < 0 else 8, 0), textcoords="offset points",
                    color=INK, fontsize=11, va="center",
                    ha="right" if v < 0 else "left", fontweight="semibold")
    ax.axvline(0, color=INK, lw=1.0)
    _style(ax, "Immediate fall in the reference-node LMP ($/MWh)", "", "")
    # Which impulse hits the price harder is a RESULT, not a given: it differs
    # across market structures, so the title is read off the numbers.
    bigger = "shock" if shk_p0 < dev_p0 else "deviation"
    ax.set_title(f"The {bigger} is the bigger price drop",
                 color=INK, fontsize=11, loc="left", pad=8, fontweight="semibold")
    ax.set_xlim(min(vals) * 1.35, abs(min(vals)) * 0.35)

    # --- panel 2: the response --------------------------------------------
    ax = axes[1]
    ax.axhline(0, color=INK, lw=1.0)
    ax.plot(t, dev_g, color=SERIES[0], lw=2.4, marker="o", ms=4,
            solid_capstyle="round")
    ax.plot(t, shk_g, color=SERIES[2], lw=2.4, marker="o", ms=4,
            solid_capstyle="round")
    ax.annotate("after a DEVIATION\n→ price war", xy=(t[1], dev_g[1]),
                xytext=(14, 12), textcoords="offset points",
                color=SERIES[0], fontsize=10, fontweight="semibold")
    ax.annotate("after an adverse SHOCK\n→ no war", xy=(t[2], shk_g[2]),
                xytext=(18, -30), textcoords="offset points",
                color=SERIES[2], fontsize=10, fontweight="semibold")
    _style(ax, "Periods after the perturbation",
           "Extra total generation vs the paired counterfactual (MW)", "")
    ax.set_title("but only the deviation is punished",
                 color=INK, fontsize=11, loc="left", pad=8, fontweight="semibold")
    ax.set_xticks(t[::2])

    fig.suptitle("Figure 6 — the algorithms tell a deviation from a demand shock, "
                 "though they only observe the price",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.02,
                 fontweight="semibold")
    exp_f, con_f, flat_f, med = _shock_response_split(mk, space, greedy, start)
    fig.text(0.005, -0.10,
             f"Both panels are paired experiments on the same converged sessions "
             f"({demand} demand for the deviation). This is the one qualitative "
             f"result that does NOT replicate Calvano et al. (2021),\nwho report "
             f"price wars triggered by adverse shocks as well as by deviations. "
             f"Here the {bigger} is the bigger price drop "
             f"(deviation ${dev_p0:.2f}, shock ${shk_p0:.2f}) yet the shock provokes "
             f"no expansion\n({100*exp_f:.0f}% expand, {100*con_f:.0f}% contract, "
             f"{100*flat_f:.0f}% do not move; median {med:+.2f} MW).",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "fig6_deviation_vs_shock.png")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "high")
