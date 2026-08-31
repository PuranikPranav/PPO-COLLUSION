"""Cross-market figures: the LMP path, the per-agent learning path, and what
kind of punishment the algorithms actually learned.

Everything here puts the THREE-FIRM hub market and the TWO-FIRM companion
market side by side, and every curve is an average over the 1,000 independent
sessions the paper's protocol runs.

    fig_lmp.png          hub LMP vs. iterations, with the Cournot-Nash and
                         joint-monopoly benchmarks
    fig_learning.png     per-agent output and per-agent profit gain vs.
                         iterations, plus how many sessions are still learning
    fig_punishment.png   is the learned punishment a GRIM TRIGGER? (no) and
                         does it deter cheating anyway? (in low demand)

The first two read the saved run artefacts straight off disk, so they need no
market object and no MARKET_CONFIG. The third replays the learned strategies,
which does need the market, and `iso_market.node_network` reads MARKET_CONFIG
once at import — so its inputs are cached per market by a separate pass:

    MARKET_CONFIG=three_firm python -m qlearning_collusion.fig_lmp_learning --cache
    MARKET_CONFIG=two_firm   python -m qlearning_collusion.fig_lmp_learning --cache
    python -m qlearning_collusion.fig_lmp_learning

Colour policy is `figures.py`'s: categorical slots 1-3 assigned to firms in
fixed order and never cycled, benchmarks as recessive grey rules, every series
direct-labelled as well as legended so identity is never colour-alone.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlearning_collusion.figures import (        # noqa: E402
    SERIES, INK, INK2, MUTED, RULE_N, RULE_M, _style, _millions, _despread,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures_combined")

# The baseline cell of the paper's Table I: imperfect monitoring (firms see only
# the price) + stochastic demand (which is what makes it imperfect).
CELL = "imperfect_stochastic"

# (results subdir, MARKET_CONFIG value, panel title, short tag)
# These figures are two-panel comparisons and every layout below is built for
# exactly TWO markets, so pick the pair with FIG_MARKETS rather than adding a
# third entry here. The default pair is the change the redistribution made:
# one plant per node versus the old all-at-the-hub siting.
ALL_MARKETS = {
    "three_firm_dist": ("results_three_firm_dist", "three_firm_dist",
                        "Three firms, one plant per node", "per-node"),
    "three_firm": ("results", "three_firm", "Three firms at the hub", "hub"),
    "two_firm": ("results_two_firm", "two_firm", "Two-firm market", "2-firm"),
}
_want = os.environ.get("FIG_MARKETS", "three_firm_dist,three_firm").split(",")
MARKETS = [ALL_MARKETS[w.strip()] for w in _want if w.strip() in ALL_MARKETS]
if len(MARKETS) != 2:
    raise SystemExit(
        "FIG_MARKETS must name exactly two of "
        f"{list(ALL_MARKETS)} (got {_want!r}) — these figures are two-panel "
        "comparisons."
    )

# Text colours for direct labels: the series hues darkened so label TEXT clears
# the contrast floor on white (slot 3 in particular does not, as a hue).
SERIES_TEXT = ["#1f5aa8", "#006b00", "#b8477a"]

IMPULSE_HORIZON = 25          # periods of the impulse response to plot
VALUE_HORIZON = 200           # periods the deterrence verdict is computed over
VALUE_PLOT = 60               # …of which this many are drawn
SETTLE, N_PHASES, CYCLE_W = 300, 12, 50


def _load(subdir: str, cell: str = CELL):
    """Read one run's artefacts straight off disk (no market object needed)."""
    base = os.path.join(HERE, subdir)
    with open(os.path.join(base, f"{cell}.json")) as fh:
        meta = json.load(fh)
    return meta, np.load(os.path.join(base, f"{cell}.npz"))


def _save(fig, name, figdir=None):
    figdir = figdir or FIGDIR
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# ===========================================================================
# 1. THE LMP PATH
# ===========================================================================
def fig_lmp(cell=CELL, figdir=None, label=""):
    """Hub LMP averaged over all sessions, against the two benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    nses = []

    for ax, (subdir, _cfg, title, tag) in zip(axes, MARKETS):
        meta, z = _load(subdir, cell)
        nses.append(meta["config"]["n_sessions"])
        gb = meta["market"]["grid_benchmarks"]
        it, p = z["log_iters"], z["log_price"]
        pN, pM = gb["nash"]["hub_price"], gb["monopoly"]["hub_price"]
        nS = meta["config"]["n_sessions"]

        # the corridor the collusion index normalises by
        ax.axhspan(pN, pM, color=RULE_M, alpha=0.07, lw=0)
        ax.axhline(pN, color=RULE_N, lw=1.3, ls="--")
        ax.axhline(pM, color=RULE_M, lw=1.3, ls="--")
        ax.plot(it, p, color=SERIES[0], lw=2.1, solid_capstyle="round",
                label=f"learned LMP (mean of {nS:,} session{'s' if nS != 1 else ''})")
        ax.scatter([it[-1]], [p[-1]], s=36, color=SERIES[0], zorder=5,
                   edgecolor="white", lw=1.5)

        lo = min(p.min(), pN) - 0.9
        hi = pM + 0.9
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, it[-1] * 1.10)

        xr = it[-1]
        ax.annotate(f"joint monopoly  ${pM:.2f}", xy=(xr * 0.015, pM),
                    xytext=(0, -13), textcoords="offset points",
                    color=INK2, fontsize=9)
        ax.annotate(f"Cournot–Nash  ${pN:.2f}", xy=(xr * 0.015, pN),
                    xytext=(0, 5), textcoords="offset points",
                    color=INK2, fontsize=9)
        closed = (p[-1] - pN) / (pM - pN)
        ax.annotate(f"${p[-1]:.2f}\n{100*closed:.0f}% of the way\nto monopoly",
                    xy=(it[-1], p[-1]), xytext=(-6, -6), textcoords="offset points",
                    color=SERIES_TEXT[0], fontsize=9.5, ha="right", va="top",
                    fontweight="semibold")

        _style(ax, "Time step (Q-learning iteration)", "Hub LMP ($/MWh)", "")
        ax.set_title(f"{title}   ·   Δ = {100*meta['result']['delta']:.1f}%",
                     color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_millions))
        leg = ax.legend(loc="lower right", frameon=False, fontsize=8.8,
                        handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(INK2)

    fig.suptitle("The price the market clears at, averaged over every session"
                 + (f" — {label}" if label else
                    " — it climbs from below Cournot–Nash to about halfway to monopoly"),
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.02,
                 fontweight="semibold")
    fig.text(0.006, -0.068,
             f"Hub LMP produced by the ISO's DC-OPF at the greedy (highest-Q) output profile, averaged over all independent sessions "
             f"({nses[0]:,} / {nses[1]:,}) and over the demand shock.\n"
             "Benchmarks are the grid-restricted stage-game Cournot–Nash and joint-monopoly profiles — the same two points the profit gain Δ is normalised by.\n"
             "The price and Δ close different fractions of their respective Nash → monopoly gaps: profit is convex in the output cut, so the two need not agree.\n"
             f"Cell: {_cell_label(cell)}.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, "fig_lmp.png", figdir)


def _cell_label(cell: str) -> str:
    mon, dem = cell.split("_")
    a = ("imperfect monitoring (firms observe only the price)" if mon == "imperfect"
         else "perfect monitoring (firms observe the whole past output profile)")
    b = ("stochastic demand" if dem == "stochastic" else
         "deterministic demand — no shock")
    return f"{a} with {b}"


# ===========================================================================
# 2. THE LEARNING PATH, AGENT BY AGENT
# ===========================================================================
def _series_label(ax, it, xmax, values, texts, min_gap):
    """Direct-label each firm's curve at the right margin, pushed apart."""
    for i, y in _despread(list(values), min_gap=min_gap):
        ax.annotate(texts[i], xy=(it[-1], values[i]), xytext=(xmax * 1.005, y),
                    textcoords="data", color=SERIES_TEXT[i], fontsize=9.5,
                    va="center", fontweight="semibold", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=SERIES[i], lw=0.7,
                                    alpha=0.5, shrinkA=0, shrinkB=2))


def fig_learning(cell=CELL, figdir=None, label=""):
    """Per-agent output, per-agent profit, and the collusion index over training.

    Per-firm profit is shown in dollars rather than normalised to each firm's
    own Nash → monopoly range: in the two-firm market that range is only $73
    wide for firm 0 (the cartel's gain there comes almost entirely from shutting
    its peaker, which barely moves its own profit), so the normalised version
    divides by ~0 and is unreadable. The combined index in row 3 is the paper's
    Δ, which divides by the JOINT range and is well conditioned in both markets.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(4, 2, figsize=(13.4, 15.0),
                             gridspec_kw=dict(height_ratios=[1.15, 1.15, 1.0, 0.55]))
    nses = []

    for c, (subdir, _cfg, title, tag) in enumerate(MARKETS):
        meta, z = _load(subdir, cell)
        nses.append(meta["config"]["n_sessions"])
        gb = meta["market"]["grid_benchmarks"]
        it, q, pi = z["log_iters"], z["log_q"], z["log_profit"]
        n = q.shape[1]
        qN, qM = np.array(gb["nash"]["gens"]), np.array(gb["monopoly"]["gens"])
        piN = np.array(gb["nash"]["profits"])
        piM = np.array(gb["monopoly"]["profits"])
        nS = meta["config"]["n_sessions"]
        xmax = it[-1] * 1.10

        # ---- row 1: output --------------------------------------------------
        ax = axes[0, c]
        for i in range(n):
            ax.axhline(qN[i], color=SERIES[i], lw=1.0, ls=":", alpha=0.5)
            ax.axhline(qM[i], color=SERIES[i], lw=1.0, ls="--", alpha=0.5)
            ax.plot(it, q[:, i], color=SERIES[i], lw=2.2, solid_capstyle="round")
        lo = min(q.min(), qM.min()) - 2
        hi = max(q.max(), qN.max()) + 2
        ax.set_ylim(lo, hi)
        _series_label(ax, it, xmax, q[-1], [f"Firm {i}" for i in range(n)],
                      (hi - lo) * 0.07)
        _style(ax, "", "Output (MW)", "")
        ax.set_title(f"{title} — what each agent learns to produce",
                     color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")

        # ---- row 2: per-firm profit -----------------------------------------
        ax = axes[1, c]
        for i in range(n):
            ax.axhline(piN[i], color=SERIES[i], lw=1.0, ls=":", alpha=0.5)
            ax.axhline(piM[i], color=SERIES[i], lw=1.0, ls="--", alpha=0.5)
            ax.plot(it, pi[:, i], color=SERIES[i], lw=2.2, solid_capstyle="round")
        lo = min(pi.min(), piN.min()) - 60
        hi = max(pi.max(), piM.max()) + 60
        ax.set_ylim(lo, hi)
        _series_label(ax, it, xmax, pi[-1],
                      [f"Firm {i}  ${pi[-1, i]:,.0f}" for i in range(n)],
                      (hi - lo) * 0.07)
        _style(ax, "", "Profit ($ per period)", "")
        ax.set_title("…and what that earns it", color=INK, fontsize=10.5,
                     loc="left", pad=8, fontweight="semibold")

        # ---- row 3: the collusion index -------------------------------------
        ax = axes[2, c]
        d = z["log_delta"]
        ax.axhspan(0, 1, color=RULE_M, alpha=0.07, lw=0)
        ax.axhline(0.0, color=RULE_N, lw=1.3, ls="--")
        ax.axhline(1.0, color=RULE_M, lw=1.3, ls="--")
        ax.plot(it, d, color=INK, lw=2.2, solid_capstyle="round")
        ax.scatter([it[-1]], [d[-1]], s=34, color=INK, zorder=5,
                   edgecolor="white", lw=1.4)
        ax.annotate("joint monopoly  (Δ = 1)", xy=(it[-1] * 0.015, 1.0),
                    xytext=(0, -13), textcoords="offset points", color=INK2,
                    fontsize=9)
        ax.annotate("Cournot–Nash  (Δ = 0)", xy=(it[-1] * 0.015, 0.0),
                    xytext=(0, 6), textcoords="offset points", color=INK2,
                    fontsize=9)
        ax.annotate(f"Δ = {100*meta['result']['delta']:.1f}%",
                    xy=(it[-1], d[-1]), xytext=(-8, 13), textcoords="offset points",
                    color=INK, fontsize=11, ha="right", fontweight="semibold")
        ax.set_ylim(min(-0.55, d.min() - 0.08), 1.15)
        _style(ax, "", "Combined profit gain  Δ", "")
        ax.set_title("…which together closes this much of the Nash → monopoly gap",
                     color=INK, fontsize=10.5, loc="left", pad=8,
                     fontweight="semibold")

        # ---- row 4: how many sessions are still learning ---------------------
        ax = axes[3, c]
        act = 100.0 * z["log_active"] / nS
        ax.fill_between(it, 0, act, color=SERIES[0], alpha=0.16, lw=0)
        ax.plot(it, act, color=SERIES[0], lw=1.8, solid_capstyle="round")
        med = meta["result"]["median_conv_iter"]
        if med:
            ax.axvline(med, color=INK2, lw=1.0, ls=":")
            ax.annotate(f"median session stops learning\nat {med/1e6:.1f}M steps",
                        xy=(med, 58), xytext=(8, 0), textcoords="offset points",
                        color=INK2, fontsize=8.6, va="center")
        ax.set_ylim(0, 108)
        _style(ax, "Time step (Q-learning iteration)",
               "Sessions still learning (%)", "")

        for ax in axes[:, c]:
            ax.set_xlim(0, xmax)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_millions))

    handles = [Line2D([], [], color=SERIES[i], lw=2.2, label=f"Firm {i}")
               for i in range(3)]
    handles += [
        Line2D([], [], color=MUTED, lw=1.2, ls=":",
               label="that firm's Cournot–Nash level"),
        Line2D([], [], color=MUTED, lw=1.2, ls="--",
               label="that firm's joint-monopoly level"),
    ]
    leg = fig.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.006, 0.995), ncol=5, frameon=False,
                     fontsize=9.2, handlelength=1.9, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("How each agent learns — every curve is a mean over the "
                 f"independent sessions ({nses[0]:,} / {nses[1]:,})"
                 + (f"  ·  {label}" if label else ""),
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.018,
                 fontweight="semibold")
    fig.text(0.006, -0.055,
             "Greedy (highest-Q) action of each agent, averaged across sessions; the actions actually played differ from these while ε-greedy exploration is still live.\n"
             "Δ = (Σπᵢ − Σπᵢ^Nash)/(Σπᵢ^Monopoly − Σπᵢ^Nash), against the grid-restricted benchmarks.\n"
             + ("Firm 2 exists only in the three-firm markets. In the two-firm market firm 0 can settle ABOVE its own joint-monopoly line: the cartel profile shuts firm 0's\n"
                "peaker, so firm 0's own Nash → monopoly range is only \\$73 wide there and the learned split favours it. That is a property of the split, not of Δ,\n"
                "whose denominator is the joint range.\n"
                if any(m[1] == "two_firm" for m in MARKETS) else "")
             + f"A session is counted as done once its greedy policy has been unchanged for 100,000 consecutive periods (the paper's criterion). Cell: {_cell_label(cell)}.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.978))
    return _save(fig, "fig_learning.png", figdir)


# ===========================================================================
# 3. IS THE PUNISHMENT A GRIM TRIGGER?
# ===========================================================================
def cache_impulse(cell=CELL):
    """Replay the learned strategies for THIS market and cache the response.

    Needs the market object, so `MARKET_CONFIG` must already be set to the
    market whose results are being replayed. Writes `impulse_cache_<cell>.npz`
    next to that market's results.
    """
    from qlearning_collusion import experiments as X
    from qlearning_collusion.qlearn import STATE_SPACES

    cfg = os.environ.get("MARKET_CONFIG", "three_firm").strip().lower()
    subdir = {m[1]: m[0] for m in MARKETS}[cfg]
    meta, z = _load(subdir, cell)
    det = bool(meta["market"]["deterministic"])
    mk = X.build_market(deterministic=det, k=meta["market"]["k"],
                        xi=meta["market"]["xi"], m=meta["market"]["m"],
                        h=meta["market"]["h"])
    space = STATE_SPACES[meta["config"]["monitoring"]](mk)
    n = mk.n_agents
    gb = meta["market"]["grid_benchmarks"]
    qN = np.array(gb["nash"]["gens"])

    # With no demand shock there is only one demand state, so "high" and "low"
    # are the same experiment — run it once.
    demands = ("low",) if det else ("low", "high")
    out = {"demands": np.array(demands)}
    for demand in demands:
        # (a) the impulse response, split by whether the cartel comes back
        ret_paths, never_paths, fracs, grims = [], [], [], []
        for dev in range(n):
            r = _impulse_split(mk, space, z, demand, dev)
            ret_paths.append(r["riv_return"])
            never_paths.append(r["riv_never"])
            fracs.append(r["frac_return"])
            grims.append(float(np.mean(np.delete(qN, dev)) - r["riv_base"]))
            out[f"{demand}_t"] = r["t"]
        out[f"{demand}_return"] = np.nanmean(ret_paths, axis=0)
        out[f"{demand}_never"] = np.nanmean(never_paths, axis=0)
        out[f"{demand}_frac"] = float(np.mean(fracs))
        out[f"{demand}_grim"] = float(np.mean(grims))

        # (b) is cheating deterred? cumulative discounted gain from deviating
        cums, deterred, one_shot = [], [], []
        for dev in range(n):
            v = X.deviation_value_test(cell, deviator=dev, demand=demand,
                                       horizon=VALUE_HORIZON, n_phases=N_PHASES)
            d = v["delta"]
            gap = v["profit_path_dev"] - v["profit_path_stay"]
            cums.append(np.cumsum((d ** np.arange(len(gap))) * gap))
            deterred.append(v["deterred_frac"])
            one_shot.append(v["one_period_gain"])
        out[f"{demand}_cum"] = np.mean(cums, axis=0)
        out[f"{demand}_deterred"] = float(np.mean(deterred))
        out[f"{demand}_oneshot"] = float(np.mean(one_shot))

    p = os.path.join(HERE, subdir, f"impulse_cache_{cell}.npz")
    np.savez_compressed(p, **out)
    print(f"  wrote {p}")
    for demand in demands:
        print(f"    {cfg:>10s} / {demand:>4s} demand: punishment peaks at "
              f"{out[f'{demand}_return'].max():+.2f} MW vs a grim-trigger step of "
              f"{out[f'{demand}_grim']:+.2f} MW | cartel returns in "
              f"{100*out[f'{demand}_frac']:.0f}% | deterred in "
              f"{100*out[f'{demand}_deterred']:.0f}%")
    return p


def _impulse_split(mk, space, z, demand: str, deviator: int):
    """Rivals' output response to a one-period forced deviation, split by
    whether that session's play returns to its pre-deviation cycle.

    Same construction as `fig4b_punishment_split.run`, but it keeps BOTH groups:
    the sessions whose cartel survives and the sessions it does not, because the
    difference between them is exactly the grim-trigger question.
    """
    n = mk.n_agents
    conv = z["converged"].astype(bool)
    greedy, start = z["greedy"][conv], z["final_state"][conv]
    S0 = greedy.shape[0]
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
    for p in range(N_PHASES):                    # spread the cycle phases out
        _, _, nxt = step(st)
        st = np.where(offset > p, nxt, st)

    s_d, s_b = st.copy(), st.copy()
    rd, rb = [], []
    a, _, s_d = step(s_d, force=True)            # period 0 = the deviation
    rd.append(mk.q_agent[ar[None, :], a][:, others].mean(axis=1))
    a, _, s_b = step(s_b)
    rb.append(mk.q_agent[ar[None, :], a][:, others].mean(axis=1))
    for _ in range(IMPULSE_HORIZON):
        a, _, s_d = step(s_d)
        rd.append(mk.q_agent[ar[None, :], a][:, others].mean(axis=1))
        a, _, s_b = step(s_b)
        rb.append(mk.q_agent[ar[None, :], a][:, others].mean(axis=1))
    riv = np.array(rd) - np.array(rb)            # (T, S) MW above the twin
    base = np.array(rb)

    # Run far out, then compare the cycles the two paths settled into: a session
    # "returns" iff the deviation path ends up cycling through the same states.
    for _ in range(SETTLE):
        _, _, s_d = step(s_d)
        _, _, s_b = step(s_b)
    cd, cb = [], []
    for _ in range(CYCLE_W):
        _, _, s_d = step(s_d); cd.append(s_d.copy())
        _, _, s_b = step(s_b); cb.append(s_b.copy())
    ret = np.array([x == y for x, y in zip([set(v) for v in np.array(cb).T],
                                           [set(v) for v in np.array(cd).T])])

    def _mean(mask):
        return riv[:, mask].mean(axis=1) if mask.any() else np.full(riv.shape[0], np.nan)

    return dict(t=np.arange(riv.shape[0]),
                riv_return=_mean(ret), riv_never=_mean(~ret),
                riv_base=float(base.mean()), frac_return=float(ret.mean()))


def fig_punishment(cell=CELL, figdir=None, label=""):
    """Not a grim trigger — a short price war that still (mostly) deters."""
    caches = []
    for subdir, cfg, title, tag in MARKETS:
        p = os.path.join(HERE, subdir, f"impulse_cache_{cell}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing — run:\n"
                f"    MARKET_CONFIG={cfg} python -m "
                f"qlearning_collusion.fig_lmp_learning --cache --cell {cell}")
        caches.append((np.load(p), title, tag))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8))

    for c, (Z, title, tag) in enumerate(caches):
        # ---- row 1: the punishment, and what a grim trigger would look like
        ax = axes[0, c]
        t = Z["low_t"]
        grim = float(Z["low_grim"])
        ax.axhline(0, color=INK, lw=1.1)
        ax.axhline(grim, color=RULE_N, lw=1.4, ls="-.")
        ax.annotate(f"a GRIM TRIGGER would sit here forever  (+{grim:.0f} MW, "
                    "permanent reversion to Cournot–Nash)",
                    xy=(0.2, grim), xytext=(0, 6), textcoords="offset points",
                    color=INK2, fontsize=8.8)
        frac = float(Z["low_frac"])
        ax.plot(t, Z["low_return"], color=SERIES[0], lw=2.4, marker="o", ms=4.0,
                solid_capstyle="round",
                label=f"cartel survives  ({100*frac:.0f}% of sessions)")
        never = Z["low_never"]
        top = grim
        # Only worth drawing when that group is a real share of sessions —
        # otherwise it is a handful of paths labelled "0%".
        if np.isfinite(never).any() and (1 - frac) >= 0.01:
            ax.plot(t, never, color=SERIES[2], lw=2.2, ls="--", marker="o",
                    ms=3.6, solid_capstyle="round",
                    label=f"cartel never returns  ({100*(1-frac):.0f}%)")
            top = max(top, float(np.nanmax(never)))
        else:
            ax.annotate(f"{100*frac:.1f}% of cartels came back — with no demand\n"
                        "shock there is no phantom cheat to get stuck punishing",
                        xy=(t[-1] * 0.98, grim * 0.72), color=INK2, fontsize=8.8,
                        ha="right", va="center")
        ax.annotate("punishment has\nfaded to zero",
                    xy=(t[-1], Z["low_return"][-1]), xytext=(-6, 26),
                    textcoords="offset points", color=SERIES_TEXT[0],
                    fontsize=9, ha="right", fontweight="semibold",
                    arrowprops=dict(arrowstyle="-", color=SERIES[0], lw=0.8,
                                    alpha=0.6))
        ax.set_ylim(-0.06 * top, top * 1.30)
        ax.set_xlim(-0.4, t[-1] + 0.4)
        _style(ax, "Periods after the one-period deviation",
               "Rivals' output above the no-deviation twin (MW)", "")
        ax.set_title(f"{title} — the punishment after one firm cheats",
                     color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")
        leg = ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.40),
                        frameon=False, fontsize=8.8, handlelength=1.9)
        for x in leg.get_texts():
            x.set_color(INK2)

        # ---- row 2: does the fading punishment still deter cheating? -------
        ax = axes[1, c]
        ax.axhline(0, color=INK, lw=1.1)
        dems = [str(d) for d in Z["demands"]] if "demands" in Z.files \
            else ["low", "high"]
        styles = {"low": (SERIES[0], "-"), "high": (SERIES[2], "--")}
        for dem in dems:
            col, ls = styles[dem]
            cum = Z[f"{dem}_cum"][:VALUE_PLOT]
            x = np.arange(len(cum))
            lab = (f"{dem} demand — " if len(dems) > 1 else "")
            ax.plot(x, cum, color=col, lw=2.3, ls=ls, solid_capstyle="round",
                    label=f"{lab}deterred in "
                          f"{100*float(Z[f'{dem}_deterred']):.0f}% of sessions")
            ax.scatter([x[-1]], [cum[-1]], s=30, color=col, zorder=5,
                       edgecolor="white", lw=1.3)
        lowcum = Z["low_cum"][:VALUE_PLOT]
        ax.fill_between(x, 0, lowcum, where=lowcum < 0,
                        color=SERIES[0], alpha=0.12, lw=0)
        _style(ax, "Periods after the one-period deviation",
               "Cumulative discounted gain from cheating ($)", "")
        ax.set_title("…and whether cheating still pays once it is priced in",
                     color=INK, fontsize=10.5, loc="left", pad=8,
                     fontweight="semibold")
        ax.set_xlim(0, VALUE_PLOT - 1)
        leg = ax.legend(loc="upper right", frameon=False, fontsize=8.8,
                        handlelength=2.0)
        for x_ in leg.get_texts():
            x_.set_color(INK2)

    fig.suptitle("What punishment did they actually learn? A short price war — "
                 "not a grim trigger"
                 + (f"  ·  {label}" if label else ""),
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.01,
                 fontweight="semibold")
    stuck = any(1 - float(Z["low_frac"]) >= 0.01 for Z, _, _ in caches)
    fig.text(0.006, -0.05,
             "One firm is forced into its static best response for a single period, then reverts to its learned strategy; demand is frozen (top row) so only the deviation\n"
             "moves the price, and every path is averaged over all converged sessions, all 12 phases of the limit cycle, and all choices of which firm cheats.\n"
             "Top: rivals expand output for a handful of periods and then go back — a grim trigger would instead step to the dash-dot line and stay there."
             + (" The minority of\nsessions that never return are not punishing; a deviation tipped their deterministic play into a different absorbing cycle."
                if stuck else "\n")
             + f" Bottom: δ = 0.95.\nCell: {_cell_label(cell)}.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, "fig_punishment.png", figdir)


# ---------------------------------------------------------------------------
def fig_cells_compare():
    """All four Table I cells, both markets: what the shock and the state do.

    The two axes of the paper's Table I are the state variable (price only vs
    the whole past output profile) and whether demand is shocked. Killing the
    shock is what actually makes monitoring perfect on this market — the
    measured non-revealing fraction falls from 0.38 to 0.007 — so the two rows
    of the table are far less different from each other than the two columns.
    """
    cells = ["imperfect_stochastic", "perfect_stochastic",
             "imperfect_deterministic", "perfect_deterministic"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
    dids, nonrev = {}, {}

    for c, (subdir, _cfg, title, tag) in enumerate(MARKETS):
        ax = axes[c]
        ax.axhline(0.0, color=RULE_N, lw=1.3, ls="--")
        ax.axhline(1.0, color=RULE_M, lw=1.3, ls="--")
        ax.annotate("joint monopoly  (Δ = 1)", xy=(0, 1.0), xytext=(2, -13),
                    textcoords="offset points", color=INK2, fontsize=9)
        ax.annotate("Cournot–Nash  (Δ = 0)", xy=(0, 0.0), xytext=(2, 6),
                    textcoords="offset points", color=INK2, fontsize=9)
        ends, final = [], {}
        for j, cell in enumerate(cells):
            meta, z = _load(subdir, cell)
            det = bool(meta["market"]["deterministic"])
            it, d = z["log_iters"], z["log_delta"]
            col = SERIES[0] if det else SERIES[2]
            ls = "-" if cell.startswith("imperfect") else "--"
            ax.plot(it, d, color=col, ls=ls, lw=2.2, solid_capstyle="round")
            final[cell] = meta["result"]["delta"]
            nonrev.setdefault(tag, {})[cell] = \
                meta["market"]["monitoring_report"]["measured_nonrevealing_fraction"]
            ends.append((d[-1], it[-1], col,
                         f"{'no shock' if det else 'shock'}, "
                         f"{'price only' if cell.startswith('imperfect') else 'full profile'}"
                         f"   Δ={100*meta['result']['delta']:.1f}%"
                         f"  (n={meta['config']['n_sessions']:,})"))
        # the paper's statistic: what the shock costs when firms can only see
        # the price, over and above what it costs when they see everything
        cost_price = 100 * (final["imperfect_stochastic"] -
                            final["imperfect_deterministic"])
        cost_prof = 100 * (final["perfect_stochastic"] -
                           final["perfect_deterministic"])
        ax.annotate(f"adding the shock costs {abs(cost_price):.1f} pp when firms "
                    f"see only the price,\nbut {abs(cost_prof):.1f} pp when they see "
                    f"the whole profile\n"
                    f"→ pure imperfect-monitoring effect: "
                    f"{cost_price - cost_prof:+.1f} pp",
                    xy=(0.03, -0.30), xycoords=("axes fraction", "data"),
                    color=INK, fontsize=9, ha="left", va="top",
                    fontweight="semibold")
        dids[tag] = cost_price - cost_prof
        xmax = max(e[1] for e in ends) * 1.06
        order = _despread([e[0] for e in ends], min_gap=0.075)
        for i, y in order:
            ax.annotate(ends[i][3], xy=(ends[i][1], ends[i][0]),
                        xytext=(xmax * 1.01, y), textcoords="data",
                        color=ends[i][2], fontsize=8.8, va="center",
                        fontweight="semibold", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=ends[i][2],
                                        lw=0.7, alpha=0.5, shrinkA=0, shrinkB=2))
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, 1.15)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_millions))
        _style(ax, "Time step (Q-learning iteration)",
               "Combined profit gain  Δ", "")
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")

    fig.suptitle("All four cells of the paper's Table I — a demand shock costs "
                 "the cartel only when firms cannot see who cheated",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.02,
                 fontweight="semibold")
    did_txt = ", ".join(f"{t} {d:+.1f} pp" for t, d in dids.items())
    sign = ("negative in both markets" if all(d < 0 for d in dids.values())
            else "negative in one market and not the other — read the signs, not the label"
            if any(d < 0 for d in dids.values()) else "positive in both markets")
    nr_txt = "; ".join(
        f"{t}: {v.get('imperfect_stochastic', float('nan')):.3f} with the shock vs "
        f"{v.get('imperfect_deterministic', float('nan')):.3f} without"
        for t, v in nonrev.items())
    fig.text(0.006, -0.16,
             "Blue = no demand shock, magenta = stochastic demand. Solid = the firms see only the price (\"imperfect monitoring\"); dashed = they see the whole past\n"
             "output profile (\"perfect monitoring\").\n"
             f"The single comparison that isolates imperfect monitoring is the difference-in-differences quoted in each panel: {did_txt} — {sign}.\n"
             "Read the levels more carefully than the gap: a no-shock cell need not beat every shocked one. What the difference-in-differences says is narrower — that\n"
             "the shock bites hardest once the price is the firms' only signal.\n"
             "Part of why the two rows can sit close: killing the shock is itself most of what makes monitoring perfect. Measured fraction of price signals that cannot\n"
             f"tell a cheat from a bad demand draw — {nr_txt} — so \"no shock, price only\" is already close to a perfect-monitoring experiment.\n"
             "Session counts are stated on each label and differ by cell; cells run at fewer sessions have correspondingly wider standard errors, so near-ties between\n"
             "them should not be read as a ranking.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, "fig_cells_compare.png", FIGDIR)


NOSHOCK_DIR = os.path.join(HERE, "figures_noshock")


def make_noshock():
    """The same three figures for the two no-shock cells, in their own folder."""
    out = [fig_cells_compare()]
    for cell, sub, lab in [
            ("imperfect_deterministic", "price_state",
             "no shock, price-only state"),
            ("perfect_deterministic", "profile_state",
             "no shock, full-profile state")]:
        d = os.path.join(NOSHOCK_DIR, sub)
        out += [fig_lmp(cell, d, lab), fig_learning(cell, d, lab),
                fig_punishment(cell, d, lab)]
    return out


if __name__ == "__main__":
    cell = CELL
    if "--cell" in sys.argv:
        cell = sys.argv[sys.argv.index("--cell") + 1]
    if "--cache" in sys.argv:
        cache_impulse(cell)
    elif "--noshock" in sys.argv:
        make_noshock()
    else:
        fig_lmp()
        fig_learning()
        fig_punishment()
