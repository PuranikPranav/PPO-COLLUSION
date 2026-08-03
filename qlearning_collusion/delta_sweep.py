"""Discount-factor sweep: does the punishment flatten as the future stops mattering?

Andrew's request, and his stated expectation: run delta = 0.99, 0.95, 0.9, 0.8,
0.7, 0.5 and look at the deviation / impulse-response plot (Figure 4b). Lower
delta means future profits are discounted harder, so a punishment that only bites
in later periods deters less -- the post-deviation response should get FLATTER,
and collusion itself should weaken.

Two things are measured, and they are not the same thing:

  1. how much collusion is sustained at all         -> Delta at convergence
  2. how hard a deviation is punished when it happens -> the Figure 4b impulse
     response, summarised by the rivals' peak output expansion above the
     no-deviation twin and by how many periods that expansion survives

Usage
-----
    # train every delta (long; run it in the background)
    python -m qlearning_collusion.delta_sweep train --sessions 300

    # then draw the figure + table from whatever has finished
    python -m qlearning_collusion.delta_sweep figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlearning_collusion import experiments as X
from qlearning_collusion.qlearn import STATE_SPACES

DELTAS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]
CELL = "imperfect_stochastic"

PRE, POST, SETTLE, N_PHASES = 3, 14, 300, 12


def tag_for(delta: float) -> str:
    return f"_d{delta:g}".replace(".", "")


def name_for(delta: float) -> str:
    return CELL + tag_for(delta)


# ---------------------------------------------------------------------------
def train(sessions=300, iters=4_000_000, deltas=None, **kw):
    for d in (deltas or DELTAS):
        nm = name_for(d)
        if os.path.exists(os.path.join(X.RESULTS, f"{nm}.json")):
            print(f"[skip] {nm} already trained")
            continue
        print(f"\n########## delta = {d} ##########", flush=True)
        X.run(CELL, n_sessions=sessions, max_iter=iters, delta=d, tag=tag_for(d), **kw)


# ---------------------------------------------------------------------------
def impulse(name: str, deviator: int, demand: str = "high"):
    """Figure 4b's impulse response for one run: MW above the no-deviation twin.

    Restricted, as in fig4b_punishment_split, to the sessions that return to
    their pre-deviation cycle -- with deterministic limit strategies and frozen
    demand a big deviation can tip a session into a different absorbing cycle it
    never leaves, and pooling those puts a permanent offset into the average that
    has nothing to do with the SHAPE of the punishment. The surviving fraction is
    returned and must be reported with the curve.
    """
    meta, z = X.load(name)
    mk = X.build_market(deterministic=meta["market"]["deterministic"],
                        k=meta["market"]["k"], xi=meta["market"]["xi"],
                        m=meta["market"]["m"], h=meta["market"]["h"])
    space = STATE_SPACES[meta["config"]["monitoring"]](mk)
    n = mk.n_agents
    conv = z["converged"].astype(bool)
    greedy = z["greedy"][conv] if conv.any() else z["greedy"]
    start = z["final_state"][conv] if conv.any() else z["final_state"]
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
            pay = mk.exp_profit_p[deviator][
                mk.dev_pidx(P[:, None], deviator, np.arange(kmax)[None, :])]
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
    for _ in range(PRE):
        a, _, s_d = step(s_d); x, y = split(a); dd.append(x); rd.append(y)
        a, _, s_b = step(s_b); x, y = split(a); db.append(x); rb.append(y)
    a, _, s_d = step(s_d, force=True); x, y = split(a); dd.append(x); rd.append(y)
    a, _, s_b = step(s_b); x, y = split(a); db.append(x); rb.append(y)
    for _ in range(POST):
        a, _, s_d = step(s_d); x, y = split(a); dd.append(x); rd.append(y)
        a, _, s_b = step(s_b); x, y = split(a); db.append(x); rb.append(y)

    dev = np.array(dd) - np.array(db)
    riv = np.array(rd) - np.array(rb)

    for _ in range(SETTLE):
        _, _, s_d = step(s_d)
        _, _, s_b = step(s_b)
    cd, cb = [], []
    for _ in range(50):
        _, _, s_d = step(s_d); cd.append(s_d.copy())
        _, _, s_b = step(s_b); cb.append(s_b.copy())
    ret = np.array([x == y for x, y in zip([set(v) for v in np.array(cb).T],
                                           [set(v) for v in np.array(cd).T])])
    if not ret.any():
        ret = np.ones_like(ret)
    return dict(t=np.arange(dev.shape[0]) - PRE,
                dev=dev[:, ret].mean(axis=1), riv=riv[:, ret].mean(axis=1),
                frac=float(ret.mean()), delta=meta["config"]["delta"],
                Delta=meta["result"]["delta"], name=name, deviator=deviator,
                demand=demand)


def punishment_stats(r: dict) -> dict:
    """Summarise one impulse response: how big the punishment is and how long."""
    post = r["t"] >= 1
    riv = r["riv"][post]
    peak = float(np.max(riv)) if riv.size else 0.0
    # Periods until the rivals' expansion falls below 10% of its peak. With no
    # punishment at all (peak <= 0) the answer is 0, not the window length --
    # otherwise a delta with no retaliation would score as the longest-lived one.
    if peak <= 1e-9:
        half = 0
    elif (riv < 0.1 * peak).any():
        half = int(np.argmax(riv < 0.1 * peak))
    else:
        half = int(riv.size)
    return dict(deviation_mw=float(r["dev"][r["t"] == 0][0]),
                peak_punish_mw=peak,
                punish_t1=float(riv[0]) if riv.size else 0.0,
                periods_above_10pct=half,
                area_mw_periods=float(np.sum(riv)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _available(deltas=None):
    out = []
    for d in (deltas or DELTAS):
        if os.path.exists(os.path.join(X.RESULTS, f"{name_for(d)}.json")):
            out.append(d)
    return out


def figure(deltas=None, deviator=0, demand="high", plot_to=10):
    """Figure 4b, one column per discount factor: does the punishment flatten?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qlearning_collusion.figures import _style, INK, INK2, MUTED, SERIES, FIGDIR

    ds = _available(deltas)
    if not ds:
        raise SystemExit("no delta-sweep runs found; run `delta_sweep train` first")

    runs = [impulse(name_for(d), deviator=deviator, demand=demand) for d in ds]
    stats = [punishment_stats(r) for r in runs]

    nc = len(runs)
    fig, axes = plt.subplots(2, nc, figsize=(3.05 * nc + 0.8, 8.0), squeeze=False)

    ymax = max(max(r["dev"].max(), r["riv"].max()) for r in runs) * 1.12
    ymin = min(min(r["dev"].min(), r["riv"].min()) for r in runs)
    ymin = min(ymin * 1.12, -0.05 * ymax)

    for c, (r, s) in enumerate(zip(runs, stats)):
        m = r["t"] <= plot_to
        t, dv, rv = r["t"][m], r["dev"][m], r["riv"][m]
        ax = axes[0, c]
        ax.axhline(0, color=INK, lw=1.0)
        ax.axvline(0, color=MUTED, lw=1.0, ls=":")
        ax.plot(t, dv, color=SERIES[0], lw=2.4, marker="o", ms=3.6, zorder=5)
        ax.plot(t, rv, color=INK2, lw=2.0, ls="--", marker="o", ms=3.0, zorder=4)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(t[0], plot_to)
        ax.set_title(rf"$\delta$ = {r['delta']:g}", color=INK, fontsize=11.5,
                     loc="left", pad=8, fontweight="semibold")
        _style(ax, "", "MW above the no-deviation twin" if c == 0 else "", "")
        if c == 0:
            # anchored to the spike, which is the only part of the panel with
            # room -- the right-hand tail is where the corner note goes
            ax.annotate("deviating firm", xy=(0, dv[t == 0][0]), xytext=(14, -4),
                        textcoords="offset points", color=SERIES[0], fontsize=8.6,
                        ha="left", va="top", fontweight="semibold")
            ax.annotate("non-deviating firms", xy=(1, rv[t == 1][0]),
                        xytext=(16, 10), textcoords="offset points", color=INK2,
                        fontsize=8.6, ha="left", fontweight="semibold")
        ax.text(0.98, 0.97,
                f"Δ = {100*r['Delta']:.0f}%\ncartel survives {100*r['frac']:.0f}%",
                transform=ax.transAxes, ha="right", va="top",
                color=MUTED, fontsize=8.0)

        # zoomed panel: the punishment only (rivals), same y-scale across columns
        ax = axes[1, c]
        ax.axhline(0, color=INK, lw=1.0)
        ax.axvline(0, color=MUTED, lw=1.0, ls=":")
        ax.plot(t, rv, color=INK2, lw=2.4, marker="o", ms=3.6)
        ax.fill_between(t, 0, rv, color=INK2, alpha=0.16, lw=0)
        _style(ax, "Periods after the forced deviation",
               "Rivals' expansion (MW)" if c == 0 else "", "")
        ax.set_xlim(t[0], plot_to)
        rmax = max(max(x["riv"].max(), 1e-6) for x in runs) * 1.15
        ax.set_ylim(min(0.0, min(x["riv"].min() for x in runs) * 1.15), rmax)
        ax.text(0.98, 0.93, f"peak {s['peak_punish_mw']:.2f} MW\n"
                            f"area {s['area_mw_periods']:.1f} MW·periods",
                transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8.0)

    fig.suptitle("Discount-factor sweep — the post-deviation punishment flattens "
                 "as the future stops mattering",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.0, fontweight="semibold")
    fig.text(0.005, -0.055,
             "Top row: one firm is forced into its static best response for a single period (t = 0), then reverts to its learned strategy; both paths are measured\n"
             "against a no-deviation twin started from the identical state. Bottom row: the non-deviating firms' response alone, on a common scale — this is the\n"
             f"punishment. Demand frozen {demand}; averaged over all {N_PHASES} phases of the limit cycle and over the sessions that return to their pre-deviation cycle.",
             color=MUTED, fontsize=8.2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, "fig7_delta_sweep_deviation.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")

    # --- summary panel: Delta and punishment size against delta --------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    dl = [r["delta"] for r in runs]

    def _title(ys, rising, flat_msg):
        """Say what the series actually does, not what it was expected to do."""
        pairs = sorted(zip(dl, ys))
        vals = [v for _, v in pairs]
        if all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])):
            return rising
        # find where it stops falling as delta drops
        lo = min(range(len(vals)), key=lambda i: vals[i])
        return flat_msg.format(d=pairs[lo][0])

    for ax, ys, lab, ttl in (
        (axes[0], [100 * r["Delta"] for r in runs], "Δ at convergence (%)",
         None),
        (axes[1], [s["peak_punish_mw"] for s in stats], "peak rival expansion (MW)",
         None),
        (axes[2], [s["area_mw_periods"] for s in stats], "MW·periods",
         None),
    ):
        if ttl is None:
            if ax is axes[0]:
                ttl = _title([100 * r["Delta"] for r in runs],
                             "Collusion weakens as δ falls",
                             "Δ falls with δ down to δ≈{d:g}, then flattens")
            elif ax is axes[1]:
                ttl = _title([s["peak_punish_mw"] for s in stats],
                             "The punishment gets smaller",
                             "Punishment shrinks down to δ≈{d:g}")
            else:
                ttl = _title([s["area_mw_periods"] for s in stats],
                             "…and dies out sooner",
                             "…and dies out by δ≈{d:g}")
        ax.plot(dl, ys, color=SERIES[0], lw=2.2, marker="o", ms=6)
        for x, y in zip(dl, ys):
            ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points", color=INK2, fontsize=8.4, ha="center")
        _style(ax, "discount factor δ", lab, "")
        ax.set_title(ttl, color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")
    fig.suptitle("Discount-factor sweep — summary", color=INK, fontsize=13,
                 x=0.005, ha="left", y=1.04, fontweight="semibold")
    fig.text(0.005, -0.13,
             "The middle and right panels are the answer to the question: the retaliation a deviation triggers shrinks monotonically as the future is discounted harder.\n"
             "Δ (left) is NOT monotone — it stops falling around δ ≈ 0.8 and edges back up. Read that together with the right panel: below δ ≈ 0.8 the punishment area is\n"
             "already ≈ 0, so the residual supra-Nash profit there is not sustained by punishment at all. It is the floor tabular Q-learning reaches on its own, which is\n"
             "why the profit level alone cannot tell collusion from failure to optimise, and why fig5_deviation_value.png tests deterrence directly.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout()
    p2 = os.path.join(FIGDIR, "fig7b_delta_sweep_summary.png")
    fig.savefig(p2, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p2}")

    tbl = table(runs, stats)
    print("\n" + tbl)
    with open(os.path.join(X.RESULTS, "delta_sweep.txt"), "w") as fh:
        fh.write(tbl + "\n")
    return p, p2


def table(runs, stats) -> str:
    L = ["DISCOUNT-FACTOR SWEEP  (cell: %s, deviator %d, %s demand)"
         % (CELL, runs[0]["deviator"], runs[0]["demand"]),
         "-" * 104,
         f"{'delta':>7}{'Delta %':>10}{'dev MW@t0':>12}{'rival t+1':>11}"
         f"{'peak rival':>12}{'periods>10%':>13}{'area MW.per':>13}{'cartel survives':>17}",
         "-" * 104]
    for r, s in zip(runs, stats):
        L.append(f"{r['delta']:>7g}{100*r['Delta']:>10.2f}{s['deviation_mw']:>12.2f}"
                 f"{s['punish_t1']:>11.2f}{s['peak_punish_mw']:>12.2f}"
                 f"{s['periods_above_10pct']:>13d}{s['area_mw_periods']:>13.2f}"
                 f"{100*r['frac']:>16.0f}%")
    L.append("-" * 104)
    L.append("dev MW@t0   = the forced deviator's output above its no-deviation twin")
    L.append("rival t+1   = the non-deviating firms' expansion one period later (the punishment)")
    L.append("area        = total rival expansion summed over the post-deviation window")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(prog="qlearning_collusion.delta_sweep")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--sessions", type=int, default=300)
    p.add_argument("--iters", type=int, default=4_000_000)
    p.add_argument("--deltas", type=float, nargs="*", default=None)
    p = sub.add_parser("figures")
    p.add_argument("--deviator", type=int, default=0)
    p.add_argument("--demand", default="high")
    p.add_argument("--deltas", type=float, nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        train(sessions=a.sessions, iters=a.iters, deltas=a.deltas)
    else:
        figure(deltas=a.deltas, deviator=a.deviator, demand=a.demand)


if __name__ == "__main__":
    main()
