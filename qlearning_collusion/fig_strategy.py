"""The ACTION plots: what rule did each firm actually learn, and why is it a
fading price war rather than a grim trigger?

Three things are drawn, for the three-firm and two-firm markets side by side:

  fig_reaction.png    each firm's learned action as a function of the price it
                      observed last period, against what a grim trigger would
                      look like; and the price map the three rules jointly
                      induce, with the cobweb that walks the price home.
  fig_why_not_grim.png the counterfactual test: rescale one firm's reaction
                      slope from "never punish" through "learned" to
                      "grim-like", and measure that firm's OWN long-run profit
                      — with a matched random-perturbation control, because a
                      converged greedy policy is a local optimum and any change
                      of a given size costs something.

The headline point the first figure makes is structural, and does not depend on
the second: with a one-period memory a firm's strategy IS a map from last
period's price to this period's output. A grim trigger needs a "somebody
cheated" flag that survives the price recovering, and there is nowhere to keep
one. So once the price returns to its pre-deviation bin the firm is in the same
state it was in before and must replay the same action — the punishment ends
whether or not anyone wants it to. The only way a punishment can be permanent
is if the punished price state maps to ITSELF, which is exactly the minority of
sessions whose cartel never comes back.

    MARKET_CONFIG=three_firm python -m qlearning_collusion.fig_strategy --cache
    MARKET_CONFIG=two_firm   python -m qlearning_collusion.fig_strategy --cache
    python -m qlearning_collusion.fig_strategy
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlearning_collusion.figures import (        # noqa: E402
    SERIES, INK, INK2, MUTED, RULE_N, RULE_M, _style, _despread,
)
from qlearning_collusion.fig_lmp_learning import (   # noqa: E402
    CELL, MARKETS, SERIES_TEXT, FIGDIR, _load, _save,
)

# the punishment-slope sweep: a_i'(s) = round(a_rest + LAM * (a_i(s) - a_rest))
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
SWEEP_SESSIONS = 300          # sessions used for the counterfactual sweep
SWEEP_T, SWEEP_BURN = 4000, 500


# ===========================================================================
# CACHE (needs the market object, so it runs once per MARKET_CONFIG)
# ===========================================================================
def cache_strategy(do_sweep: bool = True):
    """`do_sweep=False` recomputes only the cheap parts and carries the existing
    counterfactual sweep over, so the reaction curves can be re-derived without
    paying for the sweep again."""
    from qlearning_collusion import experiments as X
    from qlearning_collusion.qlearn import STATE_SPACES

    cfg = os.environ.get("MARKET_CONFIG", "three_firm").strip().lower()
    subdir = {m[1]: m[0] for m in MARKETS}[cfg]
    meta, z = _load(subdir)

    # ---- 1. the learned reaction function (visit-weighted, per firm) -------
    ls = X.limit_strategy(CELL)
    mk = ls["market"]
    n = mk.n_agents
    gb = meta["market"]["grid_benchmarks"]
    out = dict(price=ls["price"], q=ls["weighted_avg_q"], w=ls["state_weight"],
               q_grid=mk.q_agent, nash_gens=np.array(gb["nash"]["gens"]),
               nash_price=float(gb["nash"]["hub_price"]),
               mono_price=float(gb["monopoly"]["hub_price"]))

    # the demand schedule, for the slope comparison
    o = np.argsort(mk.total_gen_p)
    out["sched_gen"] = mk.total_gen_p[o] / n
    out["sched_price"] = mk.exp_price_p[o]

    # ---- 2. the price map p_t -> p_{t+1} the learned rules induce ----------
    # Visit-WEIGHTED, exactly as `limit_strategy` weights the actions: a session
    # that never reaches state s still has an untrained policy there, so a flat
    # average across sessions describes a market nobody is in. Weighting by who
    # actually visits s is what makes the map's fixed point line up with the
    # resting price the LMP figure reports.
    space = STATE_SPACES["imperfect"](mk)
    conv = z["converged"].astype(bool)
    greedy = z["greedy"][conv].astype(np.int64)              # (S, n, nS)
    S, _, nS = greedy.shape
    ar = np.arange(n)
    rowS = np.arange(S)[:, None]
    rng = np.random.default_rng(7)
    st = z["final_state"][conv].copy()
    acc = np.zeros(nS)
    cnt = np.zeros(nS)
    for t in range(5000):
        a = greedy[rowS, ar[None, :], st[:, None]]
        P = mk.pidx(a)
        u = rng.integers(0, mk.h, size=S) if mk.h > 1 else np.zeros(S, dtype=np.int64)
        np.add.at(acc, st, mk.exp_price_p[P])
        np.add.at(cnt, st, 1.0)
        st = space.next_state(a, P, u)
    # `limit_strategy` hands back its arrays sorted along the price axis (a rich
    # state index is not price-ordered), so apply the same permutation to the
    # state-indexed arrays built here or the two would not line up.
    perm = ls["order"]
    out["price_next"] = np.divide(acc, cnt, out=np.full(nS, np.nan), where=cnt > 0)[perm]
    out["visit"] = (cnt / cnt.sum())[perm]
    out["p_rest"] = float(np.average(ls["price"], weights=cnt[perm]))

    # ---- 3. the counterfactual punishment-slope sweep ----------------------
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), subdir,
                     "strategy_cache.npz")
    if not do_sweep:
        old = np.load(p)
        for k in ("lam", "own", "riv", "ctl", "n_sweep"):
            out[k] = old[k]
        np.savez_compressed(p, **out)
        print(f"  wrote {p}  (sweep carried over)")
        return p

    g = greedy[:SWEEP_SESSIONS]
    start = z["final_state"][conv][:SWEEP_SESSIONS].copy()
    Ssw = g.shape[0]
    rowS = np.arange(Ssw)[:, None]
    kmax = int(np.max(mk.n_actions))

    def simulate(pol, seed=3):
        rng = np.random.default_rng(seed)
        st = start.copy()
        tot = np.zeros((Ssw, n))
        for t in range(SWEEP_T + SWEEP_BURN):
            a = pol[rowS, ar[None, :], st[:, None]]
            P = mk.pidx(a)
            u = (rng.integers(0, mk.h, size=Ssw) if mk.h > 1
                 else np.zeros(Ssw, dtype=np.int64))
            if t >= SWEEP_BURN:
                tot += mk.profit_p[ar[None, :], P[:, None], u[:, None]]
            st = space.next_state(a, P, u)
        return (tot / SWEEP_T).mean(axis=0)

    # each session's own resting state, so "a_rest" is that session's normal play
    rng = np.random.default_rng(3)
    st = start.copy()
    seen = np.zeros((Ssw, nS))
    for t in range(1500):
        a = g[rowS, ar[None, :], st[:, None]]
        P = mk.pidx(a)
        u = (rng.integers(0, mk.h, size=Ssw) if mk.h > 1
             else np.zeros(Ssw, dtype=np.int64))
        if t >= 300:
            np.add.at(seen, (np.arange(Ssw), st), 1.0)
        st = space.next_state(a, P, u)
    s_rest = seen.argmax(axis=1)

    base = simulate(g)
    own = np.zeros((n, len(LAMBDAS)))
    riv = np.zeros((n, len(LAMBDAS)))
    ctl = np.zeros((n, len(LAMBDAS)))
    crng = np.random.default_rng(11)
    for i in range(n):
        a_rest = g[np.arange(Ssw), i, s_rest][:, None]
        for j, lam in enumerate(LAMBDAS):
            newa = np.clip(np.rint(a_rest + lam * (g[:, i, :] - a_rest)),
                           0, kmax - 1).astype(np.int64)
            d = newa - g[:, i, :]
            frac = float((d != 0).mean())
            mad = float(np.abs(d)[d != 0].mean()) if (d != 0).any() else 0.0
            pol = g.copy(); pol[:, i, :] = newa
            r = simulate(pol)
            own[i, j] = 100 * (r[i] / base[i] - 1)
            riv[i, j] = 100 * (np.delete(r, i).mean() /
                               np.delete(base, i).mean() - 1)
            # matched control: same share of states moved, same mean |shift|,
            # random directions -- isolates "changed at all" from "changed THIS way"
            cpol = g.copy()
            mask = crng.random((Ssw, nS)) < frac
            sign = crng.choice([-1, 1], size=(Ssw, nS))
            mag = crng.poisson(mad, size=(Ssw, nS)) if mad > 0 else 0
            cpol[:, i, :] = np.clip(g[:, i, :] + mask * sign * mag,
                                    0, kmax - 1).astype(np.int64)
            ctl[i, j] = 100 * (simulate(cpol)[i] / base[i] - 1)
            print(f"    firm {i} lambda={lam:5.2f}  own {own[i, j]:+6.2f}%  "
                  f"rivals {riv[i, j]:+6.2f}%  control {ctl[i, j]:+6.2f}%",
                  flush=True)
    out.update(lam=np.array(LAMBDAS), own=own, riv=riv, ctl=ctl,
               n_sweep=Ssw)
    np.savez_compressed(p, **out)
    print(f"  wrote {p}")
    return p


def _cache(subdir):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), subdir,
                     "strategy_cache.npz")
    if not os.path.exists(p):
        cfg = {m[0]: m[1] for m in MARKETS}[subdir]
        raise FileNotFoundError(
            f"{p} missing — run:\n    MARKET_CONFIG={cfg} python -m "
            f"qlearning_collusion.fig_strategy --cache")
    return np.load(p)


# ===========================================================================
# FIGURE 1 — the action plots
# ===========================================================================
def fig_reaction():
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))

    for c, (subdir, _cfg, title, tag) in enumerate(MARKETS):
        Z = _cache(subdir)
        p, q, w = Z["price"], Z["q"], Z["w"]
        n = q.shape[1]
        qN = Z["nash_gens"]

        # Two regions matter and they are different animals: the narrow band of
        # prices ordinary demand noise moves the market within (where a war must
        # NOT be triggered), and the low prices only a real output expansion
        # reaches (where it must).
        vis = Z["visit"]
        order = np.argsort(-vis)
        keep = np.zeros_like(vis, dtype=bool)
        keep[order[np.cumsum(vis[order]) <= 0.90]] = True
        keep[order[0]] = True
        idx = np.where(keep)[0]
        b0, b1 = p[idx.min()], p[idx.max()]
        band = (p >= b0) & (p <= b1) & (vis > 0)
        seen = vis > 5e-4                     # >=0.05% of the time: real data
        p_rest = float(Z["p_rest"])

        # ---- row 1: how far output moves ABOVE its normal-band level -------
        ax = axes[0, c]
        normal = np.array([np.average(q[band, i], weights=vis[band])
                           for i in range(n)])
        dq = q - normal[None, :]
        grim = qN - normal                    # what reverting to Nash would be
        x0, x1 = p[seen].min(), p[seen].max()

        ax.axvspan(b0, b1, color=MUTED, alpha=0.10, lw=0)
        ax.axhline(0, color=INK, lw=1.1)
        for i in range(n):
            ax.axhline(grim[i], color=SERIES[i], lw=1.2, ls="-.", alpha=0.55)
            ax.plot(p[seen], dq[seen, i], color=SERIES[i], lw=2.5,
                    solid_capstyle="round", zorder=4)
        y0 = min(dq[seen].min(), -1.5) - 1.5
        y1 = grim.max() * 1.42
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

        # the grim reference, labelled just above the highest dash-dot line
        ax.annotate("dash-dot = where a GRIM TRIGGER would put that firm —\n"
                    "reverting to Cournot–Nash and staying there",
                    xy=(x1, grim.max()), xytext=(-6, 9),
                    textcoords="offset points", color=INK2, fontsize=8.8,
                    ha="right", va="bottom")
        ax.annotate("where ordinary demand noise keeps the price\n"
                    f"({100*vis[band].sum():.0f}% of the time) — and the rule "
                    "is at its flattest here",
                    xy=(0.5 * (b0 + b1), y1), xytext=(0, -6),
                    textcoords="offset points", color=MUTED, fontsize=8.6,
                    ha="center", va="top")

        lowm, highm = seen & (p < b0), seen & (p > b1)
        punish = dq[lowm].mean(axis=0).mean()
        reward = dq[highm].mean(axis=0).mean() if highm.sum() else np.nan
        ax.annotate(f"punish: +{punish:.1f} MW on average once the price falls\n"
                    f"out of the band — {100*punish/grim.mean():.0f}% of a "
                    f"reversion to Cournot–Nash,\nand it lasts only while the "
                    f"price stays down",
                    xy=(x0, y0), xytext=(4, 6), textcoords="offset points",
                    color=INK, fontsize=8.8, ha="left", va="bottom",
                    fontweight="semibold")
        if np.isfinite(reward) and abs(reward) > 1.0:
            ax.annotate(f"and {reward:.1f} MW when the\nprice runs high",
                        xy=(x1, 0), xytext=(-6, -8), textcoords="offset points",
                        color=INK2, fontsize=8.6, ha="right", va="top")
        _series_labels(ax, x1, [dq[seen, i][-1] for i in range(n)],
                       [f"Firm {i}" for i in range(n)], (y1 - y0) * 0.06)
        _style(ax, "Price observed last period (hub LMP, $/MWh)",
               "Output above this firm's normal level (MW)", "")
        ax.set_title(f"{title} — the rule each firm learned",
                     color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")

        # ---- row 2: the price map — how many fixed points does it have? ----
        ax = axes[1, c]
        f = Z["price_next"]
        pN = float(Z["nash_price"])
        z0, z1 = min(x0, pN) - 0.4, x1 + 0.4
        ax.plot([z0, z1], [z0, z1], color=INK, lw=1.2, ls="--")
        ax.annotate("45°  (price stays put)",
                    xy=(z0 + (z1 - z0) * 0.72, z0 + (z1 - z0) * 0.72),
                    xytext=(5, -13), textcoords="offset points", color=INK2,
                    fontsize=8.6)
        ax.plot(p[seen], f[seen], color=SERIES[0], lw=2.6, solid_capstyle="round")

        # every crossing of the 45° line is a resting point of the market
        d, cx = f[seen] - p[seen], p[seen]
        xr = np.where(np.diff(np.sign(d)) != 0)[0]
        fixed = [float(np.interp(0.0, [d[i + 1], d[i]], [cx[i + 1], cx[i]]))
                 for i in xr]
        for fx in fixed:
            ax.scatter([fx], [fx], s=46, color=INK, zorder=6,
                       edgecolor="white", lw=1.4)
        ax.annotate(f"the ONLY fixed point, and it attracts:\n"
                    f"the collusive price, \\${fixed[0]:.2f}",
                    xy=(fixed[0], fixed[0]), xytext=(-10, 24),
                    textcoords="offset points", color=INK, fontsize=8.9,
                    ha="right", fontweight="semibold",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))

        ax.scatter([pN], [pN], s=70, color="white", zorder=6,
                   edgecolor=RULE_N, lw=1.8, marker="o")
        ax.annotate(f"a GRIM TRIGGER would need a SECOND fixed point\n"
                    f"here, at Cournot–Nash (\\${pN:.2f}) — the map never\n"
                    f"touches the 45° line again",
                    xy=(pN, pN), xytext=(8, -6), textcoords="offset points",
                    color=INK2, fontsize=8.8, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color=RULE_N, lw=0.9))

        ax.set_xlim(z0, z1); ax.set_ylim(z0, z1)
        _style(ax, "Price this period ($/MWh)", "Price next period ($/MWh)", "")
        ax.set_title("…and the price map those rules add up to",
                     color=INK, fontsize=10.5, loc="left", pad=8,
                     fontweight="semibold")

    fig.suptitle("The rule each firm learned IS a trigger — it just fires at "
                 "about a fifth of grim, and it does not stick",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.01,
                 fontweight="semibold")
    fig.text(0.006, -0.088,
             "Top: visit-weighted average greedy action across all 1,000 converged sessions, per firm, measured against that firm's own output inside the normal price\n"
             "band. Output rises as the observed price falls — a price drop IS punished — but the curve is at its FLATTEST inside the band ordinary demand noise moves\n"
             "the price within, which is what stops noise from starting a war, and the punishment it does reach is about a fifth of a reversion to Cournot–Nash. Only\n"
             "price states holding at least 0.05% of visit time are drawn; the rest are sampling noise.\n"
             "Bottom: the price map p(t) → p(t+1) these rules jointly induce, visit-weighted per state, averaged over the demand shock. It meets the 45° line exactly\n"
             "once, from above, at the collusive price — which is the same resting price the LMP figure reports, and the check that this panel is right. A grim trigger\n"
             "needs an ABSORBING punishment, i.e. a second fixed point at Cournot–Nash. There is none, and with a one-period price memory there is nowhere to put one:\n"
             "the state is last period's price and nothing else, so as soon as the price recovers the firm is in the state it was in before and must replay that action.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, "fig_reaction.png")


def _series_labels(ax, x_at, values, texts, min_gap):
    for i, y in _despread(list(values), min_gap=min_gap):
        ax.annotate(texts[i], xy=(x_at, values[i]),
                    xytext=(6, (y - values[i]) * 0 + 0), textcoords="offset points",
                    color=SERIES_TEXT[i], fontsize=9.5, va="center",
                    fontweight="semibold", annotation_clip=False)


# ===========================================================================
# FIGURE 2 — the counterfactual: would punishing harder pay?
# ===========================================================================
def fig_why_not_grim():
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

    for c, (subdir, _cfg, title, tag) in enumerate(MARKETS):
        Z = _cache(subdir)
        lam, own, riv, ctl = Z["lam"], Z["own"], Z["riv"], Z["ctl"]
        n = own.shape[0]
        x = np.arange(len(lam))
        ax = axes[c]

        ax.axhline(0, color=INK, lw=1.1)
        ax.fill_between(x, own.min(axis=0), own.max(axis=0),
                        color=SERIES[0], alpha=0.13, lw=0)
        ax.plot(x, own.mean(axis=0), color=SERIES[0], lw=2.6, marker="o", ms=4.6,
                solid_capstyle="round", zorder=5,
                label="the punisher's own profit")
        ax.plot(x, ctl.mean(axis=0), color=MUTED, lw=1.8, ls=":", marker="o",
                ms=3.4, zorder=4,
                label="control: a RANDOM change of the same size")
        ax.plot(x, riv.mean(axis=0), color=SERIES[2], lw=2.0, ls="--", marker="o",
                ms=3.8, zorder=4, label="its rivals' profit")

        j1 = int(np.where(np.isclose(lam, 1.0))[0][0])
        ax.axvline(j1, color=INK, lw=1.0, ls=":")
        ax.annotate("what it\nLEARNED", xy=(j1, 0), xytext=(0, 10),
                    textcoords="offset points", color=INK, fontsize=9,
                    ha="center", fontweight="semibold")
        ax.annotate("never\npunish", xy=(0, own.mean(axis=0)[0]), xytext=(6, -4),
                    textcoords="offset points", color=INK2, fontsize=8.8,
                    ha="left", va="top")
        ax.annotate("grim-like:\nany dip → flood", xy=(x[-1], own.mean(axis=0)[-1]),
                    xytext=(-6, -6), textcoords="offset points", color=INK2,
                    fontsize=8.8, ha="right", va="top")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in lam], fontsize=8.4)
        _style(ax, "Punishment slope, as a multiple of the learned one",
               "Change in long-run profit (%)", "")
        ax.set_title(f"{title}", color=INK, fontsize=11, loc="left", pad=8,
                     fontweight="semibold")
        leg = ax.legend(loc="lower center", frameon=False, fontsize=8.6,
                        handlelength=2.2)
        for t in leg.get_texts():
            t.set_color(INK2)

    fig.suptitle("Would punishing harder have paid? No — but the honest margin "
                 "is small",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.02,
                 fontweight="semibold")
    fig.text(0.006, -0.175,
             "One firm's reaction curve is rescaled about its own resting action — 0 flattens it to a constant (never punish), 1 is what it learned, 10 turns it into a step\n"
             "(any price dip → flood the market). Rivals keep their learned rules. Band = the spread across firms; line = the mean. Long-run average profit over 4,000\n"
             "periods from 300 converged sessions.\n"
             "READ THIS WITH THE CONTROL. A converged greedy policy is a local optimum, so a change of ANY kind costs something — the dotted line is a random change\n"
             "that touches the same share of states by the same mean number of rungs. Punishing harder is consistently worse than that control, but only by ~0.5–1 pp.\n"
             "What this experiment does show cleanly: harsher punishment costs the PUNISHER about as much as it costs its targets (blue vs magenta), which is why an\n"
             "unlimited punishment is not a free threat. The reason a grim trigger is not learned is structural, not marginal — see fig_reaction.png.",
             color=MUTED, fontsize=8.4, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, "fig_why_not_grim.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--cache" in sys.argv:
        cache_strategy(do_sweep="--no-sweep" not in sys.argv)
    else:
        fig_reaction()
        fig_why_not_grim()
