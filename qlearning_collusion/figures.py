"""Replications of Figures 1-4 (and Table I) of Calvano et al. (2021),
"Algorithmic collusion with imperfect monitoring", on the networked DC-OPF
market of this repo.

Colour policy: categorical slots 1-3 of the validated reference palette
(blue / green / magenta), assigned to firms in fixed order and never cycled.
Every series is ALSO direct-labelled, which is required anyway because slot 3
sits below 3:1 contrast on a light surface. Benchmarks are recessive grey rules,
not extra hues.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qlearning_collusion import experiments as X

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      os.path.basename(X.RESULTS).replace("results", "figures"))

SERIES = ["#2a78d6", "#008300", "#e87ba4"]      # slots 1, 2, 3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8c8b85"
GRID = "#e6e5e0"
RULE_N, RULE_M = "#9a9993", "#6f6e68"


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12,
                     fontweight="semibold")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def _millions(x, _):
    return "0" if x == 0 else f"{x/1e6:g}M"


# ---------------------------------------------------------------------------
def fig1_outputs(name=X.BASELINE):
    """Paper Figure 1: evolution of the greedy output levels."""
    meta, z = X.load(name)
    gb = meta["market"]["grid_benchmarks"]
    it, q = z["log_iters"], z["log_q"]
    n = q.shape[1]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    qN, qM = np.array(gb["nash"]["gens"]), np.array(gb["monopoly"]["gens"])
    ax.axhspan(qN.min(), qN.max(), color=RULE_N, alpha=0.13, lw=0)
    ax.axhspan(qM.min(), qM.max(), color=RULE_M, alpha=0.13, lw=0)
    ax.axhline(qN.mean(), color=RULE_N, lw=1.2, ls="--")
    ax.axhline(qM.mean(), color=RULE_M, lw=1.2, ls="--")
    ax.annotate("Cournot–Nash outputs", xy=(it[-1], qN.max()), xytext=(-4, 6),
                textcoords="offset points", color=INK2, fontsize=9, ha="right")
    ax.annotate("joint-monopoly (collusive) outputs", xy=(it[-1], qM.min()),
                xytext=(-4, -14), textcoords="offset points", color=INK2,
                fontsize=9, ha="right")

    for i in range(n):
        ax.plot(it, q[:, i], color=SERIES[i], lw=2.0, solid_capstyle="round")
        ax.annotate(f"Firm {i}", xy=(it[-1], q[-1, i]),
                    xytext=(6, 0), textcoords="offset points",
                    color=SERIES[i], fontsize=9.5, va="center", fontweight="semibold")

    _style(ax, "Iterations", "Output (MW)",
           "Figure 1 — greedy output levels converge below the Cournot–Nash level")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_millions))
    ax.set_xlim(0, it[-1] * 1.06)
    fig.text(0.5, -0.04,
             "Greedy output = the action with the highest Q-value in the current state; "
             "played output may differ during exploration.",
             color=MUTED, fontsize=8.5, ha="center")
    return _save(fig, "fig1_output_evolution.png")


# ---------------------------------------------------------------------------
def fig2_profits(name=X.BASELINE):
    """Paper Figure 2: evolution of the normalised profit gain Delta."""
    meta, z = X.load(name)
    it, d = z["log_iters"], z["log_delta"]
    final = meta["result"]["delta"]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(0.0, color=RULE_N, lw=1.2, ls="--")
    ax.axhline(1.0, color=RULE_M, lw=1.2, ls="--")
    ax.annotate("Cournot–Nash  (Δ = 0)", xy=(it[-1], 0.0), xytext=(-4, 7),
                textcoords="offset points", color=INK2, fontsize=9, ha="right")
    ax.annotate("joint monopoly  (Δ = 1)", xy=(it[-1], 1.0), xytext=(-4, -14),
                textcoords="offset points", color=INK2, fontsize=9, ha="right")
    ax.plot(it, d, color=SERIES[0], lw=2.0, solid_capstyle="round")
    ax.scatter([it[-1]], [d[-1]], s=34, color=SERIES[0], zorder=5,
               edgecolor="white", lw=1.4)
    ax.annotate(f"Δ = {100*final:.1f}%", xy=(it[-1], d[-1]),
                xytext=(-8, 14), textcoords="offset points",
                color=SERIES[0], fontsize=11, ha="right", fontweight="semibold")

    _style(ax, "Iterations", "Normalised profit gain  Δ",
           "Figure 2 — profits settle well above the competitive benchmark")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_millions))
    ax.set_xlim(0, it[-1] * 1.02)
    ax.set_ylim(min(-0.55, d.min() - 0.08), 1.12)
    fig.text(0.5, -0.04,
             "Δ = (π − π^Nash)/(π^Monopoly − π^Nash), evaluated at the greedy actions; "
             "benchmarks are the grid-restricted stage-game Nash and joint monopoly.",
             color=MUTED, fontsize=8.5, ha="center")
    return _save(fig, "fig2_profit_evolution.png")


# ---------------------------------------------------------------------------
def fig3_limit_strategy(name=X.BASELINE):
    """Paper Figure 3: the average limit strategy, with the demand schedules."""
    ls = X.limit_strategy(name)
    mk, meta = ls["market"], ls["meta"]
    p_states, w = ls["price"], ls["state_weight"]
    q_all = ls["weighted_avg_q"]                 # (nS, n)
    # Q-values in states the learned strategies never visit are still at their
    # initialisation, so the "strategy" there is an artefact. Draw the states
    # holding the central 98% of visit time solid, everything else faded.
    order = np.argsort(-w)
    keep = np.zeros_like(w, dtype=bool)
    keep[order[np.cumsum(w[order]) <= 0.98]] = True
    keep[order[0]] = True
    idx = np.where(keep)[0]
    lo, hi = idx.min(), idx.max()
    core = np.zeros_like(keep); core[lo:hi + 1] = True
    seen = w > 0

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    # firm-level "demand curves": the hub LMP the ISO produces as a function of
    # per-firm output, in each demand state (the network analogue of the paper's
    # re-scaled demand schedules).
    # Profiles are not stored in generation order once the J-lattice is gone
    # (two-firm market), so sort before drawing the schedule as a line.
    _ord = np.argsort(mk.total_gen_p)
    per_firm = mk.total_gen_p[_ord] / mk.n_agents
    styles = [(":", "low demand"), ("--", "high demand")] if mk.h == 2 else \
             [("--", f"demand level {i+1}") for i in range(mk.h)]
    dem_lines = []
    for iu in range(mk.h):
        st, lab = styles[iu]
        ax.plot(mk.price_p[_ord, iu], per_firm, color=MUTED, lw=1.4, ls=st)
        dem_lines.append((lab, mk.price_p[_ord, iu]))

    ends = []
    for i in range(mk.n_agents):
        ax.plot(p_states[seen], q_all[seen, i], color=SERIES[i], lw=1.0, alpha=0.28)
        ax.plot(p_states[core], q_all[core, i], color=SERIES[i], lw=2.4,
                solid_capstyle="round")
        ends.append(q_all[hi, i])
    x0, x1 = p_states[max(lo - 2, 0)], p_states[min(hi + 2, len(p_states) - 1)]
    y0, y1 = mk.q_agent.min() - 2, mk.q_agent.max() + 2
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for i, y in _despread(ends, min_gap=(y1 - y0) * 0.045):
        ax.annotate(f"Firm {i}", xy=(p_states[hi], ends[i]),
                    xytext=(x1 + (x1 - x0) * 0.02, y), textcoords="data",
                    color=SERIES[i], fontsize=9.5, va="center",
                    fontweight="semibold", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=SERIES[i], lw=0.7,
                                    alpha=0.5, shrinkA=0, shrinkB=2))
    # demand-schedule labels, anchored inside the visible box
    xin = x0 + (x1 - x0) * 0.92
    for lab, pcurve in dem_lines:
        j = int(np.argmin(np.abs(pcurve - xin)))
        ax.annotate(lab, xy=(pcurve[j], per_firm[j]), xytext=(4, 6),
                    textcoords="offset points", color=MUTED, fontsize=8.5)

    # where play actually spends its time (own y-scale, drawn behind the lines)
    ax2 = ax.twinx()
    ax2.bar(p_states, w, width=mk.price_bin_width * 0.85,
            color=SERIES[0], alpha=0.13, lw=0, zorder=0)
    ax2.set_ylim(0, max(w.max() * 5, 1e-6))
    ax2.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    _style(ax, "Price observed last period  (hub LMP, $/MWh)", "Output this period (MW)",
           "Figure 3 — the learned limit strategy is a price war that fades")
    fig.text(0.5, -0.09,
             "Output falls in the observed price: a price drop triggers an output expansion — "
             "the punishment. The strategy is FLATTER than the\ndemand schedules, so the price "
             "recovers and the punishment decays period after period. Shaded bars = time spent "
             "in each price\nstate; faded segments are price states the learned strategies "
             "essentially never visit.",
             color=MUTED, fontsize=8.5, ha="center")
    return _save(fig, "fig3_limit_strategy.png")


# ---------------------------------------------------------------------------
def fig4_deviation(name=X.BASELINE, deviator=0, mode="best_response"):
    """Paper Figure 4: response to an exogenous deviation, high vs low demand."""
    meta, _ = X.load(name)
    det = meta["market"]["deterministic"]
    panels = ["high", "low"] if not det else ["high"]
    runs = [X.deviation_experiment(name, deviator=deviator, demand=d, mode=mode,
                                   pre=12, post=60)
            for d in panels]

    nc = len(runs)
    fig, axes = plt.subplots(2, nc, figsize=(6.0 * nc, 8.2), sharex=True,
                             squeeze=False)
    for c, r in enumerate(runs):
        q, qb, dp = r["q"], r["q_nodev"], r["dev_period"]
        n = r["market"].n_firms
        others = [i for i in range(n) if i != deviator]
        x = np.arange(len(q)) - dp

        # --- outputs ------------------------------------------------------
        ax = axes[0, c]
        ax.axvline(0, color=MUTED, lw=1.0, ls=":")
        ax.plot(x, qb[:, deviator], color=MUTED, lw=1.1, alpha=0.8)
        ax.plot(x, qb[:, others].mean(axis=1), color=MUTED, lw=1.1, alpha=0.8)
        ax.plot(x, q[:, deviator], color=SERIES[0], lw=2.3, solid_capstyle="round")
        ax.plot(x, q[:, others].mean(axis=1), color=SERIES[1], lw=2.1, ls="--",
                solid_capstyle="round")
        _label(ax, x[-1], q[-1, deviator], "deviating\nagent", SERIES[0])
        _label(ax, x[-1], q[:, others].mean(axis=1)[-1], "non-deviating\nagents",
               SERIES[1])
        ax.annotate("no-deviation counterfactual", xy=(x[3], qb[3, deviator]),
                    xytext=(0, -26), textcoords="offset points", color=MUTED,
                    fontsize=8.5, ha="left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        _style(ax, "", "Output (MW)", "")
        ax.set_title(f"{r['demand'].capitalize()} demand", color=INK,
                     fontsize=11, loc="left", pad=8, fontweight="semibold")

        # --- hub price ----------------------------------------------------
        ax = axes[1, c]
        ax.axvline(0, color=MUTED, lw=1.0, ls=":")
        ax.plot(x, r["price_nodev"], color=MUTED, lw=1.1, alpha=0.8)
        ax.plot(x, r["price"], color=SERIES[0], lw=2.3, solid_capstyle="round")
        _label(ax, x[-1], r["price"][-1], "hub LMP", SERIES[0])
        _style(ax, "Period  (0 = the forced deviation)", "Hub LMP ($/MWh)", "")

    fig.suptitle("Figure 4 — a deviation is punished, then the punishment fades away",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.0,
                 fontweight="semibold")
    fig.text(0.005, -0.012,
             "One algorithm is forced to expand output for a single period (its static best "
             "response), then reverts to its learned strategy. Demand is frozen, so only the\n"
             f"deviation moves the price. Grey = the same {runs[0]['n_sessions']} sessions from "
             f"the identical state with no deviation; every path is averaged over all "
             f"{runs[0]['n_phases']} phases of the limit cycle.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save(fig, "fig4_deviation.png")


def _despread(ys, min_gap):
    """Yield (index, y) with labels pushed apart so they do not collide."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out, prev = [], -np.inf
    for i in order:
        y = max(ys[i], prev + min_gap)
        out.append((i, y))
        prev = y
    return out


def _label(ax, x, y, text, color, ha="left", dx=6):
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                color=color, fontsize=9.5, va="center", ha=ha,
                fontweight="semibold")


# ---------------------------------------------------------------------------
def fig_delta_distribution(name=X.BASELINE):
    """Extra: how the profit gain is distributed across independent sessions."""
    meta, z = X.load(name)
    d = z["delta_per_session"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.hist(d, bins=40, color=SERIES[0], alpha=0.85, edgecolor="white", lw=0.6)
    ax.axvline(float(np.mean(d)), color=INK, lw=1.6)
    ax.annotate(f"mean Δ = {100*np.mean(d):.1f}%", xy=(np.mean(d), ax.get_ylim()[1]),
                xytext=(8, -12), textcoords="offset points", color=INK,
                fontsize=10, fontweight="semibold")
    ax.axvline(0, color=RULE_N, lw=1.2, ls="--")
    ax.axvline(1, color=RULE_M, lw=1.2, ls="--")
    _style(ax, "Δ in a single session", "Sessions",
           "Profit gain across independent sessions")
    return _save(fig, "fig_delta_distribution.png")


# ---------------------------------------------------------------------------
def make_all(name=X.BASELINE):
    print(f"figures for {name!r}:")
    out = [fig1_outputs(name), fig2_profits(name)]
    if X.load(name)[0]["config"]["monitoring"] == "imperfect":
        out += [fig3_limit_strategy(name), fig4_deviation(name)]
    out.append(fig_delta_distribution(name))
    return out


# ---------------------------------------------------------------------------
def fig5_deviation_value(name=X.BASELINE, deviator=0, demand="low", horizon=60):
    """Is cheating actually deterred? Short-run gain vs discounted total."""
    r = X.deviation_value_test(name, deviator=deviator, demand=demand,
                               horizon=max(horizon, 200))
    d = r["delta"]
    pd_, pb = r["profit_path_dev"][:horizon], r["profit_path_stay"][:horizon]
    t = np.arange(horizon)
    cum = np.cumsum((d ** t) * (pd_ - pb))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    ax = axes[0]
    ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    ax.plot(t, pb, color=MUTED, lw=1.6)
    ax.plot(t, pd_, color=SERIES[0], lw=2.3, solid_capstyle="round")
    ax.fill_between(t, pb, pd_, where=pd_ >= pb, color=SERIES[0], alpha=0.18, lw=0)
    ax.fill_between(t, pb, pd_, where=pd_ < pb, color=SERIES[2], alpha=0.30, lw=0)
    _label(ax, t[-1], pd_[-1], "deviate", SERIES[0])
    _label(ax, t[-1], pb[-1] - (pd_[-1] - pb[-1]) * 0.6, "do not deviate", MUTED)
    _style(ax, "Period  (0 = the deviation)", "Deviator's profit ($/period)", "")
    ax.set_title("The cheat pays once, then costs", color=INK, fontsize=11,
                 loc="left", pad=8, fontweight="semibold")

    ax = axes[1]
    ax.axhline(0, color=INK, lw=1.0)
    ax.plot(t, cum, color=SERIES[0], lw=2.3, solid_capstyle="round")
    ax.fill_between(t, 0, cum, where=cum >= 0, color=SERIES[0], alpha=0.18, lw=0)
    ax.fill_between(t, 0, cum, where=cum < 0, color=SERIES[2], alpha=0.30, lw=0)
    below = np.where(cum < 0)[0]
    if len(below):
        b = int(below[0])
        ax.axvline(b, color=MUTED, lw=1.0, ls=":")
        ax.annotate(f"cheating is already under water\nafter {b} period{'s' if b != 1 else ''}",
                    xy=(b, 0), xytext=(14, -34), textcoords="offset points",
                    color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.scatter([t[-1]], [cum[-1]], s=34, color=SERIES[0], zorder=5,
               edgecolor="white", lw=1.4)
    ax.annotate(rf"\${cum[-1]:+,.0f}", xy=(t[-1], cum[-1]), xytext=(-8, 12),
                textcoords="offset points", color=SERIES[0], fontsize=10.5,
                ha="right", fontweight="semibold")
    _style(ax, "Periods included",
           "Cumulative discounted gain from deviating (\\$)", "")
    ax.set_title(f"Deterred in {100*r['deterred_frac']:.0f}% of sessions "
                 f"(median {np.median(r['discounted_per_session']):+,.0f})",
                 color=INK, fontsize=11, loc="left", pad=8, fontweight="semibold")

    fig.suptitle("Figure 5 — the supra-competitive profits are sustained by punishment, "
                 "not by failure to optimise",
                 color=INK, fontsize=13, x=0.005, ha="left", y=1.05,
                 fontweight="semibold")
    fig.text(0.005, -0.08,
             "One firm is forced into its static best response for a single period, "
             f"then reverts to its learned strategy ({demand} demand, frozen). "
             rf"The one-period gain averages \${r['one_period_gain']:,.0f}, but the "
             rf"discounted stream ($\delta$ = {d}) comes to \${r['discounted_gain']:,.0f}."
             f"\nAveraged over {r['n_sessions']} converged sessions and all cycle phases.",
             color=MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "fig5_deviation_value.png")
