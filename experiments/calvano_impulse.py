"""Shared Calvano impulse-response simulation and plotting utilities."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from experiments.impulse_response import _static_best_response_mw

DEVIATOR_FID = 1
NONDEVIATOR_FID = 0
PRE_PERIODS = 2


def get_avg_lmp(info):
    if "avg_lmp" in info:
        return info["avg_lmp"]
    if "lmp" in info:
        return info["lmp"]
    if "system_lmp" in info:
        return info["system_lmp"]
    return 0.0


def set_zoomed_ylim(ax, lower, upper, padding_frac=0.05):
    lo = float(np.min(lower))
    hi = float(np.max(upper))
    span = hi - lo
    if span < 1e-6:
        span = max(abs(hi), 1.0) * 0.02
    pad = span * padding_frac
    ax.set_ylim(lo - pad, hi + pad)


def apply_period_axis(ax, x_min, x_max):
    ticks = np.arange(x_min, x_max + 1)
    ax.set_xlim(x_min - 0.4, x_max + 0.4)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.35, linewidth=0.8)
    ax.minorticks_off()


def measure_resting_lmp(env, agents, normalizers, warmup=20, seed=42) -> float:
    """Collusive resting system LMP after greedy warmup (for ranking sessions)."""
    np.random.seed(seed)
    obs = env.reset()
    for _ in range(warmup):
        actions = {
            fid: agent.deterministic_action(normalizers[fid].normalize(obs[fid]))
            for fid, agent in agents.items()
        }
        obs, _, done, info = env.step(actions)
        if done:
            obs = env.reset()
    return float(get_avg_lmp(info))


def run_calvano_impulse(
    env,
    agents,
    normalizers,
    warmup=20,
    horizon=15,
    gamma=0.99,
    seed=42,
):
    """
    Calvano protocol: Firm 0 greedy at t=0; Firm 1 plays per-plant static best
    response once at t=0; both greedy from t>=1.
    """
    np.random.seed(seed)
    obs = env.reset()

    for _ in range(warmup):
        actions = {}
        for fid, agent in agents.items():
            obs_norm = normalizers[fid].normalize(obs[fid])
            actions[fid] = agent.deterministic_action(obs_norm)
        obs, rewards, done, info = env.step(actions)
        if done:
            obs = env.reset()

    base_gen_0 = float(np.sum(actions[0]))
    base_gen_1 = float(np.sum(actions[1]))
    base_lmp = get_avg_lmp(info)
    base_profit_0 = float(rewards[NONDEVIATOR_FID])
    base_profit_1 = float(rewards[DEVIATOR_FID])
    baseline_npv = sum(base_profit_1 * (gamma ** t) for t in range(horizon + 1))

    trace_gen_0 = [base_gen_0] * PRE_PERIODS
    trace_gen_1 = [base_gen_1] * PRE_PERIODS
    trace_lmp = [base_lmp] * PRE_PERIODS
    trace_profit_0 = [base_profit_0] * PRE_PERIODS
    trace_profit_1 = [base_profit_1] * PRE_PERIODS

    actions = {}
    for fid, agent in agents.items():
        obs_norm = normalizers[fid].normalize(obs[fid])
        actions[fid] = agent.deterministic_action(obs_norm).astype(np.float64)

    policy_mw_1 = float(np.sum(actions[DEVIATOR_FID]))
    br_mw = _static_best_response_mw(actions[NONDEVIATOR_FID], env)
    actions[DEVIATOR_FID] = br_mw
    opt_mult = float(np.sum(br_mw) / policy_mw_1) if policy_mw_1 > 0 else 1.0

    obs, rewards, done, info = env.step(actions)
    trace_gen_0.append(float(np.sum(actions[0])))
    trace_gen_1.append(float(np.sum(actions[1])))
    trace_lmp.append(get_avg_lmp(info))
    trace_profit_0.append(float(rewards[NONDEVIATOR_FID]))
    trace_profit_1.append(float(rewards[DEVIATOR_FID]))
    deviation_npv = rewards[DEVIATOR_FID]

    for t in range(1, horizon + 1):
        if done:
            obs = env.reset()
        actions = {}
        for fid, agent in agents.items():
            obs_norm = normalizers[fid].normalize(obs[fid])
            actions[fid] = agent.deterministic_action(obs_norm)
        obs, rewards, done, info = env.step(actions)
        trace_gen_0.append(float(np.sum(actions[0])))
        trace_gen_1.append(float(np.sum(actions[1])))
        trace_lmp.append(get_avg_lmp(info))
        trace_profit_0.append(float(rewards[NONDEVIATOR_FID]))
        trace_profit_1.append(float(rewards[DEVIATOR_FID]))
        deviation_npv += rewards[DEVIATOR_FID] * (gamma ** t)

    return {
        "gen_0": np.array(trace_gen_0),
        "gen_1": np.array(trace_gen_1),
        "lmp": np.array(trace_lmp),
        "profit_0": np.array(trace_profit_0),
        "profit_1": np.array(trace_profit_1),
        "is_unprofitable": deviation_npv < baseline_npv,
        "opt_mult": opt_mult,
        "base_lmp": base_lmp,
        "t0_profit_gain": float(trace_profit_1[PRE_PERIODS]) - base_profit_1,
    }


def aggregate_series(all_series):
    arr = np.array(all_series)
    return (
        np.mean(arr, axis=0),
        np.percentile(arr, 25, axis=0),
        np.percentile(arr, 75, axis=0),
    )


def plot_single_series_panel(
    ax,
    t_axis,
    series,
    baseline,
    title,
    color,
    ylabel,
    baseline_fmt,
    x_min,
    x_max,
    vline_label=False,
):
    ax.axhline(
        baseline,
        color="gray",
        linestyle=":",
        lw=1.5,
        alpha=0.75,
        label=f"Pre-deviation baseline ({baseline_fmt.format(baseline)})",
    )
    ax.plot(t_axis, series, color=color, lw=3, marker="o", markersize=5, label="Trace")
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        alpha=0.65,
        lw=1.5,
        label="Static BR (t=0)" if vline_label else None,
    )
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("Period (t)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    apply_period_axis(ax, x_min, x_max)
    set_zoomed_ylim(ax, series, series, padding_frac=0.05)
    ax.legend(loc="best", fontsize=9)


def plot_trace_panel(
    ax, t_axis, mu, p25, p75, baseline, title, color, ylabel, baseline_fmt, x_min, x_max,
    vline_label=False,
):
    ax.axhline(
        baseline,
        color="gray",
        linestyle=":",
        lw=1.5,
        alpha=0.75,
        label=f"Pre-deviation baseline ({baseline_fmt.format(baseline)})",
    )
    ax.plot(t_axis, mu, color=color, lw=3, marker="o", markersize=5, label="Mean")
    ax.fill_between(t_axis, p25, p75, color=color, alpha=0.22, label="25th–75th pct.")
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        alpha=0.65,
        lw=1.5,
        label="Static BR (t=0)" if vline_label else None,
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Period (t)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    apply_period_axis(ax, x_min, x_max)
    set_zoomed_ylim(ax, p25, p75, padding_frac=0.05)
    ax.legend(loc="best", fontsize=10)


def plot_lmp_panel(ax, t_axis, mu, p25, p75, baseline_lmp, x_min, x_max, subtitle=""):
    ax.axhline(
        baseline_lmp,
        color="green",
        linestyle=":",
        lw=2,
        label=f"Collusive baseline (${baseline_lmp:.2f})",
    )
    if p25 is None:
        ax.plot(t_axis, mu, color="#2ca02c", lw=3, marker="o", markersize=5, label="System average LMP")
        set_zoomed_ylim(ax, mu, mu, padding_frac=0.05)
    else:
        ax.plot(t_axis, mu, color="#2ca02c", lw=3, marker="o", markersize=5, label="System average LMP")
        ax.fill_between(t_axis, p25, p75, color="#2ca02c", alpha=0.18)
        set_zoomed_ylim(ax, p25, p75, padding_frac=0.05)
    ax.axvline(0, color="black", linestyle="--", alpha=0.65, lw=1.5)
    title = "Market Impact (System Average LMP)"
    if subtitle:
        title = f"{title} — {subtitle}"
    ax.set_title(title, fontsize=13 if subtitle else 14, pad=8)
    ax.set_xlabel("Period (t)", fontsize=11)
    ax.set_ylabel("System Average LMP ($/MWh)", fontsize=11)
    apply_period_axis(ax, x_min, x_max)
    ax.legend(loc="lower right", fontsize=9)


def plot_session_impulse_figure(
    res,
    session_name,
    rank,
    out_path,
    horizon,
    dpi=200,
):
    """Five-panel figure for one session (no cross-session averaging)."""
    x_min = -PRE_PERIODS
    x_max = horizon
    t_axis = np.arange(x_min, x_max + 1)
    baseline_idx = PRE_PERIODS - 1

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(5, 1, figsize=(12, 22))

    mult_pct = 100.0 * (res["opt_mult"] - 1.0)
    npv_tag = "unprofitable" if res["is_unprofitable"] else "profitable"
    fig.suptitle(
        f"Rank {rank}: {session_name} — Firm 1 static BR at t=0 "
        f"({res['opt_mult']:.3f}x, {mult_pct:+.1f}% MW; NPV {npv_tag})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plot_single_series_panel(
        axes[0], t_axis, res["gen_1"], float(res["gen_1"][baseline_idx]),
        "Deviating Firm (Firm 1) — Generation", "#d62728",
        "Generation (MW)", "{:.0f} MW", x_min, x_max, vline_label=True,
    )
    plot_single_series_panel(
        axes[1], t_axis, res["profit_1"], float(res["profit_1"][baseline_idx]),
        "Deviating Firm (Firm 1) — Profit", "#d62728",
        "Profit ($/step)", "${:.0f}", x_min, x_max,
    )
    plot_single_series_panel(
        axes[2], t_axis, res["gen_0"], float(res["gen_0"][baseline_idx]),
        "Non-Deviating Firm (Firm 0) — Generation", "#1f77b4",
        "Generation (MW)", "{:.0f} MW", x_min, x_max,
    )
    plot_single_series_panel(
        axes[3], t_axis, res["profit_0"], float(res["profit_0"][baseline_idx]),
        "Non-Deviating Firm (Firm 0) — Profit", "#1f77b4",
        "Profit ($/step)", "${:.0f}", x_min, x_max,
    )
    plot_lmp_panel(
        axes[4], t_axis, res["lmp"], None, None,
        float(res["lmp"][baseline_idx]), x_min, x_max,
        subtitle=f"baseline ${res['base_lmp']:.2f}",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
