"""
LMP-lowering one-shot deviation for high-Δ collusive sessions.

Protocol (same timing as Calvano impulse):
  t = -2, -1 : both on frozen greedy policy (collusive resting point)
  t = 0      : Firm 0 on greedy; Firm 1 deviates once to lower system avg LMP
  t >= 1     : both on greedy — observe recovery / retaliation dynamics

Firm 1 deviation at t=0: per-plant grid on [0, cap] (and policy scalings), chosen to
maximize (resting_LMP - avg_LMP) subject to avg_LMP < resting_LMP. At collusion,
this is typically **overproduction toward capacity**, not scaling greedy output down.

Default cohort: sessions with final_delta_combined > 0.3 (H=1 latest_results).

Usage:
    python experiments/plot_lmp_lowering_deviation.py --run-dir latest_results
    python experiments/plot_lmp_lowering_deviation.py --delta-threshold 0.3 --horizon 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv, FIRM_PLANT_IDX, NUM_NODES, PLANTS
from experiments.calvano_impulse import (
    PRE_PERIODS,
    aggregate_series,
    apply_period_axis,
    get_avg_lmp,
)
from experiments.paths import DEFAULT_RUN_DIR_NAME, deviation_figures_dir, resolve_run_dir
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers

DEVIATOR_FID = 1
NONDEVIATOR_FID = 0


def select_high_delta_sessions(sessions_root: Path, threshold: float) -> list[Path]:
    selected = []
    for s_dir in sorted(sessions_root.iterdir()):
        if not s_dir.is_dir():
            continue
        meta = s_dir / "session.json"
        if not meta.exists() or not (s_dir / "agent_0.pt").exists():
            continue
        data = json.loads(meta.read_text())
        delta = data.get("final_delta_combined")
        if delta is not None and float(delta) > threshold:
            selected.append(s_dir)
    return selected


def _clear_avg_lmp(env: ElectricityMarketEnv, actions: dict) -> float | None:
    gen_per_node = np.zeros(NUM_NODES)
    for fid, acts in actions.items():
        for j, pidx in enumerate(FIRM_PLANT_IDX[fid]):
            cap = PLANTS[pidx]["cap"]
            gen_per_node[PLANTS[pidx]["node"]] += float(np.clip(acts[j], 0.0, cap))
    lmps, demand, _, _ = env._clear_market(gen_per_node)
    if lmps is None:
        return None
    if float(np.sum(demand)) > 0:
        return float(np.sum(lmps * demand) / np.sum(demand))
    return float(np.mean(lmps))


def find_lmp_lowering_deviation(
    f0_mw: np.ndarray,
    f1_policy_mw: np.ndarray,
    env: ElectricityMarketEnv,
    resting_lmp: float,
    *,
    n_grid: int = 21,
) -> tuple[np.ndarray, float, float]:
    """
    Pick Firm 1 MW at t=0 to maximize LMP drop vs resting, requiring avg LMP to fall.

    At collusive resting points, scaling greedy output *down* often *raises* LMP
    (rival keeps high output). We therefore search per-plant fractions of capacity
    (and upward scalings of policy MW clipped to cap).
    Returns (deviation_mw, scale_vs_policy, t0_avg_lmp).
    """
    caps = np.array([PLANTS[pidx]["cap"] for pidx in FIRM_PLANT_IDX[DEVIATOR_FID]], dtype=float)
    policy_total = float(np.sum(f1_policy_mw))
    grid = np.linspace(0.0, 1.0, n_grid)

    best_mw = f1_policy_mw.copy()
    best_lmp = resting_lmp
    best_drop = 0.0

    def consider(trial: np.ndarray):
        nonlocal best_mw, best_lmp, best_drop
        trial = np.minimum(np.maximum(trial, 0.0), caps)
        actions = {NONDEVIATOR_FID: f0_mw, DEVIATOR_FID: trial}
        avg_lmp = _clear_avg_lmp(env, actions)
        if avg_lmp is None or avg_lmp >= resting_lmp - 1e-6:
            return
        drop = resting_lmp - avg_lmp
        if drop > best_drop:
            best_drop = drop
            best_mw = trial.copy()
            best_lmp = avg_lmp

    # Per-plant fractions of capacity (covers under- and over-production vs policy)
    if len(caps) == 1:
        for frac in grid:
            consider(np.array([frac * caps[0]]))
    else:
        from itertools import product

        for fracs in product(grid, repeat=len(caps)):
            consider(np.array([f * c for f, c in zip(fracs, caps)]))

    # Uniform scale on policy MW (both directions, clipped to caps)
    if policy_total > 1e-6:
        for mult in np.linspace(0.05, 2.0, 40):
            consider(np.minimum(f1_policy_mw * mult, caps))

    if best_drop <= 0:
        raise RuntimeError(
            f"No deviation lowered avg LMP (resting={resting_lmp:.2f}); "
            "try a longer warmup or different session."
        )

    scale = float(np.sum(best_mw) / policy_total) if policy_total > 0 else 1.0
    return best_mw, scale, best_lmp


def run_lmp_lowering_impulse(
    env: ElectricityMarketEnv,
    agents: dict,
    normalizers: dict,
    *,
    warmup: int = 20,
    horizon: int = 15,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)
    obs = env.reset()

    for _ in range(warmup):
        actions = {
            fid: agents[fid].deterministic_action(normalizers[fid].normalize(obs[fid]))
            for fid in agents
        }
        obs, rewards, done, info = env.step(actions)
        if done:
            obs = env.reset()

    trace_gen_0, trace_gen_1 = [], []
    trace_lmp, trace_profit_0, trace_profit_1 = [], [], []

    def record_step(actions, rewards, info):
        trace_gen_0.append(float(np.sum(actions[NONDEVIATOR_FID])))
        trace_gen_1.append(float(np.sum(actions[DEVIATOR_FID])))
        trace_lmp.append(float(get_avg_lmp(info)))
        trace_profit_0.append(float(rewards[NONDEVIATOR_FID]))
        trace_profit_1.append(float(rewards[DEVIATOR_FID]))

    # t = -2, -1
    for _ in range(PRE_PERIODS):
        actions = {
            fid: agents[fid].deterministic_action(normalizers[fid].normalize(obs[fid]))
            for fid in agents
        }
        obs, rewards, done, info = env.step(actions)
        if done:
            obs = env.reset()
        record_step(actions, rewards, info)

    resting_lmp = trace_lmp[-1]
    resting_gen_0 = trace_gen_0[-1]
    resting_gen_1 = trace_gen_1[-1]

    actions_policy = {
        fid: agents[fid].deterministic_action(normalizers[fid].normalize(obs[fid])).astype(np.float64)
        for fid in agents
    }
    f0_mw = actions_policy[NONDEVIATOR_FID]
    f1_policy = actions_policy[DEVIATOR_FID]

    dev_mw, dev_mult, predicted_lmp = find_lmp_lowering_deviation(
        f0_mw, f1_policy, env, resting_lmp
    )
    actions_exec = {
        NONDEVIATOR_FID: f0_mw,
        DEVIATOR_FID: dev_mw,
    }

    obs, rewards, done, info = env.step(actions_exec)
    record_step(actions_exec, rewards, info)
    t0_lmp = trace_lmp[-1]

    for _ in range(1, horizon + 1):
        if done:
            obs = env.reset()
        actions = {
            fid: agents[fid].deterministic_action(normalizers[fid].normalize(obs[fid]))
            for fid in agents
        }
        obs, rewards, done, info = env.step(actions)
        record_step(actions, rewards, info)

    return {
        "gen_0": np.array(trace_gen_0, dtype=float),
        "gen_1": np.array(trace_gen_1, dtype=float),
        "lmp": np.array(trace_lmp, dtype=float),
        "profit_0": np.array(trace_profit_0, dtype=float),
        "profit_1": np.array(trace_profit_1, dtype=float),
        "resting_lmp": resting_lmp,
        "t0_lmp": t0_lmp,
        "lmp_drop": resting_lmp - t0_lmp,
        "dev_mult": dev_mult,
        "resting_gen_0": resting_gen_0,
        "resting_gen_1": resting_gen_1,
        "predicted_t0_lmp": predicted_lmp,
    }


def set_yaxis_zoom_gap(ax, lower, upper, gap_frac: float = 0.25):
    lo = float(np.min(lower))
    hi = float(np.max(upper))
    span = hi - lo
    if span < 1e-9:
        span = max(abs(hi), 1.0) * 0.05
    pad = span * gap_frac
    ax.set_ylim(lo - pad, hi + pad)


def plot_aggregate_trace_panel(
    ax,
    t_axis,
    mu,
    p25,
    p75,
    baseline,
    *,
    title,
    color,
    ylabel,
    baseline_fmt,
    x_min,
    x_max,
    y_gap: float = 0.25,
    vline_label: str | None = "LMP-lowering deviation (t=0)",
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
        label=vline_label,
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Period (t)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    apply_period_axis(ax, x_min, x_max)
    set_yaxis_zoom_gap(ax, p25, p75, gap_frac=y_gap)
    ax.legend(loc="best", fontsize=10)


def plot_aggregate_lmp_panel(ax, t_axis, mu, p25, p75, baseline_lmp, x_min, x_max):
    ax.axhline(
        baseline_lmp,
        color="green",
        linestyle=":",
        lw=2,
        label=f"Collusive baseline (${baseline_lmp:.2f})",
    )
    ax.plot(t_axis, mu, color="#2ca02c", lw=3, marker="o", markersize=5, label="Mean system LMP")
    ax.fill_between(t_axis, p25, p75, color="#2ca02c", alpha=0.18, label="25th–75th pct.")
    ax.axvline(0, color="black", linestyle="--", alpha=0.65, lw=1.5, label="Deviation (t=0)")
    ax.set_title("Market Impact — System Average LMP", fontsize=14, pad=10)
    ax.set_xlabel("Period (t)", fontsize=12)
    ax.set_ylabel("System Average LMP ($/MWh)", fontsize=12)
    apply_period_axis(ax, x_min, x_max)
    set_yaxis_zoom_gap(ax, p25, p75, gap_frac=0.08)
    ax.legend(loc="lower right", fontsize=10)


def plot_aggregate_profit_panel(
    ax, t_axis, mu0, p25_0, p75_0, mu1, p25_1, p75_1, b0, b1, x_min, x_max, y_gap=0.25
):
    ax.axhline(b0, color="#1f77b4", linestyle=":", lw=1.5, alpha=0.7, label=f"Firm 0 baseline (${b0:.0f})")
    ax.axhline(b1, color="#d62728", linestyle=":", lw=1.5, alpha=0.7, label=f"Firm 1 baseline (${b1:.0f})")
    ax.plot(t_axis, mu0, color="#1f77b4", lw=2.5, marker="o", ms=4, label="Firm 0 mean")
    ax.fill_between(t_axis, p25_0, p75_0, color="#1f77b4", alpha=0.15)
    ax.plot(t_axis, mu1, color="#d62728", lw=2.5, marker="o", ms=4, label="Firm 1 mean")
    ax.fill_between(t_axis, p25_1, p75_1, color="#d62728", alpha=0.15)
    ax.axvline(0, color="black", linestyle="--", alpha=0.65, lw=1.5)
    ax.set_title("Profit Response — Both Firms", fontsize=14, pad=10)
    ax.set_xlabel("Period (t)", fontsize=12)
    ax.set_ylabel("Profit ($/step)", fontsize=12)
    apply_period_axis(ax, x_min, x_max)
    lo = min(np.min(p25_0), np.min(p25_1))
    hi = max(np.max(p75_0), np.max(p75_1))
    set_yaxis_zoom_gap(ax, lo, hi, gap_frac=y_gap)
    ax.legend(loc="best", fontsize=9)


def save_figure(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="LMP-lowering deviation impulse for high-Δ sessions."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--delta-threshold", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="lmp_lowering_delta030",
        help="Subfolder under deviation_experiment/",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    out_dir = deviation_figures_dir(run_dir) / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))

    session_dirs = select_high_delta_sessions(run_dir / "sessions", args.delta_threshold)
    if not session_dirs:
        raise SystemExit(f"No sessions with final_delta_combined > {args.delta_threshold}")

    print(
        f"LMP-lowering deviation | Δ > {args.delta_threshold} | "
        f"{len(session_dirs)} sessions | horizon={args.horizon}"
    )

    all_g0, all_g1, all_lmp, all_p0, all_p1 = [], [], [], [], []
    per_session = []

    for idx, s_dir in enumerate(session_dirs):
        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)
        seed = int(json.loads((s_dir / "session.json").read_text()).get("seed", idx))
        res = run_lmp_lowering_impulse(
            env, agents, normalizers, warmup=args.warmup, horizon=args.horizon, seed=seed
        )
        all_g0.append(res["gen_0"])
        all_g1.append(res["gen_1"])
        all_lmp.append(res["lmp"])
        all_p0.append(res["profit_0"])
        all_p1.append(res["profit_1"])
        per_session.append(
            {
                "session": s_dir.name,
                "final_delta_combined": json.loads((s_dir / "session.json").read_text())[
                    "final_delta_combined"
                ],
                "dev_mult": res["dev_mult"],
                "resting_lmp": res["resting_lmp"],
                "t0_lmp": res["t0_lmp"],
                "lmp_drop": res["lmp_drop"],
            }
        )
        sys.stdout.write(
            f"\r  {idx + 1}/{len(session_dirs)} {s_dir.name} "
            f"mult={res['dev_mult']:.3f} LMP {res['resting_lmp']:.2f}→{res['t0_lmp']:.2f}"
        )
        sys.stdout.flush()
    print()

    x_min = -PRE_PERIODS
    x_max = args.horizon
    t_axis = np.arange(x_min, x_max + 1)
    baseline_idx = PRE_PERIODS - 1

    mu_g0, p25_g0, p75_g0 = aggregate_series(all_g0)
    mu_g1, p25_g1, p75_g1 = aggregate_series(all_g1)
    mu_lmp, p25_lmp, p75_lmp = aggregate_series(all_lmp)
    mu_p0, p25_p0, p75_p0 = aggregate_series(all_p0)
    mu_p1, p25_p1, p75_p1 = aggregate_series(all_p1)

    b_g0 = float(np.mean([g[baseline_idx] for g in all_g0]))
    b_g1 = float(np.mean([g[baseline_idx] for g in all_g1]))
    b_lmp = float(np.mean([l[baseline_idx] for l in all_lmp]))
    b_p0 = float(np.mean([p[baseline_idx] for p in all_p0]))
    b_p1 = float(np.mean([p[baseline_idx] for p in all_p1]))

    plt.style.use("seaborn-v0_8-whitegrid")
    y_gap = 0.25

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    plot_aggregate_trace_panel(
        ax1, t_axis, mu_g1, p25_g1, p75_g1, b_g1,
        title=f"Firm 1 (Deviator) — Generation [{len(session_dirs)} sessions, Δ>{args.delta_threshold}]",
        color="#d62728", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
    )
    save_figure(fig1, out_dir / "firm1_deviator_generation.png", args.dpi)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    plot_aggregate_trace_panel(
        ax2, t_axis, mu_g0, p25_g0, p75_g0, b_g0,
        title=f"Firm 0 (Non-Deviator) — Generation & Retaliation [{len(session_dirs)} sessions]",
        color="#1f77b4", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
        vline_label="Firm 1 deviation (t=0); Firm 0 on policy",
    )
    save_figure(fig2, out_dir / "firm0_retaliation_generation.png", args.dpi)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    plot_aggregate_lmp_panel(ax3, t_axis, mu_lmp, p25_lmp, p75_lmp, b_lmp, x_min, x_max)
    save_figure(fig3, out_dir / "system_avg_lmp.png", args.dpi)

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    plot_aggregate_profit_panel(
        ax4, t_axis, mu_p0, p25_p0, p75_p0, mu_p1, p25_p1, p75_p1,
        b_p0, b_p1, x_min, x_max, y_gap=y_gap,
    )
    save_figure(fig4, out_dir / "firm_profits.png", args.dpi)

    # Combined 4-panel publication figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 18))
    plot_aggregate_trace_panel(
        axes[0], t_axis, mu_g1, p25_g1, p75_g1, b_g1,
        title="Firm 1 — Generation (LMP-lowering deviation at t=0)",
        color="#d62728", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
    )
    plot_aggregate_trace_panel(
        axes[1], t_axis, mu_g0, p25_g0, p75_g0, b_g0,
        title="Firm 0 — Generation (policy at t=0; market response after)",
        color="#1f77b4", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
        vline_label="Firm 1 deviation (t=0)",
    )
    plot_aggregate_lmp_panel(axes[2], t_axis, mu_lmp, p25_lmp, p75_lmp, b_lmp, x_min, x_max)
    plot_aggregate_profit_panel(
        axes[3], t_axis, mu_p0, p25_p0, p75_p0, mu_p1, p25_p1, p75_p1,
        b_p0, b_p1, x_min, x_max, y_gap=y_gap,
    )
    fig.suptitle(
        f"LMP-Lowering Deviation (Firm 1, t=0) — {len(session_dirs)} sessions with Δ>{args.delta_threshold}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    save_figure(fig, out_dir / "lmp_lowering_combined.png", args.dpi)

    summary = {
        "delta_threshold": args.delta_threshold,
        "n_sessions": len(session_dirs),
        "horizon": args.horizon,
        "mean_dev_mult": float(np.mean([r["dev_mult"] for r in per_session])),
        "mean_lmp_drop_at_t0": float(np.mean([r["lmp_drop"] for r in per_session])),
        "sessions": per_session,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Mean LMP drop at t=0: ${summary['mean_lmp_drop_at_t0']:.2f}")
    print(f"Mean deviation multiplier: {summary['mean_dev_mult']:.3f}")
    print(f"Figures → {out_dir}/")


if __name__ == "__main__":
    main()
