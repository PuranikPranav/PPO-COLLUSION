"""
Static best-response deviation for high-Δ collusive sessions (Δ > 0.3).

Protocol:
  t = -2, -1 : both on frozen greedy policy (collusive resting point)
  t = 0      : Firm 0 on greedy; Firm 1 plays per-plant static best response
               (one-period profit max given Firm 0's greedy MW)
  t >= 1     : both on greedy — recovery / retaliation dynamics

Default cohort: 21 sessions with final_delta_combined > 0.3 in latest_results.

Usage:
    python experiments/plot_static_br_high_delta.py --run-dir latest_results
    python experiments/plot_static_br_high_delta.py --delta-threshold 0.3 --horizon 15
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

from iso_market.market_env import ElectricityMarketEnv
from experiments.calvano_impulse import PRE_PERIODS, aggregate_series, run_calvano_impulse
from experiments.paths import DEFAULT_RUN_DIR_NAME, deviation_figures_dir, resolve_run_dir
from experiments.plot_lmp_lowering_deviation import (
    plot_aggregate_lmp_panel,
    plot_aggregate_profit_panel,
    plot_aggregate_trace_panel,
    save_figure,
    select_high_delta_sessions,
)
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers


def main():
    parser = argparse.ArgumentParser(
        description="Static-BR impulse for sessions with Δ above threshold."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--delta-threshold", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="static_br_delta030",
        help="Subfolder under deviation_experiment/",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    out_dir = deviation_figures_dir(run_dir) / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))
    gamma = float(config.get("gamma", 0.99))

    session_dirs = select_high_delta_sessions(run_dir / "sessions", args.delta_threshold)
    if not session_dirs:
        raise SystemExit(f"No sessions with final_delta_combined > {args.delta_threshold}")

    print(
        f"Static BR deviation | Δ > {args.delta_threshold} | "
        f"{len(session_dirs)} sessions | horizon={args.horizon}"
    )

    all_g0, all_g1, all_lmp, all_p0, all_p1 = [], [], [], [], []
    per_session = []
    profitable_t0 = 0

    for idx, s_dir in enumerate(session_dirs):
        meta = json.loads((s_dir / "session.json").read_text())
        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)
        seed = int(meta.get("seed", idx))
        res = run_calvano_impulse(
            env, agents, normalizers,
            warmup=args.warmup, horizon=args.horizon, gamma=gamma, seed=seed,
        )
        all_g0.append(res["gen_0"])
        all_g1.append(res["gen_1"])
        all_lmp.append(res["lmp"])
        all_p0.append(res["profit_0"])
        all_p1.append(res["profit_1"])

        baseline_idx = PRE_PERIODS - 1
        t0_gain = float(res["profit_1"][PRE_PERIODS]) - float(res["profit_1"][baseline_idx])
        profitable_t0 += int(t0_gain > 0)

        per_session.append(
            {
                "session": s_dir.name,
                "final_delta_combined": meta["final_delta_combined"],
                "opt_mult": res["opt_mult"],
                "t0_profit_gain": t0_gain,
                "is_unprofitable_npv": bool(res["is_unprofitable"]),
                "base_lmp": res["base_lmp"],
            }
        )
        sys.stdout.write(
            f"\r  {idx + 1}/{len(session_dirs)} {s_dir.name} "
            f"BR={res['opt_mult']:.3f} Δπ@t0={t0_gain:+.1f}"
        )
        sys.stdout.flush()
    print()

    x_min = -PRE_PERIODS
    x_max = args.horizon
    t_axis = np.arange(x_min, x_max + 1)
    baseline_idx = PRE_PERIODS - 1
    y_gap = 0.25

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
    n = len(session_dirs)
    thr = args.delta_threshold

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    plot_aggregate_trace_panel(
        ax1, t_axis, mu_g1, p25_g1, p75_g1, b_g1,
        title=f"Firm 1 (Deviator) — Generation [{n} sessions, Δ>{thr}]",
        color="#d62728", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
        vline_label="Static best response (t=0)",
    )
    save_figure(fig1, out_dir / "firm1_deviator_generation.png", args.dpi)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    plot_aggregate_trace_panel(
        ax2, t_axis, mu_g0, p25_g0, p75_g0, b_g0,
        title=f"Firm 0 (Non-Deviator) — Generation & Retaliation [{n} sessions]",
        color="#1f77b4", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
        vline_label="Firm 1 static BR (t=0); Firm 0 on policy",
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

    fig, axes = plt.subplots(4, 1, figsize=(12, 18))
    plot_aggregate_trace_panel(
        axes[0], t_axis, mu_g1, p25_g1, p75_g1, b_g1,
        title="Firm 1 — Generation (static BR at t=0)",
        color="#d62728", ylabel="Generation (MW)", baseline_fmt="{:.0f} MW",
        x_min=x_min, x_max=x_max, y_gap=y_gap,
        vline_label="Static BR (t=0)",
    )
    plot_aggregate_trace_panel(
        axes[1], t_axis, mu_g0, p25_g0, p75_g0, b_g0,
        title="Firm 0 — Generation (policy at t=0; retaliation after)",
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
        f"Static Best Response (Firm 1, t=0) — {n} sessions with Δ>{thr}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    save_figure(fig, out_dir / "static_br_combined.png", args.dpi)

    summary = {
        "delta_threshold": args.delta_threshold,
        "n_sessions": n,
        "horizon": args.horizon,
        "sessions_with_t0_profit_gain": profitable_t0,
        "mean_br_mult": float(np.mean([r["opt_mult"] for r in per_session])),
        "mean_t0_profit_gain": float(np.mean([r["t0_profit_gain"] for r in per_session])),
        "sessions": per_session,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Sessions with Firm 1 profit gain at t=0: {profitable_t0}/{n}")
    print(f"Mean t=0 profit gain (Firm 1): ${summary['mean_t0_profit_gain']:.2f}/step")
    print(f"Mean static-BR multiplier vs policy: {summary['mean_br_mult']:.3f}")
    print(f"Figures → {out_dir}/")


if __name__ == "__main__":
    main()
