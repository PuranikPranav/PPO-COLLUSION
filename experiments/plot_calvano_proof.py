"""
Publication-Quality Calvano Impulse Response for Continuous PPO.

Per session: Firm 1 plays per-plant static best response at t=0 (Firm 0 on
greedy policy), then both agents play greedy. Aggregates traces across sessions.

Usage:
    python experiments/plot_calvano_proof.py --run-dir latest_results
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv
from experiments.calvano_impulse import (
    PRE_PERIODS,
    aggregate_series,
    plot_lmp_panel,
    plot_trace_panel,
    run_calvano_impulse,
)
from experiments.paths import DEFAULT_RUN_DIR_NAME, deviation_figures_dir, resolve_run_dir
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers


def save_single_figure(build_fn, out_path, dpi):
    fig, ax = plt.subplots(figsize=(12, 5))
    build_fn(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"Training run directory (default: {DEFAULT_RUN_DIR_NAME}/)",
    )
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    fig_dir = deviation_figures_dir(run_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sessions_root = run_dir / "sessions"
    session_dirs = sorted(
        d for d in sessions_root.iterdir() if d.is_dir() and (d / "agent_0.pt").exists()
    )

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))
    gamma = float(config.get("gamma", 0.99))

    all_g0, all_g1, all_lmp = [], [], []
    all_p0, all_p1, all_mults = [], [], []
    unprofitable_count = 0

    print(
        f"Running Calvano proof (per-plant static BR, Firm 1 at t=0) "
        f"on {len(session_dirs)} sessions..."
    )

    for idx, s_dir in enumerate(session_dirs):
        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)
        res = run_calvano_impulse(env, agents, normalizers, args.warmup, args.horizon, gamma)

        all_g0.append(res["gen_0"])
        all_g1.append(res["gen_1"])
        all_lmp.append(res["lmp"])
        all_p0.append(res["profit_0"])
        all_p1.append(res["profit_1"])
        all_mults.append(res["opt_mult"])
        unprofitable_count += int(res["is_unprofitable"])

        sys.stdout.write(f"\r  Session {idx + 1}/{len(session_dirs)}  BR={res['opt_mult']:.3f}")
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

    avg_mult = float(np.mean(all_mults))
    pct_unprof = 100.0 * unprofitable_count / len(session_dirs)

    print(f"  Mean static-BR scale: {avg_mult:.3f}x ({100 * (avg_mult - 1):+.1f}% vs greedy MW)")
    print(f"  NPV punishment rate: {pct_unprof:.1f}% of sessions")

    plt.style.use("seaborn-v0_8-whitegrid")

    paths = {
        "firm1_gen": fig_dir / "calvano_proof_firm1_deviator.png",
        "firm0_gen": fig_dir / "calvano_proof_firm0_nondeviator.png",
        "firm1_profit": fig_dir / "calvano_proof_firm1_profit.png",
        "firm0_profit": fig_dir / "calvano_proof_firm0_profit.png",
        "lmp": fig_dir / "calvano_proof_lmp.png",
        "combined": fig_dir / "calvano_proof_publication.png",
        "empirical": fig_dir / "calvano_empirical_proof.png",
    }

    save_single_figure(
        lambda ax: plot_trace_panel(
            ax, t_axis, mu_g1, p25_g1, p75_g1, float(mu_g1[baseline_idx]),
            f"Deviating Firm (Firm 1) — Generation  [avg BR {avg_mult:.3f}x]",
            "#d62728", "Generation (MW)", "{:.0f} MW", x_min, x_max, vline_label=True,
        ),
        paths["firm1_gen"],
        args.dpi,
    )
    save_single_figure(
        lambda ax: plot_trace_panel(
            ax, t_axis, mu_g0, p25_g0, p75_g0, float(mu_g0[baseline_idx]),
            "Non-Deviating Firm (Firm 0) — Generation",
            "#1f77b4", "Generation (MW)", "{:.0f} MW", x_min, x_max,
        ),
        paths["firm0_gen"],
        args.dpi,
    )
    save_single_figure(
        lambda ax: plot_trace_panel(
            ax, t_axis, mu_p1, p25_p1, p75_p1, float(mu_p1[baseline_idx]),
            f"Deviating Firm (Firm 1) — Profit  [avg BR {avg_mult:.3f}x]",
            "#d62728", "Profit ($/step)", "${:.0f}", x_min, x_max, vline_label=True,
        ),
        paths["firm1_profit"],
        args.dpi,
    )
    save_single_figure(
        lambda ax: plot_trace_panel(
            ax, t_axis, mu_p0, p25_p0, p75_p0, float(mu_p0[baseline_idx]),
            "Non-Deviating Firm (Firm 0) — Profit",
            "#1f77b4", "Profit ($/step)", "${:.0f}", x_min, x_max,
        ),
        paths["firm0_profit"],
        args.dpi,
    )
    save_single_figure(
        lambda ax: plot_lmp_panel(
            ax, t_axis, mu_lmp, p25_lmp, p75_lmp, float(mu_lmp[baseline_idx]),
            x_min, x_max, f"NPV unprofitable in {pct_unprof:.1f}% of sessions",
        ),
        paths["lmp"],
        args.dpi,
    )

    fig, axes = plt.subplots(5, 1, figsize=(12, 22))
    fig.suptitle(
        f"Calvano Empirical Proof — Firm 1 static BR (mean {avg_mult:.3f}x, "
        f"NPV unprofitable in {pct_unprof:.1f}% of sessions)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plot_trace_panel(
        axes[0], t_axis, mu_g1, p25_g1, p75_g1, float(mu_g1[baseline_idx]),
        "Deviating Firm (Firm 1) — Generation", "#d62728",
        "Generation (MW)", "{:.0f} MW", x_min, x_max, vline_label=True,
    )
    plot_trace_panel(
        axes[1], t_axis, mu_p1, p25_p1, p75_p1, float(mu_p1[baseline_idx]),
        "Deviating Firm (Firm 1) — Profit", "#d62728",
        "Profit ($/step)", "${:.0f}", x_min, x_max,
    )
    plot_trace_panel(
        axes[2], t_axis, mu_g0, p25_g0, p75_g0, float(mu_g0[baseline_idx]),
        "Non-Deviating Firm (Firm 0) — Generation", "#1f77b4",
        "Generation (MW)", "{:.0f} MW", x_min, x_max,
    )
    plot_trace_panel(
        axes[3], t_axis, mu_p0, p25_p0, p75_p0, float(mu_p0[baseline_idx]),
        "Non-Deviating Firm (Firm 0) — Profit", "#1f77b4",
        "Profit ($/step)", "${:.0f}", x_min, x_max,
    )
    plot_lmp_panel(
        axes[4], t_axis, mu_lmp, p25_lmp, p75_lmp, float(mu_lmp[baseline_idx]),
        x_min, x_max, f"NPV unprofitable in {pct_unprof:.1f}% of sessions",
    )
    fig.tight_layout()
    fig.savefig(paths["combined"], dpi=args.dpi, bbox_inches="tight")
    fig.savefig(paths["empirical"], dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    for p in paths.values():
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
