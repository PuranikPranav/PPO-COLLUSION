"""
Publication-Quality Calvano Impulse Response for Continuous PPO.

For each session, computes Firm 1's static best-response multiplier (one-period
profit max given Firm 0 on greedy policy), forces that deviation at t=0, then
aggregates traces across sessions.

Outputs separate zoomed panels: deviator / non-deviator generation & profit,
plus system LMP; also a combined figure and calvano_empirical_proof.png.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv
from experiments.find_optimal_deviation import find_optimal_deviation, scale_deviator_actions
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers

DEVIATOR_FID = 1
NONDEVIATOR_FID = 0
PRE_PERIODS = 2  # t = -2, -1 before deviation at t = 0


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


def run_publication_deviation(
    env,
    agents,
    normalizers,
    opt_mult,
    warmup=20,
    horizon=15,
    gamma=0.99,
    seed=42,
):
    """Impulse test: Firm 1 deviates by session-specific static BR multiplier at t=0."""
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

    actions = scale_deviator_actions(actions, DEVIATOR_FID, opt_mult)

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
        "gen_0": trace_gen_0,
        "gen_1": trace_gen_1,
        "lmp": trace_lmp,
        "profit_0": trace_profit_0,
        "profit_1": trace_profit_1,
        "is_unprofitable": deviation_npv < baseline_npv,
        "opt_mult": opt_mult,
    }


def aggregate_series(all_series):
    arr = np.array(all_series)
    return (
        np.mean(arr, axis=0),
        np.percentile(arr, 25, axis=0),
        np.percentile(arr, 75, axis=0),
    )


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
    ax.axvline(0, color="black", linestyle="--", alpha=0.65, lw=1.5,
               label="Static BR deviation (t=0)" if vline_label else None)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Period (t)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    apply_period_axis(ax, x_min, x_max)
    set_zoomed_ylim(ax, p25, p75, padding_frac=0.05)
    ax.legend(loc="best", fontsize=10)


def plot_lmp_panel(ax, t_axis, mu, p25, p75, baseline_lmp, x_min, x_max, pct_unprof):
    ax.axhline(
        baseline_lmp,
        color="green",
        linestyle=":",
        lw=2,
        label=f"Collusive baseline (${baseline_lmp:.2f})",
    )
    ax.plot(t_axis, mu, color="#2ca02c", lw=3, marker="o", markersize=5, label="System average LMP")
    ax.fill_between(t_axis, p25, p75, color="#2ca02c", alpha=0.18)
    ax.axvline(0, color="black", linestyle="--", alpha=0.65, lw=1.5)
    ax.set_title(
        f"Market Impact — deviation unprofitable in {pct_unprof:.1f}% of sessions",
        fontsize=14,
        pad=10,
    )
    ax.set_xlabel("Period (t)", fontsize=12)
    ax.set_ylabel("System Average LMP ($/MWh)", fontsize=12)
    apply_period_axis(ax, x_min, x_max)
    set_zoomed_ylim(ax, p25, p75, padding_frac=0.05)
    ax.legend(loc="lower right", fontsize=10)


def save_single_figure(build_fn, out_path, dpi):
    fig, ax = plt.subplots(figsize=(12, 5))
    build_fn(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("h1"))
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mult-min", type=float, default=0.80)
    parser.add_argument("--mult-max", type=float, default=1.30)
    parser.add_argument("--n-grid", type=int, default=51)
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    sessions_root = args.run_dir / "sessions"
    session_dirs = sorted(
        d for d in sessions_root.iterdir() if d.is_dir() and (d / "agent_0.pt").exists()
    )

    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))
    gamma = float(config.get("gamma", 0.99))

    all_g0, all_g1, all_lmp = [], [], []
    all_p0, all_p1, all_mults = [], [], []
    unprofitable_count = 0

    print(
        f"Running exact Calvano proof (per-session static BR, Firm 1 deviates) "
        f"on {len(session_dirs)} sessions..."
    )

    for idx, s_dir in enumerate(session_dirs):
        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)

        br = find_optimal_deviation(
            env,
            agents,
            normalizers,
            deviating_fid=DEVIATOR_FID,
            warmup=args.warmup,
            mult_min=args.mult_min,
            mult_max=args.mult_max,
            n_grid=args.n_grid,
            verbose=False,
        )
        res = run_publication_deviation(
            env,
            agents,
            normalizers,
            br["best_multiplier"],
            args.warmup,
            args.horizon,
            gamma,
        )

        all_g0.append(res["gen_0"])
        all_g1.append(res["gen_1"])
        all_lmp.append(res["lmp"])
        all_p0.append(res["profit_0"])
        all_p1.append(res["profit_1"])
        all_mults.append(res["opt_mult"])
        unprofitable_count += int(res["is_unprofitable"])

        sys.stdout.write(f"\r  Session {idx + 1}/{len(session_dirs)}  "
                         f"opt_mult={br['best_multiplier']:.3f}")
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
    mult_label = f"{100 * (avg_mult - 1):+.1f}%"

    print(f"  Mean static-BR multiplier: {avg_mult:.3f}x ({mult_label} vs greedy MW)")
    print(f"  NPV punishment rate: {pct_unprof:.1f}% of sessions")

    plt.style.use("seaborn-v0_8-whitegrid")

    paths = {
        "firm1_gen": args.run_dir / "calvano_proof_firm1_deviator.png",
        "firm0_gen": args.run_dir / "calvano_proof_firm0_nondeviator.png",
        "firm1_profit": args.run_dir / "calvano_proof_firm1_profit.png",
        "firm0_profit": args.run_dir / "calvano_proof_firm0_profit.png",
        "lmp": args.run_dir / "calvano_proof_lmp.png",
        "combined": args.run_dir / "calvano_proof_publication.png",
        "empirical": args.run_dir / "calvano_empirical_proof.png",
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
            x_min, x_max, pct_unprof,
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
        x_min, x_max, pct_unprof,
    )
    fig.tight_layout()
    fig.savefig(paths["combined"], dpi=args.dpi, bbox_inches="tight")
    fig.savefig(paths["empirical"], dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    for p in paths.values():
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
