"""
Top-4 collusive sessions: per-session static best response at t=0, individual plots.

1. Rank all sessions by resting system LMP (highest = most collusive).
2. For each of the top 4, compute Firm 1's per-plant static best response at t=0
   (Firm 0 on greedy policy); both agents play greedy from t>=1.
3. Save one 5-panel PNG per session (gen/profit/LMP for each firm).

Usage:
    python experiments/plot_top4_cartels.py --run-dir latest_results
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv
from experiments.calvano_impulse import measure_resting_lmp, plot_session_impulse_figure, run_calvano_impulse
from experiments.paths import DEFAULT_RUN_DIR_NAME, deviation_figures_dir, resolve_run_dir
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers


def main():
    parser = argparse.ArgumentParser(
        description="Static-BR impulse plots for the top 4 most collusive sessions."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"Training run directory (default: {DEFAULT_RUN_DIR_NAME}/)",
    )
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    fig_dir = deviation_figures_dir(run_dir)

    sessions_root = run_dir / "sessions"
    session_dirs = sorted(
        d for d in sessions_root.iterdir() if d.is_dir() and (d / "agent_0.pt").exists()
    )

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))
    gamma = float(config.get("gamma", 0.99))

    print(f"Ranking {len(session_dirs)} sessions by resting LMP...")
    ranked = []
    for idx, s_dir in enumerate(session_dirs):
        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)
        resting_lmp = measure_resting_lmp(env, agents, normalizers, args.warmup)
        ranked.append((resting_lmp, s_dir))
        sys.stdout.write(f"\r  Scanned {idx + 1}/{len(session_dirs)}")
        sys.stdout.flush()

    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[: args.top_k]

    print(f"\nTop {args.top_k} sessions (static BR impulse, Firm 1 deviates at t=0):")
    out_dir = fig_dir / "top4_static_br"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (resting_lmp, s_dir) in enumerate(top, start=1):
        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)

        res = run_calvano_impulse(
            env, agents, normalizers, args.warmup, args.horizon, gamma
        )
        out_path = out_dir / f"{s_dir.name}_static_br_impulse.png"
        plot_session_impulse_figure(res, s_dir.name, rank, out_path, args.horizon, args.dpi)

        print(
            f"  {rank}. {s_dir.name}: resting LMP ${resting_lmp:.2f}, "
            f"BR {res['opt_mult']:.3f}x, t=0 profit {res['t0_profit_gain']:+.0f}, "
            f"NPV {'unprofitable' if res['is_unprofitable'] else 'profitable'}"
        )
        print(f"     -> {out_path}")

    print(f"\nDone. {args.top_k} files in {out_dir}/")


if __name__ == "__main__":
    main()
