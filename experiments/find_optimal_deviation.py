"""
Find Firm 1's one-period profit-maximizing deviation (static best response).

At t=0, Firm 0 plays its frozen greedy policy. Firm 1's output is scaled by a
multiplier on top of its greedy MW (then clipped to plant caps). We search
multipliers and pick the one with highest immediate profit.

This is a restricted static best response (uniform scale on policy MW). The
full per-plant grid search lives in impulse_response._static_best_response_mw.

Usage:
    python experiments/find_optimal_deviation.py --run-dir h1
    python experiments/find_optimal_deviation.py --run-dir h1 --session 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv, FIRM_PLANT_IDX, PLANTS
from experiments.impulse_response import _static_best_response_mw, profit_firm
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers

DEVIATOR_FID = 1
NONDEVIATOR_FID = 0


def warmup_to_resting_state(env, agents, normalizers, warmup: int, seed: int = 42):
    """Roll greedy policies to a collusive resting state; return obs and greedy actions."""
    np.random.seed(seed)
    obs = env.reset()
    for _ in range(warmup):
        actions = {}
        for fid, agent in agents.items():
            obs_norm = normalizers[fid].normalize(obs[fid])
            actions[fid] = agent.deterministic_action(obs_norm)
        obs, _, done, _ = env.step(actions)
        if done:
            obs = env.reset()

    actions = {}
    for fid, agent in agents.items():
        obs_norm = normalizers[fid].normalize(obs[fid])
        actions[fid] = agent.deterministic_action(obs_norm).astype(np.float64)
    return obs, actions


def scale_deviator_actions(policy_actions: dict, deviating_fid: int, multiplier: float) -> dict:
    """Scale deviator MW; rival keeps greedy output."""
    out = {fid: policy_actions[fid].copy() for fid in policy_actions}
    scaled = out[deviating_fid] * multiplier
    for j, pidx in enumerate(FIRM_PLANT_IDX[deviating_fid]):
        scaled[j] = min(float(scaled[j]), PLANTS[pidx]["cap"])
    out[deviating_fid] = scaled
    return out


def one_period_profit(env, actions: dict, firm_id: int) -> float | None:
    """Profit from a single DC-OPF clear at these actions (no env state advance)."""
    gen_node = np.zeros(5)
    for fid, mw in actions.items():
        for j, pidx in enumerate(FIRM_PLANT_IDX[fid]):
            gen_node[PLANTS[pidx]["node"]] += float(mw[j])
    lmps, _, _, _ = env._clear_market(gen_node)
    if lmps is None:
        return None
    return profit_firm({"lmps": lmps}, actions, firm_id)


def find_optimal_deviation(
    env,
    agents,
    normalizers,
    deviating_fid: int = DEVIATOR_FID,
    warmup: int = 20,
    mult_min: float = 0.50,
    mult_max: float = 2.00,
    n_grid: int = 76,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Search multiplier m so Firm `deviating_fid` maximizes one-period profit at t=0
    while the rival plays greedy policy output.

    Returns dict with best_multiplier, baseline_profit, best_profit, etc.
    """
    if deviating_fid not in (0, 1):
        raise ValueError("deviating_fid must be 0 or 1")

    _, policy_actions = warmup_to_resting_state(env, agents, normalizers, warmup, seed)
    rival_fid = 1 - deviating_fid

    test_multipliers = np.linspace(mult_min, mult_max, n_grid)
    best_profit = -float("inf")
    best_multiplier = 1.0
    baseline_profit = one_period_profit(env, policy_actions, deviating_fid)
    if baseline_profit is None:
        raise RuntimeError("DC-OPF infeasible at collusive greedy actions")

    if verbose:
        print(f"Searching static best response (uniform multiplier) for Firm {deviating_fid}...")
        print(f"  Rival Firm {rival_fid} fixed at greedy MW = {np.sum(policy_actions[rival_fid]):.1f}")

    for mult in test_multipliers:
        trial_actions = scale_deviator_actions(policy_actions, deviating_fid, float(mult))
        dev_profit = one_period_profit(env, trial_actions, deviating_fid)
        if dev_profit is None:
            continue
        if dev_profit > best_profit:
            best_profit = dev_profit
            best_multiplier = float(mult)

    policy_mw = float(np.sum(policy_actions[deviating_fid]))
    best_mw = policy_mw * best_multiplier

    # Full per-plant static BR (reference)
    if deviating_fid == DEVIATOR_FID:
        br_mw_arr = _static_best_response_mw(policy_actions[NONDEVIATOR_FID], env)
        br_actions = {
            NONDEVIATOR_FID: policy_actions[NONDEVIATOR_FID],
            DEVIATOR_FID: br_mw_arr,
        }
        br_profit = one_period_profit(env, br_actions, deviating_fid)
        br_mw = float(np.sum(br_mw_arr))
    else:
        br_profit = None
        br_mw = None

    gain_pct = 100.0 * (best_profit - baseline_profit) / max(abs(baseline_profit), 1e-9)

    if verbose:
        print(f"  Baseline profit (multiplier 1.0x): ${baseline_profit:,.2f}")
        print(
            f"  Best uniform multiplier: {best_multiplier:.3f}x "
            f"({100 * (best_multiplier - 1):+.1f}% vs greedy MW)"
        )
        print(f"  Deviator MW: {policy_mw:.1f} -> {best_mw:.1f}")
        print(f"  Max one-period profit: ${best_profit:,.2f} ({gain_pct:+.2f}% vs baseline)")
        if br_profit is not None:
            br_gain = 100.0 * (br_profit - baseline_profit) / max(abs(baseline_profit), 1e-9)
            print(
                f"  Per-plant static BR (reference): MW={br_mw:.1f}, "
                f"profit=${br_profit:,.2f} ({br_gain:+.2f}%)"
            )
        if best_profit <= baseline_profit + 1e-6:
            print(
                "  NOTE: No uniform multiplier beats collusive greedy output. "
                "Policies may be near Nash or collusion is too weak for a one-period gain."
            )

    return {
        "deviating_fid": deviating_fid,
        "best_multiplier": best_multiplier,
        "baseline_profit": baseline_profit,
        "best_profit": best_profit,
        "profit_gain_pct": gain_pct,
        "policy_mw": policy_mw,
        "best_mw": best_mw,
        "per_plant_br_profit": br_profit,
        "per_plant_br_mw": br_mw,
        "is_profitable": best_profit > baseline_profit + 1e-6,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find one-period profit-maximizing deviation multiplier for Firm 1."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("h1"))
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        help="Single session id (default: summarize all sessions)",
    )
    parser.add_argument("--deviating-fid", type=int, default=DEVIATOR_FID, choices=(0, 1))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mult-min", type=float, default=0.50)
    parser.add_argument("--mult-max", type=float, default=2.00)
    parser.add_argument("--n-grid", type=int, default=76)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))

    sessions_root = args.run_dir / "sessions"
    if args.session is not None:
        session_dirs = [sessions_root / f"session_{args.session}"]
    else:
        session_dirs = sorted(
            d for d in sessions_root.iterdir() if d.is_dir() and (d / "agent_0.pt").exists()
        )

    env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
    results = []

    for s_dir in session_dirs:
        if not (s_dir / "agent_0.pt").exists():
            print(f"Skipping missing session: {s_dir}")
            continue
        if not args.quiet:
            print(f"\n=== {s_dir.name} ===")
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)
        res = find_optimal_deviation(
            env,
            agents,
            normalizers,
            deviating_fid=args.deviating_fid,
            warmup=args.warmup,
            mult_min=args.mult_min,
            mult_max=args.mult_max,
            n_grid=args.n_grid,
            verbose=not args.quiet,
        )
        res["session"] = s_dir.name
        results.append(res)

    if len(results) > 1:
        mults = np.array([r["best_multiplier"] for r in results])
        gains = np.array([r["profit_gain_pct"] for r in results])
        profitable = sum(r["is_profitable"] for r in results)
        print("\n=== Summary over sessions ===")
        print(f"  Sessions analyzed: {len(results)}")
        print(f"  Profitable uniform deviation: {profitable}/{len(results)}")
        print(f"  Best multiplier: mean={mults.mean():.3f}, median={np.median(mults):.3f}")
        print(f"  One-period profit gain: mean={gains.mean():+.2f}%, max={gains.max():+.2f}%")


if __name__ == "__main__":
    main()
