"""
Driver: run repeated electricity-market games where each firm is an IBM Granite
LLM agent. Reuses the PPO project's ElectricityMarketEnv (DC-OPF clearing) and
benchmarks so results are directly comparable and plottable with
experiments/plot_results.py.

Per period, both firms' prompts are built and sent to the model in a SINGLE
batched call (one prompt per agent), then parsed into MW and cleared by the ISO.

Example (local pipeline test, no GPU):
    python llm_market/run_llm_market.py --backend mock \
        --num-sessions 3 --num-periods 60 --output-dir results/llm_mock

Example (Granite on the A100 via vLLM):
    python llm_market/run_llm_market.py --backend vllm \
        --model ibm-granite/granite-3.3-8b-instruct \
        --num-sessions 20 --num-periods 300 --history-window 10 \
        --output-dir results/llm_granite_h1

Then plot exactly like the PPO runs:
    python experiments/plot_results.py results/llm_granite_h1 --calvano-paper --save figures/llm/
    python experiments/plot_results.py results/llm_granite_h1 --per-firm-profit --save figures/llm/
    python experiments/plot_results.py results/llm_granite_h1 --variance-funnel --save figures/llm/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv, NUM_FIRMS, FIRM_PLANT_IDX
from experiments.ppo import (
    compute_competitive_benchmark,
    compute_cournot_nash_benchmark,
    compute_monopoly_benchmark,
    compute_combined_delta,
    _benchmark_profits_by_firm,
)

from llm_market.granite_engine import GraniteEngine, DEFAULT_MODEL
from llm_market.state_translator import competitive_default_mw
from llm_market.action_parser import action_json_schema
from llm_market.llm_dynamics import (
    firm_total_gen as _firm_total_gen,
    fresh_memories,
    initial_latest_state,
    select_actions_llm,
    update_memories,
    run_deviation_experiment_llm,
    compute_limit_strategy_llm,
)


def _stationarity(delta_history, window_frac=0.25, threshold=0.05):
    """Flag whether the *repeated-game dynamics* have settled (not weight training).

    Granite is frozen — there is no gradient training to converge. This instead
    reports whether Δ has stabilised: std of Δ over the final ``window_frac`` of
    periods is below ``threshold``. Reported alongside a PPO-compatible ``converged``
    flag so the existing plots/tables keep working.
    """
    if len(delta_history) < 4:
        return False, float("nan")
    w = max(2, int(len(delta_history) * window_frac))
    tail = np.asarray(delta_history[-w:], dtype=float)
    std = float(np.std(tail))
    return bool(std < threshold), std


def run_session(env, engine, benchmarks, args, session_id, pi_nash, pi_mono):
    """Run one repeated-game session; return a session dict in plot-compatible form."""
    base_seed = args.seed + 100003 * session_id  # distinct, reproducible per session
    obs = env.reset()
    ppo_parity = args.ppo_parity
    include_strategy = not ppo_parity

    memories = None if ppo_parity else fresh_memories(args)
    latest_state = None if ppo_parity else initial_latest_state(benchmarks)

    schemas = [
        action_json_schema(f, include_strategy=include_strategy)
        for f in range(NUM_FIRMS)
    ]
    last_actions = {f: competitive_default_mw(env, f) for f in range(NUM_FIRMS)}

    metrics = []
    delta_history = []
    parse_failures = 0
    transcripts = [] if args.save_transcripts else None

    # ---------- t=0 anchor: the COMPETITIVE benchmark ----------
    # Mirrors experiments/ppo.py: every plotted trajectory STARTS at the competitive
    # generation / price / profit (the market the agents are "born into" — env.reset()
    # seeds the observation history with exactly this baseline). Periods then log at
    # step t+1, so the figures read: t=0 -> competition, t>=1 -> what the LLM chooses.
    comp_b = benchmarks["competitive"]
    comp_profits = {f: float(comp_b["profits"][str(f)]) for f in range(NUM_FIRMS)}
    anchor = {
        "step": 0,
        "ppo_update": 0,
        "ppo_updates_total": args.num_periods,
        "avg_lmp": float(comp_b["avg_lmp"]),
        "delta_combined": float(
            compute_combined_delta(comp_profits, pi_nash, pi_mono)
        ),
    }
    for f in range(NUM_FIRMS):
        anchor[f"firm_{f}_avg_gen"] = float(
            sum(comp_b["gens"][pidx] for pidx in FIRM_PLANT_IDX[f])
        )
        anchor[f"firm_{f}_avg_step_profit"] = comp_profits[f]
    metrics.append(anchor)

    for t in range(args.num_periods):
        if t < args.warmup_competitive:
            # Seed the trajectory at the competitive baseline so the run visibly starts
            # from competition and the LLM's first real decisions already have a sensible
            # price/profit history to reason from (these rounds are NOT model choices).
            actions_mw = {f: competitive_default_mw(env, f) for f in range(NUM_FIRMS)}
            parsed_by_firm = {
                f: {"mw": actions_mw[f], "reasoning": "", "strategy": "", "parse_ok": True}
                for f in range(NUM_FIRMS)
            }
        else:
            # Build prompts -> ONE batched LLM call -> parse JSON -> MW per firm.
            # ``period=t`` gates --goals-start (asymmetric goals switch on at T).
            actions_mw, parsed_by_firm = select_actions_llm(
                engine, env, obs, benchmarks, args, schemas, last_actions,
                memories=memories, latest_state=latest_state, seed=base_seed + t,
                period=t,
            )
        for f in range(NUM_FIRMS):
            if memories is not None:
                memories[f].set_strategy(parsed_by_firm[f].get("strategy", ""))
            if not parsed_by_firm[f]["parse_ok"]:
                parse_failures += 1
        last_actions = {f: actions_mw[f].copy() for f in range(NUM_FIRMS)}

        # ISO clears the market; env rolls obs_history like PPO; memory is updated.
        obs, rewards, done, info = env.step(actions_mw)
        if done and info.get("error"):
            obs = env.reset()
            continue
        update_memories(memories, latest_state, info, t)

        gen = info.get("gen", {})
        avg_lmp = float(info.get("avg_lmp", 0.0))

        per_firm_profit = {f: float(rewards[f]) for f in range(NUM_FIRMS)}
        delta_now = compute_combined_delta(per_firm_profit, pi_nash, pi_mono)
        delta_history.append(delta_now)

        # ---- Log a plot_results-compatible row ----
        # INSTANTANEOUS realized values for this single period — no within-session
        # averaging anywhere; the plot layer averages ACROSS sessions per step.
        row = {
            "step": t + 1,  # t=0 is the competitive anchor row
            "ppo_update": t + 1,
            "ppo_updates_total": args.num_periods,
            "avg_lmp": avg_lmp,
            "delta_combined": float(delta_now),
        }
        for f in range(NUM_FIRMS):
            row[f"firm_{f}_avg_gen"] = _firm_total_gen(gen, f)
            row[f"firm_{f}_avg_step_profit"] = per_firm_profit[f]
        metrics.append(row)

        # ---- Qualitative evidence: the model's own words (per period) ----
        if transcripts is not None and t >= args.warmup_competitive:
            transcripts.append({
                "period": t + 1,
                "avg_lmp": avg_lmp,
                "delta_combined": float(delta_now),
                "firms": {
                    str(f): {
                        "mw": [float(v) for v in np.asarray(actions_mw[f]).ravel()],
                        "profit": per_firm_profit[f],
                        "reasoning": parsed_by_firm[f].get("reasoning", ""),
                        "strategy": parsed_by_firm[f].get("strategy", ""),
                        "parse_ok": bool(parsed_by_firm[f].get("parse_ok", False)),
                    }
                    for f in range(NUM_FIRMS)
                },
            })

        if args.verbose and (t % max(1, args.log_interval) == 0 or t == args.num_periods - 1):
            g0 = _firm_total_gen(gen, 0)
            g1 = _firm_total_gen(gen, 1)
            print(
                f"  S{session_id + 1} t={t:>3d} | LMP ${avg_lmp:5.2f} | "
                f"g0={g0:6.1f} g1={g1:6.1f} | Δ={delta_now:+.3f}"
            )

    tail = delta_history[len(delta_history) // 2:] or delta_history or [0.0]
    final_delta = float(np.mean(tail))
    converged, conv_std = _stationarity(delta_history)

    # ---- Post-hoc analysis (mirrors ppo.train_session) ----------------------
    # Punishment / impulse-response: the figure that demonstrates retaliation.
    run_dev = (not args.no_deviation) and (session_id < args.deviation_max_sessions)
    deviation_exp = (
        run_deviation_experiment_llm(
            env, engine, benchmarks, args, schemas, base_seed + 500000,
            deviation_frac=args.deviation_frac,
            warmup=args.deviation_warmup,
            horizon=args.deviation_horizon,
            pre=args.deviation_pre,
        )
        if run_dev else {}
    )
    # Reaction function (collusive limit strategy); parity mode only.
    limit_strategy = (
        compute_limit_strategy_llm(
            env, engine, benchmarks, args, schemas, base_seed + 900000,
            num_points=args.limit_points,
        )
        if (args.limit_strategy and session_id < args.deviation_max_sessions)
        else {}
    )

    return {
        "session_id": session_id,
        "seed": base_seed,
        "converged": converged,
        "convergence_std": conv_std,
        "convergence_step": args.num_periods,
        "final_delta_combined": final_delta,
        "metrics": metrics,
        "parse_failures": parse_failures,
        "ppo_parity": ppo_parity,
        "final_strategies": (
            {f: memories[f].latest_strategy for f in range(NUM_FIRMS)}
            if memories is not None else {}
        ),
        "final_cumulative_profit": (
            {f: memories[f].cumulative_profit for f in range(NUM_FIRMS)}
            if memories is not None else {}
        ),
        "limit_strategy": limit_strategy,
        "deviation_experiment": deviation_exp,
        # Written to a separate transcripts.jsonl by main() (kept out of session.json).
        "_transcripts": transcripts,
    }


def main(args):
    env = ElectricityMarketEnv(
        history_len=args.history_len,
        episode_len=args.episode_len,
        include_past_gen=args.include_past_gen,
        include_prev_reward=args.include_prev_reward,
    )

    print("\n=== Benchmarks (Cournot-Nash MCP may take a few seconds) ===")
    benchmarks = {
        "competitive": compute_competitive_benchmark(env),
        "cournot_nash": compute_cournot_nash_benchmark(env),
        "monopoly": compute_monopoly_benchmark(env),
    }
    cn, mono = benchmarks["cournot_nash"], benchmarks["monopoly"]
    print(
        f"  Competitive avg LMP ${benchmarks['competitive']['avg_lmp']:.2f} | "
        f"Nash ${cn['avg_lmp']:.2f} (π={cn['total_profit']:.0f}) | "
        f"Monopoly ${mono['avg_lmp']:.2f} (π={mono['total_profit']:.0f})"
    )

    pi_nash = _benchmark_profits_by_firm(benchmarks, "cournot_nash")
    pi_mono = _benchmark_profits_by_firm(benchmarks, "monopoly")

    print(f"\nLoading reasoning engine (backend={args.backend}, model={args.model}) ...")
    t0 = time.time()
    engine = GraniteEngine(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        seed=args.seed,
        mock_target_fraction=args.mock_target_fraction,
    )
    print(f"  engine ready in {time.time() - t0:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sessions").mkdir(exist_ok=True)

    config = vars(args).copy()
    config["benchmarks"] = benchmarks
    config["agent_type"] = "llm_granite_ppo_parity" if args.ppo_parity else "llm_granite"
    if args.goals:
        # Asymmetric-objective (seeded-colluder) runs are labeled so figures/tables
        # can't be mistaken for the emergent-collusion arms.
        config["agent_type"] += "_asym_" + "+".join(args.goals)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_final_delta = []
    all_converged = []
    total_parse_fail = 0
    for s in range(args.num_sessions):
        print(f"\n--- Session {s + 1}/{args.num_sessions} (seed={args.seed + 100003 * s}) ---")
        result = run_session(env, engine, benchmarks, args, s, pi_nash, pi_mono)

        sess_dir = out_dir / "sessions" / f"session_{s}"
        sess_dir.mkdir(parents=True, exist_ok=True)
        transcripts = result.pop("_transcripts", None)
        if transcripts:
            # One JSON object per period: the model's own reasoning/strategy text —
            # the qualitative evidence of (non-)collusion to quote in the paper.
            with open(sess_dir / "transcripts.jsonl", "w") as f:
                for entry in transcripts:
                    f.write(json.dumps(entry) + "\n")
        with open(sess_dir / "session.json", "w") as f:
            json.dump(result, f, indent=2)

        all_final_delta.append(result["final_delta_combined"])
        all_converged.append(bool(result.get("converged", False)))
        total_parse_fail += result["parse_failures"]
        print(
            f"  Final Δ_combined={result['final_delta_combined']:.3f} "
            f"(stationary={'yes' if result.get('converged') else 'no'}) | "
            f"parse failures={result['parse_failures']}"
        )
        de = result.get("deviation_experiment") or {}
        for dstr, entry in de.items():
            p = (entry or {}).get("punishment") or {}
            if not p:
                continue
            verdict = "PUNISH" if p.get("punished") else "ACCOMMODATE"
            print(
                f"  Deviation (firm {dstr} cheats): rivals {verdict} "
                f"(combined ΔMW={p.get('rival_output_increase_mw', 0):+.1f}, "
                f"LMP drop=${p.get('lmp_drop', 0):.2f})"
            )

    aggregate = {
        "num_sessions": args.num_sessions,
        "num_periods": args.num_periods,
        "backend": args.backend,
        "model": args.model,
        "agent_type": config.get("agent_type"),
        "ppo_parity": args.ppo_parity,
        "history_len": args.history_len,
        "delta_combined_mean": float(np.mean(all_final_delta)) if all_final_delta else 0.0,
        "delta_combined_std": float(np.std(all_final_delta)) if all_final_delta else 0.0,
        "stationary_fraction": float(np.mean(all_converged)) if all_converged else 0.0,
        "total_parse_failures": total_parse_fail,
    }
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Results -> {out_dir}")
    print(
        f"  Δ_combined: {aggregate['delta_combined_mean']:.3f} "
        f"± {aggregate['delta_combined_std']:.3f}  "
        f"({args.num_sessions} sessions x {args.num_periods} periods)"
    )
    print(f"  Stationary sessions: {aggregate['stationary_fraction'] * 100:.0f}%")
    print(f"  Parse failures: {total_parse_fail}")
    print(
        "\nPlot with:\n"
        f"  python experiments/plot_results.py {out_dir} --calvano-paper   --save figures/llm/\n"
        f"  python experiments/plot_results.py {out_dir} --per-firm-profit --save figures/llm/\n"
        f"  python experiments/plot_results.py {out_dir} --variance-funnel --save figures/llm/"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Repeated electricity-market game with IBM Granite LLM agents."
    )
    # Backend / model
    p.add_argument("--backend", type=str, default="mock",
                   choices=("mock", "vllm", "transformers"),
                   help="mock = no GPU (pipeline test); vllm = A100 batched; transformers = HF fallback.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.4,
                   help="Lower = steadier round-to-round choices (less bounce around "
                        "the profit peak); keep >0 for cross-session variance.")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Token budget per response; needs headroom for the reasoning field.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="GPUs for vLLM tensor parallelism (use 2 for the 30B model).")
    p.add_argument("--quantization", type=str, default=None,
                   help="e.g. 'fp8' to fit the 30B model on one A100.")
    p.add_argument("--mock-target-fraction", type=float, default=0.62,
                   help="Mock backend: generation as a fraction of capacity.")

    # Game / experiment
    p.add_argument("--num-sessions", type=int, default=3)
    p.add_argument("--num-periods", type=int, default=60,
                   help="Decision periods per session (repeated interactions).")
    p.add_argument("--warmup-competitive", type=int, default=0,
                   help="Force both firms to the competitive-baseline output for the first "
                        "N rounds, so the run starts from competition and the LLM's first "
                        "real decisions already have a price/profit history to reason from.")
    p.add_argument("--history-window", type=int, default=10,
                   help="Memory mode: number of past rounds shown in the prompt.")
    p.add_argument("--ppo-parity", action="store_true", default=True,
                   help="Feed the raw 19-number PPO observation (fair-information "
                        "comparison). Poor at eliciting reasoning — collusion rarely emerges.")
    p.add_argument("--legacy-memory", dest="ppo_parity", action="store_false",
                   help="RECOMMENDED for the collusion study: narrative history + carried "
                        "strategy note, which lets the model reason about price impact.")
    p.add_argument("--goal", type=str, default="own_profit",
                   choices=("own_profit", "joint_profit"),
                   help="Default objective for EVERY firm (overridden per firm by --goals).")
    p.add_argument(
        "--goals", type=str, default=None,
        help="Optional per-firm objectives, comma-separated (firm0,firm1). "
             "Example: collude,own_profit = firm 0 is a seeded colluder (explicit "
             "price-leader prompt) while firm 1 stays selfish — the 'what happens "
             "if ONE agent colludes' treatment. joint_profit is the softer variant. "
             "Each entry must be own_profit, joint_profit or collude.",
    )
    p.add_argument(
        "--goals-start", type=int, default=0,
        help="Period at which the per-firm --goals switch ON. Before this, every "
             "firm plays the symmetric --goal objective (in-session baseline / "
             "trial phase), giving a before/after contrast around the moment one "
             "agent starts colluding. 0 = asymmetric goals from the first period.",
    )

    # Post-hoc analysis: punishment / impulse-response + limit strategy
    # (analogues of experiments/ppo.run_deviation_experiment / compute_limit_strategy).
    p.add_argument("--no-deviation", action="store_true", default=False,
                   help="Skip the deviation/punishment experiment (saves GPU time).")
    p.add_argument("--deviation-frac", type=float, default=0.2,
                   help="One-period forced over-production (fraction above the chosen output).")
    p.add_argument("--deviation-warmup", type=int, default=20,
                   help="Periods of normal LLM play before the forced deviation "
                        "(must be long enough for a tacit resting point; ≥ history window).")
    p.add_argument("--deviation-horizon", type=int, default=20,
                   help="Periods of normal play observed after the deviation (the punishment window).")
    p.add_argument("--deviation-pre", type=int, default=4,
                   help="Resting periods recorded BEFORE the forced deviation (flat "
                        "collusive baseline in the punishment figure).")
    p.add_argument("--deviation-max-sessions", type=int, default=5,
                   help="Run the deviation/limit experiments only on the first N sessions.")
    p.add_argument("--limit-strategy", action="store_true", default=False,
                   help="Also sweep avg-LMP to record each firm's reaction function (parity mode only).")
    p.add_argument("--limit-points", type=int, default=12,
                   help="Grid points for the limit-strategy sweep.")

    # Environment (kept consistent with the PPO runs)
    p.add_argument("--history-len", type=int, default=1)
    p.add_argument("--episode-len", type=int, default=168)
    p.add_argument("--no-past-gen", dest="include_past_gen", action="store_false")
    p.add_argument("--no-prev-reward", dest="include_prev_reward", action="store_false")
    p.set_defaults(include_past_gen=True, include_prev_reward=True)

    # System / logging
    p.add_argument("--save-transcripts", action="store_true", default=True,
                   help="Write each period's reasoning/strategy text to "
                        "sessions/session_*/transcripts.jsonl (qualitative evidence).")
    p.add_argument("--no-transcripts", dest="save_transcripts", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/llm_mock")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")

    args = p.parse_args()
    # Normalize --goals into a length-NUM_FIRMS list (or leave None = use --goal).
    if args.goals:
        parts = [g.strip() for g in args.goals.split(",") if g.strip()]
        allowed = {"own_profit", "joint_profit", "collude"}
        bad = [g for g in parts if g not in allowed]
        if bad:
            p.error(f"--goals entries must be in {sorted(allowed)}; got {bad}")
        if len(parts) == 1:
            parts = parts * NUM_FIRMS
        if len(parts) != NUM_FIRMS:
            p.error(
                f"--goals needs 1 or {NUM_FIRMS} entries (got {len(parts)}): {args.goals!r}"
            )
        args.goals = parts
    else:
        args.goals = None
    return args


if __name__ == "__main__":
    main(parse_args())
