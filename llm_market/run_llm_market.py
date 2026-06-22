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

from iso_market.market_env import (
    ElectricityMarketEnv,
    NUM_FIRMS,
    FIRM_PLANT_IDX,
    PLANTS,
)
from experiments.ppo import (
    compute_competitive_benchmark,
    compute_cournot_nash_benchmark,
    compute_monopoly_benchmark,
    compute_combined_delta,
    _benchmark_profits_by_firm,
)

from llm_market.granite_engine import GraniteEngine, DEFAULT_MODEL
from llm_market.prompt_builder import (
    AgentMemory,
    build_messages,
    first_period_state_text,
    latest_state_text_from_info,
)
from llm_market.action_parser import parse_action, action_json_schema, firm_caps


def _firm_total_gen(gen_per_plant: dict, firm_id: int) -> float:
    return float(sum(gen_per_plant.get(pidx, 0.0) for pidx in FIRM_PLANT_IDX[firm_id]))


def run_session(env, engine, benchmarks, args, session_id, pi_nash, pi_mono):
    """Run one repeated-game session; return a session dict in plot-compatible form."""
    env.reset()
    memories = {f: AgentMemory(window=args.history_window) for f in range(NUM_FIRMS)}
    schemas = [action_json_schema(f) for f in range(NUM_FIRMS)]

    # Seed each firm's "latest state" text from the competitive baseline.
    latest_state = {f: first_period_state_text(benchmarks, f) for f in range(NUM_FIRMS)}
    last_actions = {f: 0.5 * firm_caps(f) for f in range(NUM_FIRMS)}

    metrics = []
    delta_history = []
    parse_failures = 0

    for t in range(args.num_periods):
        # ---- Steps 1-2: translate state + build per-agent prompts ----
        batch_messages = [
            build_messages(
                f, memories[f], latest_state[f],
                benchmarks=benchmarks, goal=args.goal,
            )
            for f in range(NUM_FIRMS)
        ]

        # ---- Step 3: ONE batched LLM call (one prompt per agent) ----
        completions = engine.chat(batch_messages, schemas=schemas)

        # ---- Step 4: parse actions ----
        actions_mw = {}
        reasonings = {}
        for f in range(NUM_FIRMS):
            parsed = parse_action(
                completions[f], f, default_mw=last_actions[f]
            )
            actions_mw[f] = parsed["mw"]
            reasonings[f] = parsed["reasoning"]
            if not parsed["parse_ok"]:
                parse_failures += 1
        last_actions = {f: actions_mw[f].copy() for f in range(NUM_FIRMS)}

        # ---- Clear the market ----
        obs, rewards, done, info = env.step(actions_mw)
        if done and info.get("error"):
            env.reset()
            continue

        gen = info.get("gen", {})
        avg_lmp = float(info.get("avg_lmp", 0.0))
        lmps = np.asarray(info["lmps"], dtype=float)

        # Per-firm profit this period (= env reward), and combined Δ.
        per_firm_profit = {f: float(rewards[f]) for f in range(NUM_FIRMS)}
        delta_now = compute_combined_delta(per_firm_profit, pi_nash, pi_mono)
        delta_history.append(delta_now)

        # ---- Update each firm's memory ----
        for f in range(NUM_FIRMS):
            own_plant_gen = [gen.get(pidx, 0.0) for pidx in FIRM_PLANT_IDX[f]]
            own_total = _firm_total_gen(gen, f)
            rival_total = sum(
                _firm_total_gen(gen, g) for g in range(NUM_FIRMS) if g != f
            )
            nodes = sorted({PLANTS[pidx]["node"] for pidx in FIRM_PLANT_IDX[f]})
            own_price = float(np.mean([lmps[n] for n in nodes]))
            memories[f].add(
                period=t,
                own_gen_total=own_total,
                own_plant_gen=own_plant_gen,
                price=own_price,
                profit=per_firm_profit[f],
                rival_total=rival_total,
            )
            latest_state[f] = latest_state_text_from_info(info, f)

        # ---- Log a plot_results-compatible row ----
        row = {
            "step": t,
            "ppo_update": t + 1,
            "ppo_updates_total": args.num_periods,
            "avg_lmp": avg_lmp,
            "delta_combined": float(delta_now),
        }
        for f in range(NUM_FIRMS):
            row[f"firm_{f}_avg_gen"] = _firm_total_gen(gen, f)
            row[f"firm_{f}_avg_step_profit"] = per_firm_profit[f]
        metrics.append(row)

        if args.verbose and (t % max(1, args.log_interval) == 0 or t == args.num_periods - 1):
            g0 = _firm_total_gen(gen, 0)
            g1 = _firm_total_gen(gen, 1)
            print(
                f"  S{session_id + 1} t={t:>3d} | LMP ${avg_lmp:5.2f} | "
                f"g0={g0:6.1f} g1={g1:6.1f} | Δ={delta_now:+.3f}"
            )

    tail = delta_history[len(delta_history) // 2:] or delta_history or [0.0]
    final_delta = float(np.mean(tail))
    return {
        "session_id": session_id,
        "seed": args.seed + session_id,
        "converged": False,
        "convergence_step": args.num_periods,
        "final_delta_combined": final_delta,
        "metrics": metrics,
        "parse_failures": parse_failures,
        # empty placeholders so plot_results' optional panels degrade gracefully
        "limit_strategy": {},
        "deviation_experiment": {},
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
    config["agent_type"] = "llm_granite"
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_final_delta = []
    total_parse_fail = 0
    for s in range(args.num_sessions):
        print(f"\n--- Session {s + 1}/{args.num_sessions} (seed={args.seed + s}) ---")
        result = run_session(env, engine, benchmarks, args, s, pi_nash, pi_mono)

        sess_dir = out_dir / "sessions" / f"session_{s}"
        sess_dir.mkdir(parents=True, exist_ok=True)
        with open(sess_dir / "session.json", "w") as f:
            json.dump(result, f, indent=2)

        all_final_delta.append(result["final_delta_combined"])
        total_parse_fail += result["parse_failures"]
        print(
            f"  Final Δ_combined={result['final_delta_combined']:.3f} | "
            f"parse failures={result['parse_failures']}"
        )

    aggregate = {
        "num_sessions": args.num_sessions,
        "num_periods": args.num_periods,
        "backend": args.backend,
        "model": args.model,
        "delta_combined_mean": float(np.mean(all_final_delta)) if all_final_delta else 0.0,
        "delta_combined_std": float(np.std(all_final_delta)) if all_final_delta else 0.0,
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
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
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
    p.add_argument("--history-window", type=int, default=10,
                   help="Sliding-window memory length (periods shown in the prompt).")
    p.add_argument("--goal", type=str, default="own_profit",
                   choices=("own_profit", "joint_profit"))

    # Environment (kept consistent with the PPO runs)
    p.add_argument("--history-len", type=int, default=1)
    p.add_argument("--episode-len", type=int, default=168)
    p.add_argument("--no-past-gen", dest="include_past_gen", action="store_false")
    p.add_argument("--no-prev-reward", dest="include_prev_reward", action="store_false")
    p.set_defaults(include_past_gen=True, include_prev_reward=True)

    # System / logging
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/llm_mock")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
