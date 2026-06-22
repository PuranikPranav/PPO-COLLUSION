"""
Repeated-game dynamics for the LLM market — shared by the main session loop and
the post-hoc analysis.

Two roles:

  1. ``select_actions_llm`` / ``play_period`` — one decision period: build prompts,
     batch them to the engine, parse JSON into MW, clear the market, update memory.
     The main loop and every analysis below go through these so behaviour is
     identical everywhere (no drift between "training" and "evaluation").

  2. ``run_deviation_experiment_llm`` — the punishment / impulse-response experiment
     (analogue of Calvano Fig. 4 and ``experiments/ppo.run_deviation_experiment``).
     After the agents settle, one firm is FORCED to over-produce for a single
     period; we then let both play their normal LLM policy and record how the rival
     reacts. A rival that *cuts output / lets price fall* in the periods after the
     deviation is exhibiting a reward–punishment (trigger) strategy — the direct
     evidence of tacit collusion sustained by the threat of retaliation.

  3. ``compute_limit_strategy_llm`` — the deterministic reaction function: sweep the
     observed average price and record how much each firm chooses to generate.
     Upward-sloping cooperation (more output only when price is already high) is the
     collusive "limit strategy"; this is the LLM analogue of ppo.compute_limit_strategy.

Nothing here trains the model — Granite is frozen. Collusion is elicited *in
context* from the repeated game + memory, not by gradient updates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from iso_market.market_env import NUM_FIRMS, FIRM_PLANT_IDX, PLANTS
from llm_market.prompt_builder import AgentMemory, build_messages
from llm_market.state_translator import (
    competitive_default_mw,
    initial_state_text,
    market_outcome_text,
)
from llm_market.action_parser import parse_action


def firm_total_gen(gen_per_plant: dict, firm_id: int) -> float:
    return float(sum(gen_per_plant.get(pidx, 0.0) for pidx in FIRM_PLANT_IDX[firm_id]))


# ---------------------------------------------------------------------------
# One decision period (shared by the main loop and every experiment)
# ---------------------------------------------------------------------------
def fresh_memories(args):
    """Per-firm sliding-window memory (legacy / collusion mode only)."""
    return {f: AgentMemory(window=args.history_window) for f in range(NUM_FIRMS)}


def initial_latest_state(benchmarks):
    return {f: initial_state_text(benchmarks, f) for f in range(NUM_FIRMS)}


def build_batch_messages(
    env, obs, benchmarks, args, memories, latest_state, last_actions=None
):
    """One chat prompt per firm for the current state.

    ``last_actions`` (rival's most recent MW) drives the what-if profit table; pass
    None to omit it (e.g. the synthetic limit-strategy sweep).
    """
    if args.ppo_parity:
        return [
            build_messages(
                f, env=env, obs=obs[f], benchmarks=benchmarks,
                goal=args.goal, ppo_parity=True, last_actions=last_actions,
            )
            for f in range(NUM_FIRMS)
        ]
    return [
        build_messages(
            f, env=env, memory=memories[f], latest_state_text=latest_state[f],
            benchmarks=benchmarks, goal=args.goal, ppo_parity=False,
            last_actions=last_actions,
        )
        for f in range(NUM_FIRMS)
    ]


def select_actions_llm(
    engine, env, obs, benchmarks, args, schemas, last_actions,
    memories=None, latest_state=None, seed=None,
):
    """Build prompts -> ONE batched LLM call -> parse -> MW per firm.

    Returns (actions_mw, parsed_by_firm). ``parsed_by_firm[f]`` carries the raw
    reasoning/strategy text so callers can log qualitative evidence.
    """
    batch = build_batch_messages(
        env, obs, benchmarks, args, memories, latest_state, last_actions=last_actions
    )
    completions = engine.chat(batch, schemas=schemas, seed=seed)

    actions_mw, parsed_by_firm = {}, {}
    for f in range(NUM_FIRMS):
        parsed = parse_action(completions[f], f, default_mw=last_actions[f])
        actions_mw[f] = parsed["mw"]
        parsed_by_firm[f] = parsed
    return actions_mw, parsed_by_firm


def update_memories(memories, latest_state, info, period):
    """Append this period's joint outcome to each firm's memory (legacy mode)."""
    if memories is None:
        return
    gen = info.get("gen", {})
    lmps = np.asarray(info["lmps"], dtype=float)
    for f in range(NUM_FIRMS):
        own_plant_gen = [gen.get(pidx, 0.0) for pidx in FIRM_PLANT_IDX[f]]
        own_total = firm_total_gen(gen, f)
        rival_total = sum(
            firm_total_gen(gen, g) for g in range(NUM_FIRMS) if g != f
        )
        nodes = sorted({PLANTS[pidx]["node"] for pidx in FIRM_PLANT_IDX[f]})
        own_price = float(np.mean([lmps[n] for n in nodes]))
        # profit for firm f this period
        profit = 0.0
        for pidx in FIRM_PLANT_IDX[f]:
            p = PLANTS[pidx]
            g = gen.get(pidx, 0.0)
            profit += lmps[p["node"]] * g - p["mc"] * g - 0.5 * p["qc"] * g * g
        memories[f].add(
            period=period, own_gen_total=own_total, own_plant_gen=own_plant_gen,
            price=own_price, profit=profit, rival_total=rival_total,
        )
        latest_state[f] = market_outcome_text(info, f)


def play_period(
    engine, env, obs, benchmarks, args, schemas, last_actions,
    memories=None, latest_state=None, seed=None, period=0,
    force=None,
):
    """Run ONE period of LLM play.

    ``force=(firm_id, multiplier)`` overrides the deviating firm's chosen output
    (capped at capacity) for this period only — used by the deviation experiment.

    Returns (obs_next, rewards, done, info, actions_mw, parsed_by_firm).
    """
    actions_mw, parsed_by_firm = select_actions_llm(
        engine, env, obs, benchmarks, args, schemas, last_actions,
        memories=memories, latest_state=latest_state, seed=seed,
    )
    if memories is not None:
        for f in range(NUM_FIRMS):
            memories[f].set_strategy(parsed_by_firm[f].get("strategy", ""))
    if force is not None:
        dev_fid, mult = force
        deviated = actions_mw[dev_fid] * float(mult)
        for j, pidx in enumerate(FIRM_PLANT_IDX[dev_fid]):
            deviated[j] = min(deviated[j], PLANTS[pidx]["cap"])
        actions_mw[dev_fid] = deviated

    obs_next, rewards, done, info = env.step(actions_mw)
    if not done:
        update_memories(memories, latest_state, info, period)
    return obs_next, rewards, done, info, actions_mw, parsed_by_firm


# ---------------------------------------------------------------------------
# Punishment / impulse-response experiment
# ---------------------------------------------------------------------------
def run_deviation_experiment_llm(
    env, engine, benchmarks, args, schemas, base_seed,
    deviation_frac=0.2, warmup=8, horizon=20,
):
    """Force a one-period deviation, then watch the rival's reaction.

    For each firm as the deviator:
      1. Warm up ``warmup`` periods of normal LLM play (reach a resting point).
      2. One deviation period: deviator's output is scaled by (1 + deviation_frac).
      3. ``horizon`` periods of normal LLM play; record generation + price.

    The returned structure matches ppo.run_deviation_experiment so the existing
    ``plot_results.py --deviation-explainer`` figure works unchanged.
    """
    results = {}
    for deviating_fid in range(NUM_FIRMS):
        obs = env.reset()
        memories = None if args.ppo_parity else fresh_memories(args)
        latest_state = None if args.ppo_parity else initial_latest_state(benchmarks)
        last_actions = {f: competitive_default_mw(env, f) for f in range(NUM_FIRMS)}

        # 1. Warm up to a resting point.
        for k in range(warmup):
            obs, _r, done, info, actions_mw, _p = play_period(
                engine, env, obs, benchmarks, args, schemas, last_actions,
                memories=memories, latest_state=latest_state,
                seed=base_seed + 1000 * (deviating_fid + 1) + k, period=k,
            )
            if done:
                obs = env.reset()
                continue
            last_actions = {f: actions_mw[f].copy() for f in range(NUM_FIRMS)}

        resting = {str(f): firm_total_gen(
            {pidx: last_actions[f][j] for j, pidx in enumerate(FIRM_PLANT_IDX[f])}, f
        ) for f in range(NUM_FIRMS)}

        trace_gen = {str(f): [] for f in range(NUM_FIRMS)}
        trace_lmp = []

        # 2. Deviation period.
        obs, _r, done, info, actions_mw, _p = play_period(
            engine, env, obs, benchmarks, args, schemas, last_actions,
            memories=memories, latest_state=latest_state,
            seed=base_seed + 1000 * (deviating_fid + 1) + warmup, period=warmup,
            force=(deviating_fid, 1.0 + deviation_frac),
        )
        if done:
            obs = env.reset()
        else:
            last_actions = {f: actions_mw[f].copy() for f in range(NUM_FIRMS)}
        for f in range(NUM_FIRMS):
            trace_gen[str(f)].append(firm_total_gen(info.get("gen", {}), f))
        trace_lmp.append(float(info.get("avg_lmp", 0.0)))

        # 3. Post-deviation: both play their normal LLM policy.
        for k in range(horizon):
            obs, _r, done, info, actions_mw, _p = play_period(
                engine, env, obs, benchmarks, args, schemas, last_actions,
                memories=memories, latest_state=latest_state,
                seed=base_seed + 1000 * (deviating_fid + 1) + warmup + 1 + k,
                period=warmup + 1 + k,
            )
            if done:
                obs = env.reset()
                continue
            last_actions = {f: actions_mw[f].copy() for f in range(NUM_FIRMS)}
            for f in range(NUM_FIRMS):
                trace_gen[str(f)].append(firm_total_gen(info.get("gen", {}), f))
            trace_lmp.append(float(info.get("avg_lmp", 0.0)))

        results[str(deviating_fid)] = {
            "resting": resting,
            "gen": trace_gen,
            "lmp": trace_lmp,
        }
    return results


# ---------------------------------------------------------------------------
# Limit strategy (reaction function) — parity mode only
# ---------------------------------------------------------------------------
def compute_limit_strategy_llm(
    env, engine, benchmarks, args, schemas, base_seed, num_points=12,
):
    """Sweep average LMP; record each firm's chosen total generation.

    Reuses ppo's synthetic-observation builders so the LLM sees the same kind of
    state the PPO limit-strategy sweep uses. Only meaningful in ppo_parity mode
    (the state is the 19-number observation); returns ``{}`` otherwise.
    """
    if not args.ppo_parity:
        return {}
    from experiments.ppo import (
        _public_vector_with_scaled_lmps,
        _per_firm_reference_obs,
    )

    lmp_grid = np.linspace(15.0, 38.0, num_points)
    strategies = {str(f): [] for f in range(NUM_FIRMS)}
    defaults = {f: competitive_default_mw(env, f) for f in range(NUM_FIRMS)}

    for gi, target in enumerate(lmp_grid):
        public_vec = _public_vector_with_scaled_lmps(env, float(target))
        obs = {f: _per_firm_reference_obs(env, public_vec, f) for f in range(NUM_FIRMS)}
        batch = build_batch_messages(env, obs, benchmarks, args, None, None)
        completions = engine.chat(batch, schemas=schemas, seed=base_seed + gi)
        for f in range(NUM_FIRMS):
            parsed = parse_action(completions[f], f, default_mw=defaults[f])
            strategies[str(f)].append(float(np.sum(parsed["mw"])))

    return {"lmp_grid": lmp_grid.tolist(), "strategies": strategies}
