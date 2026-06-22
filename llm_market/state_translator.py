"""
Step 1 — State Translator: PPO observation vector -> prompt text.

The LLM agents see the SAME information as the PPO policy network: the flattened
observation from ``ElectricityMarketEnv._get_obs()``. For the default setup
(history_len=1, past_gen=True, prev_reward=True) that is 19 numbers per firm:

  LMPs (5) + line flows (5) + shadow prices (5) + past plant generation (3)
  + your previous-period profit (1).

Each number is an ISO clearing outcome (or lagged action/reward), translated here
into labeled text for the Granite model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from iso_market.market_env import (
    PLANTS,
    FIRM_PLANT_IDX,
    NUM_FIRMS,
    NUM_NODES,
    OBS_MARKET_FEATURES_PER_STEP,
)


def plant_economics_text(firm_id: int) -> str:
    """Static description of the plants a firm controls (capacity + cost curve)."""
    lines = []
    for local_idx, pidx in enumerate(FIRM_PLANT_IDX[firm_id]):
        p = PLANTS[pidx]
        lines.append(
            f"  - Plant {local_idx} (grid node {p['node'] + 1}): "
            f"capacity {p['cap']:.0f} MW, "
            f"marginal cost ${p['mc']:.0f}/MWh, "
            f"quadratic cost coefficient {p['qc']:.3f} "
            f"(total cost = {p['mc']:.0f}*g + 0.5*{p['qc']:.3f}*g^2)."
        )
    return "\n".join(lines)


def firm_plant_caps(firm_id: int) -> list[float]:
    return [float(PLANTS[pidx]["cap"]) for pidx in FIRM_PLANT_IDX[firm_id]]


def firm_total_cap(firm_id: int) -> float:
    return float(sum(firm_plant_caps(firm_id)))


def competitive_default_mw(env, firm_id: int) -> np.ndarray:
    """Competitive-baseline generation per plant (matches PPO init_policy=competitive)."""
    gens = getattr(env, "_baseline_gens", None)
    if gens is None:
        return 0.5 * np.array(firm_plant_caps(firm_id), dtype=float)
    return np.array(
        [float(gens[pidx]) for pidx in FIRM_PLANT_IDX[firm_id]], dtype=float
    )


def _decode_obs_step_segment(segment: np.ndarray, env) -> dict:
    """Decode one interleaved row of length ``env.features_per_step``."""
    idx = 0
    num_lines = env.num_lines
    lmps = segment[idx : idx + NUM_NODES]
    idx += NUM_NODES
    flows = segment[idx : idx + num_lines]
    idx += num_lines
    shadow = segment[idx : idx + num_lines]
    idx += num_lines

    past_gen = None
    if env.include_past_gen:
        past_gen = segment[idx : idx + len(PLANTS)]
        idx += len(PLANTS)

    prev_reward = None
    if env.include_prev_reward:
        prev_reward = float(segment[idx])

    return {
        "lmps": lmps,
        "flows": flows,
        "shadow_prices": shadow,
        "past_gen": past_gen,
        "prev_reward": prev_reward,
    }


def _format_obs_step_text(step_idx: int, decoded: dict, firm_id: int) -> str:
    """Render one historical observation step as labeled ISO fields."""
    lmps = decoded["lmps"]
    flows = decoded["flows"]
    shadow = decoded["shadow_prices"]
    avg_lmp = float(np.mean(lmps))

    lmp_lines = ", ".join(
        f"node {n + 1}=${float(lmps[n]):.4f}/MWh" for n in range(NUM_NODES)
    )
    flow_lines = ", ".join(
        f"line {i + 1}={float(flows[i]):.4f} MW" for i in range(len(flows))
    )
    shadow_lines = ", ".join(
        f"line {i + 1}=${float(shadow[i]):.4f}/MWh" for i in range(len(shadow))
    )

    lines = [
        f"History step {step_idx + 1} (ISO clearing outcomes):",
        f"  Locational marginal prices (LMP, $/MWh): {lmp_lines}",
        f"  Average LMP: ${avg_lmp:.4f}/MWh",
        f"  Line flows (MW): {flow_lines}",
        f"  Transmission shadow prices ($/MWh): {shadow_lines}",
    ]

    if decoded["past_gen"] is not None:
        gen_parts = []
        for pidx, g in enumerate(decoded["past_gen"]):
            plant = PLANTS[pidx]
            gen_parts.append(
                f"plant {pidx} (firm {plant['firm']}, node {plant['node'] + 1})="
                f"{float(g):.4f} MW"
            )
        lines.append(f"  Past-period plant generation: {', '.join(gen_parts)}")

        own_idxs = FIRM_PLANT_IDX[firm_id]
        own_parts = [
            f"your plant {j}={float(decoded['past_gen'][pidx]):.4f} MW"
            for j, pidx in enumerate(own_idxs)
        ]
        lines.append(f"  Your plants last period: {', '.join(own_parts)}")

    if decoded["prev_reward"] is not None:
        lines.append(
            f"  Your profit in that period: ${decoded['prev_reward']:.4f}"
        )

    return "\n".join(lines)


def observation_vector_to_text(env, firm_id: int, obs: np.ndarray) -> str:
    """Translate the exact PPO observation vector into the LLM user prompt body.

    ``obs`` must be the firm-specific vector returned by ``env._get_obs()[firm_id]``.
    Layout matches ``ElectricityMarketEnv._get_obs()``: for each history step,
    [public market block, own previous reward] interleaved and flattened.
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    expected = env.obs_dim
    if obs.size != expected:
        raise ValueError(
            f"obs size {obs.size} != env.obs_dim {expected} for firm {firm_id}"
        )

    fps = env.features_per_step

    blocks = []
    for h in range(env.history_len):
        segment = obs[h * fps : (h + 1) * fps]
        decoded = _decode_obs_step_segment(segment, env)
        blocks.append(_format_obs_step_text(h, decoded, firm_id))

    header = (
        f"Your observation state ({env.obs_dim} values, "
        f"history_len={env.history_len}, features/step={fps}):\n"
        f"This is the same information the reinforcement-learning agents receive.\n"
        f"Each value comes from the ISO DC-OPF clearing (prices, flows, shadow "
        f"prices) or from lagged actions/rewards.\n"
    )
    return header + "\n\n".join(blocks)


def market_response_preview_text(
    env, firm_id: int, last_actions: dict, num_points: int = 8
) -> str:
    """A 'what-if' profit table: hold the rival at its last output, vary YOUR total
    output, and show the resulting price and YOUR profit at each level.

    This is the one piece of information a frozen model otherwise lacks — the SHAPE
    of its own profit curve. It is computed by re-clearing the market for each
    candidate output (the same DC-OPF the ISO runs), so the numbers are the market's
    real response, not a guess. The curve is hump-shaped: too little output and you
    sell too few MW; too much and you crush the price. The peak is the best reply to
    the rival's CURRENT output. Output is split across your plants in proportion to
    capacity (an approximation — you may split differently).

    Returns "" if ``last_actions`` is None (e.g. the synthetic limit-strategy sweep).
    """
    if last_actions is None:
        return ""

    own_idxs = FIRM_PLANT_IDX[firm_id]
    own_caps = np.array([PLANTS[p]["cap"] for p in own_idxs], dtype=float)
    own_total_cap = float(own_caps.sum())
    grid = np.linspace(0.15 * own_total_cap, own_total_cap, num_points)

    rows = []
    for target_total in grid:
        alloc = own_caps * (target_total / own_total_cap)
        gen_per_node = np.zeros(NUM_NODES)
        for fid, acts in last_actions.items():
            idxs = FIRM_PLANT_IDX[fid]
            use = alloc if fid == firm_id else np.asarray(acts, dtype=float)
            for j, pidx in enumerate(idxs):
                g = min(float(use[j]), PLANTS[pidx]["cap"])
                gen_per_node[PLANTS[pidx]["node"]] += g
        lmps, _d, _f, _s = env._clear_market(gen_per_node)
        if lmps is None:
            continue
        profit = 0.0
        for j, pidx in enumerate(own_idxs):
            p = PLANTS[pidx]
            g = min(float(alloc[j]), p["cap"])
            profit += lmps[p["node"]] * g - p["mc"] * g - 0.5 * p["qc"] * g * g
        own_price = float(np.mean([lmps[PLANTS[pidx]["node"]] for pidx in own_idxs]))
        rows.append((float(target_total), own_price, float(profit)))

    if not rows:
        return ""

    best = max(range(len(rows)), key=lambda i: rows[i][2])
    rival_total = sum(
        float(np.sum(last_actions[fid])) for fid in last_actions if fid != firm_id
    )

    lines = [
        f"WHAT-IF PROFIT TABLE (assumes the rival holds ~{rival_total:.0f} MW, its "
        f"most recent output):",
    ]
    for i, (tot, price, profit) in enumerate(rows):
        mark = "   <-- your best reply (peak profit)" if i == best else ""
        lines.append(
            f"  produce ~{tot:6.0f} MW  ->  price ~${price:5.1f}/MWh  ->  "
            f"your profit ~${profit:8.0f}{mark}"
        )
    lines.append(
        "  Read the shape: profit rises, peaks, then falls. The peak is your best "
        "reply to the rival RIGHT NOW. If the rival cuts output, every row's price "
        "and your profit go UP; if it floods, they go DOWN. Because the game repeats "
        "forever, restraining a little BELOW the peak — when the rival restrains too "
        "— keeps the price higher and earns BOTH of you more, round after round."
    )
    return "\n".join(lines)


def benchmark_context_text(benchmarks: dict, firm_id: int) -> Optional[str]:
    """Optional orientation: rough price range (competitive to Nash only)."""
    comp = benchmarks.get("competitive", {})
    nash = benchmarks.get("cournot_nash", {})
    comp_lmp = comp.get("avg_lmp")
    nash_lmp = nash.get("avg_lmp")
    if comp_lmp is None or nash_lmp is None:
        return None
    lo, hi = sorted([float(comp_lmp), float(nash_lmp)])
    return (
        f"For reference, average prices in this market have historically ranged "
        f"roughly between ${lo:.0f} and ${hi:.0f}/MWh depending on how much total "
        f"power is supplied."
    )


# ---- Legacy narrative helpers (used only with --legacy-memory) ----

def _firm_total_gen(gen_per_plant: dict, firm_id: int) -> float:
    return float(sum(gen_per_plant.get(pidx, 0.0) for pidx in FIRM_PLANT_IDX[firm_id]))


def market_outcome_text(info: dict, firm_id: int) -> str:
    lmps = np.asarray(info["lmps"], dtype=float)
    gen = info.get("gen", {})
    avg_lmp = float(info.get("avg_lmp", float(np.mean(lmps))))
    own_total = _firm_total_gen(gen, firm_id)
    rival_total = sum(
        _firm_total_gen(gen, f) for f in range(NUM_FIRMS) if f != firm_id
    )
    return (
        f"Last period outcome:\n"
        f"  - Average market price (LMP): ${avg_lmp:.2f}/MWh.\n"
        f"  - Your total generation: {own_total:.1f} MW.\n"
        f"  - Competitor total generation: {rival_total:.1f} MW.\n"
    )


def initial_state_text(benchmarks: dict, firm_id: int) -> str:
    comp = benchmarks.get("competitive", {})
    avg_lmp = float(comp.get("avg_lmp", 0.0))
    gens = comp.get("gens", [])
    own_total = sum(gens[pidx] for pidx in FIRM_PLANT_IDX[firm_id]) if gens else 0.0
    rival_total = (
        sum(
            gens[pidx]
            for f in range(NUM_FIRMS)
            if f != firm_id
            for pidx in FIRM_PLANT_IDX[f]
        )
        if gens
        else 0.0
    )
    return (
        f"The market is starting from the competitive baseline:\n"
        f"  - Average market price (LMP): ${avg_lmp:.2f}/MWh.\n"
        f"  - Your generation: {own_total:.1f} MW.\n"
        f"  - Competitor total generation: {rival_total:.1f} MW.\n"
    )
