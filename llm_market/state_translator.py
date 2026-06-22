"""
Step 1 — State Translator: raw market data -> human-readable text.

Turns the ISO clearing outcome (LMPs, generation, profit) into a compact natural
language summary that an LLM agent can reason over. Everything here is plain text;
no model calls.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from iso_market.market_env import PLANTS, FIRM_PLANT_IDX, NUM_FIRMS


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


def _firm_total_gen(gen_per_plant: dict, firm_id: int) -> float:
    return float(sum(gen_per_plant.get(pidx, 0.0) for pidx in FIRM_PLANT_IDX[firm_id]))


def _firm_nodal_price(lmps: np.ndarray, firm_id: int) -> float:
    """Quantity-unweighted mean LMP across the nodes where the firm has plants."""
    nodes = sorted({PLANTS[pidx]["node"] for pidx in FIRM_PLANT_IDX[firm_id]})
    return float(np.mean([lmps[n] for n in nodes]))


def market_outcome_text(info: dict, firm_id: int) -> str:
    """Describe the most recent cleared market from `firm_id`'s perspective.

    `info` is the dict returned by ElectricityMarketEnv.step(): expects keys
    'lmps', 'gen' (per-plant MW), 'avg_lmp'.
    """
    lmps = np.asarray(info["lmps"], dtype=float)
    gen = info.get("gen", {})
    avg_lmp = float(info.get("avg_lmp", float(np.mean(lmps))))

    own_total = _firm_total_gen(gen, firm_id)
    rival_total = sum(
        _firm_total_gen(gen, f) for f in range(NUM_FIRMS) if f != firm_id
    )
    own_price = _firm_nodal_price(lmps, firm_id)

    own_plants = []
    for local_idx, pidx in enumerate(FIRM_PLANT_IDX[firm_id]):
        own_plants.append(f"plant {local_idx} = {gen.get(pidx, 0.0):.1f} MW")

    return (
        f"Last period outcome:\n"
        f"  - Average market price (LMP): ${avg_lmp:.2f}/MWh.\n"
        f"  - Price at your node(s): ${own_price:.2f}/MWh.\n"
        f"  - Your generation: {', '.join(own_plants)} "
        f"(total {own_total:.1f} MW).\n"
        f"  - Competitor total generation: {rival_total:.1f} MW.\n"
    )


def initial_state_text(benchmarks: dict, firm_id: int) -> str:
    """Seed text for the very first period, using the competitive benchmark."""
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
        f"  - Your generation so far: {own_total:.1f} MW (running at full output).\n"
        f"  - Competitor total generation: {rival_total:.1f} MW.\n"
    )


def benchmark_context_text(benchmarks: dict, firm_id: int) -> Optional[str]:
    """Optional orientation: rough price range the firm has seen historically.

    Deliberately does NOT reveal the collusion/monopoly target — only the
    competitive and single-shot Nash price levels, framed as observed history,
    so emergent behavior is not contaminated by instructing the agent to collude.
    """
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
