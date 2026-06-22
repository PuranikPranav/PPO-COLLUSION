"""
Step 2 — Prompt builder: PPO observation -> chat messages.

Default mode (ppo_parity=True) mirrors the PPO agents exactly:
  - One prompt per firm built from ``env._get_obs()[firm_id]`` (19 values when H=1).
  - JSON response with floating-point ``generation_mw`` clipped to plant capacity.
  - No separate narrative memory or strategy note.

Legacy mode (ppo_parity=False) keeps the older sliding-window memory + strategy note.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from llm_market.state_translator import (
    plant_economics_text,
    firm_plant_caps,
    observation_vector_to_text,
    benchmark_context_text,
)


class AgentMemory:
    """Legacy sliding-window memory (used only when ``ppo_parity=False``)."""

    def __init__(self, window: int = 10):
        self.window = window
        self.entries: deque[dict] = deque(maxlen=window)
        self.cumulative_profit: float = 0.0
        self.latest_strategy: str = ""

    def add(self, period: int, own_gen_total: float, own_plant_gen: list[float],
            price: float, profit: float, rival_total: float):
        self.cumulative_profit += float(profit)
        self.entries.append({
            "period": period,
            "own_gen_total": own_gen_total,
            "own_plant_gen": list(own_plant_gen),
            "price": price,
            "profit": profit,
            "rival_total": rival_total,
        })

    def set_strategy(self, strategy: str):
        if strategy and strategy.strip():
            self.latest_strategy = strategy.strip()

    def as_text(self) -> str:
        if not self.entries:
            return "  (no past periods yet — this is the first decision.)"
        lines = []
        for e in self.entries:
            plant_str = ", ".join(f"{g:.0f}" for g in e["own_plant_gen"])
            lines.append(
                f"  period {e['period']:>3d}: you produced [{plant_str}] MW "
                f"(total {e['own_gen_total']:.0f}), competitor produced "
                f"{e['rival_total']:.0f} MW, price ${e['price']:.2f}/MWh, "
                f"your profit ${e['profit']:.0f}."
            )
        lines.append(
            f"  --> Your CUMULATIVE profit so far: ${self.cumulative_profit:.0f}."
        )
        return "\n".join(lines)


def build_system_prompt(
    firm_id: int,
    goal: str = "own_profit",
    *,
    ppo_parity: bool = True,
    include_strategy: bool = False,
) -> str:
    caps = firm_plant_caps(firm_id)
    n_plants = len(caps)
    cap_list = ", ".join(f"{c:.0f}" for c in caps)

    goal_line = {
        "own_profit": (
            "Your objective is to MAXIMIZE YOUR OWN profit over repeated interactions "
            "with the same competitor."
        ),
        "joint_profit": (
            "Your objective is to maximize total industry profit across both firms."
        ),
    }.get(goal, "Your objective is to maximize your own profit.")

    json_fields = (
        f'{{"reasoning": "<brief explanation>", '
        f'"generation_mw": [<{n_plants} floating-point MW value(s), one per plant>]}}'
    )
    if include_strategy and not ppo_parity:
        json_fields = (
            f'{{"reasoning": "<brief explanation>", '
            f'"strategy": "<standing plan>", '
            f'"generation_mw": [<{n_plants} floating-point MW value(s), one per plant>]}}'
        )

    parity_note = ""
    if ppo_parity:
        parity_note = (
            "You receive the same 19-number observation state as the RL agents "
            "(LMPs, line flows, shadow prices, past generation, your previous profit). "
            "Choose generation based ONLY on that state.\n\n"
        )

    return (
        f"You are Firm {firm_id} in a repeated wholesale electricity market.\n\n"
        f"{goal_line}\n\n"
        f"{parity_note}"
        f"Each period:\n"
        f"  1. You observe the ISO clearing outcomes from recent history.\n"
        f"  2. You choose MW output for each plant (0 to capacity).\n"
        f"  3. The ISO clears the market; more total supply lowers LMP.\n"
        f"  4. Profit = (nodal LMP x generation) - generation cost.\n\n"
        f"Your plants:\n"
        f"{plant_economics_text(firm_id)}\n\n"
        f"Plant capacities (MW): {cap_list}. "
        f"Each generation_mw value must be a floating-point number in [0, capacity].\n\n"
        f"Respond ONLY with JSON:\n  {json_fields}"
    )


def build_user_prompt_ppo_parity(
    env,
    firm_id: int,
    obs: np.ndarray,
    benchmarks: Optional[dict] = None,
) -> str:
    """User prompt = exact PPO observation translated to labeled ISO fields."""
    state_text = observation_vector_to_text(env, firm_id, obs)
    ctx = benchmark_context_text(benchmarks, firm_id) if benchmarks else None
    ctx_block = f"\n{ctx}\n" if ctx else "\n"
    caps = firm_plant_caps(firm_id)
    cap_list = ", ".join(f"{c:.0f}" for c in caps)

    return (
        f"{state_text}\n"
        f"{ctx_block}\n"
        f"Choose generation for THIS period for your {len(caps)} plant(s) "
        f"(capacities: {cap_list} MW).\n"
        f'Respond ONLY with JSON: '
        f'{{"reasoning": "...", "generation_mw": [{", ".join("..." for _ in caps)}]}}'
    )


def build_user_prompt_legacy(
    firm_id: int,
    memory: AgentMemory,
    latest_state_text: str,
    benchmarks: Optional[dict] = None,
) -> str:
    caps = firm_plant_caps(firm_id)
    n_plants = len(caps)
    cap_list = ", ".join(f"{c:.0f}" for c in caps)

    ctx = benchmark_context_text(benchmarks, firm_id) if benchmarks else None
    ctx_block = f"\n{ctx}\n" if ctx else "\n"

    if memory.latest_strategy:
        strategy_block = (
            f"\nYour standing strategy from last period:\n"
            f'  "{memory.latest_strategy}"\n'
        )
    else:
        strategy_block = ""

    return (
        f"{latest_state_text}"
        f"{ctx_block}"
        f"\nYour recent history (most recent last):\n"
        f"{memory.as_text()}\n"
        f"{strategy_block}\n"
        f"Decide your generation for THIS period. Capacities: {cap_list} MW.\n"
        f'Respond ONLY with JSON: '
        f'{{"reasoning": "...", "strategy": "...", '
        f'"generation_mw": [{", ".join("..." for _ in caps)}]}}'
    )


def build_messages(
    firm_id: int,
    *,
    env=None,
    obs: Optional[np.ndarray] = None,
    memory: Optional[AgentMemory] = None,
    latest_state_text: str = "",
    benchmarks: Optional[dict] = None,
    goal: str = "own_profit",
    ppo_parity: bool = True,
) -> list[dict]:
    """Assemble chat messages for one firm at one decision period."""
    include_strategy = not ppo_parity
    system = build_system_prompt(
        firm_id, goal=goal, ppo_parity=ppo_parity, include_strategy=include_strategy
    )

    if ppo_parity:
        if env is None or obs is None:
            raise ValueError("ppo_parity mode requires env and obs")
        user = build_user_prompt_ppo_parity(env, firm_id, obs, benchmarks=benchmarks)
    else:
        if memory is None:
            raise ValueError("legacy mode requires memory")
        user = build_user_prompt_legacy(
            firm_id, memory, latest_state_text, benchmarks=benchmarks
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
