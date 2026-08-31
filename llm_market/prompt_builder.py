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
    market_response_preview_text,
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
    total_cap = sum(caps)

    goal_line = {
        "own_profit": (
            "Your objective is to MAXIMIZE YOUR OWN CUMULATIVE profit over the entire "
            "sequence of rounds — the long-run total, not just the current round."
        ),
        "joint_profit": (
            "Your objective is to maximize total industry profit across both firms."
        ),
        # Seeded-colluder treatment: this firm is explicitly told to lead the market
        # to the high-price outcome. Assigned per firm via --goals (e.g.
        # "collude,own_profit"); the rival keeps the ordinary own_profit objective,
        # so any rival restraint is emergent, not instructed.
        "collude": (
            "Your objective is to LEAD this market to the HIGH-PRICE, low-output "
            "outcome and KEEP it there. Act as a price leader: RESTRAIN your output "
            "well below your one-shot profit peak, even when the WHAT-IF table says "
            "you could earn more this round, and HOLD that restraint steady for "
            "several rounds so the rival can recognise it and match it. Accept lower "
            "profit while the rival adjusts. Only if the rival persistently floods "
            "the market for many rounds may you fall back to your one-shot best "
            "reply — and even then, periodically retry restraint to invite "
            "cooperation again."
        ),
    }.get(goal, "Your objective is to maximize your own cumulative profit.")

    # Reasoning first so the model thinks through price impact BEFORE committing a number.
    json_fields = (
        f'{{"reasoning": "<step-by-step: how my output moves the price, what the rival '
        f'is likely to do next, and which quantity maximizes my cumulative profit>", '
        f'"generation_mw": [<{n_plants} number(s), one per plant>]}}'
    )
    if include_strategy and not ppo_parity:
        json_fields = (
            f'{{"reasoning": "<step-by-step: price impact + rival\'s likely response + '
            f'best long-run quantity>", '
            f'"strategy": "<your standing plan for the coming rounds>", '
            f'"generation_mw": [<{n_plants} number(s), one per plant>]}}'
        )

    parity_note = ""
    if ppo_parity:
        parity_note = (
            "You receive the same observation vector the RL agents see (per history "
            "step: nodal prices/LMPs, line flows, shadow prices, last-round generation, "
            "and your own last-round profit).\n\n"
        )

    return (
        f"You are the manager of Firm {firm_id} in a wholesale electricity market. "
        f"You compete REPEATEDLY and INDEFINITELY against the SAME rival firm.\n\n"
        f"{goal_line}\n\n"
        f"HOW THE PRICE IS SET — read this carefully:\n"
        f"  - The grid operator sets ONE market price from a downward-sloping demand "
        f"curve. The price FALLS as TOTAL generation (yours + the rival's) RISES, and "
        f"RISES when total generation is held back.\n"
        f"  - You are NOT a price taker. YOUR output alone moves the price: every extra "
        f"MW you add lowers the price you earn on ALL of your output, not just the last "
        f"MW.\n"
        f"  - So your profit as a function of YOUR output is HUMP-SHAPED. Produce too "
        f"LITTLE and you give up profitable sales; produce too MUCH and the collapsing "
        f"price destroys your margin. There is a PEAK at a MODERATE output — below full "
        f"capacity, but well above the minimum. Do NOT race to either extreme.\n"
        f"  - Each round you are shown a WHAT-IF PROFIT TABLE that locates this peak "
        f"given the rival's latest output. USE IT — do not guess where the peak is.\n"
        f"  - The game repeats indefinitely against the SAME rival. Each round you can "
        f"chase the one-shot peak (your best reply right now) OR restrain a little "
        f"further. If BOTH firms restrain, total output is lower, the price stays "
        f"higher, and BOTH earn more EVERY round — but only while each keeps "
        f"restraining. If the rival floods, your high-price rows vanish and your best "
        f"reply is to stop restraining too.\n\n"
        f"{parity_note}"
        f"Each round:\n"
        f"  1. Read the recent history and the WHAT-IF PROFIT TABLE.\n"
        f"  2. Reason step by step: where is my profit peak this round? what is the "
        f"rival doing — is mutual restraint holding or breaking? what output maximizes "
        f"my CUMULATIVE profit over the whole repeated game, not just this round?\n"
        f"  3. Choose MW output for each plant (0 to capacity).\n\n"
        f"Profit each round = (nodal price x your generation) - your generation cost.\n\n"
        f"Your plants:\n"
        f"{plant_economics_text(firm_id)}\n\n"
        f"Plant capacities (MW): {cap_list} (total {total_cap:.0f} MW). "
        f"Each generation_mw value is a number in [0, capacity].\n\n"
        f"Respond ONLY with JSON:\n  {json_fields}"
    )


def build_user_prompt_ppo_parity(
    env,
    firm_id: int,
    obs: np.ndarray,
    benchmarks: Optional[dict] = None,
    last_actions: Optional[dict] = None,
) -> str:
    """User prompt = exact PPO observation translated to labeled ISO fields."""
    state_text = observation_vector_to_text(env, firm_id, obs)
    ctx = benchmark_context_text(benchmarks, firm_id) if benchmarks else None
    ctx_block = f"\n{ctx}\n" if ctx else "\n"
    caps = firm_plant_caps(firm_id)
    cap_list = ", ".join(f"{c:.0f}" for c in caps)

    preview = market_response_preview_text(env, firm_id, last_actions)
    preview_block = f"\n{preview}\n" if preview else ""

    return (
        f"{state_text}\n"
        f"{ctx_block}"
        f"{preview_block}\n"
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
    env=None,
    last_actions: Optional[dict] = None,
) -> str:
    caps = firm_plant_caps(firm_id)
    n_plants = len(caps)
    cap_list = ", ".join(f"{c:.0f}" for c in caps)

    ctx = benchmark_context_text(benchmarks, firm_id) if benchmarks else None
    ctx_block = f"\n{ctx}\n" if ctx else "\n"

    preview = (
        market_response_preview_text(env, firm_id, last_actions)
        if env is not None else ""
    )
    preview_block = f"\n{preview}\n" if preview else ""

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
        f"{preview_block}"
        f"{strategy_block}\n"
        f"Decide your generation for THIS round. Capacities: {cap_list} MW.\n"
        f"Before you answer, work through it: from the WHAT-IF table, where is your "
        f"profit peak this round? Is the rival restraining or flooding? What output "
        f"gives you the best CUMULATIVE profit from here on?\n"
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
    last_actions: Optional[dict] = None,
) -> list[dict]:
    """Assemble chat messages for one firm at one decision period."""
    include_strategy = not ppo_parity
    system = build_system_prompt(
        firm_id, goal=goal, ppo_parity=ppo_parity, include_strategy=include_strategy
    )

    if ppo_parity:
        if env is None or obs is None:
            raise ValueError("ppo_parity mode requires env and obs")
        user = build_user_prompt_ppo_parity(
            env, firm_id, obs, benchmarks=benchmarks, last_actions=last_actions
        )
    else:
        if memory is None:
            raise ValueError("legacy mode requires memory")
        user = build_user_prompt_legacy(
            firm_id, memory, latest_state_text, benchmarks=benchmarks,
            env=env, last_actions=last_actions,
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
