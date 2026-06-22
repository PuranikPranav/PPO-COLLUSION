"""
Step 4 — Action Parser: model JSON text -> validated MW array per plant.

Robust to imperfect model output: tries strict JSON first, then extracts the first
balanced ``{...}`` block, then falls back to a safe default. Values are coerced to
floats, padded/truncated to the firm's plant count, and clipped to [0, capacity].
"""

from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

from iso_market.market_env import FIRM_PLANT_IDX, PLANTS


def action_json_schema(firm_id: int) -> dict:
    """JSON schema for one firm's action (used for vLLM structured/guided decoding)."""
    n = len(FIRM_PLANT_IDX[firm_id])
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "generation_mw": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": n,
                "maxItems": n,
            },
        },
        "required": ["reasoning", "generation_mw"],
        "additionalProperties": False,
    }


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_numbers(value) -> list[float]:
    """Coerce a parsed generation_mw value into a list of floats."""
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    # last resort: pull all numbers out of a string
    nums = re.findall(r"-?\d+\.?\d*", str(value))
    return [float(n) for n in nums]


def firm_caps(firm_id: int) -> np.ndarray:
    return np.array([PLANTS[pidx]["cap"] for pidx in FIRM_PLANT_IDX[firm_id]], dtype=float)


def parse_action(text: str, firm_id: int,
                 default_mw: Optional[np.ndarray] = None) -> dict:
    """Parse one model response into a validated MW array.

    Returns a dict: {"mw": np.ndarray(n_plants), "reasoning": str, "parse_ok": bool}.
    On total failure, falls back to `default_mw` (or half-capacity).
    """
    caps = firm_caps(firm_id)
    n = len(caps)
    fallback = (
        np.asarray(default_mw, dtype=float) if default_mw is not None else 0.5 * caps
    )

    obj = None
    reasoning = ""
    parse_ok = False

    candidate = text if text else ""
    for attempt in (candidate, _extract_first_json_object(candidate) or ""):
        if not attempt:
            continue
        try:
            obj = json.loads(attempt)
            break
        except (json.JSONDecodeError, TypeError):
            obj = None

    if isinstance(obj, dict):
        reasoning = str(obj.get("reasoning", "")).strip()
        raw = obj.get("generation_mw", obj.get("generation", obj.get("mw")))
        nums = _coerce_numbers(raw) if raw is not None else []
        if nums:
            parse_ok = True
            mw = np.array(nums, dtype=float)
            if len(mw) < n:
                mw = np.concatenate([mw, fallback[len(mw):]])
            elif len(mw) > n:
                mw = mw[:n]
        else:
            mw = fallback.copy()
    else:
        mw = fallback.copy()

    mw = np.clip(mw, 0.0, caps)
    return {"mw": mw, "reasoning": reasoning, "parse_ok": parse_ok}
