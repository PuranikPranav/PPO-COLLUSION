"""
Sanity-trend plots for a multi-session PPO run.

Three figures — LMP, generation (both firms), profit (both firms) — with:
  * x-axis = environment time steps
  * center line = MEAN ACROSS SESSIONS of the instantaneous realized value at
    each logged step (NO within-session averaging over past m episodes)
  * shaded band = the within-rollout SAMPLED range (lo..hi) averaged across
    sessions == how widely each firm explored that rollout. Wide early
    (agents know nothing) then narrowing as the policy settles.
  * dashed reference lines = Competitive / Cournot-Nash / Monopoly benchmarks.

Usage:
    python experiments/plot_sanity_trend.py results/sanity_trend --save figures/sanity_trend
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import FIRM_PLANT_IDX, NUM_FIRMS

FIRM_COLORS = ["#1f77b4", "#ff7f0e"]
BENCH_STYLE = {
    "competitive": ("#2ca02c", (0, (5, 4)), "Competitive"),
    "cournot_nash": ("#7f7f7f", (0, (3, 3)), "Cournot-Nash"),
    "monopoly": ("#d62728", (0, (1, 2)), "Monopoly"),
}


def load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    sessions = []
    sess_root = run_dir / "sessions"
    for sdir in sorted(sess_root.glob("session_*"), key=lambda p: int(p.name.split("_")[1])):
        sj = sdir / "session.json"
        if sj.exists():
            sessions.append(json.loads(sj.read_text())["metrics"])
    if not sessions:
        raise SystemExit(f"No sessions found under {sess_root}")
    return config, sessions


def _common_steps(sessions):
    """Steps present in every session (they share the logging schedule)."""
    step_sets = [{row["step"] for row in metrics} for metrics in sessions]
    common = sorted(set.intersection(*step_sets))
    return np.array(common, dtype=float)


def _stack(sessions, steps, key, fallback_key=None):
    """Matrix [n_sessions x n_steps] for `key`, aligned on `steps`.

    Missing values fall back to `fallback_key` (e.g. lo/hi -> center at the
    t=0 anchor row) or NaN.
    """
    out = np.full((len(sessions), len(steps)), np.nan)
    step_index = {s: i for i, s in enumerate(steps)}
    for si, metrics in enumerate(sessions):
        for row in metrics:
            j = step_index.get(float(row["step"]))
            if j is None:
                continue
            if key in row and row[key] is not None:
                out[si, j] = float(row[key])
            elif fallback_key is not None and fallback_key in row:
                out[si, j] = float(row[fallback_key])
    return out


def _bench_total_gen(bench, key):
    return float(sum(bench[key]["gens"]))


def _bench_firm_gen(bench, key, fid):
    return float(sum(bench[key]["gens"][pidx] for pidx in FIRM_PLANT_IDX[fid]))


def _bench_firm_profit(bench, key, fid):
    return float(bench[key]["profits"][str(fid)])


def _plot_series(ax, steps, center_mat, lo_mat, hi_mat, color, label):
    """Mean-across-sessions center line + exploration band (mean lo..hi)."""
    center = np.nanmean(center_mat, axis=0)
    lo = np.nanmean(lo_mat, axis=0)
    hi = np.nanmean(hi_mat, axis=0)
    ax.fill_between(steps, lo, hi, color=color, alpha=0.18, lw=0,
                    label=f"{label}: exploration range (sampled lo-hi)")
    ax.plot(steps, center, color=color, lw=2.2, label=f"{label}: mean over sessions", zorder=4)


def _bench_lines(ax, bench, value_fn):
    for key, (color, dash, name) in BENCH_STYLE.items():
        if key in bench:
            ax.axhline(value_fn(key), color=color, ls=dash, lw=1.4, alpha=0.9, label=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    config, sessions = load_run(args.run_dir)
    bench = config.get("benchmarks", {})
    n_sess = len(sessions)
    steps = _common_steps(sessions)
    sub = f"{n_sess} sessions  ·  mean across sessions, shaded = within-session exploration range"

    save_dir = Path(args.save) if args.save else args.run_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. LMP ----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    center = _stack(sessions, steps, "avg_lmp")
    lo = _stack(sessions, steps, "lmp_lo", fallback_key="avg_lmp")
    hi = _stack(sessions, steps, "lmp_hi", fallback_key="avg_lmp")
    _plot_series(ax, steps, center, lo, hi, "#6a2c91", "Avg LMP")
    _bench_lines(ax, bench, lambda k: bench[k]["avg_lmp"])
    ax.set_title("Average LMP vs time steps\n" + sub, fontsize=11)
    ax.set_xlabel("Environment time steps")
    ax.set_ylabel("Average LMP ($/MWh)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "lmp_trend.png", dpi=160)
    plt.close(fig)

    # ---------------- 2. Generation (both firms) ----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for fid in range(NUM_FIRMS):
        c = _stack(sessions, steps, f"firm_{fid}_avg_gen")
        lo = _stack(sessions, steps, f"firm_{fid}_gen_lo", fallback_key=f"firm_{fid}_avg_gen")
        hi = _stack(sessions, steps, f"firm_{fid}_gen_hi", fallback_key=f"firm_{fid}_avg_gen")
        _plot_series(ax, steps, c, lo, hi, FIRM_COLORS[fid], f"Firm {fid}")
    # Benchmark TOTAL-generation references (context for combined output level).
    for key, (color, dash, name) in BENCH_STYLE.items():
        if key in bench:
            ax.axhline(_bench_total_gen(bench, key), color=color, ls=dash, lw=1.2,
                       alpha=0.8, label=f"{name} (total)")
    ax.set_title("Firm generation vs time steps\n" + sub, fontsize=11)
    ax.set_xlabel("Environment time steps")
    ax.set_ylabel("Total generation (MW)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(save_dir / "generation_trend.png", dpi=160)
    plt.close(fig)

    # ---------------- 3. Profit (both firms) ----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for fid in range(NUM_FIRMS):
        c = _stack(sessions, steps, f"firm_{fid}_avg_step_profit")
        lo = _stack(sessions, steps, f"firm_{fid}_profit_lo", fallback_key=f"firm_{fid}_avg_step_profit")
        hi = _stack(sessions, steps, f"firm_{fid}_profit_hi", fallback_key=f"firm_{fid}_avg_step_profit")
        _plot_series(ax, steps, c, lo, hi, FIRM_COLORS[fid], f"Firm {fid}")
        for key, (color, dash, name) in BENCH_STYLE.items():
            if key in bench:
                ax.axhline(_bench_firm_profit(bench, key, fid), color=FIRM_COLORS[fid],
                           ls=dash, lw=1.0, alpha=0.6)
    ax.set_title("Per-firm profit vs time steps ($/step)\n" + sub
                 + "\n(thin dashed = each firm's Competitive/Nash/Monopoly profit)", fontsize=10)
    ax.set_xlabel("Environment time steps")
    ax.set_ylabel("Profit ($/step)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(save_dir / "profit_trend.png", dpi=160)
    plt.close(fig)

    print(f"Saved 3 figures to {save_dir}/")
    for name in ("lmp_trend.png", "generation_trend.png", "profit_trend.png"):
        print("  ", save_dir / name)


if __name__ == "__main__":
    main()
