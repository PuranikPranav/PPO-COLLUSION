"""
Calvano-style plots for PPO collusion experiments.

Per run directory (2x3 figure):
  1. Generation evolution
  2. Δ (normalized profit gain)
  3. Policy KL divergence (old → new after each PPO update)
  4. Learned limit strategy
  5–6. Impulse responses (each firm deviates)

Cross-history comparison:
  --compare          → 6-panel dashboard (Δ, LMP, KL, generation) → comparison_h1_2_3.png
  --compare-calvano  → two Calvano-style PNGs (quantity + Δ vs timesteps, H overlaid)
  --compare-generation-profit
                     → one PNG: generation + profit vs PPO iterations, one row per H

Usage
-----
    python experiments/plot_results.py results/h1 --save figures/
    python experiments/plot_results.py results/h1 --save figures/ --calvano-paper
    python experiments/plot_results.py --compare results/h1 results/h2 results/h3 --save figures/
    python experiments/plot_results.py --compare-calvano results/h1 results/h2 results/h3 --save figures/
    python experiments/plot_results.py --compare-generation-profit results/h1 results/h2 results/h3 --save figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_mpl_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ppo-collusion-matplotlib-cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# X-axis ticks for Calvano-style learning curves (environment steps)
CALVANO_XTICKS = np.array([1, 500_000, 1_000_000, 1_500_000, 2_000_000], dtype=float)


def load_sessions(run_dir: Path):
    """Load config + all session data from a run directory."""
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    sessions = []
    sess_dir = run_dir / "sessions"
    if sess_dir.exists():
        for sd in sorted(sess_dir.iterdir()):
            sf = sd / "session.json"
            if sf.exists():
                with open(sf) as f:
                    sessions.append(json.load(f))

    # Backward compat: single-session runs without sessions/ folder
    if not sessions:
        mf = run_dir / "metrics.json"
        if mf.exists():
            with open(mf) as f:
                metrics = json.load(f)
            sessions.append({"metrics": metrics, "final_delta": {}})

    return config, sessions


def _finite_interp_on_steps(ref_steps, steps, vals):
    """Linear interp; non-finite vals replaced via 1d fill before interp."""
    vals = np.asarray(vals, dtype=float)
    steps = np.asarray(steps, dtype=float)
    if len(steps) == 0:
        return np.zeros(len(ref_steps))
    good = np.isfinite(vals)
    if not good.any():
        return np.zeros(len(ref_steps))
    if not good.all():
        idx = np.arange(len(vals))
        vals = np.interp(
            idx,
            idx[good],
            vals[good],
            left=vals[np.argmax(good)],
            right=vals[len(vals) - 1 - np.argmax(good[::-1])],
        )
    return np.interp(
        ref_steps,
        steps,
        vals,
        left=vals[0],
        right=vals[-1],
    )


def aggregate_metric(sessions, key, max_steps=None, default_for_missing=0):
    """Collect a metric across sessions, aligned by step. Returns (steps, mean, std)."""
    all_series = []
    for sess in sessions:
        m = sess.get("metrics") or []
        steps = [r["step"] for r in m]
        vals = [r.get(key, default_for_missing) for r in m]
        all_series.append((steps, vals))

    if not all_series:
        return [], [], []

    # Use the longest session's step grid
    ref_steps = max(all_series, key=lambda x: len(x[0]))[0]
    if max_steps is not None:
        ref_steps = [s for s in ref_steps if s <= max_steps]

    matrix = []
    for steps, vals in all_series:
        interpolated = _finite_interp_on_steps(np.array(ref_steps, dtype=float), steps, vals)
        matrix.append(interpolated)

    matrix = np.array(matrix)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return ref_steps, mean, std


def _metric_keys(sessions):
    keys = set()
    for sess in sessions:
        for row in sess.get("metrics") or []:
            keys.update(row.keys())
    return keys


def _numeric_suffix_sort_key(key: str):
    digits = "".join(ch if ch.isdigit() else " " for ch in key).split()
    return int(digits[-1]) if digits else 10**9


def _aggregate_metric_by_iteration(sessions, key):
    """Collect a metric across sessions, aligned by PPO update number."""
    all_series = []
    for sess in sessions:
        metrics = sess.get("metrics") or []
        xs = []
        vals = []
        for idx, row in enumerate(metrics):
            if key not in row:
                continue
            val = row.get(key)
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            x_val = (
                row.get("ppo_update")
                or row.get("update")
                or row.get("upd")
                or (idx + 1)
            )
            try:
                x_val = float(x_val)
            except (TypeError, ValueError):
                x_val = float(idx + 1)
            xs.append(x_val)
            vals.append(val)
        if xs:
            all_series.append((xs, vals))

    if not all_series:
        return [], [], []

    ref_x = max(all_series, key=lambda item: len(item[0]))[0]
    matrix = []
    ref_arr = np.array(ref_x, dtype=float)
    for xs, vals in all_series:
        matrix.append(_finite_interp_on_steps(ref_arr, xs, vals))

    matrix = np.array(matrix)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return ref_x, mean, std


def _profit_key_for_firm(config, sessions, fid: int):
    """Ensure a per-step firm profit metric exists, reconstructing it from Δ if needed."""
    direct_key = f"firm_{fid}_profit"
    avg_step_key = f"firm_{fid}_avg_step_profit"
    ep_key = f"firm_{fid}_ep_profit"
    delta_key = f"firm_{fid}_delta"
    keys = _metric_keys(sessions)

    if direct_key in keys:
        return direct_key
    if avg_step_key in keys:
        return avg_step_key

    episode_len = float(config.get("episode_len", 1) or 1)
    reconstructed_key = f"firm_{fid}_profit_reconstructed"
    bench = config.get("benchmarks", {})
    comp = bench.get("competitive", {}).get("profits", {})
    mono = bench.get("monopoly", {}).get("profits", {})
    can_reconstruct_from_delta = str(fid) in comp and str(fid) in mono and delta_key in keys

    for sess in sessions:
        for row in sess.get("metrics") or []:
            if ep_key in row:
                row[reconstructed_key] = float(row[ep_key]) / episode_len
            elif can_reconstruct_from_delta and delta_key in row:
                pi_c = float(comp[str(fid)])
                pi_m = float(mono[str(fid)])
                row[reconstructed_key] = pi_c + float(row[delta_key]) * (pi_m - pi_c)

    return reconstructed_key if reconstructed_key in _metric_keys(sessions) else None


def _generation_series_specs(config, sessions):
    keys = _metric_keys(sessions)
    plant_keys = sorted(
        [k for k in keys if k.startswith("plant_") and k.endswith("_avg_gen")],
        key=_numeric_suffix_sort_key,
    )
    if plant_keys:
        return [
            (key, f"Plant {_numeric_suffix_sort_key(key)}", f"C{i}")
            for i, key in enumerate(plant_keys)
        ], "plant"

    firm_keys = sorted(
        [k for k in keys if k.startswith("firm_") and k.endswith("_avg_gen")],
        key=_numeric_suffix_sort_key,
    )
    if firm_keys:
        return [
            (key, f"Firm {_numeric_suffix_sort_key(key)}", f"C{i}")
            for i, key in enumerate(firm_keys)
        ], "firm"

    greedy_keys = sorted(
        [k for k in keys if k.startswith("firm_") and k.endswith("_greedy_gen")],
        key=_numeric_suffix_sort_key,
    )
    return [
        (key, f"Firm {_numeric_suffix_sort_key(key)} greedy", f"C{i}")
        for i, key in enumerate(greedy_keys)
    ], "firm"


def _profit_series_specs(config, sessions):
    keys = _metric_keys(sessions)
    plant_keys = sorted(
        [k for k in keys if k.startswith("plant_") and k.endswith("_profit")],
        key=_numeric_suffix_sort_key,
    )
    if plant_keys:
        return [
            (key, f"Plant {_numeric_suffix_sort_key(key)}", f"C{i}")
            for i, key in enumerate(plant_keys)
        ], "plant"

    specs = []
    for fid in range(2):
        key = _profit_key_for_firm(config, sessions, fid)
        if key:
            specs.append((key, f"Firm {fid}", f"C{fid}"))
    return specs, "firm"


def _draw_generation_benchmarks(ax, config, series_kind):
    bench = config.get("benchmarks", {})
    comp = bench.get("competitive", {}).get("gens")
    mono = bench.get("monopoly", {}).get("gens")
    if not comp or not mono:
        return

    if series_kind == "plant":
        for i, val in enumerate(comp):
            ax.axhline(val, ls="--", color=f"C{i}", alpha=0.22, linewidth=0.8)
        for i, val in enumerate(mono):
            ax.axhline(val, ls=":", color=f"C{i}", alpha=0.28, linewidth=0.9)
        return

    comp_f0 = comp[0] + comp[1]
    comp_f1 = comp[2]
    mono_f0 = mono[0] + mono[1]
    mono_f1 = mono[2]
    for fid, val in enumerate((comp_f0, comp_f1)):
        ax.axhline(val, ls="--", color=f"C{fid}", alpha=0.25, linewidth=0.8)
    for fid, val in enumerate((mono_f0, mono_f1)):
        ax.axhline(val, ls=":", color=f"C{fid}", alpha=0.32, linewidth=0.9)


def _draw_profit_benchmarks(ax, config, series_kind):
    if series_kind != "firm":
        return
    bench = config.get("benchmarks", {})
    comp = bench.get("competitive", {}).get("profits", {})
    mono = bench.get("monopoly", {}).get("profits", {})
    for fid in range(2):
        if str(fid) in comp:
            ax.axhline(float(comp[str(fid)]), ls="--", color=f"C{fid}", alpha=0.25, linewidth=0.8)
        if str(fid) in mono:
            ax.axhline(float(mono[str(fid)]), ls=":", color=f"C{fid}", alpha=0.32, linewidth=0.9)


def _metrics_has_key(sessions, key: str) -> bool:
    for sess in sessions:
        for r in sess.get("metrics") or []:
            if key in r:
                return True
    return False


def _firm_comp_mono_total_mw(config):
    bench = config["benchmarks"]
    cg = bench["competitive"]["gens"]
    mg = bench["monopoly"]["gens"]
    comp = (cg[0] + cg[1], cg[2])
    mono = (mg[0] + mg[1], mg[2])
    return comp, mono


def _calvano_xtick_formatter():
    def fmt(x, _pos):
        if abs(x - 1) < 5000:
            return "1"
        if abs(x - 500_000) < 10_000:
            return "0.5M"
        if abs(x - 1_000_000) < 10_000:
            return "1M"
        if abs(x - 1_500_000) < 10_000:
            return "1.5M"
        if abs(x - 2_000_000) < 10_000:
            return "2M"
        if x >= 1e6:
            s = f"{x / 1e6:.1f}M"
            return s.replace(".0M", "M")
        return f"{int(round(x)):,}"

    return FuncFormatter(fmt)


def plot_calvano_paper_figures(config, sessions, save_dir: Path, history_label=None):
    """
    Figure 1: greedy (deterministic-policy) mean quantity vs steps + competitive/monopoly horizontals.
    Figure 2: normalized profit Δ from greedy counterfactual clear vs steps.
    X-axis: 1, 0.5M, 1M, 1.5M, 2M timesteps (capped at 2M for display).
    """
    h = history_label if history_label is not None else config.get("history_len", "?")
    max_steps = 2_000_000
    comp, mono = _firm_comp_mono_total_mw(config)

    use_greedy = _metrics_has_key(sessions, "firm_0_greedy_gen")
    gkey = "firm_{}_greedy_gen" if use_greedy else "firm_{}_avg_gen"
    dkey = "firm_{}_greedy_delta" if _metrics_has_key(sessions, "firm_0_greedy_delta") else "firm_{}_delta"

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for fid in range(2):
        steps, mean, std = aggregate_metric(sessions, gkey.format(fid), max_steps=max_steps)
        if not steps:
            continue
        c = f"C{fid}"
        ax1.plot(steps, mean, color=c, label=f"Firm {fid} (greedy mean MW)" if use_greedy else f"Firm {fid}")
        if len(sessions) > 1:
            ax1.fill_between(steps, mean - std, mean + std, alpha=0.15, color=c)

    ax1.axhline(comp[0], ls="--", color="C0", alpha=0.55, linewidth=1.0, label="Competitive (F0)")
    ax1.axhline(comp[1], ls="--", color="C1", alpha=0.55, linewidth=1.0, label="Competitive (F1)")
    ax1.axhline(mono[0], ls=":", color="C0", alpha=0.75, linewidth=1.2, label="Monopoly (F0)")
    ax1.axhline(mono[1], ls=":", color="C1", alpha=0.75, linewidth=1.2, label="Monopoly (F1)")
    ax1.set_xlim(0, max_steps)
    ax1.set_xticks(CALVANO_XTICKS)
    ax1.xaxis.set_major_formatter(_calvano_xtick_formatter())
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Quantity (MW)")
    ax1.set_title(
        "Evolution of greedy output levels (mean deterministic policy on rollout obs)"
        + ("" if use_greedy else " — fallback: realized avg gen (re-run with new ppo.py)")
    )
    ax1.legend(fontsize=7, loc="best")
    fig1.suptitle(f"Algorithmic collusion style — H={h} ({len(sessions)} sessions)", fontsize=12, y=1.02)
    fig1.tight_layout()
    out1 = save_dir / f"calvano_fig1_quantities_h{h}.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved → {out1}")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    y_hi = 1.15
    for fid in range(2):
        steps, mean, std = aggregate_metric(sessions, dkey.format(fid), max_steps=max_steps)
        if not steps:
            continue
        c = f"C{fid}"
        lbl = f"Firm {fid} Δ (greedy)" if "greedy" in dkey else f"Firm {fid} Δ"
        ax2.plot(steps, mean, color=c, label=lbl)
        if len(sessions) > 1:
            ax2.fill_between(steps, mean - std, mean + std, alpha=0.15, color=c)
            y_hi = max(y_hi, float(np.nanmax(mean + std)) * 1.08)
        else:
            y_hi = max(y_hi, float(np.nanmax(mean)) * 1.08)

    ax2.axhline(0, ls="--", color="grey", alpha=0.6, linewidth=0.9, label="Competitive (Δ=0)")
    ax2.axhline(1, ls="--", color="black", alpha=0.6, linewidth=0.9, label="Monopoly (Δ=1)")
    ax2.set_xlim(0, max_steps)
    ax2.set_xticks(CALVANO_XTICKS)
    ax2.xaxis.set_major_formatter(_calvano_xtick_formatter())
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Normalized profit gain Δ")
    ax2.set_title(
        "Evolution of profit gains (greedy counterfactual)"
        if "greedy" in dkey
        else "Evolution of Δ (realized episode profits — re-run with new ppo.py for greedy Δ)"
    )
    ax2.set_ylim(-0.1, max(y_hi, 1.15))
    ax2.legend(fontsize=8, loc="best")
    fig2.suptitle(f"Algorithmic collusion style — H={h} ({len(sessions)} sessions)", fontsize=12, y=1.02)
    fig2.tight_layout()
    out2 = save_dir / f"calvano_fig2_profit_gain_h{h}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")


def plot_calvano_cross_history_comparison(run_dirs, save_dir: Path):
    """
    Same style as --calvano-paper (timesteps vs quantity / vs Δ), but overlay H=1,2,…
    Two PNGs: quantities (one panel per firm) and normalized profit (one panel per firm).
    All series are mean ± band across sessions within each run.
    """
    runs = []
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        config, sessions = load_sessions(rd)
        if sessions:
            runs.append((config, sessions))

    if len(runs) < 1:
        print("No valid run directories for Calvano cross-history comparison.")
        return

    h_labels = [str(c.get("history_len", "?")) for c, _ in runs]
    tag = "_".join(h_labels)
    max_steps = 2_000_000
    comp, mono = _firm_comp_mono_total_mw(runs[0][0])

    use_greedy = all(
        _metrics_has_key(sessions, "firm_0_greedy_gen") for _, sessions in runs
    )
    gkey = "firm_{}_greedy_gen" if use_greedy else "firm_{}_avg_gen"
    dkey = (
        "firm_{}_greedy_delta"
        if all(
            _metrics_has_key(sessions, "firm_0_greedy_delta")
            for _, sessions in runs
        )
        else "firm_{}_delta"
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # —— Quantities: Firm 0 | Firm 1 ——
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    for fid in range(2):
        ax = axes1[fid]
        for i, (config, sessions) in enumerate(runs):
            h = config.get("history_len", "?")
            steps, mean, std = aggregate_metric(
                sessions, gkey.format(fid), max_steps=max_steps
            )
            if not steps:
                continue
            color = f"C{i}"
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.12, color=color)
        ax.axhline(
            comp[fid], ls="--", color="grey", alpha=0.55, linewidth=1.0, label="Competitive"
        )
        ax.axhline(
            mono[fid], ls=":", color="black", alpha=0.75, linewidth=1.1, label="Monopoly"
        )
        ax.set_xlim(0, max_steps)
        ax.set_xticks(CALVANO_XTICKS)
        ax.xaxis.set_major_formatter(_calvano_xtick_formatter())
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Quantity (MW)")
        ax.set_title(
            f"Firm {fid} — "
            + ("greedy mean MW" if use_greedy else "realized avg gen")
        )
        ax.legend(fontsize=7, loc="best")
    fig1.suptitle(
        f"Cross-history quantities (session-averaged) — H={', '.join(h_labels)}",
        fontsize=12,
        y=1.02,
    )
    fig1.tight_layout()
    out1 = save_dir / f"calvano_compare_quantities_h{tag}.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved → {out1}")

    # —— Normalized profit Δ: Firm 0 | Firm 1 ——
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    y_hi = 1.15
    for fid in range(2):
        ax = axes2[fid]
        for i, (config, sessions) in enumerate(runs):
            h = config.get("history_len", "?")
            steps, mean, std = aggregate_metric(
                sessions, dkey.format(fid), max_steps=max_steps
            )
            if not steps:
                continue
            color = f"C{i}"
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.12, color=color)
                y_hi = max(y_hi, float(np.nanmax(mean + std)) * 1.05)
            else:
                y_hi = max(y_hi, float(np.nanmax(mean)) * 1.05)
        ax.axhline(0, ls="--", color="grey", alpha=0.6, linewidth=0.9, label="Δ=0")
        ax.axhline(1, ls="--", color="black", alpha=0.6, linewidth=0.9, label="Δ=1")
        ax.set_xlim(0, max_steps)
        ax.set_xticks(CALVANO_XTICKS)
        ax.xaxis.set_major_formatter(_calvano_xtick_formatter())
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Normalized profit gain Δ")
        lbl = "greedy Δ" if "greedy" in dkey else "realized Δ"
        ax.set_title(f"Firm {fid} — {lbl}")
        ax.legend(fontsize=7, loc="best")
    for ax in axes2:
        ax.set_ylim(-0.1, max(y_hi, 1.15))
    fig2.suptitle(
        f"Cross-history normalized profit (session-averaged) — H={', '.join(h_labels)}",
        fontsize=12,
        y=1.02,
    )
    fig2.tight_layout()
    out2 = save_dir / f"calvano_compare_profit_h{tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")


# ====================== Figure 1: Generation evolution ======================
def plot_generation(ax, config, sessions, label_suffix=""):
    bench = config["benchmarks"]
    comp_gens = bench["competitive"]["gens"]
    mono_gens = bench["monopoly"]["gens"]

    # Firm 0 total gen: plant 0 + plant 1
    comp_g0 = comp_gens[0] + comp_gens[1]
    comp_g1 = comp_gens[2]
    mono_g0 = mono_gens[0] + mono_gens[1]
    mono_g1 = mono_gens[2]

    for fid, (_comp_g, _mono_g) in enumerate([(comp_g0, mono_g0), (comp_g1, mono_g1)]):
        steps, mean, std = aggregate_metric(sessions, f"firm_{fid}_avg_gen")
        if not steps:
            continue
        color = f"C{fid}"
        lbl = f"Firm {fid}{label_suffix}"
        ax.plot(steps, mean, color=color, label=lbl)
        if len(sessions) > 1:
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(comp_g0, ls="--", color="C0", alpha=0.4, linewidth=0.8)
    ax.axhline(comp_g1, ls="--", color="C1", alpha=0.4, linewidth=0.8)
    ax.axhline(mono_g0, ls=":", color="C0", alpha=0.4, linewidth=0.8)
    ax.axhline(mono_g1, ls=":", color="C1", alpha=0.4, linewidth=0.8)

    ax.set_ylabel("Avg Generation (MW)")
    ax.set_title("Evolution of Generation Quantities")
    ax.legend(fontsize=8)


# ====================== Figure 2: Δ evolution ======================
def plot_delta(ax, config, sessions, label_suffix=""):
    for fid in range(2):
        steps, mean, std = aggregate_metric(sessions, f"firm_{fid}_delta")
        if not steps:
            continue
        color = f"C{fid}"
        ax.plot(steps, mean, color=color, label=f"Firm {fid}{label_suffix}")
        if len(sessions) > 1:
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8, label="Competitive (Δ=0)")
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8, label="Full collusion (Δ=1)")
    ax.set_ylabel("Δ (normalized profit gain)")
    ax.set_title("Evolution of Collusion Index Δ")
    ax.legend(fontsize=8)


# ====================== Figure 3: Limit strategy ======================
def plot_limit_strategy(ax, config, sessions):
    all_grids, all_strats = {str(f): [] for f in range(2)}, None

    for sess in sessions:
        ls = sess.get("limit_strategy")
        if not ls:
            continue
        grid = ls["lmp_grid"]
        for fid_str in ["0", "1"]:
            all_grids[fid_str].append(ls["strategies"][fid_str])

    if not all_grids["0"]:
        ax.text(0.5, 0.5, "No limit strategy data", transform=ax.transAxes, ha="center")
        return

    grid = sessions[0]["limit_strategy"]["lmp_grid"]

    for fid_str, color in [("0", "C0"), ("1", "C1")]:
        matrix = np.array(all_grids[fid_str])
        mean = matrix.mean(axis=0)
        ax.plot(grid, mean, color=color, label=f"Firm {fid_str}")
        if matrix.shape[0] > 1:
            std = matrix.std(axis=0)
            ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=color)

    # Reference lines
    bench = config["benchmarks"]
    comp_gens = bench["competitive"]["gens"]
    ax.axhline(comp_gens[0] + comp_gens[1], ls="--", color="C0", alpha=0.3, linewidth=0.8)
    ax.axhline(comp_gens[2], ls="--", color="C1", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Observed Avg LMP ($/MWh)")
    ax.set_ylabel("Generation (MW)")
    ax.set_title("Learned Limit Strategy (output vs. price)")
    ax.legend(fontsize=8)


# ====================== Figure 4: Impulse response ======================
def plot_impulse_response(axes, config, sessions):
    """Two subplots: one for each firm deviating."""
    for dev_fid, ax in enumerate(axes):
        traces = {str(f): [] for f in range(2)}
        for sess in sessions:
            de = sess.get("deviation_experiment", {})
            entry = de.get(str(dev_fid))
            if not entry:
                continue
            for fid_str in ["0", "1"]:
                traces[fid_str].append(entry["gen"][fid_str])

        if not traces["0"]:
            ax.text(0.5, 0.5, "No deviation data", transform=ax.transAxes, ha="center")
            continue

        horizon = len(traces["0"][0])
        t = np.arange(horizon)

        for fid_str, color in [("0", "C0"), ("1", "C1")]:
            matrix = np.array(traces[fid_str])
            mean = matrix.mean(axis=0)
            lbl = f"Firm {fid_str}" + (" (deviator)" if fid_str == str(dev_fid) else "")
            ls = "-" if fid_str == str(dev_fid) else "--"
            ax.plot(t, mean, color=color, ls=ls, label=lbl)
            if matrix.shape[0] > 1:
                std = matrix.std(axis=0)
                ax.fill_between(t, mean - std, mean + std, alpha=0.1, color=color)

        ax.set_xlabel("Period after deviation")
        ax.set_ylabel("Generation (MW)")
        ax.set_title(f"Impulse Response: Firm {dev_fid} deviates")
        ax.legend(fontsize=8)


# ====================== Advisor-friendly deviation explainer =================
def plot_deviation_explainer(config, sessions, save_dir: Path, history_label=None):
    """
    Advisor-friendly impulse-response figure for the deviation experiment.

    Layout (one column per deviating firm):
      Row 1 — total generation MW per firm vs period
              (deviator solid + emphasized, rival dashed, resting reference, t=0 marker)
      Row 2 — quantity-weighted average LMP vs period
              (with competitive / monopoly horizontal references)

    Each curve is mean ± std across the run's sessions.
    """
    if not sessions:
        print("No sessions for deviation explainer.")
        return

    num_firms = 2
    gen_by_dev = {str(d): {str(f): [] for f in range(num_firms)} for d in range(num_firms)}
    lmp_by_dev = {str(d): [] for d in range(num_firms)}
    resting_by_dev = {str(d): {str(f): [] for f in range(num_firms)} for d in range(num_firms)}

    for sess in sessions:
        de = sess.get("deviation_experiment", {}) or {}
        for dev_str, entry in de.items():
            if not entry:
                continue
            for fstr in [str(f) for f in range(num_firms)]:
                trace = entry.get("gen", {}).get(fstr)
                if trace:
                    gen_by_dev[dev_str][fstr].append(np.asarray(trace, dtype=float))
                rest_val = entry.get("resting", {}).get(fstr)
                if rest_val is not None:
                    resting_by_dev[dev_str][fstr].append(float(rest_val))
            lmp = entry.get("lmp")
            if lmp:
                lmp_by_dev[dev_str].append(np.asarray(lmp, dtype=float))

    deviators = sorted(
        [d for d in gen_by_dev if any(gen_by_dev[d][f] for f in gen_by_dev[d])],
        key=int,
    )
    if not deviators:
        print("No deviation traces in sessions.")
        return

    h = history_label if history_label is not None else config.get("history_len", "?")
    bench = config.get("benchmarks", {})
    comp_lmp = bench.get("competitive", {}).get("avg_lmp")
    mono_lmp = bench.get("monopoly", {}).get("avg_lmp")
    dev_frac = float(config.get("deviation_frac", 0.20))

    n_cols = len(deviators)
    fig, axes = plt.subplots(2, n_cols, figsize=(7.0 * n_cols, 8.0), squeeze=False)
    fig.suptitle(
        "Deviation experiment — does a unilateral output increase trigger a price war?  "
        f"(H={h}, {len(sessions)} sessions)",
        fontsize=13,
        y=1.0,
    )

    for col, dev_str in enumerate(deviators):
        gen_ax = axes[0, col]
        lmp_ax = axes[1, col]

        # ---- Generation panel ----
        for fstr in [str(f) for f in range(num_firms)]:
            mats = gen_by_dev[dev_str][fstr]
            if not mats:
                continue
            T = min(len(arr) for arr in mats)
            matrix = np.stack([arr[:T] for arr in mats])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            t = np.arange(T)
            is_dev = fstr == dev_str
            color = f"C{int(fstr)}"
            lw = 2.2 if is_dev else 1.6
            ls = "-" if is_dev else "--"
            lbl = f"Firm {fstr}" + (" — deviator" if is_dev else " — rival (response)")
            gen_ax.plot(t, mean, color=color, lw=lw, ls=ls, label=lbl)
            if matrix.shape[0] > 1:
                gen_ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

            rests = resting_by_dev[dev_str][fstr]
            if rests:
                gen_ax.axhline(
                    float(np.mean(rests)),
                    color=color,
                    ls=":",
                    alpha=0.55,
                    lw=0.9,
                    label=f"Firm {fstr} resting"
                    + (" (pre-deviation)" if is_dev else ""),
                )

        gen_ax.axvline(0, color="red", ls="--", alpha=0.55, lw=1.0)
        gen_ax.text(
            0.02,
            0.97,
            f"t=0: deviator forced to +{dev_frac:.0%} for one step,\n"
            "then both play their learned deterministic policy.",
            transform=gen_ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.25",
        )
        gen_ax.set_title(f"Firm {dev_str} deviates  →  generation response")
        gen_ax.set_xlabel("Period after deviation")
        gen_ax.set_ylabel("Total generation (MW)")
        gen_ax.grid(alpha=0.25)
        gen_ax.legend(fontsize=8, loc="best")

        # ---- LMP panel ----
        mats = lmp_by_dev[dev_str]
        if mats:
            T = min(len(arr) for arr in mats)
            matrix = np.stack([arr[:T] for arr in mats])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            t = np.arange(T)
            lmp_ax.plot(t, mean, color="black", lw=1.8, label="Avg LMP (qty-weighted)")
            if matrix.shape[0] > 1:
                lmp_ax.fill_between(t, mean - std, mean + std, color="black", alpha=0.12)
            if len(mean):
                lmp_ax.scatter(
                    [0],
                    [mean[0]],
                    color="red",
                    zorder=5,
                    s=46,
                    label=f"At deviation: ${mean[0]:.2f}",
                )
                lmp_ax.scatter(
                    [T - 1],
                    [mean[-1]],
                    color="green",
                    zorder=5,
                    s=46,
                    label=f"After {T - 1} steps: ${mean[-1]:.2f}",
                )

        if comp_lmp is not None:
            lmp_ax.axhline(
                comp_lmp,
                color="grey",
                ls="--",
                lw=1.0,
                alpha=0.7,
                label=f"Competitive (${comp_lmp:.2f})",
            )
        if mono_lmp is not None:
            lmp_ax.axhline(
                mono_lmp,
                color="black",
                ls=":",
                lw=1.0,
                alpha=0.75,
                label=f"Monopoly (${mono_lmp:.2f})",
            )

        lmp_ax.axvline(0, color="red", ls="--", alpha=0.55, lw=1.0)
        lmp_ax.set_title(f"Firm {dev_str} deviates  →  price response")
        lmp_ax.set_xlabel("Period after deviation")
        lmp_ax.set_ylabel("Avg LMP ($/MWh)")
        lmp_ax.grid(alpha=0.25)
        lmp_ax.legend(fontsize=8, loc="best")

    fig.text(
        0.01,
        0.005,
        "Reading guide: at t=0 the deviator is forced above its resting output. "
        "If the rival accommodates (no punishment) its curve stays near its resting line "
        "and avg LMP reverts toward the pre-deviation level within a few periods. "
        "Persistent price drops well below the resting LMP would indicate a price-war response.",
        fontsize=8,
        color="0.30",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"deviation_explainer_h{h}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ====================== Figure 5: KL divergence evolution ======================
def _positive_series_for_log(y, lo=1e-12):
    """Avoid log-scale warnings / invalid values from zeros or missing metrics."""
    y = np.asarray(y, dtype=float)
    return np.clip(y, lo, None)


def plot_kl(ax, config, sessions, label_suffix=""):
    lag_k = int(config.get("policy_kl_lag", 0) or 0)
    use_lag = lag_k > 0 and _metrics_has_key(sessions, "firm_0_kl_lag")

    for fid in range(2):
        steps, mean, std = aggregate_metric(sessions, f"firm_{fid}_kl")
        if not steps:
            continue
        color = f"C{fid}"
        m = _positive_series_for_log(mean)
        ax.plot(steps, m, color=color, label=f"F{fid} intra-update{label_suffix}")
        if len(sessions) > 1:
            s_lo = _positive_series_for_log(mean - std)
            s_hi = _positive_series_for_log(mean + std)
            ax.fill_between(steps, s_lo, s_hi, alpha=0.15, color=color)

        if use_lag:
            steps_l, mean_l, std_l = aggregate_metric(
                sessions, f"firm_{fid}_kl_lag", default_for_missing=float("nan")
            )
            if steps_l:
                ml = _positive_series_for_log(mean_l)
                ax.plot(
                    steps_l,
                    ml,
                    color=color,
                    ls="--",
                    label=f"F{fid} π_{{t−{lag_k}}}‖π_t{label_suffix}",
                )
                if len(sessions) > 1:
                    ax.fill_between(
                        steps_l,
                        _positive_series_for_log(mean_l - std_l),
                        _positive_series_for_log(mean_l + std_l),
                        alpha=0.08,
                        color=color,
                    )

    kl_thresh = config.get("kl_threshold", 0.01)
    ax.axhline(kl_thresh, ls="--", color="red", alpha=0.5, linewidth=0.8,
               label=f"KL threshold ({kl_thresh})")
    ax.set_ylabel("KL divergence")
    ax.set_yscale("log")
    title = "Policy KL (solid: intra-update; dashed: lagged)" if use_lag else "Policy KL (intra-update)"
    ax.set_title(title)
    ax.legend(fontsize=7)


# ====================== Comparison across history lengths ======================
def plot_comparison(run_dirs, save_dir=None):
    """Generate a single figure comparing key metrics across history lengths."""
    runs = []
    for rd in run_dirs:
        config, sessions = load_sessions(rd)
        if sessions:
            runs.append((config, sessions))

    if not runs:
        print("No valid run directories found for comparison.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    h_labels = [str(c.get("history_len", "?")) for c, _ in runs]
    fig.suptitle(f"Cross-History Comparison — H={', '.join(h_labels)}", fontsize=14)

    colors_h = {str(c.get("history_len", i)): f"C{i}" for i, (c, _) in enumerate(runs)}

    # --- (0,0): Δ evolution per H (Firm 0) ---
    ax = axes[0, 0]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_0_delta")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Δ (Firm 0)")
    ax.set_title("Collusion Index Δ — Firm 0")
    ax.legend(fontsize=8)

    # --- (0,1): Δ evolution per H (Firm 1) ---
    ax = axes[0, 1]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_1_delta")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Δ (Firm 1)")
    ax.set_title("Collusion Index Δ — Firm 1")
    ax.legend(fontsize=8)

    # --- (0,2): Avg LMP evolution per H ---
    ax = axes[0, 2]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "avg_lmp")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    bench = runs[0][0].get("benchmarks", {})
    if bench:
        ax.axhline(bench["competitive"]["avg_lmp"], ls="--", color="green", alpha=0.5,
                    linewidth=0.8, label="Competitive LMP")
        ax.axhline(bench["monopoly"]["avg_lmp"], ls=":", color="red", alpha=0.5,
                    linewidth=0.8, label="Monopoly LMP")
    ax.set_ylabel("Avg LMP ($/MWh)")
    ax.set_title("Average LMP")
    ax.legend(fontsize=8)

    # --- (1,0): KL divergence per H (max of both firms) ---
    ax = axes[1, 0]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "max_kl")
        if steps:
            m = _positive_series_for_log(mean)
            ax.plot(steps, m, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(
                    steps,
                    _positive_series_for_log(mean - std),
                    _positive_series_for_log(mean + std),
                    alpha=0.1,
                    color=color,
                )
    kl_thresh = runs[0][0].get("kl_threshold", 0.01)
    ax.axhline(kl_thresh, ls="--", color="red", alpha=0.5, linewidth=0.8,
               label=f"Threshold ({kl_thresh})")
    ax.set_ylabel("Max KL divergence")
    ax.set_yscale("log")
    ax.set_title("Policy KL Convergence")
    ax.legend(fontsize=8)

    # --- (1,1): Generation per H (Firm 0) ---
    ax = axes[1, 1]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_0_avg_gen")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.set_ylabel("Avg Generation (MW)")
    ax.set_title("Generation — Firm 0")
    ax.legend(fontsize=8)

    # --- (1,2): Generation per H (Firm 1) ---
    ax = axes[1, 2]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_1_avg_gen")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.set_ylabel("Avg Generation (MW)")
    ax.set_title("Generation — Firm 1")
    ax.legend(fontsize=8)

    for row in axes:
        for a in row:
            a.set_xlabel("Timesteps")

    fig.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fname = save_path / f"comparison_h{'_'.join(h_labels)}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved → {fname}")
    else:
        plt.show()


def plot_comparison_delta(run_dirs, save_dir=None):
    """
    Cross-history dashboard for delta-mode runs only.

    Identical layout to plot_comparison except the KL panel is removed and
    replaced by `delta_max_jump` — the actual convergence diagnostic used by
    --convergence-mode delta (max_f |Δ_f − Δ_f,prev| over firms, log-y).

    Output: figures/.../comparison_delta_h{H_LIST}.png
    """
    runs = []
    for rd in run_dirs:
        config, sessions = load_sessions(rd)
        if sessions:
            runs.append((config, sessions))

    if not runs:
        print("No valid run directories found for delta comparison.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    h_labels = [str(c.get("history_len", "?")) for c, _ in runs]
    fig.suptitle(
        f"Cross-History Comparison (delta mode) — H={', '.join(h_labels)}",
        fontsize=14,
    )

    colors_h = {str(c.get("history_len", i)): f"C{i}" for i, (c, _) in enumerate(runs)}

    # --- (0,0): Δ Firm 0 ---
    ax = axes[0, 0]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_0_delta")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Δ (Firm 0)")
    ax.set_title("Collusion Index Δ — Firm 0")
    ax.legend(fontsize=8)

    # --- (0,1): Δ Firm 1 ---
    ax = axes[0, 1]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_1_delta")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Δ (Firm 1)")
    ax.set_title("Collusion Index Δ — Firm 1")
    ax.legend(fontsize=8)

    # --- (0,2): Avg LMP ---
    ax = axes[0, 2]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "avg_lmp")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    bench = runs[0][0].get("benchmarks", {})
    if bench:
        ax.axhline(bench["competitive"]["avg_lmp"], ls="--", color="green", alpha=0.5,
                   linewidth=0.8, label="Competitive LMP")
        ax.axhline(bench["monopoly"]["avg_lmp"], ls=":", color="red", alpha=0.5,
                   linewidth=0.8, label="Monopoly LMP")
    ax.set_ylabel("Avg LMP ($/MWh)")
    ax.set_title("Average LMP")
    ax.legend(fontsize=8)

    # --- (1,0): Δ-jump diagnostic — replaces KL panel ---
    ax = axes[1, 0]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(
            sessions, "delta_max_jump", default_for_missing=float("nan")
        )
        if steps:
            m = _positive_series_for_log(mean)
            ax.plot(steps, m, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(
                    steps,
                    _positive_series_for_log(mean - std),
                    _positive_series_for_log(mean + std),
                    alpha=0.1,
                    color=color,
                )
    delta_thresh = runs[0][0].get("delta_convergence_threshold", 0.01)
    ax.axhline(
        delta_thresh,
        ls="--",
        color="red",
        alpha=0.5,
        linewidth=0.8,
        label=f"Δ-threshold ({delta_thresh})",
    )
    ax.set_ylabel("max_f |Δ_f − Δ_f, prev|")
    ax.set_yscale("log")
    ax.set_title("Δ-Stability Convergence Diagnostic")
    ax.legend(fontsize=8)

    # --- (1,1): Generation Firm 0 ---
    ax = axes[1, 1]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_0_avg_gen")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.set_ylabel("Avg Generation (MW)")
    ax.set_title("Generation — Firm 0")
    ax.legend(fontsize=8)

    # --- (1,2): Generation Firm 1 ---
    ax = axes[1, 2]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, "firm_1_avg_gen")
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.set_ylabel("Avg Generation (MW)")
    ax.set_title("Generation — Firm 1")
    ax.legend(fontsize=8)

    for row in axes:
        for a in row:
            a.set_xlabel("Timesteps")

    fig.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fname = save_path / f"comparison_delta_h{'_'.join(h_labels)}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved → {fname}")
    else:
        plt.show()


def plot_generation_profit_comparison(run_dirs, save_dir: Path):
    """One figure: generation quantity and profit against PPO iterations for each H."""
    runs = []
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        config, sessions = load_sessions(rd)
        if sessions:
            runs.append((config, sessions))

    if not runs:
        print("No valid run directories found for generation/profit comparison.")
        return

    h_labels = [str(c.get("history_len", "?")) for c, _ in runs]
    fig_h = max(5.0, 3.8 * len(runs))
    fig, axes = plt.subplots(len(runs), 2, figsize=(16, fig_h), squeeze=False)
    fig.suptitle(
        f"Generation and Profit vs PPO Iteration — H={', '.join(h_labels)}",
        fontsize=14,
        y=0.995,
    )

    used_firm_fallback = False
    for row_idx, (config, sessions) in enumerate(runs):
        h = config.get("history_len", "?")
        n_sessions = len(sessions)

        gen_ax = axes[row_idx, 0]
        gen_specs, gen_kind = _generation_series_specs(config, sessions)
        used_firm_fallback |= gen_kind == "firm"
        for key, label, color in gen_specs:
            x, mean, std = _aggregate_metric_by_iteration(sessions, key)
            if not x:
                continue
            gen_ax.plot(x, mean, color=color, linewidth=1.6, label=label)
            if n_sessions > 1:
                gen_ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        _draw_generation_benchmarks(gen_ax, config, gen_kind)
        if not gen_specs:
            gen_ax.text(0.5, 0.5, "No generation metric found", ha="center", va="center", transform=gen_ax.transAxes)
        gen_ax.set_title(f"Generation quantity — H={h}")
        gen_ax.set_xlabel("PPO iteration")
        gen_ax.set_ylabel("Generation (MW)")
        gen_ax.grid(alpha=0.25)
        gen_ax.legend(fontsize=8, loc="best")

        profit_ax = axes[row_idx, 1]
        profit_specs, profit_kind = _profit_series_specs(config, sessions)
        used_firm_fallback |= profit_kind == "firm"
        for key, label, color in profit_specs:
            x, mean, std = _aggregate_metric_by_iteration(sessions, key)
            if not x:
                continue
            profit_ax.plot(x, mean, color=color, linewidth=1.6, label=label)
            if n_sessions > 1:
                profit_ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        _draw_profit_benchmarks(profit_ax, config, profit_kind)
        if not profit_specs:
            profit_ax.text(0.5, 0.5, "No profit metric found", ha="center", va="center", transform=profit_ax.transAxes)
        profit_ax.set_title(f"Profit — H={h}")
        profit_ax.set_xlabel("PPO iteration")
        profit_ax.set_ylabel("Profit ($/step)")
        profit_ax.grid(alpha=0.25)
        profit_ax.legend(fontsize=8, loc="best")

    if used_firm_fallback:
        fig.text(
            0.01,
            0.01,
            "Note: saved comparison metrics are firm-level unless plant_* metrics are present; "
            "Firm 0 aggregates its two plants.",
            fontsize=8,
            color="0.35",
        )

    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"generation_profit_h{'_'.join(h_labels)}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Calvano-style plots for PPO collusion")
    parser.add_argument("run_dirs", nargs="*", type=Path,
                        help="One or more run directories to plot")
    parser.add_argument("--compare", action="store_true",
                        help="6-panel dashboard (Δ, LMP, KL, gen) across history lengths")
    parser.add_argument(
        "--compare-delta",
        action="store_true",
        help="6-panel dashboard for delta-mode runs only "
        "(KL panel replaced by Δ-jump convergence diagnostic).",
    )
    parser.add_argument(
        "--compare-calvano",
        action="store_true",
        help="Two Calvano-style figures across H: quantities + normalized Δ (session-averaged)",
    )
    parser.add_argument(
        "--compare-generation-profit",
        action="store_true",
        help="One PNG across H: generation quantity + profit vs PPO iterations",
    )
    parser.add_argument("--calvano-paper", action="store_true",
                        help="Save only Calvano-style Fig 1 (quantities) and Fig 2 (Δ vs timesteps)")
    parser.add_argument(
        "--deviation-explainer",
        action="store_true",
        help="Advisor-friendly impulse-response figure: generation + LMP per deviator (one PNG per run dir).",
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save figures (PNG). If omitted, shows interactively.")
    args = parser.parse_args()

    run_dirs = [rd for rd in args.run_dirs if rd is not None and str(rd).strip()]
    if not run_dirs:
        parser.error("Provide at least one run directory.")

    if args.compare_calvano:
        missing = [str(rd) for rd in run_dirs if not rd.is_dir()]
        if missing:
            parser.error(f"Not a directory: {', '.join(missing)}")
        if not args.save:
            parser.error("--compare-calvano requires --save DIR")
        plot_calvano_cross_history_comparison(run_dirs, Path(args.save))
        return

    if args.compare_generation_profit:
        missing = [str(rd) for rd in run_dirs if not rd.is_dir()]
        if missing:
            parser.error(f"Not a directory: {', '.join(missing)}")
        if not args.save:
            parser.error("--compare-generation-profit requires --save DIR")
        plot_generation_profit_comparison(run_dirs, Path(args.save))
        return

    if args.compare:
        missing = [str(rd) for rd in run_dirs if not rd.is_dir()]
        if missing:
            parser.error(f"Not a directory: {', '.join(missing)}")
        plot_comparison(run_dirs, save_dir=args.save)
        return

    if args.compare_delta:
        missing = [str(rd) for rd in run_dirs if not rd.is_dir()]
        if missing:
            parser.error(f"Not a directory: {', '.join(missing)}")
        plot_comparison_delta(run_dirs, save_dir=args.save)
        return

    if args.deviation_explainer:
        missing = [str(rd) for rd in run_dirs if not rd.is_dir()]
        if missing:
            parser.error(f"Not a directory: {', '.join(missing)}")
        if not args.save:
            parser.error("--deviation-explainer requires --save DIR")
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for rd in run_dirs:
            config, sessions = load_sessions(rd)
            h = config.get("history_len", "?")
            plot_deviation_explainer(config, sessions, save_dir, history_label=h)
        return

    for rd in run_dirs:
        config, sessions = load_sessions(rd)
        h = config.get("history_len", "?")
        n = len(sessions)

        if args.calvano_paper:
            if not args.save:
                parser.error("--calvano-paper requires --save DIR")
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_calvano_paper_figures(config, sessions, save_dir, history_label=h)
            continue

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"PPO Collusion — H={h}  ({n} session{'s' if n>1 else ''})", fontsize=14)

        plot_generation(axes[0, 0], config, sessions)
        plot_delta(axes[0, 1], config, sessions)
        plot_kl(axes[0, 2], config, sessions)
        plot_limit_strategy(axes[1, 0], config, sessions)

        plot_impulse_response([axes[1, 1], axes[1, 2]], config, sessions)

        fig.tight_layout()

        if args.save:
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = save_dir / f"collusion_h{h}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved → {fname}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
