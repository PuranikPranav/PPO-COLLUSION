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
    python experiments/plot_results.py latest_results --save latest_results/figures/
    python experiments/plot_results.py latest_results --save latest_results/figures/ --calvano-paper
    python experiments/plot_results.py --compare \
        old_results/delta_crosshistory/h1 old_results/delta_crosshistory/h2 old_results/delta_crosshistory/h3 \
        --save old_results/figures/crosshistory/
    python experiments/plot_results.py --compare-calvano \
        old_results/delta_crosshistory/h1 old_results/delta_crosshistory/h2 old_results/delta_crosshistory/h3 \
        --save old_results/figures/crosshistory/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_mpl_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ppo-collusion-matplotlib-cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# X-axis ticks for Calvano-style learning curves (environment steps)
CALVANO_XTICKS = np.array([1, 500_000, 1_000_000, 1_500_000, 2_000_000], dtype=float)

# ============================================================================
#  House style — one cohesive, publication-quality look shared by every figure.
#  Every plot is a SINGLE standalone axis (no mixed-scale subplot grids), so each
#  figure carries exactly one message and one y-axis meaning.
# ============================================================================
FIRM_COLORS = ("#1b6ca8", "#e08a1e")   # Firm 0 = deep blue, Firm 1 = warm amber
FIRM_NAMES = ("Firm 0", "Firm 1")
# Benchmark reference lines: (color, linestyle) — identical everywhere they appear.
BENCH_STYLE = {
    "competitive": ("#2e8b57", (0, (6, 3))),       # sea green, dashed
    "nash":        ("#b8860b", (0, (5, 2, 1, 2))),  # dark goldenrod, dash-dot
    "monopoly":    ("#b3322c", (0, (1, 2))),        # brick red, dotted
}
ANCHOR_RED = "#d62728"   # the t=0 competitive start marker / deviation instant


def _install_house_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 12.5,
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#9aa0a6",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cfd4da",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#d4d4d4",
        "legend.fontsize": 9.5,
        "legend.borderpad": 0.6,
        "xtick.color": "#3c3c3c",
        "ytick.color": "#3c3c3c",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.prop_cycle": plt.cycler(color=list(FIRM_COLORS)),
    })


_install_house_style()


def _polish(ax):
    """Drop the top/right spines and lighten the remaining frame — the clean look
    every figure shares."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9aa0a6")
    ax.grid(True, which="major", alpha=0.7)
    return ax


def _benchmark_hline(ax, value, kind, label):
    """Draw a benchmark horizontal reference line in the shared house style."""
    if value is None:
        return
    color, ls = BENCH_STYLE[kind]
    ax.axhline(float(value), color=color, ls=ls, lw=1.7, alpha=0.95, zorder=2,
               label=label)


def _title_block(ax, main, sub=None):
    """Bold main title + optional grey sub-line, placed above the axes with clear
    vertical separation (no overlap). Saved figures use bbox_inches='tight', so the
    titles are never clipped."""
    ax.set_title("")  # clear any rcParams title slot
    ax.text(0.5, 1.060, main, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=15, fontweight="bold", color="#1a1a1a")
    if sub:
        ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color="#6a6a6a")


def _settled_badge(ax, text):
    """Small rounded annotation box (top-left) summarising the settled level."""
    ax.text(
        0.015, 0.97, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color="#1a1a1a",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f6f7f9", ec="#c9cdd3", lw=1.0,
                  alpha=0.95),
    )


def _legend(ax, loc="best", ncol=1):
    leg = ax.legend(loc=loc, ncol=ncol)
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)
    return leg


def _save(fig, save_dir: Path, name: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


def _firm_benchmark_levels(config):
    """Per-firm (Competitive, Nash, Monopoly) generation levels in MW, or None."""
    bench = config.get("benchmarks", {})

    def lvl(key, fid):
        g = bench.get(key, {}).get("gens")
        if not g:
            return None
        return (g[0] + g[1]) if fid == 0 else g[2]

    return {
        fid: {
            "competitive": lvl("competitive", fid),
            "nash": lvl("cournot_nash", fid),
            "monopoly": lvl("monopoly", fid),
        }
        for fid in range(2)
    }


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
    cn = bench.get("cournot_nash", {}).get("profits", {})
    mono = bench.get("monopoly", {}).get("profits", {})
    comp = bench.get("competitive", {}).get("profits", {})
    delta_combined_key = "delta_combined"
    can_reconstruct_combined = (
        cn and mono
        and delta_combined_key in keys
        and all(str(f) in cn and str(f) in mono for f in range(2))
    )

    for sess in sessions:
        for row in sess.get("metrics") or []:
            if ep_key in row:
                row[reconstructed_key] = float(row[ep_key]) / episode_len
            elif can_reconstruct_combined and delta_combined_key in row:
                pi_n = sum(float(cn[str(f)]) for f in range(2))
                pi_m = sum(float(mono[str(f)]) for f in range(2))
                total = pi_n + float(row[delta_combined_key]) * (pi_m - pi_n)
                row[reconstructed_key] = total * (
                    float(cn.get(str(fid), 0)) / pi_n if pi_n > 1e-8 else 0.5
                )
            elif delta_key in keys and str(fid) in mono:
                pi_c = float(cn.get(str(fid), comp.get(str(fid), 0)))
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


def _calvano_x_axis(sessions, hard_cap=2_000_000):
    """Return (display_max, xticks) for the Calvano time figures.

    A full-length (≈2M-timestep) run reproduces the original fixed 2M axis and
    ticks exactly. A SHORTER run (e.g. a quick demo or a partial/early-stopped
    run) gets an axis scaled to the data so the curves are not crammed into the
    far-left sliver of a 2M-wide plot. The shared tick formatter already renders
    arbitrary sub-2M ticks (e.g. "50,000"), so only the limit and tick positions
    need to adapt.
    """
    max_step = 0
    for sess in sessions:
        for row in sess.get("metrics") or []:
            s = row.get("step")
            if isinstance(s, (int, float)) and s > max_step:
                max_step = s
    max_step = min(max_step, hard_cap)
    # Long runs (or no data) keep the original fixed 2M axis unchanged.
    if max_step >= 1_500_000 or max_step <= 0:
        return float(hard_cap), CALVANO_XTICKS
    # Pick a "nice" tick step (~max/4 rounded to 1/2/2.5/5 ×10^k) and a clean max.
    raw = max_step / 4.0
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    display_max = math.ceil(max_step / step) * step
    ticks = np.arange(0.0, display_max + step / 2, step)
    ticks[0] = 1.0  # anchor the first tick at the t=1 start (matches 2M-run convention)
    return float(display_max), ticks


def _calvano_time_axis(ax, max_steps, calvano_xticks):
    """Shared time x-axis (timesteps, adaptive ticks) for the Calvano figures."""
    ax.set_xlim(0, max_steps)
    ax.set_xticks(calvano_xticks)
    ax.xaxis.set_major_formatter(_calvano_xtick_formatter())
    ax.set_xlabel("Timesteps")


def _settled_level(values, frac=0.2):
    """Mean of the final `frac` of a series — the converged resting level."""
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float("nan")
    return float(np.mean(arr[-max(1, int(len(arr) * frac)):]))


def plot_calvano_paper_figures(config, sessions, save_dir: Path, history_label=None):
    """Three SEPARATE, single-axis figures (each a standalone PNG):

      fig1 — output quantity (MW) per firm vs timesteps, with the true sampled
             exploration envelope shaded.
      fig2 — the combined collusion index Δ (0 = Nash, 1 = Monopoly).
      fig3 — the average clearing price (LMP), shaded with the SAME within-rollout
             sampling envelope as the generation plot (the price-side exploration).

    All three start at the competitive t=0 anchor and carry consistent
    Competitive / Nash / Monopoly references.
    """
    h = history_label if history_label is not None else config.get("history_len", "?")
    n = len(sessions)
    max_steps, calvano_xticks = _calvano_x_axis(sessions)
    comp, mono = _firm_comp_mono_total_mw(config)
    _ng = config.get("benchmarks", {}).get("cournot_nash", {}).get("gens")
    nash = (_ng[0] + _ng[1], _ng[2]) if _ng else None

    use_greedy = _metrics_has_key(sessions, "firm_0_greedy_gen")
    gkey = "firm_{}_greedy_gen" if use_greedy else "firm_{}_avg_gen"
    if _metrics_has_key(sessions, "greedy_delta_combined"):
        dkey, dkey_per_firm = "greedy_delta_combined", None
    elif _metrics_has_key(sessions, "delta_combined"):
        dkey, dkey_per_firm = "delta_combined", None
    else:
        dkey = None
        dkey_per_firm = (
            "firm_{}_greedy_delta"
            if _metrics_has_key(sessions, "firm_0_greedy_delta")
            else "firm_{}_delta"
        )

    sub = f"H={h}  ·  {n} session{'s' if n != 1 else ''}"

    # ------------------------------------------------------------------ Fig 1
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    has_spread = _metrics_has_key(sessions, "firm_0_gen_lo")
    settled_txt = []
    for fid in range(2):
        steps, mean, std = aggregate_metric(sessions, gkey.format(fid), max_steps=max_steps)
        if not steps:
            continue
        c = FIRM_COLORS[fid]
        if has_spread:
            xs, lo, _ = aggregate_metric(sessions, f"firm_{fid}_gen_lo", max_steps=max_steps)
            _, hi, _ = aggregate_metric(sessions, f"firm_{fid}_gen_hi", max_steps=max_steps)
            if xs:
                ax1.fill_between(xs, lo, hi, color=c, alpha=0.10, lw=0,
                                 label=f"{FIRM_NAMES[fid]} sampled range (exploration)")
        if n > 1:
            ax1.fill_between(steps, mean - std, mean + std, alpha=0.20, color=c, lw=0)
        ax1.plot(steps, mean, color=c, lw=2.4, label=f"{FIRM_NAMES[fid]} (mean output)", zorder=4)
        ax1.scatter([steps[0]], [mean[0]], color=ANCHOR_RED, s=42, zorder=6,
                    edgecolor="white", linewidth=0.8)
        settled_txt.append(f"{FIRM_NAMES[fid]} ≈ {_settled_level(mean):.0f} MW")

    # Per-firm benchmark lines (firm-coloured; linestyle encodes the benchmark type).
    for fid, (cmp_v, nsh_v, mono_v) in enumerate(
        ((comp[0], nash[0] if nash else None, mono[0]),
         (comp[1], nash[1] if nash else None, mono[1]))
    ):
        for v, kind in ((cmp_v, "competitive"), (nsh_v, "nash"), (mono_v, "monopoly")):
            if v is not None:
                _, ls = BENCH_STYLE[kind]
                ax1.axhline(v, color=FIRM_COLORS[fid], ls=ls, lw=1.3, alpha=0.65, zorder=1)
    bench_proxies = [
        Line2D([0], [0], color="#555", ls=BENCH_STYLE[k][1], lw=1.4, label=lbl)
        for k, lbl in (("competitive", "Competitive"), ("nash", "Nash"), ("monopoly", "Monopoly"))
    ]
    _calvano_time_axis(ax1, max_steps, calvano_xticks)
    ax1.set_ylabel("Output quantity (MW)")
    _title_block(ax1, "Firm output converges into the Nash↔Monopoly band", sub)
    _polish(ax1)
    if settled_txt:
        _settled_badge(ax1, "Settled output\n" + "\n".join(settled_txt))
    handles, labels = ax1.get_legend_handles_labels()
    leg = ax1.legend(handles + bench_proxies, labels + [p.get_label() for p in bench_proxies],
                     loc="best", ncol=2, fontsize=9)
    leg.get_frame().set_linewidth(0.8)
    fig1.tight_layout()
    _save(fig1, save_dir, f"calvano_fig1_quantities_h{h}.png")

    # ------------------------------------------------------------------ Fig 2
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    y_hi = 1.15
    final_delta = None
    if dkey_per_firm is None:
        steps, mean, std = aggregate_metric(sessions, dkey, max_steps=max_steps)
        if steps:
            if n > 1:
                ax2.fill_between(steps, mean - std, mean + std, alpha=0.18,
                                 color="#3a3a3a", lw=0, label="±1 std across sessions")
                y_hi = max(y_hi, float(np.nanmax(mean + std)) * 1.08)
            else:
                y_hi = max(y_hi, float(np.nanmax(mean)) * 1.08)
            ax2.plot(steps, mean, color="#222222", lw=2.6, label="Δ combined (market)", zorder=4)
            final_delta = _settled_level(mean)
    else:
        for fid in range(2):
            steps, mean, std = aggregate_metric(sessions, dkey_per_firm.format(fid), max_steps=max_steps)
            if not steps:
                continue
            ax2.plot(steps, mean, color=FIRM_COLORS[fid], lw=2.2, label=f"{FIRM_NAMES[fid]} Δ (legacy)")
            if n > 1:
                ax2.fill_between(steps, mean - std, mean + std, alpha=0.16, color=FIRM_COLORS[fid], lw=0)
                y_hi = max(y_hi, float(np.nanmax(mean + std)) * 1.08)

    ax2.axhline(0, color=BENCH_STYLE["nash"][0], ls=BENCH_STYLE["nash"][1], lw=1.7, alpha=0.9,
                label="Nash (Δ = 0)")
    ax2.axhline(1, color=BENCH_STYLE["monopoly"][0], ls=BENCH_STYLE["monopoly"][1], lw=1.7, alpha=0.9,
                label="Monopoly (Δ = 1)")
    _calvano_time_axis(ax2, max_steps, calvano_xticks)
    ax2.set_ylabel("Combined collusion index  Δ")
    _title_block(
        ax2, "Profits settle between competition and full collusion",
        r"$\Delta=\sum(\pi-\pi^{Nash})\,/\,\sum(\pi^{Mono}-\pi^{Nash})$   ·   " + sub,
    )
    ax2.set_ylim(-0.1, max(y_hi, 1.15))
    _polish(ax2)
    if final_delta is not None and np.isfinite(final_delta):
        _settled_badge(ax2, f"Settled Δ ≈ {final_delta:.2f}\n({final_delta*100:.0f}% of the way\nto full monopoly)")
    _legend(ax2, loc="center right")
    fig2.tight_layout()
    _save(fig2, save_dir, f"calvano_fig2_profit_gain_h{h}.png")

    # ------------------------------------------------------------------ Fig 3
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    steps, mean, std = aggregate_metric(sessions, "avg_lmp", max_steps=max_steps)
    has_lmp_spread = _metrics_has_key(sessions, "lmp_lo")
    lmp_color = "#6a2c91"
    if steps:
        # Same shading recipe as the generation plot: the within-rollout sampled
        # min–max envelope (true price exploration), then the cross-session ±std.
        if has_lmp_spread:
            xs, lo, _ = aggregate_metric(sessions, "lmp_lo", max_steps=max_steps)
            _, hi, _ = aggregate_metric(sessions, "lmp_hi", max_steps=max_steps)
            if xs:
                ax3.fill_between(xs, lo, hi, color=lmp_color, alpha=0.12, lw=0,
                                 label="sampled range (price exploration)")
        if n > 1:
            ax3.fill_between(steps, mean - std, mean + std, alpha=0.22, color=lmp_color, lw=0,
                             label="±1 std across sessions")
        ax3.plot(steps, mean, color=lmp_color, lw=2.6, label="Avg LMP (clearing price)", zorder=4)
        ax3.scatter([steps[0]], [mean[0]], color=ANCHOR_RED, s=42, zorder=6,
                    edgecolor="white", linewidth=0.8, label="t=0 competitive start")

    bench = config.get("benchmarks", {})
    for key, kind, nm in (("competitive", "competitive", "Competitive"),
                          ("cournot_nash", "nash", "Nash"),
                          ("monopoly", "monopoly", "Monopoly")):
        v = bench.get(key, {}).get("avg_lmp")
        if v is not None:
            _benchmark_hline(ax3, v, kind, f"{nm} (${float(v):.1f})")
    _calvano_time_axis(ax3, max_steps, calvano_xticks)
    ax3.set_ylabel("Average LMP ($/MWh)")
    _title_block(ax3, "Clearing price rises above competition and holds", sub)
    _polish(ax3)
    if steps:
        _settled_badge(ax3, f"Settled price ≈ ${_settled_level(mean):.1f}/MWh")
    _legend(ax3, loc="best")
    fig3.tight_layout()
    _save(fig3, save_dir, f"calvano_fig3_lmp_h{h}.png")


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
    if _metrics_has_key(sessions, "delta_combined"):
        steps, mean, std = aggregate_metric(sessions, "delta_combined")
        if steps:
            ax.plot(steps, mean, color="black", label=f"Δ combined{label_suffix}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color="black")
    else:
        for fid in range(2):
            steps, mean, std = aggregate_metric(sessions, f"firm_{fid}_delta")
            if not steps:
                continue
            color = f"C{fid}"
            ax.plot(steps, mean, color=color, label=f"Firm {fid}{label_suffix}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8, label="Nash (Δ=0)")
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8, label="Monopoly (Δ=1)")
    ax.set_ylabel("Δ (combined profit gain)")
    ax.set_title("Evolution of Combined Collusion Index Δ")
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
        dev_idxs = []
        for sess in sessions:
            de = sess.get("deviation_experiment", {})
            entry = de.get(str(dev_fid))
            if not entry:
                continue
            for fid_str in ["0", "1"]:
                traces[fid_str].append(entry["gen"][fid_str])
            if entry.get("dev_index") is not None:
                dev_idxs.append(int(entry["dev_index"]))

        if not traces["0"]:
            ax.text(0.5, 0.5, "No deviation data", transform=ax.transAxes, ha="center")
            continue

        horizon = len(traces["0"][0])
        di = int(round(np.mean(dev_idxs))) if dev_idxs else 0
        t = np.arange(horizon) - di
        ax.axvline(0, color="red", ls="--", alpha=0.45, lw=0.9)

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
    devidx_by_dev = {str(d): [] for d in range(num_firms)}
    punish_by_dev = {str(d): [] for d in range(num_firms)}

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
            if entry.get("dev_index") is not None:
                devidx_by_dev[dev_str].append(int(entry["dev_index"]))
            if entry.get("punishment") is not None:
                punish_by_dev[dev_str].append(entry["punishment"])

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

    n = len(sessions)
    sub = f"H={h}  ·  {n} session{'s' if n != 1 else ''}"

    for dev_str in deviators:
        # Deviation index (pre-deviation resting periods sit at negative time).
        di = int(round(np.mean(devidx_by_dev[dev_str]))) if devidx_by_dev[dev_str] else 0

        # Punishment summary across sessions (the rival's reaction to the cheat).
        punishes = punish_by_dev[dev_str]
        punish_txt = ""
        accommodates = True
        if punishes:
            frac = float(np.mean([1.0 if p.get("punished") else 0.0 for p in punishes]))
            inc = float(np.mean([p.get("rival_output_increase_mw", 0.0) for p in punishes]))
            drop = float(np.mean([p.get("lmp_drop", 0.0) for p in punishes]))
            accommodates = frac < 0.5
            verdict = ("PUNISHMENT — rival floods, price war" if not accommodates
                       else "ACCOMMODATION — no retaliation")
            punish_txt = (
                f"Rival retaliated in {frac:.0%} of seeds\n"
                f"rival output {inc:+.1f} MW · price -${drop:.2f}\n{verdict}"
            )

        # ----------------------------- Generation response figure -----------------
        fig, ax = plt.subplots(figsize=(11, 6))
        for fstr in [str(f) for f in range(num_firms)]:
            mats = gen_by_dev[dev_str][fstr]
            if not mats:
                continue
            T = min(len(arr) for arr in mats)
            matrix = np.stack([arr[:T] for arr in mats])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            t = np.arange(T) - di
            is_dev = fstr == dev_str
            color = FIRM_COLORS[int(fstr)]
            lbl = f"{FIRM_NAMES[int(fstr)]}" + (" — deviator" if is_dev else " — rival (response)")
            if matrix.shape[0] > 1:
                ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15, lw=0)
            ax.plot(t, mean, color=color, lw=2.6 if is_dev else 1.8,
                    ls="-" if is_dev else (0, (5, 2)), label=lbl, zorder=4 if is_dev else 3)
            rests = resting_by_dev[dev_str][fstr]
            if rests:
                ax.axhline(float(np.mean(rests)), color=color, ls=":", alpha=0.55, lw=1.0,
                           label=f"{FIRM_NAMES[int(fstr)]} resting" + (" (pre-deviation)" if is_dev else ""))
        ax.axvline(0, color=ANCHOR_RED, ls="--", alpha=0.65, lw=1.3)
        _title_block(ax, f"{FIRM_NAMES[int(dev_str)]} deviates → does the rival retaliate?", sub)
        ax.set_xlabel("Period after deviation")
        ax.set_ylabel("Total output (MW)")
        _polish(ax)
        badge = f"t=0: deviator forced +{dev_frac:.0%} for one step"
        if punish_txt:
            badge += "\n" + punish_txt
        _settled_badge(ax, badge)
        _legend(ax, loc="lower right" if accommodates else "best")
        fig.tight_layout()
        _save(fig, save_dir, f"deviation_gen_firm{dev_str}_h{h}.png")

        # ----------------------------- Price (LMP) response figure -----------------
        fig, ax = plt.subplots(figsize=(11, 6))
        mats = lmp_by_dev[dev_str]
        if mats:
            T = min(len(arr) for arr in mats)
            matrix = np.stack([arr[:T] for arr in mats])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            t = np.arange(T) - di
            if matrix.shape[0] > 1:
                ax.fill_between(t, mean - std, mean + std, color="#6a2c91", alpha=0.14, lw=0,
                                label="±1 std across sessions")
            ax.plot(t, mean, color="#6a2c91", lw=2.6, label="Avg LMP (qty-weighted)", zorder=4)
            if di < len(mean):
                ax.scatter([0], [mean[di]], color=ANCHOR_RED, zorder=6, s=60, edgecolor="white",
                           linewidth=0.9, label=f"At deviation: ${mean[di]:.2f}")
                ax.scatter([T - 1 - di], [mean[-1]], color="#2e8b57", zorder=6, s=60,
                           edgecolor="white", linewidth=0.9,
                           label=f"After {T - 1 - di} periods: ${mean[-1]:.2f}")
        _benchmark_hline(ax, comp_lmp, "competitive",
                         f"Competitive (${comp_lmp:.2f})" if comp_lmp is not None else "Competitive")
        _benchmark_hline(ax, mono_lmp, "monopoly",
                         f"Monopoly (${mono_lmp:.2f})" if mono_lmp is not None else "Monopoly")
        ax.axvline(0, color=ANCHOR_RED, ls="--", alpha=0.65, lw=1.3)
        _title_block(ax, f"{FIRM_NAMES[int(dev_str)]} deviates → price response", sub)
        ax.set_xlabel("Period after deviation")
        ax.set_ylabel("Average LMP ($/MWh)")
        _polish(ax)
        _legend(ax, loc="best")
        fig.tight_layout()
        _save(fig, save_dir, f"deviation_lmp_firm{dev_str}_h{h}.png")


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

    delta_metric = (
        "delta_combined"
        if any(_metrics_has_key(s, "delta_combined") for _, s in runs)
        else "firm_0_delta"
    )

    # --- (0,0): Combined Δ (Nash floor) ---
    ax = axes[0, 0]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, delta_metric)
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Δ combined" if delta_metric == "delta_combined" else "Δ (legacy F0)")
    ax.set_title("Combined Collusion Index Δ")
    ax.legend(fontsize=8)

    # --- (0,1): Greedy Δ or per-firm step profit ---
    ax = axes[0, 1]
    greedy_key = "greedy_delta_combined"
    if any(_metrics_has_key(s, greedy_key) for _, s in runs):
        for config, sessions in runs:
            h = config.get("history_len", "?")
            color = colors_h[str(h)]
            steps, mean, std = aggregate_metric(
                sessions, greedy_key, default_for_missing=float("nan")
            )
            if steps:
                ax.plot(steps, mean, color=color, label=f"H={h}")
                if len(sessions) > 1:
                    ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
        ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
        ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
        ax.set_ylabel("Greedy Δ_comb")
        ax.set_title("Greedy-policy Δ (one-shot clear)")
    else:
        for config, sessions in runs:
            h = config.get("history_len", "?")
            color = colors_h[str(h)]
            for fid, ls in ((0, "-"), (1, "--")):
                steps, mean, std = aggregate_metric(
                    sessions, f"firm_{fid}_avg_step_profit", default_for_missing=float("nan")
                )
                if steps:
                    ax.plot(steps, mean, color=color, ls=ls, label=f"H={h} F{fid}")
        ax.set_ylabel("Avg profit ($/step)")
        ax.set_title("Per-firm step profit")
    ax.legend(fontsize=7)

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
    _add_lmp_benchmark_lines(ax, runs[0][0].get("benchmarks", {}))
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


def _add_lmp_benchmark_lines(ax, bench: dict):
    if not bench:
        return
    ax.axhline(
        bench["competitive"]["avg_lmp"],
        ls="--",
        color="green",
        alpha=0.5,
        linewidth=0.8,
        label="Competitive LMP",
    )
    if "cournot_nash" in bench:
        ax.axhline(
            bench["cournot_nash"]["avg_lmp"],
            ls="-.",
            color="orange",
            alpha=0.6,
            linewidth=0.8,
            label="Cournot–Nash LMP",
        )
    ax.axhline(
        bench["monopoly"]["avg_lmp"],
        ls=":",
        color="red",
        alpha=0.5,
        linewidth=0.8,
        label="Monopoly LMP",
    )


def plot_comparison_delta(run_dirs, save_dir=None):
    """
    Cross-history dashboard for delta-mode runs only.

    Uses combined Δ and |Δ_comb − Δ_comb,prev| convergence diagnostic
    (--convergence-mode delta).

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

    delta_metric = (
        "delta_combined"
        if any(_metrics_has_key(s, "delta_combined") for _, s in runs)
        else "firm_0_delta"
    )

    # --- (0,0): Combined Δ ---
    ax = axes[0, 0]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        steps, mean, std = aggregate_metric(sessions, delta_metric)
        if steps:
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
    ax.axhline(0, ls="--", color="grey", alpha=0.5, linewidth=0.8)
    ax.axhline(1, ls="--", color="black", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Δ combined")
    ax.set_title("Combined Collusion Index Δ")
    ax.legend(fontsize=8)

    # --- (0,1): Δ-comb jump (convergence) ---
    ax = axes[0, 1]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        jump_key = (
            "delta_combined_jump"
            if _metrics_has_key(sessions, "delta_combined_jump")
            else "delta_max_jump"
        )
        steps, mean, std = aggregate_metric(
            sessions, jump_key, default_for_missing=float("nan")
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
        label=f"threshold ({delta_thresh})",
    )
    ax.set_ylabel("|Δ_comb − Δ_comb,prev|")
    ax.set_yscale("log")
    ax.set_title("Δ-Stability Convergence Diagnostic")
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
    _add_lmp_benchmark_lines(ax, runs[0][0].get("benchmarks", {}))
    ax.set_ylabel("Avg LMP ($/MWh)")
    ax.set_title("Average LMP")
    ax.legend(fontsize=8)

    # --- (1,0): Generation Firm 0 ---
    ax = axes[1, 0]
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

    # --- (1,1): Generation Firm 1 ---
    ax = axes[1, 1]
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

    # --- (1,2): total profit per step (F0 + F1 episode profit / episode_len) ---
    ax = axes[1, 2]
    for config, sessions in runs:
        h = config.get("history_len", "?")
        color = colors_h[str(h)]
        ep_len = float(config.get("episode_len", 168) or 168)
        steps_list, totals = [], []
        for sess in sessions:
            for row in sess.get("metrics") or []:
                if "step" not in row:
                    continue
                p0 = row.get("firm_0_ep_profit", 0) / ep_len
                p1 = row.get("firm_1_ep_profit", 0) / ep_len
                steps_list.append(row["step"])
                totals.append(p0 + p1)
        if steps_list:
            ax.plot(steps_list, totals, color=color, alpha=0.5, label=f"H={h} (per session)")
    bench = runs[0][0].get("benchmarks", {})
    if bench and "cournot_nash" in bench:
        ax.axhline(
            bench["cournot_nash"]["total_profit"],
            ls="--",
            color="grey",
            alpha=0.6,
            label="Nash total π",
        )
        ax.axhline(
            bench["monopoly"]["total_profit"],
            ls=":",
            color="black",
            alpha=0.6,
            label="Monopoly total π",
        )
    ax.set_ylabel("Total profit ($/step)")
    ax.set_title("Industry profit")
    ax.legend(fontsize=7)

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


def plot_variance_funnel(config, sessions, save_dir: Path, history_label=None):
    """Cross-session generation band vs PPO iteration (log-x), plus band width (std).

    Surfaces the early high-variance fan-out: every session starts at the (pinned)
    competitive point, so the band is thin at iteration 1, widens sharply as sessions
    diverge during early learning, then narrows as they converge. Top row = mean ± std
    bands; bottom row = the cross-session std (band width) over iterations.
    """
    if not sessions:
        print("No sessions for variance funnel.")
        return
    h = history_label if history_label is not None else config.get("history_len", "?")
    n = len(sessions)
    use_greedy = (
        not _metrics_has_key(sessions, "firm_0_avg_gen")
        and _metrics_has_key(sessions, "firm_0_greedy_gen")
    )
    gkey = "firm_{}_greedy_gen" if use_greedy else "firm_{}_avg_gen"
    levels = _firm_benchmark_levels(config)
    sub = f"H={h}  ·  {n} session{'s' if n != 1 else ''}"

    series = {}
    for fid in range(2):
        x, mean, std = _aggregate_metric_by_iteration(sessions, gkey.format(fid))
        series[fid] = (np.asarray(x, float), np.asarray(mean, float), np.asarray(std, float)) if x else None

    # --- One standalone cross-session BAND figure per firm (its own MW scale) ---
    for fid in range(2):
        if series[fid] is None:
            continue
        x, mean, std = series[fid]
        c = FIRM_COLORS[fid]
        fig, ax = plt.subplots(figsize=(11, 6))
        if n > 1:
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.22, lw=0,
                            label="±1 std across sessions")
        ax.plot(x, mean, color=c, lw=2.4, label=f"{FIRM_NAMES[fid]} mean (across sessions)", zorder=4)
        for kind, nm in (("competitive", "Competitive"), ("nash", "Nash"), ("monopoly", "Monopoly")):
            _benchmark_hline(ax, levels[fid][kind], kind, nm)
        ax.set_xscale("log")
        _title_block(ax, f"{FIRM_NAMES[fid]} cross-session fan-out: diverge early, re-converge late", sub)
        ax.set_xlabel("PPO iteration (log scale)")
        ax.set_ylabel("Output quantity (MW)")
        ax.grid(True, which="both", alpha=0.5)
        _polish(ax)
        _legend(ax, loc="best")
        fig.tight_layout()
        _save(fig, save_dir, f"variance_band_firm{fid}_h{h}.png")

    # --- One standalone BAND-WIDTH (std) figure — both firms share MW-of-std units ---
    fig, ax = plt.subplots(figsize=(11, 6))
    drew = False
    for fid in range(2):
        if series[fid] is None:
            continue
        x, _, std = series[fid]
        c = FIRM_COLORS[fid]
        ax.fill_between(x, 0, std, color=c, alpha=0.16, lw=0)
        ax.plot(x, std, color=c, lw=2.4, label=f"{FIRM_NAMES[fid]} band width (std)", zorder=4)
        drew = True
    if drew:
        ax.set_xscale("log")
    _title_block(ax, "Cross-session disagreement collapses as the seeds converge", sub)
    ax.set_xlabel("PPO iteration (log scale)")
    ax.set_ylabel("Std of output across sessions (MW)")
    ax.grid(True, which="both", alpha=0.5)
    _polish(ax)
    _legend(ax, loc="best")
    fig.tight_layout()
    _save(fig, save_dir, f"variance_std_h{h}.png")


def plot_per_firm_profit_vs_benchmarks(config, sessions, save_dir: Path, history_label=None):
    """Per-firm profit vs PPO iteration with that firm's OWN competitive/Nash/monopoly lines.

    Makes the asymmetry explicit: Firm 1's Nash profit can exceed its monopoly profit, so the
    joint-monopoly allocation is not individually rational for it — which is exactly what
    bounds the combined Δ below 1.
    """
    if not sessions:
        print("No sessions for per-firm profit plot.")
        return
    if not _metrics_has_key(sessions, "firm_0_avg_step_profit"):
        print("No per-step profit metric (firm_*_avg_step_profit) in sessions.")
        return

    h = history_label if history_label is not None else config.get("history_len", "?")
    n = len(sessions)
    bench = config.get("benchmarks", {})
    comp = bench.get("competitive", {}).get("profits", {})
    nash = bench.get("cournot_nash", {}).get("profits", {})
    mono = bench.get("monopoly", {}).get("profits", {})
    sub = f"H={h}  ·  {n} session{'s' if n != 1 else ''}"

    # One standalone figure per firm — each firm's profit has its own scale, so they
    # never share an axis.
    for fid in range(2):
        fig, ax = plt.subplots(figsize=(11, 6))
        c = FIRM_COLORS[fid]
        x, mean, std = _aggregate_metric_by_iteration(sessions, f"firm_{fid}_avg_step_profit")
        settled = None
        if x:
            x = np.asarray(x, float)
            mean = np.asarray(mean, float)
            std = np.asarray(std, float)
            if n > 1:
                ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.20, lw=0,
                                label="±1 std across sessions")
            ax.plot(x, mean, color=c, lw=2.4, label=f"{FIRM_NAMES[fid]} profit (mean)", zorder=4)
            settled = _settled_level(mean)
        for table, kind, name in ((comp, "competitive", "Competitive"),
                                  (nash, "nash", "Nash"),
                                  (mono, "monopoly", "Monopoly")):
            if str(fid) in table:
                _benchmark_hline(ax, float(table[str(fid)]), kind,
                                 f"{name} (${float(table[str(fid)]):,.0f})")
        _title_block(ax, f"{FIRM_NAMES[fid]} profit vs its own competitive / Nash / monopoly benchmarks", sub)
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel("Profit ($/step)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        _polish(ax)
        if settled is not None and np.isfinite(settled):
            _settled_badge(ax, f"Settled ≈ ${settled:,.0f}/step")
        _legend(ax, loc="best")
        fig.tight_layout()
        _save(fig, save_dir, f"per_firm_profit_firm{fid}_h{h}.png")


def plot_exploration_funnel(config, sessions, save_dir: Path, history_label=None):
    """Per-firm exploration funnel: the ACTUAL within-rollout sampled-output spread
    (min–max shaded + ±std), wide early (haphazard exploration) and narrowing late
    (exploitation). Starts at the competitive t=0 anchor.

    Uses firm_*_gen_lo/hi/std — what the agents actually TRIED — not the smoothed greedy
    center or cross-session variance. Produces a full-range figure and an 'extremely
    zoomed' figure (tight y around the Nash↔Monopoly collusion band) for the advisor.
    """
    if not sessions:
        print("No sessions for exploration funnel.")
        return
    if not _metrics_has_key(sessions, "firm_0_gen_std"):
        print("No sampled-spread metrics (firm_*_gen_std). Re-run ppo.py (it logs them now).")
        return
    h = history_label if history_label is not None else config.get("history_len", "?")
    n = len(sessions)
    caps = (200.0, 100.0)
    levels = _firm_benchmark_levels(config)
    sub = f"H={h}  ·  {n} session{'s' if n != 1 else ''}"

    def agg(fid, k):
        steps, mean, _ = aggregate_metric(sessions, f"firm_{fid}_{k}")
        return np.asarray(steps, float), np.asarray(mean, float)

    # One standalone figure per firm, at full range and zoomed to the collusion band.
    for fid in range(2):
        x, center = agg(fid, "avg_gen")
        _, lo = agg(fid, "gen_lo")
        _, hi = agg(fid, "gen_hi")
        _, sd = agg(fid, "gen_std")
        c = FIRM_COLORS[fid]
        comp = levels[fid]["competitive"]
        nash = levels[fid]["nash"]
        mono = levels[fid]["monopoly"]
        settled = _settled_level(center)

        for zoom in (False, True):
            fig, ax = plt.subplots(figsize=(11, 6))
            mlh = np.isfinite(lo) & np.isfinite(hi)
            ax.fill_between(x[mlh], lo[mlh], hi[mlh], color=c, alpha=0.12, lw=0,
                            label="sampled range (min–max per rollout)")
            ms = np.isfinite(sd) & np.isfinite(center)
            ax.fill_between(x[ms], (center - sd)[ms], (center + sd)[ms], color=c, alpha=0.30, lw=0,
                            label="±1 std (exploration width)")
            ax.plot(x, center, color=c, lw=2.4, label=f"{FIRM_NAMES[fid]} mean sampled output", zorder=4)
            for v, kind, nm in ((comp, "competitive", "Competitive"),
                                (nash, "nash", "Nash"),
                                (mono, "monopoly", "Monopoly")):
                _benchmark_hline(ax, v, kind, f"{nm} ({v:.0f} MW)" if v is not None else nm)
            if len(center):
                ax.scatter([x[0]], [center[0]], color=ANCHOR_RED, zorder=6, s=46,
                           edgecolor="white", linewidth=0.8,
                           label=f"t=0 competitive ({center[0]:.0f} MW)")
            if zoom:
                vals = [v for v in (nash, mono, settled) if v is not None and np.isfinite(v)]
                pad = max(8.0, 0.15 * (max(vals) - min(vals))) if len(vals) > 1 else 10.0
                ax.set_ylim(min(vals) - pad, max(vals) + pad)
                title = f"{FIRM_NAMES[fid]} — zoom on the Nash↔Monopoly collusion band"
            else:
                ax.set_ylim(-4, caps[fid] + 6)
                title = f"{FIRM_NAMES[fid]} exploration funnel: wide explore → narrow exploit"
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Output quantity (MW)")
            _title_block(ax, title, sub)
            _polish(ax)
            _settled_badge(ax, f"Settled ≈ {settled:.0f} MW")
            _legend(ax, loc="best")
            fig.tight_layout()
            tag = "_zoom" if zoom else ""
            _save(fig, save_dir, f"exploration_funnel_firm{fid}{tag}_h{h}.png")


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
    parser.add_argument(
        "--variance-funnel",
        action="store_true",
        help="Cross-session generation band + std vs PPO iteration (log-x): shows the early "
        "high-variance fan-out narrowing on convergence (one PNG per run dir).",
    )
    parser.add_argument(
        "--per-firm-profit",
        action="store_true",
        help="Per-firm profit vs PPO iteration with each firm's own competitive/Nash/monopoly "
        "lines (one PNG per run dir).",
    )
    parser.add_argument(
        "--exploration-funnel",
        action="store_true",
        help="Per-firm exploration funnel from the ACTUAL sampled-output spread "
        "(firm_*_gen_lo/hi/std): wide haphazard exploration narrowing to exploitation. "
        "Writes a full-range PNG and an extremely-zoomed PNG (tight on the Nash↔Monopoly "
        "band) per run dir.",
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

    if args.variance_funnel:
        if not args.save:
            parser.error("--variance-funnel requires --save DIR")
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for rd in run_dirs:
            config, sessions = load_sessions(rd)
            h = config.get("history_len", "?")
            plot_variance_funnel(config, sessions, save_dir, history_label=h)
        return

    if args.per_firm_profit:
        if not args.save:
            parser.error("--per-firm-profit requires --save DIR")
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for rd in run_dirs:
            config, sessions = load_sessions(rd)
            h = config.get("history_len", "?")
            plot_per_firm_profit_vs_benchmarks(config, sessions, save_dir, history_label=h)
        return

    if args.exploration_funnel:
        if not args.save:
            parser.error("--exploration-funnel requires --save DIR")
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for rd in run_dirs:
            config, sessions = load_sessions(rd)
            h = config.get("history_len", "?")
            plot_exploration_funnel(config, sessions, save_dir, history_label=h)
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
