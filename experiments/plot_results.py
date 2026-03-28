"""
Calvano-style plots for PPO collusion experiments.

Generates four figures:
  1. Evolution of generation quantities (cf. Calvano Fig 1)
  2. Evolution of Δ normalized profit gain (cf. Calvano Fig 2)
  3. Learned limit strategy: generation vs observed LMP (cf. Calvano Fig 3)
  4. Impulse response after forced deviation (cf. Calvano Fig 4)

Usage
-----
    python experiments/plot_results.py results/h1
    python experiments/plot_results.py results/h1 results/h3 --save figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def aggregate_metric(sessions, key, max_steps=None):
    """Collect a metric across sessions, aligned by step. Returns (steps, mean, std)."""
    all_series = []
    for sess in sessions:
        steps = [r["step"] for r in sess["metrics"]]
        vals = [r.get(key, 0) for r in sess["metrics"]]
        all_series.append((steps, vals))

    if not all_series:
        return [], [], []

    # Use the longest session's step grid
    ref_steps = max(all_series, key=lambda x: len(x[0]))[0]
    if max_steps:
        ref_steps = [s for s in ref_steps if s <= max_steps]

    matrix = []
    for steps, vals in all_series:
        interpolated = np.interp(ref_steps, steps, vals,
                                 left=vals[0] if vals else 0,
                                 right=vals[-1] if vals else 0)
        matrix.append(interpolated)

    matrix = np.array(matrix)
    return ref_steps, matrix.mean(axis=0), matrix.std(axis=0)


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

    for fid, (comp_g, mono_g) in enumerate([(comp_g0, comp_g0), (comp_g1, comp_g1)]):
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


# ====================== Figure 5: KL divergence evolution ======================
def plot_kl(ax, config, sessions, label_suffix=""):
    for fid in range(2):
        steps, mean, std = aggregate_metric(sessions, f"firm_{fid}_kl")
        if not steps:
            continue
        color = f"C{fid}"
        ax.plot(steps, mean, color=color, label=f"Firm {fid}{label_suffix}")
        if len(sessions) > 1:
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    kl_thresh = config.get("kl_threshold", 0.01)
    ax.axhline(kl_thresh, ls="--", color="red", alpha=0.5, linewidth=0.8,
               label=f"KL threshold ({kl_thresh})")
    ax.set_ylabel("KL divergence")
    ax.set_yscale("log")
    ax.set_title("Policy KL Divergence (old → new)")
    ax.legend(fontsize=8)


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
            ax.plot(steps, mean, color=color, label=f"H={h}")
            if len(sessions) > 1:
                ax.fill_between(steps, mean - std, mean + std, alpha=0.1, color=color)
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


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Calvano-style plots for PPO collusion")
    parser.add_argument("run_dirs", nargs="*", type=Path,
                        help="One or more run directories to plot")
    parser.add_argument("--compare", action="store_true",
                        help="Generate a cross-history comparison figure")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save figures (PNG). If omitted, shows interactively.")
    args = parser.parse_args()

    if not args.run_dirs:
        parser.error("Provide at least one run directory.")

    if args.compare:
        plot_comparison(args.run_dirs, save_dir=args.save)
        return

    for rd in args.run_dirs:
        config, sessions = load_sessions(rd)
        h = config.get("history_len", "?")
        n = len(sessions)

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
