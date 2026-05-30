"""
KKT verification for the ISO welfare-maximization (DC-OPF) problem.

Motivation (advisor meeting, 2026-05-19)
---------------------------------------
Early training shows avg LMP rising *above* the monopoly-LMP benchmark. The
advisor asked for a hand-check that this region is actually feasible / optimal
for the welfare problem and not an artifact of the solver. Concretely:

    "the only way you can check is to see if the KKT holds, right?"

This script does that.

Welfare problem (given firms' per-plant generation g)
-----------------------------------------------------
    max_{d >= 0}  Σ_i [ P0_i d_i  -  ½ (P0_i / Q0_i) d_i² ]
    s.t.   Σ_i (g_i - d_i)   = 0                 (dual: λ, free)
           PTDF (g - d)     <= L                 (dual: μ ≥ 0)
           PTDF (g - d)     >= -L                (dual: ν ≥ 0)
           d                >= 0                 (dual: α ≥ 0)

LMP_i = P0_i - (P0_i / Q0_i) d_i

Stationarity → for every node i:
    LMP_i = λ + [(μ - ν) PTDF]_i - α_i

CLI modes
---------
* single-point check
    python experiments/kkt_check.py check --plant-gen 115,50,30
    python experiments/kkt_check.py check --firm-gen 165,30
    python experiments/kkt_check.py check --preset monopoly
    python experiments/kkt_check.py check --preset competitive

* scan over firm totals → average-LMP heatmap with monopoly contour
    python experiments/kkt_check.py scan \
        --grid 41 --save figures/kkt/avg_lmp_scan.png

* read a trained session's early-iteration generation and verify KKT there
    python experiments/kkt_check.py from-session \
        --session results/delta/h1/sessions/session_0 --iteration 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import cvxpy as cp

_mpl_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ppo-collusion-matplotlib-cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.node_network import P0, Q0, get_ptdf_matrix, LINE_LIMITS  # noqa: E402
from iso_market.market_env import (  # noqa: E402
    NUM_FIRMS,
    NUM_NODES,
    FIRM_PLANT_IDX,
    PLANTS,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def firm_to_plant_mw(g_f0_total: float, g_f1_total: float) -> np.ndarray:
    """
    Marginal-cost-equalizing allocation across each firm's plants.

    Firm 0 owns plants 0 and 1 (both: MC=15, QC=0.02; caps 150 and 50).
    Equal MC ⇒ equalize g across the two plants up to the smaller cap (50),
    then push the residual into plant 0.

    Firm 1 owns plant 2 (cap 100).
    """
    cap = np.array([p["cap"] for p in PLANTS], dtype=np.float64)

    g0 = float(np.clip(g_f0_total, 0.0, cap[0] + cap[1]))
    if g0 <= 2 * cap[1]:
        p0, p1 = g0 / 2, g0 / 2
    else:
        p1 = cap[1]
        p0 = min(g0 - p1, cap[0])

    p2 = float(np.clip(g_f1_total, 0.0, cap[2]))
    return np.array([p0, p1, p2], dtype=np.float64)


def aggregate_plant_to_node(g_per_plant: np.ndarray) -> np.ndarray:
    g_per_node = np.zeros(NUM_NODES)
    for pidx, plant in enumerate(PLANTS):
        g_per_node[plant["node"]] += float(g_per_plant[pidx])
    return g_per_node


def firm_profits(g_per_plant: np.ndarray, lmps: np.ndarray) -> dict:
    out = {}
    for fid in range(NUM_FIRMS):
        pi = 0.0
        for pidx in FIRM_PLANT_IDX[fid]:
            p = PLANTS[pidx]
            g = float(g_per_plant[pidx])
            pi += lmps[p["node"]] * g - p["mc"] * g - 0.5 * p["qc"] * g * g
        out[fid] = pi
    return out


def quantity_weighted_lmp(lmps: np.ndarray, d: np.ndarray) -> float:
    s = float(np.sum(d))
    return float(np.sum(lmps * d) / s) if s > 1e-10 else 0.0


# ----------------------------------------------------------------------
# Core: welfare solve with explicit duals, and KKT residual report
# ----------------------------------------------------------------------
def solve_welfare_with_duals(g_per_plant: np.ndarray) -> dict | None:
    """Solve the welfare DC-OPF; return primal + dual variables."""
    ptdf = get_ptdf_matrix()
    line_limits = LINE_LIMITS.astype(np.float64)
    g_per_node = aggregate_plant_to_node(g_per_plant)

    # d declared FREE (not nonneg) so the dual α of d ≥ 0 is explicit.
    d = cp.Variable(NUM_NODES)
    y = g_per_node - d

    benefit = cp.sum(cp.multiply(P0, d) - 0.5 * cp.multiply(P0 / Q0, cp.square(d)))

    c_balance = cp.sum(y) == 0
    c_flow_up = ptdf @ y <= line_limits
    c_flow_lo = ptdf @ y >= -line_limits
    c_d_nonneg = d >= 0

    prob = cp.Problem(
        cp.Maximize(benefit),
        [c_balance, c_flow_up, c_flow_lo, c_d_nonneg],
    )
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate") or d.value is None:
        return None

    flows = ptdf @ (g_per_node - d.value)
    lmps = P0 - (P0 / Q0) * d.value
    return {
        "status": prob.status,
        "g_per_plant": g_per_plant.astype(np.float64).copy(),
        "g_per_node": g_per_node,
        "d": np.asarray(d.value, dtype=np.float64),
        "flows": np.asarray(flows, dtype=np.float64),
        "line_limits": line_limits,
        "lmps": np.asarray(lmps, dtype=np.float64),
        "avg_lmp_qty_weighted": quantity_weighted_lmp(lmps, d.value),
        "ptdf": ptdf,
        # Duals
        "lambda": float(c_balance.dual_value),
        "mu_up": np.asarray(c_flow_up.dual_value, dtype=np.float64),
        "mu_lo": np.asarray(c_flow_lo.dual_value, dtype=np.float64),
        "alpha": np.asarray(c_d_nonneg.dual_value, dtype=np.float64),
    }


def check_kkt(sol: dict, *, tol: float = 1e-5) -> dict:
    """Verify primal, dual, stationarity, complementary-slackness conditions."""
    d = sol["d"]
    lmps = sol["lmps"]
    flows = sol["flows"]
    L = sol["line_limits"]
    g_per_node = sol["g_per_node"]
    lam = float(sol["lambda"])
    mu = sol["mu_up"]
    nu = sol["mu_lo"]
    alpha = sol["alpha"]
    ptdf = sol["ptdf"]
    y = g_per_node - d

    rpt = {}

    # 1. Primal feasibility
    rpt["primal/balance_residual"] = float(abs(np.sum(y)))
    rpt["primal/flow_up_max_viol"] = float(max(0.0, float(np.max(flows - L))))
    rpt["primal/flow_lo_max_viol"] = float(max(0.0, float(np.max(-L - flows))))
    rpt["primal/d_neg_max_viol"] = float(max(0.0, -float(np.min(d))))

    # 2. Dual feasibility
    rpt["dual/mu_up_min"] = float(np.min(mu))
    rpt["dual/mu_lo_min"] = float(np.min(nu))
    rpt["dual/alpha_min"] = float(np.min(alpha))

    # 3. Stationarity:  LMP_i = λ − ((μ−ν) PTDF)_i − α_i
    # CVXPY reports the balance-constraint dual with a sign flip relative to
    # the textbook system price; we restore the textbook convention here.
    lam_textbook = -lam
    cong = (mu - nu) @ ptdf
    pred_lmps = lam_textbook - cong - alpha
    rpt["stationarity/per_node_residual_max"] = float(np.max(np.abs(lmps - pred_lmps)))
    rpt["stationarity/per_node_residual_l2"] = float(np.linalg.norm(lmps - pred_lmps))
    rpt["KKT/lambda_cvxpy"] = float(lam)
    rpt["KKT/lambda_textbook"] = float(lam_textbook)

    # 4. Complementary slackness  (component-wise products should be ≈ 0)
    rpt["cs/mu_up_max"] = float(np.max(np.abs(mu * (L - flows))))
    rpt["cs/mu_lo_max"] = float(np.max(np.abs(nu * (L + flows))))
    rpt["cs/alpha_max"] = float(np.max(np.abs(alpha * d)))

    # Summary: collapse to a single worst-residual / pass-fail
    residual_keys = [
        "primal/balance_residual",
        "primal/flow_up_max_viol",
        "primal/flow_lo_max_viol",
        "primal/d_neg_max_viol",
        "stationarity/per_node_residual_max",
        "cs/mu_up_max",
        "cs/mu_lo_max",
        "cs/alpha_max",
    ]
    sign_keys = ["dual/mu_up_min", "dual/mu_lo_min", "dual/alpha_min"]
    worst_residual = max(rpt[k] for k in residual_keys)
    worst_sign = max(-rpt[k] for k in sign_keys)  # positive means a sign violation
    rpt["KKT/max_residual"] = float(worst_residual)
    rpt["KKT/max_sign_violation"] = float(worst_sign)
    rpt["KKT/satisfied"] = bool(worst_residual < tol and worst_sign < tol)
    rpt["KKT/pred_lmps"] = pred_lmps.tolist()
    return rpt


# ----------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------
def _fmt_vec(v, w=8, p=3):
    return "[" + ", ".join(f"{x:>{w}.{p}f}" for x in v) + "]"


def print_kkt_report(sol: dict, rpt: dict, *, header: str = "") -> None:
    g_per_plant = sol["g_per_plant"]
    g_per_node = sol["g_per_node"]
    d = sol["d"]
    lmps = sol["lmps"]
    flows = sol["flows"]
    L = sol["line_limits"]
    profits = firm_profits(g_per_plant, lmps)

    bar = "=" * 78
    print()
    print(bar)
    if header:
        print(f"  {header}")
        print(bar)

    print("\nGeneration (per plant, MW):")
    for pidx, plant in enumerate(PLANTS):
        print(
            f"  plant {pidx}  (Firm {plant['firm']} @ Node {plant['node']+1}, "
            f"cap={plant['cap']:.0f}, mc={plant['mc']}, qc={plant['qc']}): "
            f"{g_per_plant[pidx]:7.3f} MW"
        )
    print(f"  total firm 0       = {sum(g_per_plant[i] for i in FIRM_PLANT_IDX[0]):.3f} MW")
    print(f"  total firm 1       = {sum(g_per_plant[i] for i in FIRM_PLANT_IDX[1]):.3f} MW")
    print(f"  total system gen   = {float(g_per_plant.sum()):.3f} MW")

    print("\nGeneration (per node, MW):  " + _fmt_vec(g_per_node))
    print("Demand     (per node, MW):  " + _fmt_vec(d))
    print("Net inj    y = g − d     :  " + _fmt_vec(sol["g_per_node"] - d))
    print(f"Balance Σy               :  {float(np.sum(sol['g_per_node'] - d)):.3e}")

    print("\nLine flows vs. ±limit (MW):")
    for k in range(len(flows)):
        bind = " *" if abs(flows[k]) >= L[k] - 1e-6 else "  "
        print(f"  line {k}: flow={flows[k]:8.3f}  limit=±{L[k]:.1f}{bind}")

    print("\nLMPs (per node, $/MWh):     " + _fmt_vec(lmps, p=4))
    print(f"Avg LMP (qty-weighted)     : ${sol['avg_lmp_qty_weighted']:.4f}/MWh")

    print(
        "\nFirm profit per step ($):"
        f"   Firm 0 = {profits[0]:9.3f}"
        f"   Firm 1 = {profits[1]:9.3f}"
        f"   total = {profits[0] + profits[1]:9.3f}"
    )

    print("\nDuals:")
    print(f"  λ (balance)                : {sol['lambda']:.6f}")
    print("  μ (flow upper)              : " + _fmt_vec(sol["mu_up"], p=4))
    print("  ν (flow lower)              : " + _fmt_vec(sol["mu_lo"], p=4))
    print("  α (d ≥ 0, per node)         : " + _fmt_vec(sol["alpha"], p=4))

    cong = (sol["mu_up"] - sol["mu_lo"]) @ sol["ptdf"]
    print(
        "\nStationarity reconstruction  LMP_i = λ − ((μ−ν) PTDF)_i − α_i"
        "   (λ here = textbook system price = −CVXPY's balance-dual):"
    )
    print(f"  λ_cvxpy                     : {rpt['KKT/lambda_cvxpy']:.6f}")
    print(f"  λ_textbook = −λ_cvxpy       : {rpt['KKT/lambda_textbook']:.6f}")
    print("  (μ−ν) PTDF                  : " + _fmt_vec(cong, p=4))
    print("  reconstructed LMPs          : " + _fmt_vec(np.array(rpt["KKT/pred_lmps"]), p=4))
    print("  realized LMPs               : " + _fmt_vec(lmps, p=4))
    print(f"  max per-node residual       : {rpt['stationarity/per_node_residual_max']:.3e}")

    print("\nKKT residuals (smaller is better; sign should be ≥ 0 for duals):")
    width = 38
    for k, v in rpt.items():
        if k.startswith("KKT/"):
            continue
        print(f"  {k:<{width}} {v:.3e}")

    print("\n" + "—" * 78)
    flag = "YES" if rpt["KKT/satisfied"] else "NO"
    print(
        f"  KKT satisfied : {flag}    "
        f"max residual = {rpt['KKT/max_residual']:.3e}    "
        f"max sign violation = {rpt['KKT/max_sign_violation']:.3e}"
    )
    print(bar)


# ----------------------------------------------------------------------
# Preset generation vectors
# ----------------------------------------------------------------------
def preset_competitive() -> np.ndarray:
    """Welfare-max (perfect competition) solve over both g and d, return g."""
    ptdf = get_ptdf_matrix()
    line_limits = LINE_LIMITS.astype(np.float64)
    g_vars = [cp.Variable(nonneg=True) for _ in PLANTS]
    d = cp.Variable(NUM_NODES, nonneg=True)

    gen_per_node = [0.0] * NUM_NODES
    for pidx, plant in enumerate(PLANTS):
        gen_per_node[plant["node"]] = gen_per_node[plant["node"]] + g_vars[pidx]
    y = cp.hstack([gen_per_node[i] - d[i] for i in range(NUM_NODES)])

    benefit = cp.sum(cp.multiply(P0, d) - 0.5 * cp.multiply(P0 / Q0, cp.square(d)))
    cost = sum(
        plant["mc"] * g_vars[pidx] + 0.5 * plant["qc"] * cp.square(g_vars[pidx])
        for pidx, plant in enumerate(PLANTS)
    )
    cons = [
        cp.sum(y) == 0,
        ptdf @ y <= line_limits,
        ptdf @ y >= -line_limits,
    ]
    cons += [g_vars[pidx] <= plant["cap"] for pidx, plant in enumerate(PLANTS)]
    cp.Problem(cp.Maximize(benefit - cost), cons).solve(solver=cp.CLARABEL)
    return np.array([float(g.value) for g in g_vars])


def preset_monopoly() -> np.ndarray:
    """Joint-profit-max generation (numerical, via the env's clear)."""
    from scipy.optimize import minimize as scipy_minimize
    caps = np.array([p["cap"] for p in PLANTS])

    def neg_total_profit(g_flat):
        g_per_plant = np.clip(g_flat, 0, caps)
        sol = solve_welfare_with_duals(g_per_plant)
        if sol is None:
            return 1e6
        prof = firm_profits(g_per_plant, sol["lmps"])
        return -(prof[0] + prof[1])

    rng = np.random.default_rng(0)
    best = None
    for _ in range(50):
        x0 = rng.uniform(0, 1, len(PLANTS)) * caps
        res = scipy_minimize(
            neg_total_profit, x0, method="L-BFGS-B",
            bounds=[(0, c) for c in caps],
        )
        if best is None or res.fun < best.fun:
            best = res
    return np.clip(best.x, 0, caps)


def preset_half_capacity() -> np.ndarray:
    return np.array([p["cap"] * 0.5 for p in PLANTS])


# ----------------------------------------------------------------------
# Single-point check
# ----------------------------------------------------------------------
def parse_gen_spec(args) -> tuple[np.ndarray, str]:
    if args.plant_gen:
        vals = np.array([float(x) for x in args.plant_gen.split(",")], dtype=np.float64)
        if vals.size != len(PLANTS):
            raise ValueError(f"--plant-gen needs {len(PLANTS)} values: P0,P1,P2")
        return vals, f"plant_gen={vals.tolist()}"

    if args.firm_gen:
        f = [float(x) for x in args.firm_gen.split(",")]
        if len(f) != NUM_FIRMS:
            raise ValueError(f"--firm-gen needs {NUM_FIRMS} values: F0_total,F1_total")
        g = firm_to_plant_mw(f[0], f[1])
        return g, f"firm_gen={f} → plant={g.tolist()}"

    if args.preset:
        if args.preset == "competitive":
            g = preset_competitive()
        elif args.preset == "monopoly":
            g = preset_monopoly()
        elif args.preset == "half-capacity":
            g = preset_half_capacity()
        else:
            raise ValueError(f"unknown preset: {args.preset}")
        return g, f"preset={args.preset} → plant={g.tolist()}"

    raise ValueError("Specify one of --plant-gen, --firm-gen, --preset")


def run_check(args) -> int:
    g, label = parse_gen_spec(args)
    sol = solve_welfare_with_duals(g)
    if sol is None:
        print(f"Solver failed for {label}")
        return 2
    rpt = check_kkt(sol, tol=args.tol)
    print_kkt_report(sol, rpt, header=f"KKT check — {label}")
    return 0 if rpt["KKT/satisfied"] else 1


# ----------------------------------------------------------------------
# Benchmarks comparison
# ----------------------------------------------------------------------
def run_benchmarks(args) -> int:
    for name, fn in [("competitive", preset_competitive),
                     ("monopoly", preset_monopoly),
                     ("half-capacity", preset_half_capacity)]:
        g = fn()
        sol = solve_welfare_with_duals(g)
        if sol is None:
            print(f"[{name}] solver failed")
            continue
        rpt = check_kkt(sol, tol=args.tol)
        print_kkt_report(sol, rpt, header=f"KKT check — preset={name}")
    return 0


# ----------------------------------------------------------------------
# Scan over firm totals
# ----------------------------------------------------------------------
def run_scan(args) -> int:
    n = int(args.grid)
    cap0 = PLANTS[0]["cap"] + PLANTS[1]["cap"]   # Firm 0 max 200
    cap1 = PLANTS[2]["cap"]                       # Firm 1 max 100
    g0_grid = np.linspace(0.0, cap0, n)
    g1_grid = np.linspace(0.0, cap1, n)

    avg_lmp = np.full((n, n), np.nan)
    sum_profit = np.full((n, n), np.nan)
    kkt_ok = np.zeros((n, n), dtype=bool)
    max_resid = np.full((n, n), np.nan)

    for i, g0 in enumerate(g0_grid):
        for j, g1 in enumerate(g1_grid):
            g = firm_to_plant_mw(g0, g1)
            sol = solve_welfare_with_duals(g)
            if sol is None:
                continue
            rpt = check_kkt(sol, tol=args.tol)
            avg_lmp[i, j] = sol["avg_lmp_qty_weighted"]
            kkt_ok[i, j] = rpt["KKT/satisfied"]
            max_resid[i, j] = rpt["KKT/max_residual"]
            profits = firm_profits(g, sol["lmps"])
            sum_profit[i, j] = profits[0] + profits[1]

    # Benchmarks for reference markers
    g_comp = preset_competitive()
    g_mono = preset_monopoly()
    sol_comp = solve_welfare_with_duals(g_comp)
    sol_mono = solve_welfare_with_duals(g_mono)
    comp_avg = sol_comp["avg_lmp_qty_weighted"]
    mono_avg = sol_mono["avg_lmp_qty_weighted"]
    comp_total = sum(g_comp[i] for i in FIRM_PLANT_IDX[0]), sum(g_comp[i] for i in FIRM_PLANT_IDX[1])
    mono_total = sum(g_mono[i] for i in FIRM_PLANT_IDX[0]), sum(g_mono[i] for i in FIRM_PLANT_IDX[1])

    # KKT-OK fraction across the grid
    n_evaluated = int(np.sum(~np.isnan(avg_lmp)))
    n_ok = int(np.sum(kkt_ok & ~np.isnan(avg_lmp)))
    print(f"KKT satisfied at {n_ok}/{n_evaluated} grid points (tol={args.tol:.0e})")
    print(f"Competitive  avg LMP = ${comp_avg:.2f}    "
          f"Monopoly     avg LMP = ${mono_avg:.2f}")

    if args.save is None:
        print("Pass --save <path.png> to render the heatmap.")
        return 0

    # Plot heatmap with iso-LMP contours
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    G1, G0 = np.meshgrid(g1_grid, g0_grid)  # rows -> g0, cols -> g1
    extent = [g1_grid.min(), g1_grid.max(), g0_grid.min(), g0_grid.max()]

    ax = axes[0]
    im = ax.imshow(
        avg_lmp, origin="lower", extent=extent, aspect="auto",
        cmap="viridis",
    )
    cs_mono = ax.contour(G1, G0, avg_lmp, levels=[mono_avg],
                         colors="red", linewidths=2.0)
    ax.clabel(cs_mono, fmt={mono_avg: f"monopoly LMP = ${mono_avg:.2f}"})
    cs_comp = ax.contour(G1, G0, avg_lmp, levels=[comp_avg],
                         colors="white", linewidths=1.6)
    ax.clabel(cs_comp, fmt={comp_avg: f"competitive LMP = ${comp_avg:.2f}"})
    ax.scatter([comp_total[1]], [comp_total[0]],
               marker="x", s=110, color="white", linewidths=2.5,
               label=f"competitive ({comp_total[0]:.0f}, {comp_total[1]:.0f})")
    ax.scatter([mono_total[1]], [mono_total[0]],
               marker="o", s=110, edgecolor="white", facecolor="red",
               linewidths=1.8,
               label=f"monopoly ({mono_total[0]:.0f}, {mono_total[1]:.0f})")
    ax.set_xlabel("Firm 1 total generation (MW)")
    ax.set_ylabel("Firm 0 total generation (MW)")
    ax.set_title("Quantity-weighted avg LMP from welfare DC-OPF")
    plt.colorbar(im, ax=ax, label="avg LMP ($/MWh)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    pos = avg_lmp > mono_avg
    overlay = np.where(pos, 1.0, 0.0)
    im = ax.imshow(
        avg_lmp, origin="lower", extent=extent, aspect="auto",
        cmap="viridis",
    )
    ax.contourf(
        G1, G0, overlay, levels=[0.5, 1.5],
        colors=["red"], alpha=0.35,
    )
    cs = ax.contour(G1, G0, avg_lmp, levels=[mono_avg],
                    colors="red", linewidths=2.0)
    ax.clabel(cs, fmt={mono_avg: f"LMP = ${mono_avg:.2f} (monopoly)"})
    ax.scatter([mono_total[1]], [mono_total[0]],
               marker="o", s=110, edgecolor="white", facecolor="red", linewidths=1.8)
    ax.set_xlabel("Firm 1 total generation (MW)")
    ax.set_ylabel("Firm 0 total generation (MW)")
    ax.set_title("Region where avg LMP > monopoly LMP  (red overlay)")
    plt.colorbar(im, ax=ax, label="avg LMP ($/MWh)")

    fig.suptitle(
        f"KKT scan — {n}×{n} grid, all points satisfy KKT to tol={args.tol:.0e}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")
    return 0


# ----------------------------------------------------------------------
# Read a trained session's early-iteration generation
# ----------------------------------------------------------------------
def run_from_session(args) -> int:
    sess_dir = Path(args.session)
    sjson = sess_dir / "session.json"
    if not sjson.exists():
        print(f"No session.json under {sess_dir}", file=sys.stderr)
        return 2
    metrics = json.loads(sjson.read_text()).get("metrics", [])
    if not metrics:
        print("No metrics in session.json", file=sys.stderr)
        return 2
    if args.iteration >= len(metrics):
        print(f"--iteration {args.iteration} out of range (have {len(metrics)})",
              file=sys.stderr)
        return 2

    row = metrics[args.iteration]
    f0 = float(row.get("firm_0_avg_gen", 0.0))
    f1 = float(row.get("firm_1_avg_gen", 0.0))
    g = firm_to_plant_mw(f0, f1)
    sol = solve_welfare_with_duals(g)
    if sol is None:
        print("Solver failed for that iteration's generation.")
        return 2
    rpt = check_kkt(sol, tol=args.tol)
    print_kkt_report(
        sol, rpt,
        header=(
            f"KKT check — session={sess_dir.name}  iter#{args.iteration}  "
            f"firm0_avg_gen={f0:.2f}  firm1_avg_gen={f1:.2f}  "
            f"logged_avg_lmp={row.get('avg_lmp', 0.0):.3f}"
        ),
    )
    return 0 if rpt["KKT/satisfied"] else 1


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", type=float, default=1e-5,
                   help="KKT residual tolerance (default 1e-5)")
    sub = p.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("check", help="Verify KKT for a single generation vector.")
    pc.add_argument("--plant-gen", type=str,
                    help="Comma-sep per-plant MW (P0,P1,P2)")
    pc.add_argument("--firm-gen", type=str,
                    help="Comma-sep per-firm total MW (F0_total,F1_total)")
    pc.add_argument("--preset", type=str,
                    choices=("competitive", "monopoly", "half-capacity"),
                    help="Use a built-in preset")
    pc.set_defaults(handler=run_check)

    pb = sub.add_parser("benchmarks",
                        help="Run KKT check at competitive, monopoly, half-cap.")
    pb.set_defaults(handler=run_benchmarks)

    ps = sub.add_parser("scan",
                        help="2D scan over (Firm 0 total, Firm 1 total) → LMP heatmap.")
    ps.add_argument("--grid", type=int, default=41)
    ps.add_argument("--save", type=str, default=None,
                    help="Save heatmap PNG here (without this, prints summary only)")
    ps.set_defaults(handler=run_scan)

    pf = sub.add_parser("from-session",
                        help="Take per-firm avg_gen from a logged training iteration "
                        "and check the KKT.")
    pf.add_argument("--session", type=str, required=True,
                    help="Path to results/<crit>/<H>/sessions/session_<id>/")
    pf.add_argument("--iteration", type=int, default=0,
                    help="Metrics-row index (default 0 = earliest iteration)")
    pf.set_defaults(handler=run_from_session)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
