"""Collect every finished run into one markdown results document.

    MARKET_CONFIG=three_firm_dist python -m qlearning_collusion.summarize > RESULTS_DIST.md

Everything here is read off the saved artefacts, so the document cannot drift
from the runs it describes.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.node_network import MARKET, P0, Q0, LINE_LIMITS, MC, QC, CAP, PLANT_SPECS
from qlearning_collusion import experiments as X

LINE_NAMES = ["1-2", "2-3", "3-1", "3-4", "4-5"]
ORDER = ["imperfect_stochastic", "imperfect_deterministic",
         "perfect_stochastic", "perfect_deterministic",
         "rich_stochastic", "rich_deterministic"]
LABEL = {
    "imperfect_stochastic": "imperfect (price only) · stochastic demand",
    "imperfect_deterministic": "imperfect (price only) · deterministic demand",
    "perfect_stochastic": "perfect (full profile) · stochastic demand",
    "perfect_deterministic": "perfect (full profile) · deterministic demand",
    "rich_stochastic": "**rich 19-variable state** · stochastic demand",
    "rich_deterministic": "**rich 19-variable state** · deterministic demand",
}


def _meta(name):
    p = os.path.join(X.RESULTS, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main():
    metas = {n: _meta(n) for n in ORDER}
    have = {n: m for n, m in metas.items() if m}
    if not have:
        raise SystemExit(f"no runs found in {X.RESULTS}")
    any_meta = next(iter(have.values()))
    cb = any_meta["market"]["continuous_benchmarks"]
    gb = any_meta["market"]["grid_benchmarks"]

    L = []
    A = L.append
    A(f"# Results — `MARKET_CONFIG={MARKET}`")
    A("")
    A("Everything below is generated from the saved run artefacts by")
    A("`python -m qlearning_collusion.summarize`.")
    A("")

    # ---- market -----------------------------------------------------------
    A("## 1. The market")
    A("")
    A("| node | P⁰ ($/MWh) | Q⁰ (MW) | slope ($/MWh per MW) | elasticity at $45 |")
    A("|---|---|---|---|---|")
    for i in range(len(P0)):
        d = Q0[i] * (1 - 45.0 / P0[i])
        eps = -(Q0[i] / P0[i]) * 45.0 / d
        A(f"| {i+1} | {P0[i]:.0f} | {Q0[i]:.0f} | {P0[i]/Q0[i]:.3f} | {eps:+.4f} |")
    A("")
    A("| firm | plant node | MC | QC | cap (MW) |")
    A("|---|---|---|---|---|")
    for f, n, key in PLANT_SPECS:
        A(f"| {f} | {n+1} | {MC[key]:.1f} | {QC[key]:.3f} | {CAP[key]:.0f} |")
    A("")
    A("Line limits (MW): " + ", ".join(
        f"**{LINE_NAMES[i]}** {LINE_LIMITS[i]:.0f}" for i in range(len(LINE_NAMES))))
    A("")

    # ---- benchmarks -------------------------------------------------------
    A("## 2. Benchmarks")
    A("")
    A("| outcome | total gen (MW) | avg LMP ($/MWh) | total profit ($/period) |")
    A("|---|---|---|---|")
    for key, lab in (("competitive", "perfect competition"),
                     ("nash", "Nash-Cournot — LCP (paper eqs. 39-45)"),
                     ("monopoly", "joint monopoly")):
        b = cb[key]
        A(f"| {lab} | {b['total_gen']:.1f} | {b['avg_lmp']:.2f} | {b['total_profit']:.0f} |")
    A(f"| **Nash-Cournot — best response (Δ denominator)** | "
      f"{gb['nash']['total_gen']:.1f} | {gb['nash']['avg_lmp']:.2f} | "
      f"{gb['nash']['total_profit']:.0f} |")
    A(f"| **joint monopoly on the action grid (Δ = 1)** | "
      f"{gb['monopoly']['total_gen']:.1f} | {gb['monopoly']['avg_lmp']:.2f} | "
      f"{gb['monopoly']['total_profit']:.0f} |")
    A("")
    ir = [gb["monopoly"]["profits"][i] - gb["nash"]["profits"][i]
          for i in range(len(gb["nash"]["profits"]))]
    gap = gb["monopoly"]["total_profit"] - gb["nash"]["total_profit"]
    A(f"Nash → monopoly profit gap **${gap:,.0f} ({100*gap/gb['nash']['total_profit']:.1f}%)**; "
      f"per-firm cartel IR margins {[round(x) for x in ir]} "
      f"({'all positive' if min(ir) > 0 else 'NOT all positive'}).")
    A("")
    A(f"Grid action indices: Nash at `{gb['nash']['actions']}`, "
      f"monopoly at `{gb['monopoly']['actions']}` (paper: 12 and 2).")
    A("")

    # ---- learned results --------------------------------------------------
    A("## 3. What the algorithms learn")
    A("")
    A("| cell | Δ | s.e. | converged | total gen (MW) | ref-node LMP | avg LMP | \\|S\\| |")
    A("|---|---|---|---|---|---|---|---|")
    for n in ORDER:
        m = have.get(n)
        if not m:
            continue
        r = m["result"]
        rr = m["market"].get("rich_state_report")
        ns = (rr["n_rich_states"] if rr else
              (m["market"]["monitoring_report"]["n_price_states"]
               if m["config"]["monitoring"] == "imperfect" else "k^n"))
        A(f"| {LABEL[n]} | **{100*r['delta']:.2f}%** | {100*r['delta_se']:.2f} | "
          f"{100*r['converged_frac']:.0f}% | {r['total_gen']:.1f} | "
          f"${r['hub_price']:.2f} | ${r['avg_lmp']:.2f} | {ns} |")
    A("")
    A(f"(Nash → monopoly reference: generation {gb['nash']['total_gen']:.1f} → "
      f"{gb['monopoly']['total_gen']:.1f} MW, "
      f"reference-node LMP ${gb['nash']['hub_price']:.2f} → "
      f"${gb['monopoly']['hub_price']:.2f}.)")
    A("")

    A("### Per-firm profit gain")
    A("")
    A("| cell | " + " | ".join(f"firm {i}" for i in range(len(ir))) + " |")
    A("|---|" + "---|" * len(ir))
    for n in ORDER:
        m = have.get(n)
        if not m:
            continue
        A(f"| {LABEL[n]} | " +
          " | ".join(f"{100*x:.1f}%" for x in m["result"]["delta_per_firm"]) + " |")
    A("")

    # ---- information structure -------------------------------------------
    A("### How much the state reveals")
    A("")
    A("Fraction of (own action, observed state) pairs consistent with more than "
      "one rival profile — 1.0 means the state says nothing about rivals, "
      "0.0 means it identifies them exactly.")
    A("")
    A("| state | \\|S\\| | non-revealing fraction |")
    A("|---|---|---|")
    mr = any_meta["market"]["monitoring_report"]
    A(f"| price only | {mr['n_price_states']} | "
      f"{mr['measured_nonrevealing_fraction']:.3f} |")
    rich = have.get("rich_stochastic") or have.get("rich_deterministic")
    if rich and rich["market"].get("rich_state_report"):
        rr = rich["market"]["rich_state_report"]
        A(f"| rich 19-variable [{', '.join(rr['components'])}] | "
          f"{rr['n_rich_states']} | {rr['measured_nonrevealing_fraction']:.3f} |")
    A(f"| full profile (perfect) | {any_meta['market']['k']}^{any_meta['market']['n_firms']} | 0.000 |")
    A("")

    # ---- table I ----------------------------------------------------------
    from qlearning_collusion.run import table1_text
    A("## 4. Table I")
    A("")
    A("```")
    A(table1_text())
    A("```")
    A("")

    # ---- attachments ------------------------------------------------------
    for fname, title in (("network_report.txt", "5. Transmission congestion"),
                         ("delta_sweep.txt", "6. Discount-factor sweep")):
        p = os.path.join(X.RESULTS, fname)
        if os.path.exists(p):
            A(f"## {title}")
            A("")
            A("```")
            A(open(p).read().rstrip())
            A("```")
            A("")

    print("\n".join(L))


if __name__ == "__main__":
    main()
