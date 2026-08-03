"""Side-by-side report: the two-firm/three-plant market vs the three-firm hub
market vs the paper's duopoly baseline.

    MARKET_CONFIG=two_firm python -m qlearning_collusion.compare_markets

Reads the saved artefacts of both markets (`results/` and `results_two_firm/`)
and prints every number the write-up quotes, so nothing in RESULTS_2FIRM.md is
hand-transcribed.
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["imperfect_stochastic", "imperfect_deterministic",
         "perfect_stochastic", "perfect_deterministic"]
PAPER = {"perfect_deterministic": 84.16, "perfect_stochastic": 79.72,
         "imperfect_deterministic": 89.60, "imperfect_stochastic": 76.25}


def _load(dirname):
    out = {}
    for c in CELLS:
        p = os.path.join(HERE, dirname, f"{c}.json")
        if os.path.exists(p):
            with open(p) as fh:
                out[c] = json.load(fh)
    return out


def did(cells):
    """Difference-in-differences: the pure imperfect-monitoring effect."""
    need = set(CELLS)
    if not need <= set(cells):
        return None
    d = {c: cells[c]["result"]["delta"] for c in CELLS}
    return 100 * ((d["imperfect_stochastic"] - d["imperfect_deterministic"])
                  - (d["perfect_stochastic"] - d["perfect_deterministic"]))


def critical_deltas(mk, gb):
    """delta* per firm for sustaining joint monopoly with Nash reversion."""
    N, M = gb["nash"], gb["monopoly"]
    rows = []
    for i in range(mk.n_agents):
        ka = int(mk.n_actions[i])
        dev = mk.dev_pidx(np.array([M["profile"]]), i, np.arange(ka))
        pid = float(mk.exp_profit_p[i][dev].max())
        piM, piN = M["profits"][i], N["profits"][i]
        rows.append({
            "firm": i, "pi_M": piM, "pi_dev": pid, "pi_N": piN,
            "rent": piM - piN, "temptation": pid - piM,
            "delta_star": (pid - piM) / (pid - piN) if pid > piN else float("nan"),
        })
    return rows


def main():
    two, three = _load("results_two_firm"), _load("results")

    print("=" * 78)
    print("TABLE I -- profit gain Delta (%), by market")
    print("=" * 78)
    hdr = f"{'cell':<26}{'2-firm/3-plant':>16}{'3-firm hub':>14}{'paper (duopoly)':>18}"
    print(hdr); print("-" * 78)
    for c in CELLS:
        a = f"{100*two[c]['result']['delta']:.2f}%" if c in two else "--"
        b = f"{100*three[c]['result']['delta']:.2f}%" if c in three else "--"
        print(f"{c:<26}{a:>16}{b:>14}{PAPER[c]:>17.2f}%")
    print("-" * 78)
    d2, d3 = did(two), did(three)
    print(f"{'difference-in-differences':<26}"
          f"{(f'{d2:+.2f} pp' if d2 is not None else '--'):>16}"
          f"{(f'{d3:+.2f} pp' if d3 is not None else '--'):>14}"
          f"{'-8.91 pp':>18}")

    print()
    print("=" * 78)
    print("BASELINE CELL (imperfect monitoring + stochastic demand)")
    print("=" * 78)
    for lbl, cells in (("2-firm/3-plant", two), ("3-firm hub", three)):
        if "imperfect_stochastic" not in cells:
            continue
        r = cells["imperfect_stochastic"]["result"]
        m = cells["imperfect_stochastic"]["market"]
        b = r["bench"]
        print(f"\n{lbl}:")
        print(f"  Delta                 : {100*r['delta']:.2f}% +/- {100*r['delta_se']:.2f}")
        print(f"  Delta per firm        : {[round(100*x,1) for x in r['delta_per_firm']]} %")
        print(f"  converged             : {100*r['converged_frac']:.1f}%  "
              f"median {r['median_conv_iter']:,.0f} iters")
        print(f"  total generation      : {r['total_gen']:.1f} MW "
              f"(Nash {b['nash_gen']:.1f} -> monopoly {b['monopoly_gen']:.1f})")
        print(f"  hub LMP               : ${r['hub_price']:.2f} "
              f"(Nash ${b['nash_hub_price']:.2f} -> monopoly ${b['monopoly_hub_price']:.2f})")
        print(f"  qty-weighted avg LMP  : ${r['avg_lmp']:.2f}")
        print(f"  non-revealing prices  : "
              f"{m['monitoring_report']['measured_nonrevealing_fraction']:.3f} "
              f"(paper closed form {m['monitoring_report']['paper_closed_form_fraction']:.3f})")

    print()
    print("=" * 78)
    print("WHY THE TWO MARKETS DIFFER: cartel incentives at joint monopoly")
    print("=" * 78)
    from qlearning_collusion.experiments import build_market
    from iso_market.node_network import MARKET
    mk = build_market()
    gb = mk.grid_benchmarks()
    print(f"\nMARKET_CONFIG={MARKET}  (run once per market)")
    print(f"  {'firm':<6}{'pi^N':>10}{'pi^M':>10}{'rent':>10}"
          f"{'best cheat':>12}{'temptation':>12}{'delta*':>9}")
    for r in critical_deltas(mk, gb):
        print(f"  {r['firm']:<6}{r['pi_N']:>10.1f}{r['pi_M']:>10.1f}{r['rent']:>+10.1f}"
              f"{r['pi_dev']:>12.1f}{r['temptation']:>+12.1f}{r['delta_star']:>9.4f}")
    gap = gb["monopoly"]["total_profit"] - gb["nash"]["total_profit"]
    print(f"  Nash->monopoly rent: ${gap:.1f} "
          f"({100*gap/gb['nash']['total_profit']:.1f}% of Nash profit)")


if __name__ == "__main__":
    main()
