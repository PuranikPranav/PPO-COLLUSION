"""CLI for the Q-learning collusion replication.

    python -m qlearning_collusion.run market                 # describe the game
    python -m qlearning_collusion.run train imperfect_stochastic --sessions 1000
    python -m qlearning_collusion.run table1                 # all four cells
    python -m qlearning_collusion.run figures                # all plots
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from qlearning_collusion import experiments as X


def _train_args(p):
    p.add_argument("--sessions", type=int, default=1000)
    p.add_argument("--iters", type=int, default=4_000_000)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--xi", type=float, default=0.2)
    p.add_argument("--m", type=float, default=8.0, help="shock size in output steps")
    p.add_argument("--h", type=int, default=2, help="number of demand levels")
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--beta", type=float, default=4e-6)
    p.add_argument("--delta", type=float, default=0.95)
    p.add_argument("--conv-window", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dtype", type=str, default="float32")


def _kw(a):
    return dict(n_sessions=a.sessions, max_iter=a.iters, k=a.k, xi=a.xi, m=a.m,
                h=a.h, alpha=a.alpha, beta=a.beta, delta=a.delta,
                conv_window=a.conv_window, seed=a.seed, tag=a.tag, dtype=a.dtype)


def main():
    ap = argparse.ArgumentParser(prog="qlearning_collusion.run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("market")

    p = sub.add_parser("train"); p.add_argument("name", choices=list(X.CELLS)); _train_args(p)
    p = sub.add_parser("table1"); _train_args(p)
    p = sub.add_parser("figures"); p.add_argument("--name", default=X.BASELINE)
    p = sub.add_parser("report"); p.add_argument("--name", default=X.BASELINE)

    a = ap.parse_args()

    if a.cmd == "market":
        print(X.build_market().describe())

    elif a.cmd == "train":
        X.run(a.name, **_kw(a))

    elif a.cmd == "table1":
        for name in X.TABLE1_CELLS:
            X.run(name, **_kw(a))
        print(table1_text(tag=a.tag))

    elif a.cmd == "figures":
        from qlearning_collusion import figures
        figures.make_all(a.name)

    elif a.cmd == "report":
        meta, _ = X.load(a.name)
        print(f"\n{a.name}:")
        print(X.summary_text(meta))
        print(table1_text())
        if meta["config"]["monitoring"] == "imperfect":
            print("\nForced-deviation incentive test (is cheating deterred?)")
            print("-" * 62)
            for dem in ("high", "low"):
                r = X.deviation_value_test(a.name, demand=dem, horizon=250)
                d = r["discounted_per_session"]
                print(f"  {dem:>4} demand: one-period gain ${r['one_period_gain']:+,.0f} | "
                      f"discounted ${r['discounted_gain']:+,.0f} "
                      f"(median ${float(np.median(d)):+,.0f}) | "
                      f"deterred in {100*r['deterred_frac']:.1f}% of sessions")


def table1_text(tag: str = "") -> str:
    """The paper's Table I layout, filled with this market's numbers."""
    cells, conv = {}, {}
    for name in X.TABLE1_CELLS + ["rich_stochastic", "rich_deterministic"]:
        path = os.path.join(X.RESULTS, f"{name}{tag}.json")
        if os.path.exists(path):
            with open(path) as fh:
                r = json.load(fh)["result"]
            cells[name] = r["delta"]
            conv[name] = r["converged_frac"]
    def g(k):
        return f"{100*cells[k]:.2f}%" if k in cells else "   --  "
    L = [
        "",
        "TABLE I - the impact of imperfect monitoring (profit gain Delta)",
        "-" * 62,
        f"{'':<22}{'Deterministic Demand':>20}{'Stochastic Demand':>20}",
        f"{'Perfect Monitoring':<22}{g('perfect_deterministic'):>20}{g('perfect_stochastic'):>20}",
        f"{'Imperfect Monitoring':<22}{g('imperfect_deterministic'):>20}{g('imperfect_stochastic'):>20}",
        "-" * 62,
    ]
    if "rich_stochastic" in cells or "rich_deterministic" in cells:
        L.append(f"{'Rich 19-var state':<22}{g('rich_deterministic'):>20}"
                 f"{g('rich_stochastic'):>20}")
        L.append("-" * 62)
        L.append("(rich = the PPO public observation — nodal LMPs, congestion /")
        L.append(" shadow prices, realised demand — instead of one binned price;")
        L.append(" still imperfect monitoring: rivals' outputs are NOT in the state)")
    need = set(X.TABLE1_CELLS)
    if need <= set(cells):
        did = ((cells["imperfect_stochastic"] - cells["imperfect_deterministic"])
               - (cells["perfect_stochastic"] - cells["perfect_deterministic"]))
        L.append(f"difference-in-differences (pure imperfect-monitoring effect): "
                 f"{100*did:+.2f} pp")
        L.append("(paper baseline: -8.91 pp; 76.25 / 89.60 / 79.72 / 84.16)")
    # A cell whose sessions never met the paper's convergence criterion is not
    # a comparable number, and the difference-in-differences inherits that.
    weak = {k: v for k, v in conv.items() if v < 0.90}
    if weak:
        L.append("")
        L.append("CAUTION - these cells did NOT converge for most sessions, so their")
        L.append("Delta is a snapshot at the iteration cap, not a limit strategy:")
        for k, v in sorted(weak.items()):
            L.append(f"    {k:<26} {100*v:5.1f}% of sessions converged")
        L.append("Any difference-in-differences involving them inherits that caveat.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
