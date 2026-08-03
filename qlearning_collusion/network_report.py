"""Which transmission lines bind, in every market outcome.

Andrew's question at the last meeting was: *in your results, is any line binding
-- in Cournot, in monopoly, and under collusion?* This module answers it for all
four, side by side, from the shadow prices of the ISO's DC-OPF:

    perfect competition     welfare-maximising dispatch
    Nash-Cournot            the paper's LCP (eqs. 39-45) and, where that is not
                            an equilibrium of the game the algorithms play, the
                            iterated best-response Nash of the DC-OPF game
    joint monopoly          the cartel benchmark
    LEARNED COLLUSION       the converged Q-learning strategies, played forward

A line is BINDING when its congestion rent (the signed shadow price) is nonzero.
A line can sit exactly at its limit with a zero shadow price -- that is a
degenerate, non-binding case and is reported separately as "at limit, no rent",
because it constrains nothing at the margin.

For the learned outcome a single dispatch is not enough: the strategies cycle and
the demand shock keeps moving, so we play the converged policies forward and
report, per line, the FRACTION OF PERIODS it binds and the mean congestion rent.

    MARKET_CONFIG=three_firm_dist python -m qlearning_collusion.network_report
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv, PLANTS, NUM_FIRMS, FIRM_PLANT_IDX
from iso_market.node_network import MARKET, LINE_LIMITS
from qlearning_collusion import experiments as X
from qlearning_collusion.qlearn import STATE_SPACES

LINE_NAMES = ["1-2", "2-3", "3-1", "3-4", "4-5"]
BIND_TOL = 1e-3          # $/MWh of congestion rent


# ---------------------------------------------------------------------------
def clear_at(env: ElectricityMarketEnv, plant_gen, u: float = 0.0) -> dict:
    """Clear the DC-OPF at a committed per-plant dispatch."""
    node_gen = np.zeros(env.P0.shape[0])
    for p, g in enumerate(plant_gen):
        node_gen[PLANTS[p]["node"]] += float(g)
    env._demand_u = float(u)
    lmps, demand, flows, shadow = env._clear_market(node_gen)
    env._demand_u = 0.0
    if lmps is None:
        raise RuntimeError("DC-OPF infeasible")
    g = np.asarray(plant_gen, float)
    mc = np.array([p["mc"] for p in PLANTS])
    qc = np.array([p["qc"] for p in PLANTS])
    node = np.array([p["node"] for p in PLANTS])
    plant_pi = lmps[node] * g - (mc * g + 0.5 * qc * g ** 2)
    firm_pi = np.array([plant_pi[FIRM_PLANT_IDX[f]].sum() for f in range(NUM_FIRMS)])
    return dict(gens=g, lmps=lmps, demand=demand, flows=flows, shadow=shadow,
                plant_profit=plant_pi, firm_profit=firm_pi,
                total_gen=float(g.sum()), total_profit=float(firm_pi.sum()),
                avg_lmp=float((lmps * demand).sum() / demand.sum()))


def line_rows(res: dict) -> list:
    out = []
    for l, nm in enumerate(LINE_NAMES):
        fl, sh, lim = float(res["flows"][l]), float(res["shadow"][l]), float(LINE_LIMITS[l])
        binding = abs(sh) > BIND_TOL
        at_lim = abs(abs(fl) - lim) < 1e-4
        out.append(dict(line=nm, flow=fl, limit=lim, shadow=sh, binding=binding,
                        at_limit_no_rent=(at_lim and not binding),
                        util=abs(fl) / lim if lim > 0 else 0.0))
    return out


# ---------------------------------------------------------------------------
def benchmark_outcomes() -> dict:
    """Competitive / Nash / monopoly dispatches, cleared and annotated."""
    env = ElectricityMarketEnv()
    bm = X.build_market()                      # for the BR-Nash and grid points
    cont = bm.bench_continuous
    out = {}
    out["competitive"] = clear_at(env, cont["competitive"]["gens"])
    out["nash_lcp"] = clear_at(env, cont["nash"]["gens"])
    out["monopoly"] = clear_at(env, cont["monopoly"]["gens"])
    if hasattr(bm, "br_nash"):
        out["nash_br"] = clear_at(env, bm.br_nash["plant_gens"])
    # the two GRID points that actually define Delta
    gb = bm.grid_benchmarks()
    for nm in ("nash", "monopoly"):
        pg = gb[nm].get("plant_gens")
        if pg is not None:
            out[f"grid_{nm}"] = clear_at(env, pg)
    return out, bm


# ---------------------------------------------------------------------------
def collusion_outcome(name: str = X.BASELINE, periods: int = 20_000,
                      burn_in: int = 1_000, seed: int = 4242) -> dict:
    """Play the converged strategies forward; average the network state.

    Returns per-line binding FREQUENCY and mean rent, plus the mean dispatch,
    nodal LMPs and demand under the learned (collusive) behaviour.
    """
    meta, z = X.load(name)
    mk = X.build_market(deterministic=meta["market"]["deterministic"],
                        k=meta["market"]["k"], xi=meta["market"]["xi"],
                        m=meta["market"]["m"], h=meta["market"]["h"])
    space = STATE_SPACES[meta["config"]["monitoring"]](mk)
    greedy = z["greedy"]
    conv = z["converged"].astype(bool)
    start = z["final_state"]
    if conv.any():
        greedy, start = greedy[conv], start[conv]
    S, n, _ = greedy.shape
    rng = np.random.default_rng(seed)
    rowS = np.arange(S)[:, None]
    ar = np.arange(n)
    state = start.copy()

    L = mk.flows_tab.shape[2]
    acc = dict(flows=np.zeros(L), shadow=np.zeros(L), bind=np.zeros(L),
               lmps=np.zeros(mk.nodal_lmps.shape[2]),
               demand=np.zeros(mk.demand_tab.shape[2]),
               gen=np.zeros(mk.profile_plant_gen.shape[1]
                            if hasattr(mk, "profile_plant_gen") else n))
    cnt = 0
    for step in range(periods + burn_in):
        a = greedy[rowS, ar[None, :], state[:, None]].astype(np.int64)
        P = mk.pidx(a)
        u = rng.integers(0, mk.h, size=S) if mk.h > 1 else np.zeros(S, dtype=np.int64)
        if step >= burn_in:
            acc["flows"] += mk.flows_tab[P, u].mean(axis=0)
            sh = mk.shadow_tab[P, u]
            acc["shadow"] += sh.mean(axis=0)
            acc["bind"] += (np.abs(sh) > BIND_TOL).mean(axis=0)
            acc["lmps"] += mk.nodal_lmps[P, u].mean(axis=0)
            acc["demand"] += mk.demand_tab[P, u].mean(axis=0)
            if hasattr(mk, "profile_plant_gen"):
                acc["gen"] += mk.profile_plant_gen[P].mean(axis=0)
            cnt += 1
        state = space.next_state(a, P, u)

    for k in acc:
        acc[k] = acc[k] / cnt
    acc["n_sessions"] = int(S)
    acc["name"] = name
    acc["market"] = mk
    return acc


# ---------------------------------------------------------------------------
def text_report(name: str = X.BASELINE, periods: int = 20_000) -> str:
    bench, bm = benchmark_outcomes()
    col = collusion_outcome(name, periods=periods)

    L = []
    L.append("=" * 96)
    L.append(f"TRANSMISSION CONGESTION ACROSS MARKET OUTCOMES  (market: {MARKET})")
    L.append("=" * 96)
    L.append(f"line limits (MW): " +
             "  ".join(f"{nm}={LINE_LIMITS[i]:.1f}" for i, nm in enumerate(LINE_NAMES)))
    L.append("")

    order = [("competitive", "perfect competition"),
             ("nash_lcp", "Nash-Cournot (LCP, paper eqs. 39-45)"),
             ("nash_br", "Nash-Cournot (iterated best response)"),
             ("monopoly", "joint monopoly"),
             ("grid_nash", "Nash on the action grid (Delta = 0)"),
             ("grid_monopoly", "monopoly on the action grid (Delta = 1)")]

    L.append(f"{'outcome':<40}{'gen':>8}{'avgLMP':>9}{'profit':>10}   binding lines (rent $/MWh)")
    L.append("-" * 96)
    for key, label in order:
        if key not in bench:
            continue
        r = bench[key]
        rows = line_rows(r)
        b = ", ".join(f"{x['line']} ({x['shadow']:+.2f})" for x in rows if x["binding"])
        atl = [x["line"] for x in rows if x["at_limit_no_rent"]]
        if not b:
            b = "NONE"
        if atl:
            b += f"   [at limit, zero rent: {', '.join(atl)}]"
        L.append(f"{label:<40}{r['total_gen']:8.1f}{r['avg_lmp']:9.2f}"
                 f"{r['total_profit']:10.1f}   {b}")

    cb = ", ".join(f"{LINE_NAMES[i]} ({100*col['bind'][i]:.0f}% of periods, "
                   f"mean rent {col['shadow'][i]:+.2f})"
                   for i in range(len(LINE_NAMES)) if col["bind"][i] > 0.005) or "NONE"
    tot_gen = float(np.sum(col["gen"]))
    avg_lmp = float((col["lmps"] * col["demand"]).sum() / col["demand"].sum())
    L.append(f"{'LEARNED COLLUSION (' + name + ')':<40}{tot_gen:8.1f}{avg_lmp:9.2f}"
             f"{'':>10}   {cb}")
    L.append("-" * 96)
    L.append("")

    L.append("Per-line detail (flow / limit, utilisation, congestion rent):")
    for key, label in order:
        if key not in bench:
            continue
        rows = line_rows(bench[key])
        L.append(f"  {label}")
        for x in rows:
            tag = "BINDING" if x["binding"] else ("at limit, no rent"
                                                 if x["at_limit_no_rent"] else "slack")
            L.append(f"      {x['line']:>4}  flow {x['flow']:+8.2f} / {x['limit']:5.1f} MW"
                     f"  ({100*x['util']:5.1f}%)   rent {x['shadow']:+8.3f}   {tag}")
    L.append(f"  LEARNED COLLUSION ({name}, {col['n_sessions']} converged sessions)")
    for i, nm in enumerate(LINE_NAMES):
        L.append(f"      {nm:>4}  mean flow {col['flows'][i]:+8.2f} / {LINE_LIMITS[i]:5.1f} MW"
                 f"  ({100*abs(col['flows'][i])/LINE_LIMITS[i]:5.1f}%)   "
                 f"mean rent {col['shadow'][i]:+8.3f}   binds {100*col['bind'][i]:5.1f}% of periods")
    L.append("")

    L.append("Nodal LMPs ($/MWh) and demand (MW):")
    for key, label in order:
        if key not in bench:
            continue
        r = bench[key]
        L.append(f"  {label:<40} LMP={np.round(r['lmps'],2).tolist()}")
        L.append(f"  {'':<40} d  ={np.round(r['demand'],2).tolist()}")
    L.append(f"  {'LEARNED COLLUSION':<40} LMP={np.round(col['lmps'],2).tolist()}")
    L.append(f"  {'':<40} d  ={np.round(col['demand'],2).tolist()}")
    L.append("=" * 96)
    return "\n".join(L)


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else X.BASELINE
    txt = text_report(nm)
    print(txt)
    out = os.path.join(X.RESULTS, "network_report.txt")
    os.makedirs(X.RESULTS, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(txt + "\n")
    print(f"\nwrote {out}")
