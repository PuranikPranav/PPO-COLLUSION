"""
Search 2-firm (Firm0: 2 plants @ nodes 0,1; Firm1: 1 plant @ node 1) parameters
for a wide Nash→Monopoly collusion gap with per-firm IR and correct orderings.

Uses the same competitive / MCP-Nash / wheeling-MPEC monopoly formulations as
experiments/ppo.py, specialized to the original plant siting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize, root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NUM_NODES = 5
# Firm0 plant @ node1 (idx0), Firm0 plant @ node2 (idx1), Firm1 plant @ node2 (idx2)
PLANT_NODES = np.array([0, 1, 1], dtype=int)
FIRM_OF = [0, 0, 1]


def get_ptdf_matrix():
    lines = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    n, L = 5, len(lines)
    B = np.zeros((n, n))
    for u, v in lines:
        B[u, v] = B[v, u] = -1.0
        B[u, u] += 1.0
        B[v, v] += 1.0
    Br = np.linalg.inv(np.delete(np.delete(B, 4, 0), 4, 1))
    Binv = np.zeros((n, n))
    Binv[:4, :4] = Br
    A = np.zeros((L, n))
    for i, (u, v) in enumerate(lines):
        A[i, u], A[i, v] = 1, -1
    return A @ Binv


PTDF = get_ptdf_matrix()


def gen_per_node(g):
    gn = np.zeros(NUM_NODES)
    for k, node in enumerate(PLANT_NODES):
        gn[node] += g[k]
    return gn


def firm_profits(lmps, g, mc, qc):
    pi = np.zeros(2)
    for k in range(3):
        node = PLANT_NODES[k]
        pi[FIRM_OF[k]] += lmps[node] * g[k] - mc[k] * g[k] - 0.5 * qc[k] * g[k] ** 2
    return pi


def competitive(P):
    p0, q0 = P["P0"], P["Q0"]
    mc, qc, cap, lim = P["MC"], P["QC"], P["CAP"], P["LIM"]
    g = [cp.Variable(nonneg=True) for _ in range(3)]
    d = cp.Variable(NUM_NODES, nonneg=True)
    benefit = cp.sum(cp.multiply(p0, d) - 0.5 * cp.multiply(p0 / q0, cp.square(d)))
    cost = sum(mc[k] * g[k] + 0.5 * qc[k] * cp.square(g[k]) for k in range(3))
    gn = [0.0] * NUM_NODES
    for k in range(3):
        gn[PLANT_NODES[k]] = gn[PLANT_NODES[k]] + g[k]
    y = cp.hstack([gn[i] - d[i] for i in range(NUM_NODES)])
    cons = [cp.sum(y) == 0, PTDF @ y <= lim, PTDF @ y >= -lim] + [
        g[k] <= cap[k] for k in range(3)
    ]
    prob = cp.Problem(cp.Maximize(benefit - cost), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()
    if prob.status not in ("optimal", "optimal_inaccurate") or d.value is None:
        return None
    lmps = p0 - (p0 / q0) * d.value
    gv = np.array([float(v.value) for v in g])
    demand = d.value
    flows = PTDF @ (gen_per_node(gv) - demand)
    pi = firm_profits(lmps, gv, mc, qc)
    avg = float(np.sum(lmps * demand) / np.sum(demand))
    return dict(gens=gv, profits=pi, avg_lmp=avg, total_gen=float(gv.sum()),
                total_profit=float(pi.sum()), flows=flows, lmps=lmps)


def nash(P, attempts=30):
    p0, q0 = P["P0"], P["Q0"]
    beta = p0 / q0
    mc, qc, cap, lim = P["MC"], P["QC"], P["CAP"], P["LIM"]
    nP = 3

    def fb(a, b):
        return a + b - np.sqrt(a * a + b * b + 1e-18)

    i_y0, i_mu = 2 * nP, 2 * nP + NUM_NODES

    def unpack(z):
        return (z[0:nP], z[nP:i_y0], z[i_y0:i_mu], z[i_mu],
                z[i_mu + 1:i_mu + 6], z[i_mu + 6:i_mu + 11])

    def residual(z):
        g, rho, y, mu, lp, lm = unpack(z)
        p = p0 - beta * (gen_per_node(g) + y)
        pp = p[PLANT_NODES]
        # Paper eq (31): own-node slope only
        f_stat = -(pp - beta[PLANT_NODES] * g) + (mc + qc * g) + rho
        f_cap = cap - g
        f_disp = p - mu - PTDF.T @ (lm - lp)
        flow = PTDF @ y
        return np.concatenate([
            fb(g, f_stat), fb(rho, f_cap), f_disp, np.array([np.sum(y)]),
            fb(lp, lim + flow), fb(lm, lim - flow),
        ])

    best_z, best_res = None, np.inf
    base = np.concatenate([0.5 * cap, np.zeros(nP), np.zeros(5),
                           np.array([30.0]), np.zeros(5), np.zeros(5)])
    rng = np.random.default_rng(0)
    for attempt in range(attempts):
        z0 = base.copy()
        if attempt:
            z0[0:nP] = rng.uniform(0.05, 1.0) * cap
            z0[i_y0:i_mu] = rng.uniform(-30, 30, 5)
            z0[i_mu] = rng.uniform(20, 50)
        sol = root(residual, z0, method="hybr", tol=1e-12)
        res = float(np.max(np.abs(residual(sol.x))))
        g = sol.x[0:nP]
        if np.all(g >= -1e-6) and np.all(g <= cap + 1e-6) and res < best_res:
            best_res, best_z = res, sol.x.copy()
    if best_z is None or best_res > 1e-5:
        return None
    g, rho, y, mu, lp, lm = unpack(best_z)
    g = np.clip(g, 0, cap)
    lmps = p0 - beta * (gen_per_node(g) + y)
    demand = gen_per_node(g) + y
    pi = firm_profits(lmps, g, mc, qc)
    avg = float(np.sum(lmps * demand) / max(np.sum(demand), 1e-9))
    return dict(gens=g.copy(), profits=pi, avg_lmp=avg, total_gen=float(g.sum()),
                total_profit=float(pi.sum()), flows=PTDF @ y, lmps=lmps,
                residual=best_res)


def monopoly(P, attempts=12, seed_g=None):
    p0, q0 = P["P0"], P["Q0"]
    beta = p0 / q0
    mc, qc, cap, lim = P["MC"], P["QC"], P["CAP"], P["LIM"]
    nP, L, nS = 3, len(lim), NUM_NODES - 1
    Psp = PTDF[:, :nS]

    i_g = slice(0, nP)
    i_y = slice(nP, nP + nS)
    i_ph = nP + nS
    i_w = slice(nP + nS + 1, nP + 2 * nS + 1)
    i_lp = slice(nP + 2 * nS + 1, nP + 2 * nS + 1 + L)
    i_lm = slice(nP + 2 * nS + 1 + L, nP + 2 * nS + 1 + 2 * L)
    dim = nP + 2 * nS + 1 + 2 * L

    def w_full(z):
        return np.concatenate([z[i_w], [0.0]])

    def y_full(z):
        y = z[i_y]
        return np.concatenate([y, [-np.sum(y)]])

    def neg_profit(z):
        lmp = z[i_ph] + w_full(z)
        g = z[i_g]
        return -(np.sum(lmp * gen_per_node(g)) - np.sum(mc * g + 0.5 * qc * g ** 2))

    def eq_con(z):
        mkt = p0 - beta * (gen_per_node(z[i_g]) + y_full(z)) - (z[i_ph] + w_full(z))
        whl = z[i_w] - Psp.T @ (z[i_lp] - z[i_lm])
        return np.concatenate([mkt, whl])

    bounds = ([(0.0, float(cap[k])) for k in range(nP)]
              + [(-400.0, 400.0)] * nS + [(0.0, 400.0)] + [(-150.0, 150.0)] * nS
              + [(0.0, 2000.0)] * L + [(0.0, 2000.0)] * L)
    rng = np.random.default_rng(1)
    best_z, best_obj = None, -np.inf
    for eps in (1e-1, 1e-3, 1e-5, 1e-7):
        cons = [
            {"type": "eq", "fun": eq_con},
            {"type": "ineq", "fun": lambda z: lim - Psp @ z[i_y]},
            {"type": "ineq", "fun": lambda z: lim + Psp @ z[i_y]},
            {"type": "ineq", "fun": lambda z, e=eps: e - z[i_lp] * (lim - Psp @ z[i_y])},
            {"type": "ineq", "fun": lambda z, e=eps: e - z[i_lm] * (lim + Psp @ z[i_y])},
        ]
        for attempt in range(attempts):
            z0 = np.zeros(dim)
            if attempt == 0 and seed_g is not None:
                z0[i_g] = np.clip(0.7 * np.asarray(seed_g), 0, cap)
            elif attempt == 1 and seed_g is not None:
                z0[i_g] = np.clip(0.55 * np.asarray(seed_g), 0, cap)
            else:
                z0[i_g] = rng.uniform(0.1, 0.9) * cap
            z0[i_y] = rng.uniform(-40, 40, nS)
            z0[i_ph] = rng.uniform(28, 70)
            r = minimize(neg_profit, z0, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 700, "ftol": 1e-10})
            if r.success and np.max(np.abs(eq_con(r.x))) < 1e-5 and -r.fun > best_obj:
                best_obj, best_z = -r.fun, r.x.copy()
    if best_z is None:
        return None
    g = np.clip(best_z[i_g], 0.0, cap)
    lmps = best_z[i_ph] + w_full(best_z)
    demand = gen_per_node(g) + y_full(best_z)
    pi = firm_profits(lmps, g, mc, qc)
    avg = float(np.sum(lmps * demand) / max(np.sum(demand), 1e-9))
    return dict(gens=g.copy(), profits=pi, avg_lmp=avg, total_gen=float(g.sum()),
                total_profit=float(pi.sum()), flows=Psp @ best_z[i_y], lmps=lmps)


def evaluate(P, fast=True):
    c = competitive(P)
    if c is None:
        return None
    n = nash(P, attempts=18 if fast else 40)
    if n is None:
        return None
    m = monopoly(P, attempts=8 if fast else 18, seed_g=n["gens"])
    if m is None:
        return None
    return c, n, m


def check(P, c, n, m):
    lim, cap = P["LIM"], P["CAP"]
    info = {}
    # Generation: Comp > Nash > Mono with clear gaps
    info["gen_order"] = (c["total_gen"] > n["total_gen"] + 8
                         and n["total_gen"] > m["total_gen"] + 8)
    # LMP: Mono > Nash > Comp
    info["lmp_order"] = (m["avg_lmp"] > n["avg_lmp"] + 1.5
                         and n["avg_lmp"] > c["avg_lmp"] + 1.5)
    # Profit: Mono > Nash > Comp (industry)
    info["profit_order"] = (m["total_profit"] > n["total_profit"] * 1.12
                            and n["total_profit"] > c["total_profit"] * 1.02)
    # Per-firm cartel IR
    ir = m["profits"] / np.maximum(n["profits"], 1.0)
    info["ir0"], info["ir1"] = float(ir[0]), float(ir[1])
    info["per_firm_ir"] = bool(np.all(ir > 1.08) and np.all(m["profits"] > 50))
    # Headroom vs Nash (cheat / punish)
    firm_nash = np.array([n["gens"][0] + n["gens"][1], n["gens"][2]])
    firm_cap = np.array([cap[0] + cap[1], cap[2]])
    head = firm_cap / np.maximum(firm_nash, 1.0)
    info["head0"], info["head1"] = float(head[0]), float(head[1])
    info["headroom"] = bool(np.all(head > 1.15))
    # Monopoly unbound preferred
    info["mono_unbound"] = bool(np.all(np.abs(m["flows"]) < lim - 0.3))
    # Neither firm nearly zeroed under monopoly
    info["mono_alive"] = bool(np.all(m["gens"][[0, 2]] > 5) or
                              (m["gens"][0] + m["gens"][1] > 20 and m["gens"][2] > 8))
    info["gap_pct"] = 100.0 * (m["total_profit"] / max(n["total_profit"], 1.0) - 1.0)
    info["gap_abs"] = float(m["total_profit"] - n["total_profit"])
    ok = all(info[k] for k in [
        "gen_order", "lmp_order", "profit_order", "per_firm_ir",
        "headroom", "mono_unbound", "mono_alive",
    ])
    return ok, info


def fmt(P, c, n, m, info):
    lines = [
        f"MC={np.round(P['MC'],3).tolist()} QC={np.round(P['QC'],4).tolist()} "
        f"CAP={np.round(P['CAP'],1).tolist()}",
        f"P0={np.round(P['P0'],1).tolist()} Q0={np.round(P['Q0'],1).tolist()} "
        f"LIM={np.round(P['LIM'],1).tolist()}",
    ]
    for name, b in [("Comp", c), ("Nash", n), ("Mono", m)]:
        lines.append(
            f"  {name:4s} gen={b['total_gen']:6.1f}  LMP=${b['avg_lmp']:5.2f}  "
            f"π=${b['total_profit']:7.0f}  "
            f"(π0=${b['profits'][0]:6.0f} π1=${b['profits'][1]:6.0f})  "
            f"g=[{b['gens'][0]:5.1f},{b['gens'][1]:5.1f},{b['gens'][2]:5.1f}]"
        )
    lines.append(
        f"  gap={info['gap_pct']:+.1f}% (${info['gap_abs']:+.0f})  "
        f"IR=({info['ir0']:.2f},{info['ir1']:.2f})  "
        f"head=({info['head0']:.2f},{info['head1']:.2f})  "
        f"monoUnbound={info['mono_unbound']}"
    )
    return "\n".join(lines)


def make_P(P0, Q0, MC, QC, CAP, LIM):
    return dict(
        P0=np.asarray(P0, float),
        Q0=np.asarray(Q0, float),
        MC=np.asarray(MC, float),
        QC=np.asarray(QC, float),
        CAP=np.asarray(CAP, float),
        LIM=np.asarray(LIM, float),
    )


def baseline_candidates():
    """Hand + grid candidates, all keeping the loop topology."""
    cands = []

    # Known good-ish from git HEAD (claims ~52% gap)
    cands.append(make_P(
        [55, 50, 32, 30, 40], [250, 200, 320, 300, 200],
        [22, 22, 25], [0.05, 0.05, 0.025],
        [150, 50, 100], [40, 40, 40, 40, 30],
    ))
    # Conversation retune attempt
    cands.append(make_P(
        [55, 50, 75, 70, 70], [250, 200, 25, 20, 15],
        [18, 18, 22], [0.04, 0.04, 0.02],
        [150, 50, 100], [40, 15, 25, 20, 10],
    ))
    # Liu-Hobbs original costs with raised gen-bus intercepts
    cands.append(make_P(
        [55, 50, 32, 30, 40], [250, 200, 320, 300, 200],
        [15, 15, 18], [0.02, 0.02, 0.01],
        [150, 50, 100], [40, 40, 40, 40, 30],
    ))

    # Systematic grid: load mostly at demand nodes (3,4,5) OR balanced
    for p_gen in ([55, 50], [60, 55], [50, 45]):
        for p_load in ([32, 30, 40], [70, 65, 65], [80, 75, 70], [45, 42, 50]):
            for q_gen in ([250, 200], [200, 180], [300, 250]):
                for q_load in ([320, 300, 200], [80, 70, 60], [40, 35, 30], [150, 140, 120]):
                    for mc0, mc1 in ((18, 22), (20, 25), (15, 20), (22, 28)):
                        for qc0, qc1 in ((0.04, 0.02), (0.05, 0.025), (0.03, 0.015), (0.06, 0.03)):
                            for lims in (
                                [40, 40, 40, 40, 30],
                                [50, 35, 35, 25, 20],
                                [45, 25, 30, 20, 15],
                                [60, 40, 40, 30, 20],
                                [35, 20, 25, 20, 12],
                            ):
                                for caps in (
                                    [150, 50, 100],
                                    [160, 60, 110],
                                    [140, 50, 120],
                                    [180, 60, 100],
                                ):
                                    P0 = list(p_gen) + list(p_load)
                                    Q0 = list(q_gen) + list(q_load)
                                    MC = [mc0, mc0, mc1]
                                    QC = [qc0, qc0, qc1]
                                    cands.append(make_P(P0, Q0, MC, QC, caps, lims))
    # Deduplicate
    seen = set()
    uniq = []
    for P in cands:
        key = (
            tuple(np.round(P["P0"], 2)), tuple(np.round(P["Q0"], 2)),
            tuple(np.round(P["MC"], 3)), tuple(np.round(P["QC"], 4)),
            tuple(np.round(P["CAP"], 1)), tuple(np.round(P["LIM"], 1)),
        )
        if key not in seen:
            seen.add(key)
            uniq.append(P)
    return uniq


if __name__ == "__main__":
    # First evaluate the known git baseline thoroughly
    print("=== Evaluating git HEAD (70dbbb4) params thoroughly ===")
    P0 = make_P(
        [55, 50, 32, 30, 40], [250, 200, 320, 300, 200],
        [22, 22, 25], [0.05, 0.05, 0.025],
        [150, 50, 100], [40, 40, 40, 40, 30],
    )
    r = evaluate(P0, fast=False)
    if r:
        c, n, m = r
        ok, info = check(P0, c, n, m)
        print(fmt(P0, c, n, m, info))
        print("PASS" if ok else "FAIL keys:", {k: info[k] for k in info if isinstance(info[k], bool)})
    else:
        print("baseline failed to solve")

    cands = baseline_candidates()
    print(f"\n=== Searching {len(cands)} candidates (fast) ===")
    results = []
    near = []
    for i, P in enumerate(cands):
        try:
            r = evaluate(P, fast=True)
        except Exception:
            continue
        if r is None:
            continue
        c, n, m = r
        ok, info = check(P, c, n, m)
        score = info["gap_pct"]
        # soft score for near-misses
        soft = (
            int(info["gen_order"]) + int(info["lmp_order"]) + int(info["profit_order"])
            + int(info["per_firm_ir"]) + int(info["headroom"]) + int(info["mono_unbound"])
            + int(info["mono_alive"])
        )
        if ok:
            results.append((score, P, c, n, m, info))
            print(f"[{i}] PASS gap={score:.1f}% IR=({info['ir0']:.2f},{info['ir1']:.2f})")
        elif soft >= 5 and info["per_firm_ir"] and info["gap_pct"] > 15:
            near.append((score, soft, P, c, n, m, info))
        if i % 200 == 0:
            print(f"...{i}/{len(cands)} pass={len(results)} near={len(near)}", file=sys.stderr)

    results.sort(key=lambda t: -t[0])
    near.sort(key=lambda t: (-t[1], -t[0]))
    print(f"\n===== {len(results)} PASS; top 10 =====")
    for score, P, c, n, m, info in results[:10]:
        print(fmt(P, c, n, m, info))
        print()
    if not results:
        print("===== No full PASS; top near-misses =====")
        for score, soft, P, c, n, m, info in near[:12]:
            print(f"soft={soft}/7")
            print(fmt(P, c, n, m, info))
            fails = [k for k, v in info.items() if isinstance(v, bool) and not v]
            print("  FAIL:", fails)
            print()
