"""
Sannikov-Skrzypacz (2007) on a congested network: the geometry that decides it.

S&S kill collusion by RELAXING the N incentive constraints to their SUM. That
step is licensed by one sentence in the proof of Prop. 2:

    "Each deviation has the same effect on the distribution of prices."

Summing is what makes balanced transfers cancel. On a network the object that
sentence is about is firm i's DEVIATION FINGERPRINT

    d_i := d(LMP vector) / d(q_i)   in R^{NUM_NODES}

and the sentence says d_i = d_j. Under a DC-OPF,

    LMP = lambda * 1 + sum_{k in B} mu_k * PTDF_k

so d_i - d_j = (dlambda/dq_i - dlambda/dq_j) 1
              + sum_{k in B} (dmu_k/dq_i - dmu_k/dq_j) PTDF_{k,.}

which is identically 0 when the binding set B is empty. Congestion is the ONLY
thing on this network that can separate the firms.

What this script measures, at every operating point along competitive -> Nash ->
monopoly:

  1. B, the binding set, and the fingerprints d_i.
  2. rho_ij, the cosine between d_i and d_j in the Sigma^{-1} metric (Sigma =
     nodal price-noise covariance). rho = 1 is exactly the S&S hypothesis.
  3. COST(q) := min { sum_i beta_i' Sigma beta_i : sum_i beta_i = 0,
                      beta_i' d_i = -a_i }
     the per-unit-time volatility a cartel must load onto continuation values to
     enforce q with BALANCED transfers (no value burnt, no money). Infeasible
     <=> S&S impossibility bites. For two firms COST = c' G^{-1} c with
     G = [d_i' Sigma^{-1} d_j] and c = (-a_1, +a_2), which equals
     2a^2 / (s^2 (1 - rho)) in the symmetric case -- it blows up as rho -> 1.
  4. a_i := d(pi_i)/d(q_i), the static deviation slope. S&S Lemma 4 needs
     a_i >= eps_pi > 0 uniformly on the collusive set; a saturated line can
     drive a_i <= 0, which is an independent break.

Run:  PYTHONPATH=. python theory/fingerprint_geometry.py
"""

from __future__ import annotations

import sys

import numpy as np

from iso_market.market_env import (
    ElectricityMarketEnv,
    NUM_NODES,
    NUM_FIRMS,
    PLANTS,
    FIRM_PLANT_IDX,
)

FD_STEP = 0.25          # MW, finite-difference step
BIND_TOL = 1e-6         # $/MWh on a line's shadow price
RANK_REL_TOL = 1e-4     # singular values below this * sigma_1 are noise


# ----------------------------------------------------------------------------
# Market primitives
# ----------------------------------------------------------------------------
def firm_nodes():
    """Node each firm injects at. Multi-plant firms are reported per plant."""
    return {f: [PLANTS[p]["node"] for p in FIRM_PLANT_IDX[f]] for f in range(NUM_FIRMS)}


def gen_by_node(q):
    """Per-firm output vector q (one entry per PLANT) -> per-node injection."""
    g = np.zeros(NUM_NODES)
    for p, val in enumerate(q):
        g[PLANTS[p]["node"]] += val
    return g


def clear(env, q, u=0.0):
    env._demand_u = float(u)
    lmps, demand, flows, shadow = env._clear_market(gen_by_node(q))
    if lmps is None:
        raise RuntimeError(f"DC-OPF infeasible at q={q}")
    return np.asarray(lmps), np.asarray(shadow)


def profits(env, q):
    lmps, _ = clear(env, q)
    out = np.zeros(NUM_FIRMS)
    for f in range(NUM_FIRMS):
        for p in FIRM_PLANT_IDX[f]:
            g = q[p]
            out[f] += lmps[PLANTS[p]["node"]] * g - (
                PLANTS[p]["mc"] * g + 0.5 * PLANTS[p]["qc"] * g**2
            )
    return out


# ----------------------------------------------------------------------------
# Fingerprints and deviation slopes
# ----------------------------------------------------------------------------
def fingerprints(env, q, h=FD_STEP):
    """d_p = d(LMP vector)/d(q_p) for every plant p, plus the binding set."""
    base_lmp, shadow = clear(env, q)
    binding = tuple(np.where(np.abs(shadow) > BIND_TOL)[0].tolist())

    D = np.zeros((NUM_NODES, len(q)))
    for p in range(len(q)):
        hb = min(h, float(q[p]))          # generation is nonnegative
        qp = np.array(q, float); qp[p] += h
        lp, _ = clear(env, qp)
        if hb > 0:
            qm = np.array(q, float); qm[p] -= hb
            lm, _ = clear(env, qm)
            D[:, p] = (lp - lm) / (h + hb)
        else:
            D[:, p] = (lp - base_lmp) / h
    return D, binding, base_lmp


def deviation_slopes(env, q, h=FD_STEP):
    """a_f = d(pi_f)/d(q_f), firm f raising ALL its plants by h/n_plants."""
    a = np.zeros(NUM_FIRMS)
    for f in range(NUM_FIRMS):
        idx = FIRM_PLANT_IDX[f]
        qp = np.array(q, float)
        for p in idx:
            qp[p] += h / len(idx)
        qm = np.array(q, float)
        step_back = min(h / len(idx), min(q[p] for p in idx))
        for p in idx:
            qm[p] -= step_back
        a[f] = (profits(env, qp)[f] - profits(env, qm)[f]) / (
            h / len(idx) + step_back
        )
    return a


def firm_fingerprints(D):
    """Collapse plant fingerprints to firm fingerprints (sum over own plants)."""
    return np.column_stack(
        [D[:, FIRM_PLANT_IDX[f]].sum(axis=1) for f in range(NUM_FIRMS)]
    )


# ----------------------------------------------------------------------------
# The two quantities the theorem turns on
# ----------------------------------------------------------------------------
def rho_matrix(Dfirm, Sinv):
    """Pairwise cosines of the fingerprints in the Sigma^{-1} inner product."""
    n = Dfirm.shape[1]
    G = Dfirm.T @ Sinv @ Dfirm
    s = np.sqrt(np.maximum(np.diag(G), 0.0))
    R = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if s[i] > 0 and s[j] > 0:
                R[i, j] = G[i, j] / (s[i] * s[j])
    return R, s, G


def balanced_transfer_cost(Dfirm, a, Sigma):
    """
    min  sum_i beta_i' Sigma beta_i
    s.t. sum_i beta_i = 0            (balanced: value transferred, never burnt)
         beta_i' d_i  = -a_i         (firm i's local IC against expanding)

    Returns (cost, feasible). cost = inf  <=>  the S&S summing step still binds
    and only value DESTRUCTION can provide incentives.

    Solved as an equality-constrained QP via the KKT system; infeasibility is
    detected from the rank of the constraint matrix against its augmented form.
    """
    n_nodes, n = Dfirm.shape
    dim = n_nodes * n

    # Constraint matrix A x = b, with x = vec(beta_1, ..., beta_n).
    rows, rhs = [], []
    for k in range(n_nodes):                      # sum_i beta_i = 0
        r = np.zeros(dim)
        for i in range(n):
            r[i * n_nodes + k] = 1.0
        rows.append(r); rhs.append(0.0)
    for i in range(n):                            # beta_i' d_i = -a_i
        r = np.zeros(dim)
        r[i * n_nodes:(i + 1) * n_nodes] = Dfirm[:, i]
        rows.append(r); rhs.append(-a[i])
    A = np.array(rows)
    b = np.array(rhs)

    # Feasibility: rank(A) == rank([A|b]).
    ra = np.linalg.matrix_rank(A, tol=1e-9 * max(1.0, np.abs(A).max()))
    rab = np.linalg.matrix_rank(
        np.column_stack([A, b]), tol=1e-9 * max(1.0, np.abs(A).max())
    )
    if ra != rab:
        return np.inf, False

    # Minimum-Sigma-norm solution: x = Sblk^{-1} A' (A Sblk^{-1} A')^+ b.
    Sblk_inv = np.kron(np.eye(n), np.linalg.inv(Sigma))
    Mid = A @ Sblk_inv @ A.T
    lam = np.linalg.pinv(Mid) @ b
    x = Sblk_inv @ A.T @ lam
    Sblk = np.kron(np.eye(n), Sigma)
    return float(x @ Sblk @ x), True


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------
def joint_monopoly(env, caps):
    from scipy.optimize import minimize

    f = lambda q: -profits(env, np.clip(q, 0, caps)).sum()
    best, bq = np.inf, None
    for frac in (0.3, 0.5, 0.7):
        r = minimize(f, frac * caps, method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-6})
        if r.fun < best:
            best, bq = r.fun, np.clip(r.x, 0, caps)
    return bq


def best_response_nash(env, caps, iters=400, damp=0.5):
    from scipy.optimize import minimize_scalar

    q = 0.5 * caps
    for _ in range(iters):
        q_new = q.copy()
        for f in range(NUM_FIRMS):
            idx = FIRM_PLANT_IDX[f]

            def negp(t):
                qq = q.copy()
                for p in idx:
                    qq[p] = np.clip(t, 0, caps[p])
                return -profits(env, qq)[f]

            r = minimize_scalar(negp, bounds=(0.0, float(min(caps[p] for p in idx))),
                                method="bounded", options={"xatol": 1e-4})
            for p in idx:
                q_new[p] = damp * r.x + (1 - damp) * q[p]
        if np.max(np.abs(q_new - q)) < 1e-5:
            q = q_new
            break
        q = q_new
    return q


# ----------------------------------------------------------------------------
def report(env, label, q, Sigmas):
    D, binding, lmp = fingerprints(env, q)
    Df = firm_fingerprints(D)
    a = deviation_slopes(env, q)
    sv = np.linalg.svd(Df, compute_uv=False)
    rank = int((sv > RANK_REL_TOL * sv[0]).sum()) if sv[0] > 0 else 0

    print(f"\n--- {label}")
    print(f"    q            = {np.round(q, 2).tolist()}   (total {q.sum():.1f} MW)")
    print(f"    LMPs         = {np.round(lmp, 2).tolist()}")
    print(f"    binding lines= {binding if binding else '(none)'}")
    print(f"    profits      = {np.round(profits(env, q), 1).tolist()}")
    print(f"    a_i = dpi_i/dq_i = {np.round(a, 4).tolist()}"
          f"    {'<-- SOME a_i <= 0: Lemma 4 fails' if np.any(a <= 1e-6) else ''}")
    print("    firm fingerprints d_i (rows = price node):")
    for k in range(NUM_NODES):
        print("      " + "  ".join(f"{Df[k, i]:+.6f}" for i in range(NUM_FIRMS)))
    print(f"    rank[d_1..d_N] = {rank}   sv = {np.round(sv, 6).tolist()}")

    for name, Sigma in Sigmas.items():
        Sinv = np.linalg.inv(Sigma)
        R, s, _ = rho_matrix(Df, Sinv)
        cost, feas = balanced_transfer_cost(Df, a, Sigma)
        pair = [(i, j, R[i, j]) for i in range(NUM_FIRMS) for j in range(i + 1, NUM_FIRMS)]
        pretty = "  ".join(f"rho_{i}{j}={r:+.6f}" for i, j, r in pair)
        print(f"    [Sigma={name}] {pretty}")
        print(f"    [Sigma={name}] min 1-|rho| = {min(1 - abs(r) for _, _, r in pair):.3e}"
              f"    balanced-transfer cost = "
              f"{'INFEASIBLE (S&S bites)' if not feas else f'{cost:.4g}'}")


def main():
    env = ElectricityMarketEnv()
    caps = np.array([p["cap"] for p in PLANTS], float)

    print("=" * 78)
    print("FINGERPRINT GEOMETRY OF THE S&S SUMMING STEP ON A CONGESTED NETWORK")
    print("=" * 78)
    print(f"firms -> nodes: {firm_nodes()}")

    # Noise covariances for the nodal price vector.
    #   iid    : full-rank isotropic measurement noise (the honest benchmark).
    #   shock  : the model's OWN scalar demand shock, u shifting every intercept
    #            -- a RANK-1 covariance, and therefore degenerate monitoring.
    q_mid = 0.5 * caps
    l_hi, _ = clear(env, q_mid, u=+1.0)
    l_lo, _ = clear(env, q_mid, u=-1.0)
    v = (l_hi - l_lo) / 2.0                       # dLMP/du
    Sigma_shock_rank1 = np.outer(v, v)
    print(f"\ndLMP/du = {np.round(v, 4).tolist()}   "
          f"rank of the model's own price-noise covariance = "
          f"{np.linalg.matrix_rank(Sigma_shock_rank1, tol=1e-9)}")
    Sigmas = {
        "iid": np.eye(NUM_NODES),
        "shock+iid": Sigma_shock_rank1 + 1e-2 * np.eye(NUM_NODES),
    }

    q_mono = joint_monopoly(env, caps)
    q_nash = best_response_nash(env, caps)

    print("\n" + "=" * 78)
    print("BENCHMARKS")
    print("=" * 78)
    report(env, "JOINT MONOPOLY (the collusive target)", q_mono, Sigmas)
    report(env, "STATIC NASH (best-response)", q_nash, Sigmas)

    print("\n" + "=" * 78)
    print("PATH  Nash -> monopoly   (t=0 Nash, t=1 monopoly): does the")
    print("congestion that separates the firms survive the withholding?")
    print("=" * 78)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = (1 - t) * q_nash + t * q_mono
        D, binding, _ = fingerprints(env, q)
        Df = firm_fingerprints(D)
        a = deviation_slopes(env, q)
        R, _, _ = rho_matrix(Df, np.eye(NUM_NODES))
        worst = min(1 - abs(R[i, j]) for i in range(NUM_FIRMS)
                    for j in range(i + 1, NUM_FIRMS))
        cost, feas = balanced_transfer_cost(Df, a, np.eye(NUM_NODES))
        print(f"  t={t:.2f}  Q={q.sum():7.2f} MW  binding={str(binding) if binding else '()':<10}"
              f"  min(1-|rho|)={worst:.3e}"
              f"  cost={'INF' if not feas else f'{cost:9.4g}'}"
              f"  min a_i={a.min():+.4f}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    sys.exit(main())
