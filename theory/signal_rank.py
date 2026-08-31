"""
Is the Sannikov-Skrzypacz observational-equivalence step recoverable on this
network?

Sannikov & Skrzypacz (2007), Prop. 2, relaxes the two firms' incentive
constraints to their SUM, licensed by the sentence

    "Each deviation has the same effect on the distribution of prices,
     decreasing the mean of the observed price to p(Q + e)."

Summing is what makes balanced transfers cancel. It is valid iff the two
deviations shift the distribution of the PUBLIC SIGNAL identically, i.e. iff

    d_i = d_j,      d_i := d(LMP vector) / d(q_i)

In a DC-OPF the LMP vector is  LMP = lambda*1 + sum_{k in B} mu_k * PTDF_k ,
so

    d_i - d_j = (dlambda/dq_i - dlambda/dq_j)*1
                + sum_{k in B} (dmu_k/dq_i - dmu_k/dq_j) * PTDF_{k,.}

With no binding line (B empty) every nodal price equals lambda, which depends
on total injection only, so d_i = d_j and the step is valid. Congestion is
therefore the only thing that can separate the firms' fingerprints.

Because d_i depends on WHERE firm i injects and not on its costs, the whole
question reduces to the rank of the nodal sensitivity matrix

    M[p, n] = d(LMP_p) / d(g_n)

restricted to the columns the firms occupy. This script computes M across
operating points and reports, for every siting of n firms on the 5 nodes:

  * rank of [d_1 ... d_n]                       (>= 2 breaks the Prop-2 step)
  * the second singular value sigma_2           (HOW separated they are --
    rank 2 with sigma_2 ~ 0 is the paper's Sec. VII alpha -> 1/2 case, where
    collusion still dies)
  * which lines are binding, since B drives everything

Run:  PYTHONPATH=. python theory/signal_rank.py
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from iso_market.market_env import ElectricityMarketEnv, NUM_NODES

# Total generation levels to probe, spanning the three benchmarks of the
# three-firm hub market (monopoly 233, Nash 326, competitive 417 MW).
G_LEVELS = [233.0, 280.0, 326.0, 380.0, 417.0]

FD_STEP = 0.25          # MW, finite-difference step for the Jacobian
KINK_TOL = 1e-3         # $/MWh/MW gap between one-sided derivatives => kink
# Rank must be judged RELATIVE to the leading singular value: the Jacobian is a
# finite difference, so an absolute tolerance counts differencing noise (~1e-7)
# as signal and reports rank 5 on a matrix that is plainly rank 2.
RANK_REL_TOL = 1e-4


def _rank(sv):
    return int((sv > RANK_REL_TOL * sv[0]).sum()) if sv[0] > 0 else 0


def clear(env, gen_by_node, u=0.0):
    """Clear the DC-OPF for a per-node generation vector. Returns (lmps, shadow)."""
    env._demand_u = float(u)
    lmps, demand, flows, shadow = env._clear_market(np.asarray(gen_by_node, dtype=float))
    if lmps is None:
        raise RuntimeError(f"DC-OPF infeasible at gen={gen_by_node}")
    return np.asarray(lmps), np.asarray(shadow)


def lmp_jacobian(env, gen_by_node, h=FD_STEP):
    """M[p, n] = d(LMP_p)/d(g_n), central difference, plus a kink flag per node.

    Returns (M, M_left, M_right, binding_set).
    """
    base_lmp, shadow = clear(env, gen_by_node)
    binding = tuple(np.where(np.abs(shadow) > 1e-6)[0].tolist())

    M = np.zeros((NUM_NODES, NUM_NODES))
    ML = np.zeros_like(M)
    MR = np.zeros_like(M)
    for n in range(NUM_NODES):
        g0 = float(gen_by_node[n])
        # Generation is constrained nonnegative, so the backward step is only
        # available where the node already hosts output. At an empty node the
        # right derivative IS the economically meaningful one (a firm sited
        # there can only inject upward from zero).
        h_back = min(h, g0)
        gp = np.array(gen_by_node, dtype=float)
        gp[n] += h
        lp, _ = clear(env, gp)
        MR[:, n] = (lp - base_lmp) / h
        if h_back > 0:
            gm = np.array(gen_by_node, dtype=float)
            gm[n] -= h_back
            lm, _ = clear(env, gm)
            ML[:, n] = (base_lmp - lm) / h_back
            M[:, n] = (lp - lm) / (h + h_back)
        else:
            ML[:, n] = MR[:, n]
            M[:, n] = MR[:, n]
    return M, ML, MR, binding


def fingerprint_rank(M, nodes):
    """Rank and singular values of the firms' deviation fingerprints."""
    D = np.column_stack([M[:, n] for n in nodes])
    sv = np.linalg.svd(D, compute_uv=False)
    return _rank(sv), sv, D


def main():
    env = ElectricityMarketEnv()
    n_firms = 3

    print("=" * 78)
    print("SIGNAL-RANK SEARCH  --  can congestion separate the firms' fingerprints?")
    print("=" * 78)
    print(f"nodes = {NUM_NODES}, firms = {n_firms}, finite-difference step = {FD_STEP} MW")
    print("d_i = d(LMP vector)/d(q_i) depends only on firm i's NODE, so a siting")
    print("is just an assignment of firms to nodes.\n")

    # ---- Part 1: the nodal sensitivity matrix at each operating point -------
    per_point = {}
    for G in G_LEVELS:
        # Spread the generation evenly across nodes that can host it; the
        # operating point (hence the binding set) is what we are probing.
        gen = np.zeros(NUM_NODES)
        gen[1] = G                      # baseline: everything at the hub
        try:
            M, ML, MR, binding = lmp_jacobian(env, gen)
        except RuntimeError as e:
            print(f"G={G:6.1f}: {e}")
            continue
        per_point[G] = (M, binding)
        kinked = [n for n in range(NUM_NODES)
                  if np.max(np.abs(MR[:, n] - ML[:, n])) > KINK_TOL]
        print(f"--- G = {G:6.1f} MW at the hub | binding lines {binding if binding else '(none)'}"
              f" | kinked injection nodes {kinked if kinked else '(none)'}")
        print("    M[p,n] = dLMP_p/dg_n   (rows = price node, cols = injection node)")
        for p in range(NUM_NODES):
            print("      " + "  ".join(f"{M[p, n]:+.5f}" for n in range(NUM_NODES)))
        sv = np.linalg.svd(M, compute_uv=False)
        print(f"    rank(M) = {_rank(sv)}   singular values = {np.round(sv, 6).tolist()}")
        print(f"    theory: rank <= 1 + |B| = {1 + len(binding)}")
        print()

    # ---- Part 2: every siting of 3 firms on 5 nodes -------------------------
    print("=" * 78)
    print("SITINGS  --  rank >= 2 breaks the Prop-2 summing step")
    print("=" * 78)
    header = f"{'siting':<14}" + "".join(f"{f'G={g:.0f}':>22}" for g in per_point)
    print(header)
    print("-" * len(header))

    viable = []
    for nodes in itertools.combinations_with_replacement(range(NUM_NODES), n_firms):
        cells = []
        best_sigma2 = 0.0
        for G, (M, binding) in per_point.items():
            r, sv, _ = fingerprint_rank(M, nodes)
            s2 = float(sv[1]) if len(sv) > 1 else 0.0
            best_sigma2 = max(best_sigma2, s2)
            cells.append(f"rank {r}, s2={s2:.2e}".rjust(22))
        line = f"{str(nodes):<14}" + "".join(cells)
        if best_sigma2 > 1e-3:
            viable.append((nodes, best_sigma2))
            line += "   <== SEPARATES"
        print(line)

    print()
    print("=" * 78)
    if viable:
        print(f"{len(viable)} siting(s) produce distinguishable fingerprints.")
        print("Ranked by separation (larger sigma_2 = easier to tell deviators apart):")
        for nodes, s2 in sorted(viable, key=lambda t: -t[1]):
            print(f"   firms at nodes {nodes}:  sigma_2 = {s2:.6e}")
        print()
        print("NOTE: rank >= 2 only removes the obstruction. sigma_2 must also be")
        print("large relative to the price noise, else this is Sannikov-Skrzypacz")
        print("Sec. VII (alpha -> 1/2), where collusion dies anyway.")
    else:
        print("NO siting separates the firms on this network.")
        print("=> the Prop-2 summing step is valid for EVERY siting here, and the")
        print("   impossibility result holds on the whole Liu & Hobbs 5-node")
        print("   topology, not just the hub configuration.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
