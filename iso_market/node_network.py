import os

import numpy as np

# ============================================================================
# MARKET SELECTOR — MARKET_CONFIG picks the ownership structure. The topology
# (nodes / lines / PTDF) is IDENTICAL in all three; only the scalars change.
#
#   three_firm_dist  (default)  Structure 2, redistributed: three firms, one
#                    plant each, one per node (nodes 1, 2, 3), with the MOST
#                    EXPENSIVE plant at node 3 — and node 5 an inelastic load
#                    pocket behind the 3–4–5 tail.
#   three_firm       Structure 2 as it was: the same three firms, all sited at
#                    the node-2 hub. Kept so the earlier results stay reproducible.
#   two_firm         Structure 1: the paper's duopoly layout (Firm 0 = cheap base
#                    @ node 1 + peaker @ node 2; Firm 1 = one plant @ node 2).
# ============================================================================
MARKET = os.environ.get("MARKET_CONFIG", "three_firm_dist").strip().lower()
if MARKET not in ("three_firm_dist", "three_firm", "two_firm"):
    raise ValueError(
        f"MARKET_CONFIG={MARKET!r} is not one of "
        "'three_firm_dist', 'three_firm', 'two_firm'"
    )

# 1. Network Parameters (topology fixed from Liu & Hobbs 2013; ONLY these scalars change).
#
# THREE ASYMMETRIC FIRMS, one plant each, ALL at node 2 (the generation hub);
# nodes 1, 3, 4, 5 are pure demand/import centers served over the network.
#
# WHY all firms share node 2: in the paper's Nash–Cournot LCP (eqs. 39-45)
# each firm's stationarity uses its OWN node's inverse-demand slope. A firm
# that sits alone on a node is therefore a local monopolist ALREADY AT NASH,
# which makes the Nash outcome nearly as profitable as the cartel (tiny
# Nash→Monopoly gap) and breaks per-firm cartel rationality. Placing the
# three firms on the same demand-heavy hub makes Nash a genuine three-way
# Cournot race — far from the joint monopoly — which is exactly the wide gap
# the collusion index needs. (Verified across ~2,600 swept parameter settings:
# every 2+1 siting caps at ~+5-7% profit gap; the shared hub reaches +22.5%.)
#
# Design goals (verified by competitive / MCP-Nash / joint-monopoly benchmarks):
#   1. WIDE Nash → Monopoly gap in avg LMP, generation AND total profit
#      (the combined-Δ denominator, paper eq. 47).
#   2. Generation ordering: Competitive > Nash > Monopoly.
#   3. PER-FIRM individual rationality of the cartel: EACH firm's joint-monopoly
#      profit strictly exceeds its Nash profit (margins ≈ +20-24%), so tacit
#      collusion is incentive-compatible for ALL THREE firms and the learned
#      combined Δ can settle in the interior of (0, 1).
#   4. Caps sit 25-38% above Nash output, so every firm has headroom to CHEAT
#      (raise output) and to PUNISH (flood after a rival cheats) — required
#      for the deviation / impulse-response experiment.
#
# With the parameters below (verified via the Python DC-OPF / MCP / MPEC trio):
#   Competitive : gen ≈ 417 MW,  π ≈ $2084, avg LMP ≈ $30.3  (caps + exports bind)
#   MCP Nash    : gen ≈ 326 MW,  π ≈ $5429, avg LMP ≈ $39.5  (line 1-2 binds)
#   Monopoly    : gen ≈ 233 MW,  π ≈ $6653, avg LMP ≈ $47.7  (NO binding lines)
#   Nash→Mono gaps: LMP +$8.2/MWh, profit +$1224/step (+22.5%)
#   Per-firm cartel IR margins: F0 +$435 (+24%), F1 +$437 (+24%), F2 +$352 (+20%)
if MARKET == "three_firm_dist":
    # ------------------------------------------------------------------
    # STRUCTURE 2, REDISTRIBUTED (default) — one plant per node, costliest
    # at node 3, and node 5 turned into a real load pocket.
    #
    # WHAT CHANGED AND WHY (both points are Andrew's, from the 27 Jul meeting):
    #
    # 1. "Redistribute the three plants — one at each node, with the most
    #    expensive plant at node 3."  Firm 0 (MC 15) sits at node 1, Firm 1
    #    (MC 16) at node 2, Firm 2 (MC 18) at node 3 — the head of the radial
    #    3–4–5 tail that feeds the load pocket. Marginal-cost ordering
    #    15 < 16 < 18 and average cost at the collusive dispatch
    #    $18.23 < $18.73 < $19.73 /MWh, so node 3 is unambiguously the
    #    expensive unit.
    #
    #    The cost SPREAD is deliberately moderate. Widening it to MC 14/16/20
    #    makes the cartel cut the node-3 plant from 69 MW to 41 MW, at which
    #    point that firm earns LESS under joint monopoly than at Nash and has
    #    no reason to collude at all (per-firm IR fails, verified). The
    #    quadratic terms run the other way (0.090 > 0.075 > 0.050) for the same
    #    reason: they stop the cartel concentrating output in one firm.
    #
    # 2. "Make node 5's demand inelastic — the whole point of the topology is
    #    for node 5 to be a load pocket."  With linear demand
    #    d_i = Q0_i (1 − p/P0_i), inelasticity means a LARGE P0_i relative to
    #    Q0_i: node 5 gets P0 = 900, Q0 = 11, so its slope is 81.8 $/MWh per MW
    #    and its point elasticity is −0.06, against −0.7…−1.1 before. Its load
    #    is ~10.3–10.4 MW no matter what the price does — must-serve load whose
    #    intercept is effectively the value of lost load. All the demand
    #    response in this market now has to come from nodes 1–4.
    #
    # WHICH NASH. Every firm sitting alone on its own node breaks the paper's
    # LCP benchmark (eqs. 39-45), which gives each firm the inverse-demand slope
    # of ITS OWN node. Nodes 1 and 3 are steep locally but price-coupled to the
    # rest of the network whenever their lines are slack, so the LCP has them
    # withhold as if they were local monopolists: at the LCP point a firm gains
    # $1,288 by deviating, and LCP "Nash" generation (207.6 MW) is BELOW joint
    # monopoly (213.4 MW) — it is more collusive than the cartel. Δ is therefore
    # measured against the iterated best-response Nash of the DC-OPF game
    # (`market_multi.best_response_nash`), which IS an equilibrium here (max
    # unilateral deviation gain $0.00). Both are always reported side by side.
    #
    # VERIFIED BENCHMARKS (Python DC-OPF / MCP / direct joint-profit trio):
    #   Competitive : gen 395.4 MW, π $1831, avg LMP $27.03
    #                 binding: 2–3 (rent −2.24) and 3–4 (rent +27.62)
    #   Nash (BR)   : gen 306.6 MW, π $6515, avg LMP $41.83
    #                 binding: 3–4 only (rent +11.30)
    #   Monopoly    : gen 213.4 MW, π $8169, avg LMP $57.16   binding: NONE
    #   Nash→Mono   : gen −93.2 MW, LMP +$15.33, profit +$1654 (+25.4%)
    #   Per-firm cartel IR margins: F0 +$617, F1 +$609, F2 +$428 (ALL POSITIVE)
    #   Caps sit 35–48% above Nash output — headroom to cheat and to punish.
    #
    # THE LOAD POCKET IN ONE LINE: under competition and Nash the 3–4 corridor
    # is congested, so nodes 4 and 5 pay $52.54 while nodes 1–3 pay $25–$42.
    # When the cartel withholds, the system price rises ABOVE $52.54, the
    # corridor un-congests, and the pocket's premium disappears — collusion
    # de-congests the network and levels every nodal price at $57.16.
    # ------------------------------------------------------------------
    P0 = np.array([85, 95, 90, 88, 900])   # node 5: value-of-lost-load intercept
    Q0 = np.array([140, 300, 90, 14, 11])  # node 5: ~11 MW of must-serve load

    MC = {
        'Firm1_Node1': 15.0,   # Firm 0 — cheapest, node 1
        'Firm2_Node2': 16.0,   # Firm 1 — mid, node 2 (largest load centre)
        'Firm3_Node3': 18.0,   # Firm 2 — MOST EXPENSIVE, node 3 (feeds the pocket)
    }
    QC = {
        'Firm1_Node1': 0.090,
        'Firm2_Node2': 0.075,
        'Firm3_Node3': 0.050,
    }
    CAP = {
        'Firm1_Node1': 135.0,
        'Firm2_Node2': 145.0,
        'Firm3_Node3': 155.0,
    }
    # Line indices 0:(1-2), 1:(2-3), 2:(3-1), 3:(3-4), 4:(4-5).
    # 3–4 is the load-pocket corridor and is the constraint that discriminates
    # between the market outcomes; 3–1 and 4–5 never bind at any benchmark.
    LINE_LIMITS = np.array([36.0, 45.0, 38.0, 16.0, 12.0])
    PLANT_SPECS = [
        (0, 0, 'Firm1_Node1'),
        (1, 1, 'Firm2_Node2'),
        (2, 2, 'Firm3_Node3'),
    ]
elif MARKET == "two_firm":
    # ------------------------------------------------------------------
    # TWO-FIRM COMPANION MARKET (paper's original 2-firm/3-plant layout,
    # retuned for wide gaps). Verified benchmarks:
    #   Competitive : gen 448.7, π $4160, LMP $30.18
    #   MCP Nash    : gen 324.1, π $6295, LMP $40.39
    #   Monopoly    : gen 267.5, π $7408, LMP $45.03  (cartel maxes base 54→115,
    #                 SHUTS the peaker 107→0; no line binds in any benchmark)
    #   N→M gaps: LMP +$4.64 (+11.5%), gen −56.6 MW (−17.5%), π +$1114 (+17.7%)
    #   Firm IR: F0 +$591 (+19.6%), F1 +$523 (+15.9%)
    # ------------------------------------------------------------------
    P0 = np.array([70, 65, 95, 90, 90])
    Q0 = np.array([140, 630, 20, 15, 12])
    MC = {
        'Firm1_Node1': 10.0,   # Firm 0 cheap base plant @ node 1
        'Firm1_Node2': 24.0,   # Firm 0 expensive peaker @ node 2 (hub)
        'Firm2_Node2': 17.0,   # Firm 1 mid-cost plant @ node 2 (hub)
    }
    QC = {
        'Firm1_Node1': 0.065,
        'Firm1_Node2': 0.050,
        'Firm2_Node2': 0.040,
    }
    CAP = {
        'Firm1_Node1': 115.0,
        'Firm1_Node2': 125.0,
        'Firm2_Node2': 210.0,
    }
    LINE_LIMITS = np.array([55.0, 40.0, 40.0, 20.0, 12.0])
    # (firm, node, cost-key) per plant
    PLANT_SPECS = [
        (0, 0, 'Firm1_Node1'),
        (0, 1, 'Firm1_Node2'),
        (1, 1, 'Firm2_Node2'),
    ]
else:
    # ------------------------------------------------------------------
    # THREE-FIRM HUB MARKET (default)
    # ------------------------------------------------------------------
    P0 = np.array([70, 65, 95, 90, 90])  # not constant elasticity
    Q0 = np.array([200, 550, 20, 15, 12])

    # Marginal and quadratic cost: C(g) = MC*g + 0.5*QC*g^2.
    # Asymmetry: MC and QC both differ; the cheap-MC firm has the steepest QC, so
    # the joint-monopoly allocation cannot concentrate output in one firm
    # (per-firm IR, goal 3).
    MC = {
        'Firm1_Node2': 15.0,
        'Firm2_Node2': 16.0,
        'Firm3_Node2': 18.0
    }
    QC = {
        'Firm1_Node2': 0.090,
        'Firm2_Node2': 0.075,
        'Firm3_Node2': 0.050
    }

    # Capacity (MW) per firm/plant. Binding only under perfect competition;
    # ~25-38% above Nash output so deviation/punishment headroom exists.
    CAP = {
        'Firm1_Node2': 135.0,
        'Firm2_Node2': 145.0,
        'Firm3_Node2': 155.0
    }

    # Thermal limits (MW): Line indices 0:(1-2), 1:(2-3), 2:(3-1), 3:(3-4), 4:(4-5)
    # Export lines out of the hub bind under competitive/Nash play (congestion is
    # part of the story); the joint monopoly withholds enough that no line binds
    # with a positive shadow price (line 1-2 sits AT its limit, zero rent).
    LINE_LIMITS = np.array([50.0, 42.0, 38.0, 20.0, 12.0])
    PLANT_SPECS = [
        (0, 1, 'Firm1_Node2'),
        (1, 1, 'Firm2_Node2'),
        (2, 1, 'Firm3_Node2'),
    ]

def get_ptdf_matrix():
    # 1. Define Topology: (From, To) using 0-based indexing
    # Loop: (1-2, 2-3, 3-1) | Radial: (3-4, 4-5)
    lines = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    num_nodes, num_lines = 5, len(lines)

    # 2. Build B-bus (Susceptance) Matrix
    B_bus = np.zeros((num_nodes, num_nodes))
    for u, v in lines:
        B_bus[u, v] = B_bus[v, u] = -1.0  # x=1 assumption
        B_bus[u, u] += 1.0
        B_bus[v, v] += 1.0

    # 3. Invert Reduced B-bus (Reference Node 5 at Index 4)
    B_inv_reduced = np.linalg.inv(np.delete(np.delete(B_bus, 4, 0), 4, 1))

    # 4. Expand back to 5x5
    B_inv = np.zeros((num_nodes, num_nodes))
    B_inv[:4, :4] = B_inv_reduced

    # 5. Build Incidence Matrix A
    A = np.zeros((num_lines, num_nodes))
    for i, (u, v) in enumerate(lines):
        A[i, u], A[i, v] = 1, -1

    # 6. Final PTDF
    return A @ B_inv


# Verification print (optional)
if __name__ == "__main__":
    matrix = get_ptdf_matrix()
    print("PTDF Matrix (Rows: Lines, Cols: Nodes):")
    print(np.round(matrix, 2))
