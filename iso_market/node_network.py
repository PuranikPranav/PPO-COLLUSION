import os

import numpy as np

# ============================================================================
# MARKET SELECTOR — set MARKET_CONFIG=two_firm to run the two-firm companion
# market (Firm 0: cheap base @ node 1 + expensive peaker @ node 2; Firm 1: one
# mid-cost plant @ node 2). Default (unset or "three_firm") is the three-firm
# hub market below. Topology (nodes/lines/PTDF) is identical in both.
# ============================================================================
MARKET = os.environ.get("MARKET_CONFIG", "three_firm").strip().lower()

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
if MARKET == "two_firm":
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
