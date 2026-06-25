import numpy as np

# 1. Network Parameters (base topology from Liu & Hobbs 2013).
# Demand price-intercepts at the two generation buses (nodes 0,1) are raised to 55/50
# to open headroom between marginal cost (~$15-18) and willingness-to-pay. This widens
# the LCP-Nash -> joint-monopoly profit gap from ~12% to ~52% AND makes the cartel
# individually rational for BOTH firms (firm 1 GAINS under monopoly instead of losing),
# so collusion is sustainable and the deviation/punishment experiment can actually
# exhibit retaliation. 55/50 is the widest bump that preserves the economic ordering
# Competitive < Nash < Monopoly (at 60/55 the competitive solution becomes
# capacity-constrained and out-earns Nash, which is nonsensical). Firms remain
# asymmetric (firm 0 stays dominant).
P0 = np.array([55, 50, 32, 30, 40]) #not constant elasticity
Q0 = np.array([250, 200, 320, 300, 200])

# Marginal and quadratic cost coefficients: C(g) = MC*g + 0.5*QC*g^2.
# MC raised (15->22, 18->25) and QC made more convex (0.02->0.05, 0.01->0.025) so the
# marginal-cost curve rises to meet the price BEFORE full capacity. This pulls the
# COMPETITIVE solution off the capacity ceiling (firm 0's big plant runs ~128/150 MW
# instead of flooding to 150), which makes the competitive start economically natural
# (price-takers no longer max out). It also keeps Competitive < Nash < Monopoly and the
# cartel individually rational for both firms. (Firm 1 stays near its cap at competitive
# by design: that is what keeps collusion individually rational for the small firm while
# the firms remain asymmetric.)
MC = {
    'Firm1_Node1': 22.0,
    'Firm1_Node2': 22.0,
    'Firm2_Node2': 25.0
}
QC = {
    'Firm1_Node1': 0.05,
    'Firm1_Node2': 0.05,
    'Firm2_Node2': 0.025
}

# Thermal limits: Line indices 0:(1-2), 1:(2-3), 2:(3-1), 3:(3-4), 4:(4-5)
LINE_LIMITS = np.array([40.0, 40.0, 40.0, 40.0, 30.0])

def get_ptdf_matrix():
    # 1. Define Topology: (From, To) using 0-based indexing
    # Loop: (1-2, 2-3, 3-1) | Radial: (3-4, 4-5)
    lines = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    num_nodes, num_lines = 5, len(lines)
    
    # 2. Build B-bus (Susceptance) Matrix
    B_bus = np.zeros((num_nodes, num_nodes))
    for u, v in lines:
        B_bus[u, v] = B_bus[v, u] = -1.0 # x=1 assumption
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