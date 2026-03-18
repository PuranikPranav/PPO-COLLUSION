import numpy as np

# 1. Network Parameters from Table 1
P0 = np.array([40, 35, 32, 30, 40])
Q0 = np.array([250, 200, 320, 300, 200])

# Marginal costs for Firm 1 (Nodes 1, 2) and Firm 2 (Node 2)
MC = {
    'Firm1_Node1': 15.0,
    'Firm1_Node2': 15.0,
    'Firm2_Node2': 18.0
}

# Thermal limits for critical arcs
LINE_LIMITS = {
    'line_1_2': 40.0,  # Line 0 (Node 1-2)
    'line_2_3': 40.0,  # Line 1 (Node 2-3)
    'line_1_3': 40.0,  # Line 2 (Node 1-3) -- Note: check your line indexing order!
    'line_3_4': 40.0,  # Line 3 (Node 3-4)
    'line_4_5': 30.0   # Line 4 (Node 4-5)
}

import numpy as np

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