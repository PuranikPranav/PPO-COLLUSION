import numpy as np

# 1. Network Parameters from Table 1 (Liu & Hobbs 2013)
P0 = np.array([40, 35, 32, 30, 40])
Q0 = np.array([250, 200, 320, 300, 200])

# Marginal and quadratic cost coefficients: C(g) = MC*g + 0.5*QC*g^2
MC = {
    'Firm1_Node1': 15.0,
    'Firm1_Node2': 15.0,
    'Firm2_Node2': 18.0
}
QC = {
    'Firm1_Node1': 0.02,
    'Firm1_Node2': 0.02,
    'Firm2_Node2': 0.01
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