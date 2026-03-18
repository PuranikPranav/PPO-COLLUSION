import sys, os
import numpy as np
import cvxpy as cp
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements

# Add path to find iso_market logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iso_market.node_network import get_ptdf_matrix

# # --- INPUT DATA ---
# P0_VALS = np.array([40.0, 35.0, 32.0, 30.0, 40.0]) 
# Q0_VALS = np.array([250.0, 200.0, 320.0, 300.0, 200.0])

# class EquilibriumSolver:
#     def __init__(self):
#         self.ptdf = get_ptdf_matrix() 
#         # Line Indices: 0:(1-2), 1:(2-3), 2:(1-3), 3:(3-4), 4:(4-5)
#         self.line_limits = np.array([40.0, 40.0, 40.0, 40.0, 30.0]) 

#     # =========================================================================
#     # 1. PERFECT COMPETITION (CVXPY)
#     # =========================================================================
#     def solve_competitive(self):
#         print("\n--- 1. Solving Perfect Competition (CVXPY) ---")
#         g1_n1 = cp.Variable(nonneg=True) 
#         g1_n2 = cp.Variable(nonneg=True) 
#         g2_n2 = cp.Variable(nonneg=True) 
#         d = cp.Variable(5, nonneg=True)

#         # Objective: Total Welfare (Benefit - Cost)
#         benefit = cp.sum(P0_VALS @ d - 0.5 * (P0_VALS / Q0_VALS) @ cp.square(d))
#         cost = (15*g1_n1 + 0.01*g1_n1**2) + (15*g1_n2 + 0.01*g1_n2**2) + (18*g2_n2 + 0.005*g2_n2**2)
        
#         # Physics: Net Injection & Line Flows
#         # Node 1: g1_n1 - d[0]
#         # Node 2: (g1_n2 + g2_n2) - d[1]
#         y = cp.vstack([g1_n1, g1_n2 + g2_n2, 0, 0, 0]) - cp.reshape(d, (5,1))
#         flows = self.ptdf @ y
        
#         # Flatten flows for easier constraint handling
#         flows_flat = cp.reshape(flows, (5,))

#         # --- CONSTRAINTS ---
#         # We split thermal limits into Pos/Neg to get clear Shadow Prices
#         c_bal = (cp.sum(y) == 0)
#         c_therm_max = (flows_flat <= self.line_limits)
#         c_therm_min = (flows_flat >= -self.line_limits)
#         c_cap_g1n1 = (g1_n1 <= 150)
#         c_cap_g1n2 = (g1_n2 <= 50)
#         c_cap_g2n2 = (g2_n2 <= 100)

#         constrs = [c_bal, c_therm_max, c_therm_min, c_cap_g1n1, c_cap_g1n2, c_cap_g2n2]
        
#         # Solve
#         prob = cp.Problem(cp.Maximize(benefit - cost), constrs)
#         prob.solve()
        
#         # --- EXTRACT RESULTS ---
#         lmps = P0_VALS - (P0_VALS / Q0_VALS) * d.value
        
#         print(f"  > Solver Status: {prob.status}")
        
#         print("\n  --- GENERATION QUANTITIES (MW) ---")
#         print(f"  Firm 1 (Node 1): {g1_n1.value:.2f} MW")
#         print(f"  Firm 1 (Node 2): {g1_n2.value:.2f} MW")
#         print(f"  Firm 2 (Node 2): {g2_n2.value:.2f} MW")
#         print(f"  Total Gen:       {(g1_n1.value + g1_n2.value + g2_n2.value):.2f} MW")

#         print("\n  --- NODAL DEMAND (MW) ---")
#         for i in range(5):
#             print(f"  Node {i+1}: {d.value[i]:.2f} MW")

#         print("\n  --- LINE 2-3 DETAILS (Index 1) ---")
#         # Flow on Line 2-3 (Index 1)
#         flow_23 = flows_flat.value[1]
#         limit_23 = self.line_limits[1]
        
#         # Shadow Price calculation: Dual_Max - Dual_Min
#         # Note: Duals might be None if solver failed, but here we assume optimal
#         # c_therm_max is index 1 in constraints list, c_therm_min is index 2
#         lam_max = constrs[1].dual_value[1]
#         lam_min = constrs[2].dual_value[1]
#         shadow_price_23 = lam_max - lam_min

#         print(f"  Flow on Line 2-3: {flow_23:.2f} MW (Limit: {limit_23:.2f})")
#         print(f"  Shadow Price (Congestion Cost): ${shadow_price_23:.2f}/MWh")
        
#         if abs(abs(flow_23) - limit_23) < 1e-4:
#             print("  > STATUS: CONGESTED (Binding)")
#         else:
#             print("  > STATUS: UNCONGESTED")

#         return {'g_f1_n1': g1_n1.value, 'g_f1_n2': g1_n2.value, 'g_f2_n2': g2_n2.value, 'd': d.value, 'price': lmps}


# if __name__ == "__main__":
#     solver = EquilibriumSolver()
#     init_vals = solver.solve_competitive()


import numpy as np
import cvxpy as cp

# --- INPUT DATA (From Liu & Hobbs Table 1) ---
P0_VALS = np.array([40.0, 35.0, 32.0, 30.0, 40.0]) 
Q0_VALS = np.array([250.0, 200.0, 320.0, 300.0, 200.0])

class EquilibriumSolver:
    def __init__(self):
        # UNCONSTRAINED MODEL: No PTDFs or Line Limits needed.
        pass

    # =========================================================================
    # 1. PERFECT COMPETITION
    # =========================================================================
    def solve_competitive(self):
        print("\n--- 1. Solving Perfect Competition (Unconstrained) ---")
        
        # Decision Variables
        g1_n1 = cp.Variable(nonneg=True) 
        g1_n2 = cp.Variable(nonneg=True) 
        g2_n2 = cp.Variable(nonneg=True) 
        d = cp.Variable(5, nonneg=True) # Demand at 5 nodes

        # --- OBJECTIVE: Maximize Social Welfare ---
        # Consumer Benefit: Sum of areas under inverse demand curves
        # Benefit_i = P0_i * d_i - 0.5 * (P0_i / Q0_i) * d_i^2
        benefit = cp.sum(cp.multiply(P0_VALS, d) - 0.5 * cp.multiply(P0_VALS / Q0_VALS, cp.square(d)))
        
        # Total Production Cost
        cost = (15*g1_n1 + 0.01*g1_n1**2) + \
               (15*g1_n2 + 0.01*g1_n2**2) + \
               (18*g2_n2 + 0.005*g2_n2**2)
        
        # --- CONSTRAINTS ---
        # 1. Global Power Balance (Copper Plate): Total Supply == Total Demand
        c_global_balance = (g1_n1 + g1_n2 + g2_n2 == cp.sum(d))
        
        # 2. Generation Capacity Limits
        c_cap_g1n1 = (g1_n1 <= 150)
        c_cap_g1n2 = (g1_n2 <= 50)
        c_cap_g2n2 = (g2_n2 <= 100)

        constrs = [c_global_balance, c_cap_g1n1, c_cap_g1n2, c_cap_g2n2]
        
        # --- SOLVE ---
        prob = cp.Problem(cp.Maximize(benefit - cost), constrs)
        prob.solve()
        
        # --- CALCULATE SYSTEM PRICE ---
        # In unconstrained market, P_node1 = P_node2 = ... = P_system
        # We calculate it using Node 1's inverse demand: P = P0 - (P0/Q0)*d
        # (This mathematically equals the Dual Variable of the global balance constraint)
        p_system = P0_VALS[0] - (P0_VALS[0] / Q0_VALS[0]) * d.value[0]
        
        # --- PRINT RESULTS ---
        print(f"  > Solver Status: {prob.status}")
        
        print("\n  --- MARKET OUTCOMES ---")
        print(f"  System Price:    ${p_system:.2f} / MWh")
        print(f"  Total Gen:       {(g1_n1.value + g1_n2.value + g2_n2.value):.2f} MW")
        print(f"  Total Demand:    {np.sum(d.value):.2f} MW")

        print("\n  --- GENERATION DETAILS ---")
        print(f"  Firm 1 (Node 1): {g1_n1.value:.2f} MW (Max 150)")
        print(f"  Firm 1 (Node 2): {g1_n2.value:.2f} MW (Max 50)")
        print(f"  Firm 2 (Node 2): {g2_n2.value:.2f} MW (Max 100)")

        print("\n  --- NODAL DEMAND ---")
        for i in range(5):
            print(f"  Node {i+1}: {d.value[i]:.2f} MW")

        return {
            'g_f1_n1': g1_n1.value, 
            'g_f1_n2': g1_n2.value, 
            'g_f2_n2': g2_n2.value, 
            'd': d.value, 
            'price': p_system
        }

if __name__ == "__main__":
    solver = EquilibriumSolver()
    results = solver.solve_competitive()