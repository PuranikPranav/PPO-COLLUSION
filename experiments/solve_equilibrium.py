import sys, os
import numpy as np
import cvxpy as cp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iso_market.node_network import P0, Q0, get_ptdf_matrix, MC, QC

# Aliases for clarity within this file
P0_VALS = P0.astype(float)
Q0_VALS = Q0.astype(float)


class EquilibriumSolver:
    def __init__(self):
        self.ptdf = get_ptdf_matrix()
        # Line Indices: 0:(1-2), 1:(2-3), 2:(1-3), 3:(3-4), 4:(4-5)
        self.line_limits = np.array([40.0, 40.0, 40.0, 40.0, 30.0])

    # =========================================================================
    # 1. PERFECT COMPETITION (Constrained DC-OPF)
    # =========================================================================
    def solve_competitive(self):
        print("\n--- 1. Solving Perfect Competition (Constrained DC-OPF) ---")

        g1_n1 = cp.Variable(nonneg=True)
        g1_n2 = cp.Variable(nonneg=True)
        g2_n2 = cp.Variable(nonneg=True)
        d = cp.Variable(5, nonneg=True)

        # Objective: Total Welfare (Benefit - Cost)
        # Benefit_i = P0_i * d_i - 0.5 * (P0_i / Q0_i) * d_i^2
        benefit = cp.sum(cp.multiply(P0_VALS, d) - 0.5 * cp.multiply(P0_VALS / Q0_VALS, cp.square(d)))
        cost = (MC['Firm1_Node1']*g1_n1 + 0.5*QC['Firm1_Node1']*g1_n1**2) + \
               (MC['Firm1_Node2']*g1_n2 + 0.5*QC['Firm1_Node2']*g1_n2**2) + \
               (MC['Firm2_Node2']*g2_n2 + 0.5*QC['Firm2_Node2']*g2_n2**2)

        # Net Injection Vector: Node 1 has g1_n1, Node 2 has g1_n2 + g2_n2, Nodes 3-5 have no generators
        y = cp.hstack([g1_n1 - d[0], g1_n2 + g2_n2 - d[1], -d[2], -d[3], -d[4]])
        flows = self.ptdf @ y

        # --- CONSTRAINTS ---
        # We split thermal limits into pos/neg to get clear signed shadow prices
        c_bal = (cp.sum(y) == 0)
        c_therm_max = (flows <= self.line_limits)
        c_therm_min = (flows >= -self.line_limits)
        c_cap_g1n1 = (g1_n1 <= 150)
        c_cap_g1n2 = (g1_n2 <= 50)
        c_cap_g2n2 = (g2_n2 <= 100)

        constrs = [c_bal, c_therm_max, c_therm_min, c_cap_g1n1, c_cap_g1n2, c_cap_g2n2]

        prob = cp.Problem(cp.Maximize(benefit - cost), constrs)
        prob.solve()

        # LMPs: Derivative of welfare w.r.t. demand at each node
        lmps = P0_VALS - (P0_VALS / Q0_VALS) * d.value

        # Quantity-weighted average price (as defined in Liu & Hobbs footnote 14)
        avg_price = np.sum(lmps * d.value) / np.sum(d.value)

        print(f"  > Solver Status: {prob.status}")
        print(f"  > Avg Price (qty-weighted): ${avg_price:.2f} / MWh  [Paper: ~23.47]")

        print("\n  --- GENERATION QUANTITIES (MW) ---")
        print(f"  Firm 1 (Node 1): {g1_n1.value:.2f} MW")
        print(f"  Firm 1 (Node 2): {g1_n2.value:.2f} MW")
        print(f"  Firm 2 (Node 2): {g2_n2.value:.2f} MW")
        print(f"  Total Gen:       {(g1_n1.value + g1_n2.value + g2_n2.value):.2f} MW")

        print("\n  --- NODAL LMPs ($/MWh) ---")
        for i in range(5):
            print(f"  Node {i+1}: ${lmps[i]:.2f}")

        print("\n  --- NODAL DEMAND (MW) ---")
        for i in range(5):
            print(f"  Node {i+1}: {d.value[i]:.2f} MW")

        print("\n  --- LINE 2-3 DETAILS (Index 1) ---")
        flow_23 = flows.value[1]
        limit_23 = self.line_limits[1]

        # Shadow Price: Dual_Max - Dual_Min (positive = congestion rent from Node 1 to Node 2 direction)
        lam_max = c_therm_max.dual_value[1]
        lam_min = c_therm_min.dual_value[1]
        shadow_price_23 = lam_max - lam_min

        print(f"  Flow on Line 2-3: {flow_23:.2f} MW (Limit: {limit_23:.2f})")
        print(f"  Shadow Price (Congestion Cost): ${shadow_price_23:.2f}/MWh")

        if abs(abs(flow_23) - limit_23) < 1e-4:
            print("  > STATUS: CONGESTED (Binding)")
        else:
            print("  > STATUS: UNCONGESTED")

        return {
            'g_f1_n1': g1_n1.value,
            'g_f1_n2': g1_n2.value,
            'g_f2_n2': g2_n2.value,
            'd': d.value,
            'lmps': lmps,
            'avg_price': avg_price,
            'flows': flows.value,
            'shadow_price_23': shadow_price_23,
        }


class UnconstrainedEquilibriumSolver:
    def __init__(self):
        # Copper-plate model: no PTDFs or line limits
        pass

    # =========================================================================
    # 1. PERFECT COMPETITION (Copper Plate)
    # =========================================================================
    def solve_competitive(self):
        print("\n--- 1. Solving Perfect Competition (Unconstrained) ---")

        g1_n1 = cp.Variable(nonneg=True)
        g1_n2 = cp.Variable(nonneg=True)
        g2_n2 = cp.Variable(nonneg=True)
        d = cp.Variable(5, nonneg=True)

        # Consumer Benefit: Sum of areas under inverse demand curves
        # Benefit_i = P0_i * d_i - 0.5 * (P0_i / Q0_i) * d_i^2
        benefit = cp.sum(cp.multiply(P0_VALS, d) - 0.5 * cp.multiply(P0_VALS / Q0_VALS, cp.square(d)))

        cost = (MC['Firm1_Node1']*g1_n1 + 0.5*QC['Firm1_Node1']*g1_n1**2) + \
               (MC['Firm1_Node2']*g1_n2 + 0.5*QC['Firm1_Node2']*g1_n2**2) + \
               (MC['Firm2_Node2']*g2_n2 + 0.5*QC['Firm2_Node2']*g2_n2**2)

        # Global Power Balance (Copper Plate): Total Supply == Total Demand
        c_global_balance = (g1_n1 + g1_n2 + g2_n2 == cp.sum(d))
        c_cap_g1n1 = (g1_n1 <= 150)
        c_cap_g1n2 = (g1_n2 <= 50)
        c_cap_g2n2 = (g2_n2 <= 100)

        constrs = [c_global_balance, c_cap_g1n1, c_cap_g1n2, c_cap_g2n2]

        prob = cp.Problem(cp.Maximize(benefit - cost), constrs)
        prob.solve()

        # In unconstrained market, P_node1 = P_node2 = ... = P_system
        # (Mathematically equals the dual variable of the global balance constraint)
        p_system = P0_VALS[0] - (P0_VALS[0] / Q0_VALS[0]) * d.value[0]

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
            'price': p_system,
        }


if __name__ == "__main__":
    solver = EquilibriumSolver()
    results = solver.solve_competitive()
