import cvxpy as cp
import numpy as np

import sys
import os

# Add the parent directory (PPO-COLLUSION) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.node_network import P0, Q0, get_ptdf_matrix, LINE_LIMITS, MC, QC, PLANT_SPECS

class DCOPF:
    def __init__(self):
        """Initialize the ISO engine with the grid physics."""
        self.ptdf = get_ptdf_matrix()
        self.num_nodes = 5

    def solve_market(self, gen_dict):
        """
        Solves the DC-OPF to maximize social welfare given firm generation.
        
        Args:
            gen_dict (dict): Dictionary keyed by the plant cost-keys of the active
                market (see node_network.PLANT_SPECS), e.g. 'Firm1_Node2', ...
        """
        # 1. Variables: ISO decides nodal demand (d) to maximize welfare
        d = cp.Variable(self.num_nodes)

        # 2. Fixed Generation Inputs from the firms (placed per PLANT_SPECS)
        g = np.zeros(self.num_nodes)
        for _f, node, key in PLANT_SPECS:
            g[node] += gen_dict[key]

        # 3. Net Injection Vector (Injection = Generation - Demand)
        y = g - d

        # 4. Objective: Maximize Social Welfare
        # Welfare = Integral of (P0 - (P0/Q0)*d) = P0*d - 0.5*(P0/Q0)*d^2
        welfare = cp.sum(P0 @ d - 0.5 * (P0 / Q0) @ cp.square(d))
        
        # 5. Physical Constraints
        # balance: Sum of all net injections must be zero (KCL)
        balance_constraint = cp.sum(y) == 0 
        
        limits = LINE_LIMITS.astype(float)
        line_flows  = self.ptdf @ y
        flow_limit_con = line_flows <= limits
        flow_limit_min = line_flows >= -limits

        constraints = [balance_constraint, flow_limit_con, flow_limit_min]


        # 6. Solve the Optimization Problem
        prob = cp.Problem(cp.Maximize(welfare), constraints)
        prob.solve()

        # 7. Extract Market Metrics
        # LMPs = Price at each node (Derivative of welfare w.r.t. demand)
        lmps = P0 - (P0 / Q0) * d.value
        
        # Shadow Price: difference of upper and lower bound duals on line 2-3
        lam_max = flow_limit_con.dual_value[1] if flow_limit_con.dual_value is not None else 0.0
        lam_min = flow_limit_min.dual_value[1] if flow_limit_min.dual_value is not None else 0.0
        shadow_price_23 = lam_max - lam_min
        
        # Total System Production Cost
        total_cost = sum(gen_dict[key] * MC[key] for _f, _n, key in PLANT_SPECS)

        return {
            'lmps': lmps,
            'demand': d.value,
            'flows': line_flows.value,
            'shadow_price_23': shadow_price_23,
            'production_cost': total_cost
        }