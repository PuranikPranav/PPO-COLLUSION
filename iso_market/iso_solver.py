import cvxpy as cp
import numpy as np

import sys
import os

# Add the parent directory (PPO-COLLUSION) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.node_network import P0, Q0, get_ptdf_matrix, LINE_LIMITS, MC, QC

class DCOPF:
    def __init__(self):
        """Initialize the ISO engine with the grid physics."""
        self.ptdf = get_ptdf_matrix()
        self.num_nodes = 5

    def solve_market(self, gen_dict):
        """
        Solves the DC-OPF to maximize social welfare given firm generation.
        
        Args:
            gen_dict (dict): Dictionary with keys 'Firm1_Node1', 'Firm1_Node2', 'Firm2_Node2'
        """
        # 1. Variables: ISO decides nodal demand (d) to maximize welfare
        d = cp.Variable(self.num_nodes)
        
        # 2. Fixed Generation Inputs from the firms
        g = np.zeros(self.num_nodes)
        g[0] = gen_dict['Firm1_Node1']
        g[1] = gen_dict['Firm1_Node2'] + gen_dict['Firm2_Node2']
        # Nodes 2, 3, 4 (Indices 2, 3, 4) have no generators in this setup

        # 3. Net Injection Vector (Injection = Generation - Demand)
        y = g - d

        # 4. Objective: Maximize Social Welfare
        # Welfare = Integral of (P0 - (P0/Q0)*d) = P0*d - 0.5*(P0/Q0)*d^2
        welfare = cp.sum(P0 @ d - 0.5 * (P0 / Q0) @ cp.square(d))
        
        # 5. Physical Constraints
        # balance: Sum of all net injections must be zero (KCL)
        balance_constraint = cp.sum(y) == 0 
        
        limits = np.array([40.0, 40.0, 40.0, 40.0, 30.0])
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
        total_cost = (gen_dict['Firm1_Node1'] * MC['Firm1_Node1'] + 
                      gen_dict['Firm1_Node2'] * MC['Firm1_Node2'] + 
                      gen_dict['Firm2_Node2'] * MC['Firm2_Node2'])

        return {
            'lmps': lmps,
            'demand': d.value,
            'flows': line_flows.value,
            'shadow_price_23': shadow_price_23,
            'production_cost': total_cost
        }