
"""
Multi-agent electricity market environment for PPO collusion study.

Firms choose generation quantities; the ISO clears the market via DC-OPF.
Each firm observes previous LMPs and earns profit (revenue minus cost) as reward.

The market clearing is formulated as a parametric CVXPY problem: the generation
vector is a Parameter, so the problem structure is compiled once and re-solved
efficiently at each step with warm-starting.
"""

import numpy as np
import cvxpy as cp #for DC-OPF convex maxi

from iso_market.node_network import P0, Q0, get_ptdf_matrix, LINE_LIMITS, MC, QC

# ---------------------------------------------------------------------------
# Plant registry — derived from node_network constants
# ---------------------------------------------------------------------------
PLANTS = [
    {"firm": 0, "node": 0, "mc": MC["Firm1_Node1"], "qc": QC["Firm1_Node1"], "cap": 150.0},
    {"firm": 0, "node": 1, "mc": MC["Firm1_Node2"], "qc": QC["Firm1_Node2"], "cap":  50.0},
    {"firm": 1, "node": 1, "mc": MC["Firm2_Node2"], "qc": QC["Firm2_Node2"], "cap": 100.0},
]

NUM_FIRMS = 2
NUM_NODES = 5

FIRM_PLANT_IDX = {0: [0, 1], 1: [2]}


class ElectricityMarketEnv:
    """
    At each step:
      1. Each firm submits generation quantities for its plants.
      2. The ISO clears the market (DC-OPF) → LMPs, demand.
      3. Rewards = firm profit = Σ_plant [LMP_i·g − MC·g − ½·QC·g²].

    Observations are the last ``history_len`` vectors of nodal LMPs, flattened.
    Actions are generation in MW, clipped to [0, CAP] per plant.
    """

    def __init__(self, history_len: int = 1, episode_len: int = 168):
        self.history_len = history_len
        self.episode_len = episode_len

        self.P0 = P0.astype(np.float64)
        self.Q0 = Q0.astype(np.float64)
        self.ptdf = get_ptdf_matrix()
        self.line_limits = LINE_LIMITS.astype(np.float64)

        self.obs_dim = history_len * NUM_NODES
        self.action_dims = {f: len(FIRM_PLANT_IDX[f]) for f in range(NUM_FIRMS)}
        self.action_caps = {
            f: np.array([PLANTS[i]["cap"] for i in FIRM_PLANT_IDX[f]])
            for f in range(NUM_FIRMS)
        }

        self._build_cvxpy_problem()

        # Baseline LMPs (all firms at 50 % capacity) used to seed the history
        baseline_gen = np.zeros(NUM_NODES)
        for p in PLANTS:
            baseline_gen[p["node"]] += p["cap"] * 0.5
        lmps, _ = self._clear_market(baseline_gen)
        self._baseline_lmps = lmps if lmps is not None else self.P0 * 0.6

        self.reset()

    # ------------------------------------------------------------------
    # Parametric CVXPY (compiled once, re-solved with warm start)
    # ------------------------------------------------------------------
    def _build_cvxpy_problem(self):
        self._gen_param = cp.Parameter(NUM_NODES, nonneg=True)
        self._d_var = cp.Variable(NUM_NODES, nonneg=True)
        y = self._gen_param - self._d_var

        benefit = cp.sum(
            cp.multiply(self.P0, self._d_var)
            - 0.5 * cp.multiply(self.P0 / self.Q0, cp.square(self._d_var))
        )
        constraints = [
            cp.sum(y) == 0,
            self.ptdf @ y <= self.line_limits,
            self.ptdf @ y >= -self.line_limits,
        ]
        self._prob = cp.Problem(cp.Maximize(benefit), constraints)

    def _clear_market(self, gen_per_node: np.ndarray):
        self._gen_param.value = gen_per_node
        try:
            self._prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            try:
                self._prob.solve(warm_start=True)
            except Exception:
                return None, None

        if self._prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None

        demand = self._d_var.value
        lmps = self.P0 - (self.P0 / self.Q0) * demand
        return lmps, demand

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.lmp_history = np.tile(self._baseline_lmps, (self.history_len, 1))
        return self._get_obs()

    def _get_obs(self):
        """Return per-firm observations (identical — LMPs are public)."""
        flat = self.lmp_history.flatten().astype(np.float32)
        return {f: flat.copy() for f in range(NUM_FIRMS)}

    def step(self, actions: dict):
        """
        Parameters
        ----------
        actions : dict  {firm_id: np.ndarray of generation MW per plant}

        Returns
        -------
        obs, rewards, done, info
        """
        gen_per_node = np.zeros(NUM_NODES)
        gen_per_plant = {}

        for fid, acts in actions.items():
            for j, pidx in enumerate(FIRM_PLANT_IDX[fid]):
                g = float(np.clip(acts[j], 0.0, PLANTS[pidx]["cap"]))
                gen_per_node[PLANTS[pidx]["node"]] += g
                gen_per_plant[pidx] = g

        lmps, demand = self._clear_market(gen_per_node)

        if lmps is None:
            return (
                self._get_obs(),
                {f: -1e3 for f in range(NUM_FIRMS)},
                True,
                {"error": "infeasible"},
            )

        # Per-firm profit
        rewards = {}
        for fid in range(NUM_FIRMS):
            profit = 0.0
            for pidx in FIRM_PLANT_IDX[fid]:
                p = PLANTS[pidx]
                g = gen_per_plant[pidx]
                profit += lmps[p["node"]] * g - (p["mc"] * g + 0.5 * p["qc"] * g ** 2)
            rewards[fid] = profit

        # Shift history window
        self.lmp_history = np.roll(self.lmp_history, -1, axis=0)
        self.lmp_history[-1] = lmps

        self.t += 1
        done = self.t >= self.episode_len

        # Quantity-weighted average price (Liu & Hobbs fn.14)
        avg_lmp = float(np.sum(lmps * demand) / np.sum(demand)) if np.sum(demand) > 0 else 0.0

        info = {
            "lmps": lmps.copy(),
            "demand": demand.copy(),
            "gen": dict(gen_per_plant),
            "total_gen": sum(gen_per_plant.values()),
            "avg_lmp": avg_lmp,
        }
        return self._get_obs(), rewards, done, info
