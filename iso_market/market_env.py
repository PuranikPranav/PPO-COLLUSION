
"""
Multi-agent electricity market environment for PPO collusion study.

Firms choose generation quantities; the ISO clears the market via DC-OPF.
Each firm observes a public history of LMPs, line flows, and transmission shadow
prices; reward is per-period profit (revenue minus cost).

The market clearing is formulated as a parametric CVXPY problem: the generation
vector is a Parameter, so the problem structure is compiled once and re-solved
efficiently at each step with warm-starting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cvxpy as cp  # for DC-OPF convex maximization

from iso_market.node_network import (
    P0, Q0, get_ptdf_matrix, LINE_LIMITS, MC, QC, CAP, PLANT_SPECS, MARKET,
)

# ---------------------------------------------------------------------------
# Plant registry — built generically from node_network.PLANT_SPECS, so the
# same code runs the three-firm hub market (default) and the two-firm
# companion market (MARKET_CONFIG=two_firm).
# ---------------------------------------------------------------------------
PLANTS = [
    {"firm": f, "node": n, "mc": MC[key], "qc": QC[key], "cap": CAP[key]}
    for f, n, key in PLANT_SPECS
]

NUM_FIRMS = max(p["firm"] for p in PLANTS) + 1
NUM_NODES = 5
NUM_PLANTS = len(PLANTS)

FIRM_PLANT_IDX = {
    f: [i for i, p in enumerate(PLANTS) if p["firm"] == f]
    for f in range(NUM_FIRMS)
}

# Public per-timestep market signals: LMPs (5) + line flows (5) + shadow prices (5).
OBS_MARKET_FEATURES_PER_STEP = NUM_NODES + 10  # 15 when num_lines == 5
# Back-compat alias (older scripts import this name).
OBS_FEATURES_PER_STEP = OBS_MARKET_FEATURES_PER_STEP


class ElectricityMarketEnv:
    """
    At each step:
      1. Each firm submits generation quantities for its plants.
      2. The ISO clears the market (DC-OPF) → LMPs, demand, flows, duals.
      3. Rewards = firm profit = Σ_plant [LMP_i·g − MC·g − ½·QC·g²].

    Observations are the last ``history_len`` vectors of
    [LMPs, line flows, shadow prices], flattened (15 features per step).
    Actions are generation in MW, clipped to [0, CAP] per plant.
    """

    def __init__(
        self,
        history_len: int = 1,
        episode_len: int = 168,
        include_past_gen: bool = True,
        include_prev_reward: bool = True,
        obs_mode: str = "full",
        demand_shock: float = 0.0,
        shock_seed: Optional[int] = None,
        shock_persistence: float = 0.5,
    ):
        self.history_len = history_len
        # Kept for CLI/logging ("weeks"); step() does not terminate on episode_len.
        self.episode_len = episode_len

        # ------------------------------------------------------------------
        # IMPERFECT MONITORING (Green-Porter / Calvano et al. 2021):
        #   demand_shock u > 0 → every period an i.i.d. shock u_t ∈ {−u, +u}
        #   (equally likely) shifts ALL nodal inverse-demand intercepts:
        #   p_i = (P0_i + u_t) − (P0_i/Q0_i)·d_i. The shock realizes AFTER firms
        #   commit output and is NEVER observed — firms see only market signals,
        #   so a low price may be a rival deviation OR an adverse demand shock.
        #   Certainty equivalence (linear demand, symmetric i.i.d. shock) leaves
        #   the Nash / monopoly benchmark outputs and EXPECTED profits at their
        #   mean-demand values, so Δ calibration is unchanged.
        #   obs_mode="price_only" → the state is last period's nodal LMPs ONLY
        #   (paper baseline s_t = p_{t−1}): no flows, no shadow prices, no past
        #   generation, no own reward.
        # ------------------------------------------------------------------
        #   obs_mode="price_own" → LMPs + OWN last total output (the paper's
        #   variant s_i = {q_{i,t−1}, p_{t−1}}): punishment phases can persist as
        #   a self-referential state loop instead of hanging off one price reading.
        #   shock_persistence ρ → two-state Markov demand (stay-probability ρ);
        #   0.5 = i.i.d. (paper baseline). Persistent shocks are natural for
        #   electricity load.
        self.obs_mode = str(obs_mode)
        if self.obs_mode not in ("full", "price_only", "price_own"):
            raise ValueError(
                f"obs_mode must be 'full', 'price_only' or 'price_own', got {obs_mode!r}"
            )
        self.demand_shock = float(demand_shock)
        self.shock_persistence = float(shock_persistence)
        self._shock_rng = np.random.default_rng(shock_seed)
        self._demand_u = 0.0          # current-period realized shock
        self._demand_frozen = None    # None = stochastic; else fixed value

        # State-augmentation switches (Andrew's request):
        #   include_past_gen   → append last period's per-plant generation (public).
        #   include_prev_reward→ append each firm's OWN previous-period profit (private,
        #                        so observations become firm-specific).
        # price_only / price_own modes override both (no rival quantities, no
        # reward in the state — imperfect monitoring).
        if self.obs_mode in ("price_only", "price_own"):
            include_past_gen = False
            include_prev_reward = False
        self.include_past_gen = bool(include_past_gen)
        self.include_prev_reward = bool(include_prev_reward)

        self.P0 = P0.astype(np.float64)
        self.Q0 = Q0.astype(np.float64)
        self.ptdf = get_ptdf_matrix()
        self.line_limits = LINE_LIMITS.astype(np.float64)
        self.num_lines = len(self.line_limits)

        # Public per-step block = market signals (+ optionally past generation).
        if self.obs_mode in ("price_only", "price_own"):
            self.public_features_per_step = NUM_NODES  # LMPs only
        else:
            self.public_features_per_step = OBS_MARKET_FEATURES_PER_STEP + (
                NUM_PLANTS if self.include_past_gen else 0
            )
        # Full per-step features as seen by ONE firm (public block + private tail:
        # own prev reward in full mode, own prev total MW in price_own mode).
        private_tail = 1 if (self.include_prev_reward or self.obs_mode == "price_own") else 0
        self.features_per_step = self.public_features_per_step + private_tail
        self.obs_dim = history_len * self.features_per_step

        self.action_dims = {f: len(FIRM_PLANT_IDX[f]) for f in range(NUM_FIRMS)}
        self.action_caps = {
            f: np.array([PLANTS[i]["cap"] for i in FIRM_PLANT_IDX[f]])
            for f in range(NUM_FIRMS)
        }

        self._build_cvxpy_problem()

        # Seed history with the competitive baseline (LMPs, flows, shadow prices,
        # competitive per-plant generation, and competitive per-firm profit).
        self._baseline_public_vector = self._compute_competitive_baseline()
        # Back-compat: market-only baseline slice (LMPs, flows, shadow prices).
        self._baseline_obs_vector = self._baseline_public_vector[
            :OBS_MARKET_FEATURES_PER_STEP
        ]
        print(
            f"[ElectricityMarketEnv] obs_dim={self.obs_dim} "
            f"(history_len={history_len}, features/step={self.features_per_step}; "
            f"mode={self.obs_mode}, past_gen={self.include_past_gen}, "
            f"prev_reward={self.include_prev_reward}, demand_shock=±${self.demand_shock:.2f}) | "
            f"baseline avg LMP=${float(np.mean(self._baseline_public_vector[:NUM_NODES])):.2f}"
        )

        self.reset()

    @staticmethod
    def _shadow_prices_from_duals(mu_up, mu_lo) -> np.ndarray:
        """Signed congestion rent: binding upper − binding lower (per line)."""
        mu_up = np.asarray(mu_up, dtype=np.float64)
        mu_lo = np.asarray(mu_lo, dtype=np.float64)
        return np.maximum(mu_up, 0.0) - np.maximum(mu_lo, 0.0)

    def _compute_competitive_baseline(self) -> np.ndarray:
        """Welfare-max solve.

        Returns the public per-step vector used to seed history:
            [LMPs(5), Flows(5), ShadowPrices(5)] (+ per-plant gen if include_past_gen).
        Also stores `self._baseline_gens` (per plant) and `self._baseline_firm_reward`
        (per firm) so the competitive starting point can be reproduced in synthetic
        observations (limit-strategy / deviation analysis).
        """
        d = cp.Variable(NUM_NODES, nonneg=True)
        g_vars = [cp.Variable(nonneg=True) for _ in PLANTS]

        gen_per_node = [0.0] * NUM_NODES
        for pidx, plant in enumerate(PLANTS):
            gen_per_node[plant["node"]] = gen_per_node[plant["node"]] + g_vars[pidx]
        y = cp.hstack([gen_per_node[i] - d[i] for i in range(NUM_NODES)])

        benefit = cp.sum(
            cp.multiply(self.P0, d)
            - 0.5 * cp.multiply(self.P0 / self.Q0, cp.square(d))
        )
        cost = sum(
            plant["mc"] * g_vars[pidx]
            + 0.5 * plant["qc"] * cp.square(g_vars[pidx])
            for pidx, plant in enumerate(PLANTS)
        )

        c_flow_up = self.ptdf @ y <= self.line_limits
        c_flow_lo = self.ptdf @ y >= -self.line_limits
        constraints = [cp.sum(y) == 0, c_flow_up, c_flow_lo]
        constraints += [g_vars[pidx] <= plant["cap"] for pidx, plant in enumerate(PLANTS)]

        prob = cp.Problem(cp.Maximize(benefit - cost), constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            prob.solve()

        if prob.status not in ("optimal", "optimal_inaccurate") or d.value is None:
            baseline_lmps = (self.P0 * 0.6).astype(np.float64)
            self._baseline_gens = np.zeros(NUM_PLANTS, dtype=np.float64)
            self._baseline_firm_reward = {f: 0.0 for f in range(NUM_FIRMS)}
            market = np.concatenate(
                [baseline_lmps, np.zeros(self.num_lines), np.zeros(self.num_lines)]
            )
            return self._assemble_public_vector(market, self._baseline_gens)

        lmps = (self.P0 - (self.P0 / self.Q0) * d.value).astype(np.float64)
        flows = (self.ptdf @ y.value).astype(np.float64)
        mu_up = c_flow_up.dual_value
        mu_lo = c_flow_lo.dual_value
        shadow = self._shadow_prices_from_duals(
            mu_up if mu_up is not None else np.zeros(self.num_lines),
            mu_lo if mu_lo is not None else np.zeros(self.num_lines),
        )

        gens = np.array([float(g_vars[p].value) for p in range(NUM_PLANTS)], dtype=np.float64)
        self._baseline_gens = gens
        self._baseline_firm_reward = {}
        for fid in range(NUM_FIRMS):
            prof = 0.0
            for pidx in FIRM_PLANT_IDX[fid]:
                plant = PLANTS[pidx]
                g = gens[pidx]
                prof += lmps[plant["node"]] * g - plant["mc"] * g - 0.5 * plant["qc"] * g ** 2
            self._baseline_firm_reward[fid] = float(prof)

        market = np.concatenate([lmps, flows, shadow]).astype(np.float64)
        return self._assemble_public_vector(market, gens)

    def _assemble_public_vector(self, market_vec: np.ndarray, gens: np.ndarray) -> np.ndarray:
        """Concatenate market signals with per-plant generation (if enabled).

        price_only mode: the public block is the nodal LMPs alone.
        """
        market_vec = np.asarray(market_vec, dtype=np.float64)
        if self.obs_mode in ("price_only", "price_own"):
            return market_vec[:NUM_NODES].copy()
        if self.include_past_gen:
            return np.concatenate([market_vec, np.asarray(gens, dtype=np.float64)])
        return market_vec

    # ------------------------------------------------------------------
    # Demand-shock control (imperfect monitoring)
    # ------------------------------------------------------------------
    def freeze_demand(self, state):
        """Freeze the demand shock for controlled experiments.

        state: "high" (+u), "low" (−u), a numeric shock value, or None to
        restore stochastic i.i.d. draws.
        """
        if state is None:
            self._demand_frozen = None
        elif state == "high":
            self._demand_frozen = +self.demand_shock
        elif state == "low":
            self._demand_frozen = -self.demand_shock
        else:
            self._demand_frozen = float(state)

    def _draw_demand_shock(self):
        if self.demand_shock <= 0.0:
            self._demand_u = 0.0
        elif self._demand_frozen is not None:
            self._demand_u = float(self._demand_frozen)
        else:
            rho = self.shock_persistence
            if self._demand_u != 0.0 and rho != 0.5:
                # Two-state Markov chain: keep the current sign w.p. rho.
                stay = self._shock_rng.random() < rho
                sign = np.sign(self._demand_u) if stay else -np.sign(self._demand_u)
                self._demand_u = float(sign * self.demand_shock)
            else:
                self._demand_u = float(
                    self._shock_rng.choice((-self.demand_shock, self.demand_shock))
                )

    # ------------------------------------------------------------------
    # Parametric CVXPY (compiled once, re-solved with warm start)
    # ------------------------------------------------------------------
    def _build_cvxpy_problem(self):
        self._gen_param = cp.Parameter(NUM_NODES, nonneg=True)
        # Inverse-demand intercept (P0 + demand shock); slope stays P0/Q0.
        self._p0_param = cp.Parameter(NUM_NODES)
        self._p0_param.value = self.P0.copy()
        self._d_var = cp.Variable(NUM_NODES, nonneg=True)
        y = self._gen_param - self._d_var

        benefit = cp.sum(
            cp.multiply(self._p0_param, self._d_var)
            - 0.5 * cp.multiply(self.P0 / self.Q0, cp.square(self._d_var))
        )

        self._c_flow_up = self.ptdf @ y <= self.line_limits
        self._c_flow_lo = self.ptdf @ y >= -self.line_limits

        constraints = [
            cp.sum(y) == 0,
            self._c_flow_up,
            self._c_flow_lo,
        ]
        self._prob = cp.Problem(cp.Maximize(benefit), constraints)

    def _clear_market(self, gen_per_node: np.ndarray):
        """
        Clears at the CURRENT demand intercept (P0 + self._demand_u).

        Returns
        -------
        lmps, demand, flows, shadow_prices
        or (None, None, None, None) if infeasible.
        """
        self._gen_param.value = gen_per_node
        p0_now = self.P0 + self._demand_u
        self._p0_param.value = p0_now
        try:
            self._prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            try:
                self._prob.solve(warm_start=True)
            except Exception:
                return None, None, None, None

        if self._prob.status not in ("optimal", "optimal_inaccurate"):
            return None, None, None, None

        demand = self._d_var.value
        lmps = p0_now - (self.P0 / self.Q0) * demand
        net_inj = self._gen_param.value - demand
        flows = self.ptdf @ net_inj

        mu_up = self._c_flow_up.dual_value
        mu_lo = self._c_flow_lo.dual_value
        shadow = self._shadow_prices_from_duals(
            mu_up if mu_up is not None else np.zeros(self.num_lines),
            mu_lo if mu_lo is not None else np.zeros(self.num_lines),
        )

        return lmps, demand, flows, shadow

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self._demand_u = 0.0
        # Public per-step history seeded with the competitive baseline.
        self.obs_history = np.tile(
            self._baseline_public_vector, (self.history_len, 1)
        )
        # Per-firm OWN-reward history, seeded with competitive per-firm profit.
        self.reward_history = {
            f: np.full(self.history_len, self._baseline_firm_reward[f], dtype=np.float64)
            for f in range(NUM_FIRMS)
        }
        # Per-firm OWN-output history (price_own mode), seeded with baseline totals.
        self.own_gen_history = {
            f: np.full(
                self.history_len,
                float(sum(self._baseline_gens[p] for p in FIRM_PLANT_IDX[f])),
                dtype=np.float64,
            )
            for f in range(NUM_FIRMS)
        }
        return self._get_obs()

    def _get_obs(self):
        """Return per-firm observations.

        Public market signals (+ past generation) are shared; each firm additionally
        sees its OWN previous-period reward, so observations are firm-specific when
        ``include_prev_reward`` is enabled.
        """
        obs = {}
        for f in range(NUM_FIRMS):
            if self.obs_mode == "price_own":
                # [LMPs_step, own_total_MW_step] per history row (paper's
                # imperfect-monitoring variant s_i = {q_{i,t−1}, p_{t−1}}).
                rows = [
                    np.concatenate([self.obs_history[h], [self.own_gen_history[f][h]]])
                    for h in range(self.history_len)
                ]
                obs[f] = np.concatenate(rows).astype(np.float32)
            elif self.include_prev_reward:
                # Interleave per step: [public_step, own_reward_step] for each of the
                # history_len rows, then flatten.
                rows = [
                    np.concatenate([self.obs_history[h], [self.reward_history[f][h]]])
                    for h in range(self.history_len)
                ]
                obs[f] = np.concatenate(rows).astype(np.float32)
            else:
                obs[f] = self.obs_history.flatten().astype(np.float32)
        return obs

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

        # Demand shock realizes AFTER firms commit output (imperfect monitoring).
        self._draw_demand_shock()
        lmps, demand, flows, shadow_prices = self._clear_market(gen_per_node)

        if lmps is None:
            return (
                self._get_obs(),
                {f: -1e3 for f in range(NUM_FIRMS)},
                True,
                {"error": "infeasible"},
            )

        market_vec = np.concatenate([lmps, flows, shadow_prices]).astype(np.float64)
        gens_per_plant_arr = np.array(
            [gen_per_plant.get(p, 0.0) for p in range(NUM_PLANTS)], dtype=np.float64
        )
        obs_vec = self._assemble_public_vector(market_vec, gens_per_plant_arr)

        rewards = {}
        for fid in range(NUM_FIRMS):
            profit = 0.0
            for pidx in FIRM_PLANT_IDX[fid]:
                p = PLANTS[pidx]
                g = gen_per_plant[pidx]
                profit += lmps[p["node"]] * g - (p["mc"] * g + 0.5 * p["qc"] * g ** 2)
            rewards[fid] = profit

        # Roll public history and append the new period's public vector.
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = obs_vec
        # Roll each firm's own-reward and own-output histories.
        for fid in range(NUM_FIRMS):
            self.reward_history[fid] = np.roll(self.reward_history[fid], -1)
            self.reward_history[fid][-1] = rewards[fid]
            own_total = float(sum(gen_per_plant.get(p, 0.0) for p in FIRM_PLANT_IDX[fid]))
            self.own_gen_history[fid] = np.roll(self.own_gen_history[fid], -1)
            self.own_gen_history[fid][-1] = own_total

        self.t += 1
        done = False  # Continuing task: market never closes (episode_len is for logging only)

        avg_lmp = float(np.sum(lmps * demand) / np.sum(demand)) if np.sum(demand) > 0 else 0.0

        info = {
            "lmps": lmps.copy(),
            "demand": demand.copy(),
            "flows": flows.copy(),
            "shadow_prices": shadow_prices.copy(),
            "gen": dict(gen_per_plant),
            "total_gen": sum(gen_per_plant.values()),
            "avg_lmp": avg_lmp,
            "demand_u": float(self._demand_u),
        }
        return self._get_obs(), rewards, done, info
