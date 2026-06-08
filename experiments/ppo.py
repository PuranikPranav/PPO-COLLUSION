"""
Multi-Agent Independent PPO for studying tacit collusion in electricity markets.

Methodology adapted from Calvano, Calzolari, Denicolò & Pastorello (2021):
  - Combined collusion index Δ = Σ_f(π̄_f − π_f^Nash) / Σ_f(π_f^Mono − π_f^Nash)
  - Early stopping: --convergence-mode delta, kl (policy KL), or none
  - Multiple independent sessions with different seeds, averaged
  - Post-training analysis: limit strategy + impulse response to deviation

Usage
-----
    # Quick single run
    python experiments/ppo.py --history-len 1 --total-timesteps 500000

    # Δ-stability (default mode): |Δ_comb − Δ_comb,prev| < threshold for N consecutive updates
    python experiments/ppo.py --history-len 1 --convergence-mode delta \
        --convergence-patience 100 --delta-convergence-threshold 0.01 --total-timesteps 2000000

    # KL mode (optional): max KL < --kl-threshold for N consecutive PPO updates
    python experiments/ppo.py --history-len 1 --convergence-mode kl \
        --convergence-patience 100 --kl-threshold 0.01 --total-timesteps 2000000

    # Lag-k policy KL: KL(π_{t−k}‖π_t) on rollout obs (logged; kl-mode stop uses it when k>0)
    python experiments/ppo.py --history-len 1 --policy-kl-lag 10 --convergence-patience 0

    # No early stop (run full --total-timesteps) even if patience > 0
    python experiments/ppo.py --convergence-mode none --convergence-patience 100

Paper baseline (Calzolari et al., imperfect monitoring): Q-learning Cournot duopoly, discrete
actions, δ=0.95, α=0.15, β=4e-6, 1000 sessions, convergence = greedy policy unchanged for
100_000 *periods* (their time step). This repo uses independent PPO on a DC-OPF market;
``convergence`` is measured in *PPO update* units (--convergence-patience), not env steps.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from scipy.optimize import root

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import (
    ElectricityMarketEnv,
    NUM_FIRMS,
    NUM_NODES,
    FIRM_PLANT_IDX,
    PLANTS,
)


# ======================================================================
# Observation normalizer (Welford running mean / std)
# ======================================================================
class RunningNormalizer:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[np.newaxis]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict:
        """Serialize Welford running stats for later reproduction."""
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore Welford running stats from `state_dict`."""
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


# ======================================================================
# Rollout buffer (one per agent)
# ======================================================================
class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0

    def store(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        n = self.ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == n - 1 else self.values[t + 1]
            delta = (
                self.rewards[t]
                + gamma * next_value * next_nonterminal
                - self.values[t]
            )
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        return advantages, returns

    def tensors(self, device):
        n = self.ptr
        return (
            torch.from_numpy(self.obs[:n]).to(device),
            torch.from_numpy(self.actions[:n]).to(device),
            torch.from_numpy(self.log_probs[:n]).to(device),
        )

    def clear(self):
        self.ptr = 0


# ======================================================================
# Actor-Critic network (Gaussian policy + sigmoid squashing to [0, cap])
# ======================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def policy(self, obs):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


# ======================================================================
# PPO Agent
# ======================================================================
class PPOAgent:
    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        act_dim: int,
        caps: np.ndarray,
        hidden: int = 64,
        lr: float = 3e-4,
        rollout_len: int = 2048,
        device: str = "cpu",
    ):
        self.id = agent_id
        self.caps = caps.astype(np.float32)
        self.device = device

        self.ac = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = RolloutBuffer(rollout_len, obs_dim, act_dim)

    @torch.no_grad()
    def select_action(self, obs_norm: np.ndarray):
        """Sample stochastic action. Returns (raw, scaled_MW, log_prob, value)."""
        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        dist = self.ac.policy(obs_t)
        raw = dist.sample().squeeze(0)
        log_prob = dist.log_prob(raw).sum().item()
        value = self.ac.value(obs_t).item()
        raw_np = raw.cpu().numpy()
        return raw_np, self._to_mw(raw_np), log_prob, value

    @torch.no_grad()
    def deterministic_action(self, obs_norm: np.ndarray) -> np.ndarray:
        """Greedy actor mean passed through sigmoid, scaled to MW."""
        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        raw = self.ac.actor_mean(obs_t).squeeze(0).cpu().numpy()
        return self._to_mw(raw)

    @torch.no_grad()
    def get_value(self, obs_norm: np.ndarray) -> float:
        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        return self.ac.value(obs_t).item()

    def _to_mw(self, raw: np.ndarray) -> np.ndarray:
        frac = 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
        return frac * self.caps

    def update(self, last_val, gamma, gae_lambda, clip_eps, epochs,
               minibatch_size, ent_coef, vf_coef, max_grad_norm):
        adv, returns = self.buffer.compute_gae(last_val, gamma, gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t, act_t, old_lp_t = self.buffer.tensors(self.device)
        adv_t = torch.from_numpy(adv).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)

        n = obs_t.shape[0]
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch_size):
                mb = idx[start : start + minibatch_size]

                dist = self.ac.policy(obs_t[mb])
                new_lp = dist.log_prob(act_t[mb]).sum(-1)
                entropy = dist.entropy().sum(-1)
                new_val = self.ac.value(obs_t[mb])

                ratio = (new_lp - old_lp_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[mb]

                loss = (
                    -torch.min(surr1, surr2).mean()
                    + vf_coef * (new_val - ret_t[mb]).pow(2).mean()
                    - ent_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()


# ======================================================================
# Benchmarks
# ======================================================================
def compute_competitive_benchmark(env: ElectricityMarketEnv):
    """Perfect competition: maximize social welfare (consumer + producer)."""
    import cvxpy as cp
    from iso_market.node_network import P0, Q0

    P0f, Q0f = P0.astype(float), Q0.astype(float)
    g1n1 = cp.Variable(nonneg=True)
    g1n2 = cp.Variable(nonneg=True)
    g2n2 = cp.Variable(nonneg=True)
    d = cp.Variable(5, nonneg=True)

    benefit = cp.sum(cp.multiply(P0f, d) - 0.5 * cp.multiply(P0f / Q0f, cp.square(d)))
    cost = (
        PLANTS[0]["mc"] * g1n1 + 0.5 * PLANTS[0]["qc"] * g1n1 ** 2
        + PLANTS[1]["mc"] * g1n2 + 0.5 * PLANTS[1]["qc"] * g1n2 ** 2
        + PLANTS[2]["mc"] * g2n2 + 0.5 * PLANTS[2]["qc"] * g2n2 ** 2
    )
    y = cp.hstack([g1n1 - d[0], g1n2 + g2n2 - d[1], -d[2], -d[3], -d[4]])
    constraints = [
        cp.sum(y) == 0,
        env.ptdf @ y <= env.line_limits,
        env.ptdf @ y >= -env.line_limits,
        g1n1 <= 150, g1n2 <= 50, g2n2 <= 100,
    ]
    prob = cp.Problem(cp.Maximize(benefit - cost), constraints)
    prob.solve()

    lmps = P0f - (P0f / Q0f) * d.value
    avg_lmp = float(np.sum(lmps * d.value) / np.sum(d.value))
    g_vals = [g1n1.value, g1n2.value, g2n2.value]

    profits = {}
    for fid in range(NUM_FIRMS):
        p = 0.0
        for pidx in FIRM_PLANT_IDX[fid]:
            plant = PLANTS[pidx]
            g = g_vals[pidx]
            p += lmps[plant["node"]] * g - plant["mc"] * g - 0.5 * plant["qc"] * g ** 2
        profits[fid] = p

    return {
        "avg_lmp": avg_lmp,
        "lmps": lmps.tolist(),
        "gens": [float(g) for g in g_vals],
        "total_gen": float(sum(g_vals)),
        "total_profit": float(sum(profits.values())),
        "profits": {str(k): float(v) for k, v in profits.items()},
    }


def compute_monopoly_benchmark(env: ElectricityMarketEnv):
    """Joint monopoly via CVXPY: maximize total revenue minus total cost.

    Mirrors `compute_competitive_benchmark` exactly except the objective:
      - revenue at node i is the raw payment P_i * d_i (no 0.5 factor),
        where P_i = P0_i - (P0_i / Q0_i) d_i, so revenue is concave in d.
      - cost (quadratic, convex) is identical to the competitive case.
    Nodal prices are recovered from the inverse demand curve at the optimum,
    not from the balance-constraint dual (which is the system marginal price
    for the welfare problem, not the monopolist's price).
    """
    import cvxpy as cp
    from iso_market.node_network import P0, Q0

    P0f, Q0f = P0.astype(float), Q0.astype(float)
    g1n1 = cp.Variable(nonneg=True)
    g1n2 = cp.Variable(nonneg=True)
    g2n2 = cp.Variable(nonneg=True)
    d = cp.Variable(5, nonneg=True)

    revenue = cp.sum(cp.multiply(P0f, d) - cp.multiply(P0f / Q0f, cp.square(d)))
    cost = (
        PLANTS[0]["mc"] * g1n1 + 0.5 * PLANTS[0]["qc"] * g1n1 ** 2
        + PLANTS[1]["mc"] * g1n2 + 0.5 * PLANTS[1]["qc"] * g1n2 ** 2
        + PLANTS[2]["mc"] * g2n2 + 0.5 * PLANTS[2]["qc"] * g2n2 ** 2
    )
    y = cp.hstack([g1n1 - d[0], g1n2 + g2n2 - d[1], -d[2], -d[3], -d[4]])
    constraints = [
        cp.sum(y) == 0,
        env.ptdf @ y <= env.line_limits,
        env.ptdf @ y >= -env.line_limits,
        g1n1 <= 150, g1n2 <= 50, g2n2 <= 100,
    ]
    prob = cp.Problem(cp.Maximize(revenue - cost), constraints)
    prob.solve()

    lmps = P0f - (P0f / Q0f) * d.value
    avg_lmp = float(np.sum(lmps * d.value) / np.sum(d.value))
    g_vals = [g1n1.value, g1n2.value, g2n2.value]

    profits = {}
    for fid in range(NUM_FIRMS):
        p = 0.0
        for pidx in FIRM_PLANT_IDX[fid]:
            plant = PLANTS[pidx]
            g = g_vals[pidx]
            p += lmps[plant["node"]] * g - plant["mc"] * g - 0.5 * plant["qc"] * g ** 2
        profits[fid] = p

    return {
        "avg_lmp": avg_lmp,
        "lmps": lmps.tolist(),
        "gens": [float(g) for g in g_vals],
        "total_gen": float(sum(g_vals)),
        "total_profit": float(sum(profits.values())),
        "profits": {str(k): float(v) for k, v in profits.items()},
    }


def compute_cournot_nash_benchmark(env: ElectricityMarketEnv):
    """
    Nash–Cournot equilibrium: stacked KKT MCP (same as model.mod / NEOS PATH).

    Firm stationarity + capacity complementarity; ISO dispatch equalities;
    transmission complementarity. Matches AMPL formulation in repo root.
    """
    from iso_market.node_network import P0, Q0

    p0 = P0.astype(float)
    beta = p0 / Q0.astype(float)
    ptdf = env.ptdf
    line_limits = env.line_limits
    plant_node = np.array([PLANTS[i]["node"] for i in range(len(PLANTS))])
    plant_mc = np.array([PLANTS[i]["mc"] for i in range(len(PLANTS))])
    plant_qc = np.array([PLANTS[i]["qc"] for i in range(len(PLANTS))])
    plant_cap = np.array([PLANTS[i]["cap"] for i in range(len(PLANTS))])
    n_plants = len(PLANTS)

    def gen_per_node(g):
        gn = np.zeros(NUM_NODES)
        for p in range(n_plants):
            gn[plant_node[p]] += g[p]
        return gn

    def nodal_prices(g, y):
        return p0 - beta * (gen_per_node(g) + y)

    def fb(a, b):
        return a + b - np.sqrt(a * a + b * b + 1e-18)

    def unpack(z):
        return z[0:3], z[3:6], z[6:11], z[11], z[12:17], z[17:22]

    def mcp_residual(z):
        g, rho, y, mu, lp, lm = unpack(z)
        p = nodal_prices(g, y)
        pp = p[plant_node]
        f_stat = -(pp - beta[plant_node] * g) + (plant_mc + plant_qc * g) + rho
        f_cap = plant_cap - g
        f_disp = p - mu - ptdf.T @ (lm - lp)
        flow = ptdf @ y
        return np.concatenate([
            fb(g, f_stat),
            fb(rho, f_cap),
            f_disp,
            np.array([np.sum(y)]),
            fb(lp, line_limits + flow),
            fb(lm, line_limits - flow),
        ])

    best_z, best_res = None, np.inf
    base = np.concatenate([
        np.array([100.0, 40.0, 40.0]),
        np.zeros(3),
        np.zeros(5),
        np.array([28.0]),
        np.zeros(5),
        np.zeros(5),
    ])
    rng = np.random.default_rng(0)
    for attempt in range(40):
        z0 = base.copy()
        if attempt:
            z0[0:3] = rng.uniform(0, plant_cap)
            z0[6:11] = rng.uniform(-20, 20, 5)
            z0[11] = rng.uniform(15, 35)
        sol = root(mcp_residual, z0, method="hybr", tol=1e-12)
        res = float(np.max(np.abs(mcp_residual(sol.x))))
        g = sol.x[0:3]
        if np.all(g >= -1e-6) and np.all(g <= plant_cap + 1e-6) and res < best_res:
            best_res, best_z = res, sol.x.copy()
        if best_res < 1e-9:
            break

    if best_z is None:
        raise RuntimeError("Cournot–Nash MCP failed to converge.")

    g, _, y, _, _, _ = unpack(best_z)
    g = np.clip(g, 0.0, plant_cap)
    lmps = nodal_prices(g, y)
    demand = gen_per_node(g) + y
    avg_lmp = float(np.sum(lmps * demand) / np.sum(demand))
    g_vals = [float(v) for v in g]

    profits = {}
    for fid in range(NUM_FIRMS):
        p = 0.0
        for pidx in FIRM_PLANT_IDX[fid]:
            plant = PLANTS[pidx]
            gv = g_vals[pidx]
            p += (
                lmps[plant["node"]] * gv
                - plant["mc"] * gv
                - 0.5 * plant["qc"] * gv ** 2
            )
        profits[fid] = p

    return {
        "avg_lmp": avg_lmp,
        "lmps": lmps.tolist(),
        "gens": g_vals,
        "total_gen": float(sum(g_vals)),
        "total_profit": float(sum(profits.values())),
        "profits": {str(k): float(v) for k, v in profits.items()},
        "mcp_max_residual": best_res,
    }


def _benchmark_profits_by_firm(benchmarks: dict, key: str) -> dict:
    """Map firm id → profit ($/step) from a benchmark dict."""
    return {
        fid: float(benchmarks[key]["profits"][str(fid)])
        for fid in range(NUM_FIRMS)
    }


def compute_combined_delta(
    avg_step_profits: dict,
    pi_nash: dict,
    pi_mono: dict,
) -> float:
    """
    Market-wide collusion index:
      Δ_comb = Σ_f (π̄_f − π_f^Nash) / Σ_f (π_f^Mono − π_f^Nash).
    0 = static Nash; 1 = joint monopoly (on aggregate profit scale).
    """
    num = sum(avg_step_profits[fid] - pi_nash[fid] for fid in range(NUM_FIRMS))
    denom = sum(pi_mono[fid] - pi_nash[fid] for fid in range(NUM_FIRMS))
    if abs(denom) < 1e-8:
        return 0.0
    return float(num / denom)


def compute_greedy_metrics_from_obs(env, agents, obs_by_firm: dict, pi_nash, pi_mono):
    """
    Mean deterministic-policy MW per firm over stored rollout observations (post-update
    actor mean on each row), and Δ from one DC-OPF clear at mean plant outputs.
    obs_by_firm[fid] is (T, obs_dim) numpy — typically a copy of the buffer before .clear().
    Actions in the buffer are pre-squash Gaussian samples (log_prob under Normal).
    """
    if not obs_by_firm:
        return None
    n = next(iter(obs_by_firm.values())).shape[0]
    if n == 0:
        return None

    gen_per_plant = np.zeros(len(PLANTS), dtype=np.float64)
    greedy_totals = {}
    for fid, agent in agents.items():
        obs_np = obs_by_firm[fid]
        obs = torch.from_numpy(obs_np.astype(np.float32)).to(agent.device)
        with torch.no_grad():
            raw = agent.ac.actor_mean(obs).cpu().numpy()
        frac = 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
        mw = frac * agent.caps
        greedy_totals[fid] = float(np.mean(mw.sum(axis=1)))
        mean_mw = np.mean(mw, axis=0)
        for j, pidx in enumerate(FIRM_PLANT_IDX[fid]):
            gen_per_plant[pidx] = mean_mw[j]

    gen_per_node = np.zeros(NUM_NODES, dtype=np.float64)
    for pidx, plant in enumerate(PLANTS):
        gen_per_node[plant["node"]] += gen_per_plant[pidx]

    lmps, _demand, _flows, _shadow = env._clear_market(gen_per_node)
    if lmps is None:
        return {
            "greedy_totals": greedy_totals,
            "greedy_delta_combined": float("nan"),
        }

    profits = {}
    for fid in range(NUM_FIRMS):
        p = 0.0
        for pidx in FIRM_PLANT_IDX[fid]:
            plant = PLANTS[pidx]
            g = gen_per_plant[pidx]
            p += lmps[plant["node"]] * g - plant["mc"] * g - 0.5 * plant["qc"] * g * g
        profits[fid] = p

    return {
        "greedy_totals": greedy_totals,
        "greedy_delta_combined": float(
            compute_combined_delta(profits, pi_nash, pi_mono)
        ),
    }


def kl_checkpoint_vs_current_policy(agent, obs_buf: torch.Tensor, old_state_dict_cpu: dict) -> float:
    """
    KL( π_checkpoint || π_current ) on obs_buf: checkpoint is old_state_dict (CPU tensors),
    current is agent's live weights. Restores agent after.
    """
    device = agent.device
    current_sd = {k: v.clone() for k, v in agent.ac.state_dict().items()}
    old_sd_dev = {
        k: (v.to(device, dtype=torch.float32) if torch.is_tensor(v) else v)
        for k, v in old_state_dict_cpu.items()
    }
    try:
        agent.ac.load_state_dict(old_sd_dev)
        with torch.no_grad():
            old_dist = agent.ac.policy(obs_buf)
            old_mean = old_dist.loc.clone()
            old_std = old_dist.scale.clone()
    finally:
        agent.ac.load_state_dict(current_sd)

    with torch.no_grad():
        new_dist = agent.ac.policy(obs_buf)
    old_fixed = Normal(old_mean, old_std)
    kl = torch.distributions.kl_divergence(old_fixed, new_dist)
    return float(kl.sum(dim=-1).mean().item())


# ======================================================================
# Post-training analysis
# ======================================================================
def _obs_vector_with_scaled_lmps(env, target_avg_lmp: float) -> np.ndarray:
    """Build one per-step obs vector; scale LMPs, keep competitive flows/shadows."""
    base = env._baseline_obs_vector.astype(np.float64)
    comp_lmps = base[:NUM_NODES]
    tail = base[NUM_NODES:]
    comp_avg = float(np.mean(comp_lmps))
    if comp_avg > 0:
        scaled_lmps = comp_lmps * (target_avg_lmp / comp_avg)
    else:
        scaled_lmps = np.full(NUM_NODES, target_avg_lmp, dtype=np.float64)
    return np.concatenate([scaled_lmps, tail])


def build_reference_obs(env, benchmarks, num_points=20):
    """Grid of observations spanning plausible average LMP (limit-strategy sweep)."""
    comp_avg = float(np.mean(env._baseline_obs_vector[:NUM_NODES]))
    targets = np.linspace(15, 38, num_points)
    ref_obs = []
    for target in targets:
        obs_vec = _obs_vector_with_scaled_lmps(env, target)
        ref_obs.append(np.tile(obs_vec, env.history_len).astype(np.float32))
    return np.array(ref_obs)


def evaluate_deterministic(agents, obs_normalizers, ref_obs):
    """Evaluate each agent's deterministic (greedy) output on reference states."""
    result = {}
    for fid, agent in agents.items():
        normed = np.array([obs_normalizers[fid].normalize(o) for o in ref_obs])
        obs_t = torch.from_numpy(normed).to(agent.device)
        with torch.no_grad():
            raw = agent.ac.actor_mean(obs_t).cpu().numpy()
        frac = 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
        result[fid] = frac * agent.caps
    return result


def compute_limit_strategy(agents, obs_normalizers, env, benchmarks, num_points=50):
    """Evaluate the converged deterministic policy across a range of LMP levels."""
    lmp_grid = np.linspace(15, 38, num_points)

    strategies = {str(fid): [] for fid in range(NUM_FIRMS)}
    for target in lmp_grid:
        obs = np.tile(
            _obs_vector_with_scaled_lmps(env, target), env.history_len
        ).astype(np.float32)
        for fid, agent in agents.items():
            obs_norm = obs_normalizers[fid].normalize(obs)
            gen_mw = agent.deterministic_action(obs_norm)
            strategies[str(fid)].append(float(np.sum(gen_mw)))

    return {"lmp_grid": lmp_grid.tolist(), "strategies": strategies}


def run_deviation_experiment(env, agents, obs_normalizers,
                             deviation_frac=0.2, warmup=20, horizon=20):
    """
    After convergence, force one firm to increase output by deviation_frac
    for one step and observe the response — analogous to Calvano Fig 4.
    """
    results = {}

    for deviating_fid in range(NUM_FIRMS):
        obs = env.reset()

        # Warm up with deterministic policies to reach the "resting point"
        for _ in range(warmup):
            actions = {}
            for fid, agent in agents.items():
                obs_norm = obs_normalizers[fid].normalize(obs[fid])
                actions[fid] = agent.deterministic_action(obs_norm)
            obs, _, _done, _ = env.step(actions)

        # Record resting generation
        resting = {}
        for fid, agent in agents.items():
            obs_norm = obs_normalizers[fid].normalize(obs[fid])
            resting[fid] = agent.deterministic_action(obs_norm)

        trace_gen = {str(fid): [] for fid in range(NUM_FIRMS)}
        trace_lmp = []

        # --- Deviation step ---
        actions = {}
        for fid, agent in agents.items():
            obs_norm = obs_normalizers[fid].normalize(obs[fid])
            actions[fid] = agent.deterministic_action(obs_norm)
        deviated = actions[deviating_fid] * (1 + deviation_frac)
        for j, pidx in enumerate(FIRM_PLANT_IDX[deviating_fid]):
            deviated[j] = min(deviated[j], PLANTS[pidx]["cap"])
        actions[deviating_fid] = deviated

        obs, _, done, info = env.step(actions)
        if done and info.get("error"):
            obs = env.reset()
        for fid in range(NUM_FIRMS):
            trace_gen[str(fid)].append(float(np.sum(actions[fid])))
        trace_lmp.append(info.get("avg_lmp", 0))

        # --- Post-deviation: both play deterministic ---
        for _ in range(horizon):
            actions = {}
            for fid, agent in agents.items():
                obs_norm = obs_normalizers[fid].normalize(obs[fid])
                actions[fid] = agent.deterministic_action(obs_norm)
            obs, _, done, info = env.step(actions)
            for fid in range(NUM_FIRMS):
                trace_gen[str(fid)].append(float(np.sum(actions[fid])))
            trace_lmp.append(info.get("avg_lmp", 0))

        results[str(deviating_fid)] = {
            "resting": {str(fid): float(np.sum(resting[fid])) for fid in range(NUM_FIRMS)},
            "gen": trace_gen,
            "lmp": trace_lmp,
        }

    return results


def _print_paper_vs_ppo_banner():
    print(
        "\n=== Paper vs this simulation (do not expect 1:1 parameter match) ===\n"
        "  Paper: Q-learning, Cournot, discrete outputs, two demand states, γ=0.95, "
        "100k consecutive periods with unchanged greedy policy.\n"
        "  Here:  Independent PPO, continuous generation, DC-OPF/LMP, γ from --gamma, "
        "early stop = N consecutive PPO updates (--convergence-patience) under "
        "--convergence-mode delta|kl (or none).\n"
        "  Δ metric: Σ(π−π^Nash)/Σ(π^Mono−π^Nash) combined — Nash floor, monopoly ceiling.\n"
        + "=" * 66
    )


def _print_session_convergence_banner(args, session_id: int):
    use = args.convergence_patience > 0 and args.convergence_mode in ("delta", "kl")
    print(
        "\n--- Convergence / logging ---\n"
        f"  convergence_mode={args.convergence_mode}\n"
        f"  early_stop_active={'yes' if use else 'no'} "
        f"(patience={args.convergence_patience} PPO updates; ignored when mode=none)\n"
        f"  delta_convergence_threshold={args.delta_convergence_threshold}\n"
        f"  kl_threshold={args.kl_threshold}  policy_kl_lag={args.policy_kl_lag}\n"
        "  streak counts consecutive PPO updates satisfying the *active* mode only "
        "(delta: |Δ_comb−Δ_comb,prev|; kl: max_f KL per --policy-kl-lag). "
        "Printed KL is for monitoring in all modes.\n"
        "---\n"
    )


def _print_progress_line(
    args,
    session_id: int,
    num_sessions: int,
    total_steps: int,
    episode_count: int,
    update_idx: int,
    num_updates: int,
    row: dict,
    stable_count: int,
    delta_max_jump: float,
    kl_for_convergence: float,
):
    d_comb = row.get("delta_combined", float("nan"))
    mk = row.get("max_kl", 0)
    mkl = row.get("max_kl_lag")
    sess_prefix = (
        f"S{session_id + 1}/{num_sessions} "
        if num_sessions > 1
        else ""
    )
    kl_extra = ""
    if args.policy_kl_lag > 0 and mkl is not None and math.isfinite(mkl):
        kl_extra = f" KL_lag{args.policy_kl_lag}={mkl:.2e}"

    use_conv = args.convergence_patience > 0 and args.convergence_mode in ("delta", "kl")
    if args.log_format == "legacy":
        print(
            f"{sess_prefix}[{total_steps:>8d}] ep {episode_count:>4d} | "
            f"LMP ${row['avg_lmp']:.2f} | "
            f"Δ_comb={d_comb:.3f} | "
            f"g0={row['firm_0_avg_gen']:.1f} g1={row['firm_1_avg_gen']:.1f} | "
            f"KL={mk:.2e}{kl_extra} streak={stable_count}"
            + (f"  [streak=active:{args.convergence_mode}]" if use_conv else "")
        )
        return

    if use_conv:
        sm = args.convergence_mode
    elif args.convergence_mode == "none":
        sm = "none"
    else:
        sm = "inactive"  # delta/kl but patience=0
    dj = delta_max_jump if math.isfinite(delta_max_jump) else float("nan")
    kc = kl_for_convergence if math.isfinite(kl_for_convergence) else float("nan")
    parts = [
        f"{sess_prefix}PPO",
        f"conv={args.convergence_mode}",
        f"early_stop={'on' if use_conv else 'off'}",
        f"d_thr={args.delta_convergence_threshold}",
        f"kl_thr={args.kl_threshold}",
        f"step={total_steps}",
        f"upd={update_idx + 1}/{num_updates}",
        f"ep={episode_count}",
        f"LMP={row['avg_lmp']:.2f}",
        f"Δ_comb={d_comb:.3f}" if math.isfinite(d_comb) else "Δ_comb=NA",
        f"g0={row['firm_0_avg_gen']:.1f}",
        f"g1={row['firm_1_avg_gen']:.1f}",
        f"KL_intra_max={mk:.2e}",
    ]
    if args.policy_kl_lag > 0 and mkl is not None and math.isfinite(mkl):
        parts.append(f"KL_lag_k={mkl:.2e}")
    parts.append(f"d_jump={dj:.4g}" if math.isfinite(dj) else "d_jump=NA")
    parts.append(f"kl_for_conv={kc:.4g}" if math.isfinite(kc) else "kl_for_conv=NA")
    parts.append(f"streak_metric={sm}")
    if use_conv:
        parts.append(f"streak={stable_count}/{args.convergence_patience}")
    else:
        parts.append("streak=n/a")
    print(" ".join(parts))


# ======================================================================
# Single-session training
# ======================================================================
def train_session(env, benchmarks, args, session_id, device):
    seed = args.seed + session_id
    np.random.seed(seed)
    torch.manual_seed(seed)

    pi_nash = _benchmark_profits_by_firm(benchmarks, "cournot_nash")
    pi_mono = _benchmark_profits_by_firm(benchmarks, "monopoly")

    # Fresh agents
    agents = {}
    obs_normalizers = {}
    for fid in range(NUM_FIRMS):
        caps = np.array([PLANTS[i]["cap"] for i in FIRM_PLANT_IDX[fid]])
        agents[fid] = PPOAgent(
            agent_id=fid, obs_dim=env.obs_dim, act_dim=env.action_dims[fid],
            caps=caps, hidden=args.hidden_dim, lr=args.lr,
            rollout_len=args.rollout_len, device=device,
        )
        obs_normalizers[fid] = RunningNormalizer(env.obs_dim)

    # Early stopping: Δ-stability, policy KL, or none (--convergence-mode)
    use_convergence = args.convergence_patience > 0 and args.convergence_mode in (
        "delta",
        "kl",
    )
    conv_delta = args.convergence_mode == "delta"
    conv_kl = args.convergence_mode == "kl"
    stable_count = 0
    last_delta_max_jump = float("nan")
    last_kl_for_convergence = float("nan")
    last_agent_kls = {fid: 0.0 for fid in range(NUM_FIRMS)}
    last_agent_kls_lag = {fid: float("nan") for fid in range(NUM_FIRMS)}
    policy_ckpt_hist = {fid: [] for fid in agents}
    prev_delta_combined = None

    # Logging
    log_rows = []
    obs = env.reset()
    total_steps = 0
    episode_count = 0
    converged = False

    smoothing_steps = 200 * args.episode_len if args.episode_len > 0 else 33600
    recent_lmps = deque(maxlen=2000)
    recent_step_profits = {f: deque(maxlen=smoothing_steps) for f in range(NUM_FIRMS)}
    recent_gens = {f: deque(maxlen=2000) for f in range(NUM_FIRMS)}

    num_updates = args.total_timesteps // args.rollout_len
    wall_start = time.time()

    _print_session_convergence_banner(args, session_id)

    for update in range(num_updates):
        # ---------- collect rollout ----------
        for _ in range(args.rollout_len):
            actions_mw = {}
            pending = {}

            for fid, agent in agents.items():
                obs_normalizers[fid].update(obs[fid])
                obs_norm = obs_normalizers[fid].normalize(obs[fid])
                raw, scaled, lp, val = agent.select_action(obs_norm)
                actions_mw[fid] = scaled
                pending[fid] = (obs_norm.copy(), raw, lp, val)

            obs_next, rewards, done, info = env.step(actions_mw)

            for fid, agent in agents.items():
                o, a, lp, v = pending[fid]
                agent.buffer.store(o, a, lp, rewards[fid], v, done)
                recent_step_profits[fid].append(rewards[fid])

            if "lmps" in info:
                recent_lmps.append(info["avg_lmp"])
                for fid in range(NUM_FIRMS):
                    gen_total = sum(
                        info["gen"].get(pidx, 0) for pidx in FIRM_PLANT_IDX[fid]
                    )
                    recent_gens[fid].append(gen_total)

            obs = obs_next
            total_steps += 1

            if args.episode_len > 0 and total_steps % args.episode_len == 0:
                episode_count += 1

            if done:
                obs = env.reset()

        # ---------- snapshot policies before update (KL metrics + optional convergence) ----------
        old_policy_snapshots = {}
        for fid, agent in agents.items():
            with torch.no_grad():
                obs_buf = torch.from_numpy(
                    agent.buffer.obs[:agent.buffer.ptr]
                ).to(agent.device)
                old_dist = agent.ac.policy(obs_buf)
                old_policy_snapshots[fid] = {
                    "obs": obs_buf,
                    "mean": old_dist.loc.clone(),
                    "std": old_dist.scale.clone(),
                }

        # Rollout obs for greedy metrics (post-update policy mean on same states)
        rollout_obs_backup = {
            fid: agent.buffer.obs[: agent.buffer.ptr].copy()
            for fid, agent in agents.items()
        }

        # ---------- PPO update ----------
        for fid, agent in agents.items():
            obs_norm = obs_normalizers[fid].normalize(obs[fid])
            last_val = agent.get_value(obs_norm)
            agent.update(
                last_val=last_val, gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip_eps=args.clip_eps, epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size, ent_coef=args.ent_coef,
                vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
            )

        # ---------- KL vs previous policy (always, for logs; convergence uses same values) ----------
        for fid in agents:
            with torch.no_grad():
                snap = old_policy_snapshots[fid]
                new_dist = agents[fid].ac.policy(snap["obs"])
                old_dist = Normal(snap["mean"], snap["std"])
                kl = torch.distributions.kl_divergence(old_dist, new_dist)
                last_agent_kls[fid] = kl.sum(dim=-1).mean().item()

        # Lag-k KL: π at end of update (t−k) vs π now (t) on this rollout's observations
        if args.policy_kl_lag > 0:
            for fid, agent in agents.items():
                obs_buf = old_policy_snapshots[fid]["obs"]
                hist = policy_ckpt_hist[fid]
                sd_cpu = {
                    k: v.detach().cpu().clone()
                    for k, v in agent.ac.state_dict().items()
                }
                hist.append(sd_cpu)
                if len(hist) > args.policy_kl_lag + 1:
                    hist.pop(0)
                if len(hist) > args.policy_kl_lag:
                    last_agent_kls_lag[fid] = kl_checkpoint_vs_current_policy(
                        agent, obs_buf, hist[0]
                    )
                else:
                    last_agent_kls_lag[fid] = float("nan")
        else:
            for fid in agents:
                last_agent_kls_lag[fid] = float("nan")

        if use_convergence and conv_kl:
            if args.policy_kl_lag > 0:
                lag_vals = [
                    last_agent_kls_lag[f]
                    for f in agents
                    if math.isfinite(last_agent_kls_lag[f])
                ]
                if len(lag_vals) == len(agents):
                    max_kl = max(lag_vals)
                else:
                    max_kl = None
            else:
                max_kl = max(last_agent_kls.values())

            if max_kl is not None:
                last_kl_for_convergence = float(max_kl)
                if max_kl < args.kl_threshold:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= args.convergence_patience:
                    converged = True
            else:
                last_kl_for_convergence = float("nan")
        else:
            last_kl_for_convergence = float("nan")

        # Combined Δ and jump (logged every mode; streak only in delta mode)
        avg_step_profits = {}
        for fid in range(NUM_FIRMS):
            avg_step_profits[fid] = (
                float(np.mean(recent_step_profits[fid]))
                if recent_step_profits[fid]
                else 0.0
            )
        delta_combined_now = compute_combined_delta(
            avg_step_profits, pi_nash, pi_mono
        )

        if prev_delta_combined is not None:
            last_delta_max_jump = abs(delta_combined_now - prev_delta_combined)
        else:
            last_delta_max_jump = float("nan")

        if use_convergence and conv_delta:
            if prev_delta_combined is not None:
                if last_delta_max_jump < args.delta_convergence_threshold:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= args.convergence_patience:
                    converged = True

        prev_delta_combined = delta_combined_now

        # ---------- logging ----------
        if (update + 1) % args.log_interval == 0:
            avg_lmp = np.mean(recent_lmps) if recent_lmps else 0
            gr = compute_greedy_metrics_from_obs(
                env, agents, rollout_obs_backup, pi_nash, pi_mono
            )
            row = {
                "step": total_steps,
                "episodes": episode_count,
                "avg_lmp": float(avg_lmp),
                "wall_sec": time.time() - wall_start,
                "delta_combined": float(delta_combined_now),
            }
            for fid in range(NUM_FIRMS):
                avg_step_prof = (
                    float(np.mean(recent_step_profits[fid]))
                    if recent_step_profits[fid]
                    else 0.0
                )
                avg_gen = float(np.mean(recent_gens[fid])) if recent_gens[fid] else 0.0
                mock_ep_prof = avg_step_prof * args.episode_len
                row[f"firm_{fid}_ep_profit"] = mock_ep_prof
                row[f"firm_{fid}_avg_step_profit"] = avg_step_prof
                row[f"firm_{fid}_avg_gen"] = avg_gen
                row[f"firm_{fid}_kl"] = last_agent_kls.get(fid, 0)
                if args.policy_kl_lag > 0:
                    row[f"firm_{fid}_kl_lag"] = last_agent_kls_lag.get(fid, float("nan"))
                if gr is not None:
                    row[f"firm_{fid}_greedy_gen"] = gr["greedy_totals"][fid]
            if gr is not None:
                row["greedy_delta_combined"] = gr["greedy_delta_combined"]
            row["max_kl"] = max(last_agent_kls.values())
            if args.policy_kl_lag > 0:
                fin_lag = [
                    last_agent_kls_lag[f]
                    for f in agents
                    if math.isfinite(last_agent_kls_lag.get(f, float("nan")))
                ]
                row["max_kl_lag"] = max(fin_lag) if fin_lag else float("nan")
            row["kl_streak"] = stable_count
            row["convergence_streak"] = stable_count
            row["delta_combined_jump"] = float(last_delta_max_jump)
            row["delta_max_jump"] = float(last_delta_max_jump)
            row["kl_for_convergence"] = float(last_kl_for_convergence)
            row["ppo_update"] = update + 1
            row["ppo_updates_total"] = num_updates
            row["convergence_mode"] = args.convergence_mode
            row["early_stop_active"] = use_convergence
            log_rows.append(row)

            _print_progress_line(
                args,
                session_id,
                args.num_sessions,
                total_steps,
                episode_count,
                update,
                num_updates,
                row,
                stable_count,
                last_delta_max_jump,
                last_kl_for_convergence,
            )

        if converged:
            sess_prefix = (
                f"S{session_id + 1}/{args.num_sessions} "
                if args.num_sessions > 1
                else ""
            )
            if conv_kl:
                if args.policy_kl_lag > 0:
                    msg = (
                        f"max KL(π_t−{args.policy_kl_lag}‖π_t) < {args.kl_threshold} "
                        f"for {args.convergence_patience} consecutive updates"
                    )
                else:
                    msg = (
                        f"KL(intra-update) < {args.kl_threshold} "
                        f"for {args.convergence_patience} consecutive updates"
                    )
            else:
                msg = (
                    f"|Δ_comb−Δ_comb,prev| < {args.delta_convergence_threshold} "
                    f"for {args.convergence_patience} consecutive updates"
                )
            print(f"  {sess_prefix}✓ Converged at step {total_steps} ({msg})")
            break

    # ---------- post-training analysis ----------
    limit_strat = compute_limit_strategy(agents, obs_normalizers, env, benchmarks)
    deviation_exp = run_deviation_experiment(env, agents, obs_normalizers)

    final_avg_step = {}
    for fid in range(NUM_FIRMS):
        if recent_step_profits[fid]:
            final_avg_step[fid] = float(np.mean(recent_step_profits[fid]))
        else:
            final_avg_step[fid] = 0.0
    final_delta_combined = compute_combined_delta(
        final_avg_step, pi_nash, pi_mono
    )

    return {
        "session_id": session_id,
        "seed": seed,
        "converged": converged,
        "convergence_step": total_steps,
        "final_delta_combined": float(final_delta_combined),
        "final_avg_step_profits": {str(k): float(v) for k, v in final_avg_step.items()},
        "metrics": log_rows,
        "limit_strategy": limit_strat,
        "deviation_experiment": deviation_exp,
        "agents": agents,
        "obs_normalizers": obs_normalizers,
    }


# ======================================================================
# Main
# ======================================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    print(f"Device: {device}")
    _print_paper_vs_ppo_banner()

    env = ElectricityMarketEnv(
        history_len=args.history_len,
        episode_len=args.episode_len,
    )

    # Compute benchmarks
    print("\n=== Benchmarks (Cournot–Nash MCP may take a few seconds) ===")
    benchmarks = {
        "competitive": compute_competitive_benchmark(env),
        "cournot_nash": compute_cournot_nash_benchmark(env),
        "monopoly": compute_monopoly_benchmark(env),
    }

    comp = benchmarks["competitive"]
    cn = benchmarks["cournot_nash"]
    mono = benchmarks["monopoly"]
    print(f"  Competitive:    avg LMP ${comp['avg_lmp']:.2f}  |  total π={comp['profits']['0']+comp['profits']['1']:.1f}/step")
    print(f"  Cournot–Nash:   avg LMP ${cn['avg_lmp']:.2f}  |  total π={cn['total_profit']:.1f}/step  (MCP res {cn['mcp_max_residual']:.2e})")
    print(f"  Monopoly:       avg LMP ${mono['avg_lmp']:.2f}  |  total π={mono['total_profit']:.1f}/step")
    denom = mono["total_profit"] - cn["total_profit"]
    print(f"  Combined Δ: 0 at Nash, 1 at monopoly  (Σπ^Mono−Σπ^Nash = {denom:.1f} $/step)")
    print("=" * 50)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sessions").mkdir(exist_ok=True)

    config = vars(args)
    config["benchmarks"] = benchmarks
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run sessions
    all_final_delta_combined = []
    all_convergence_steps = []

    for s in range(args.num_sessions):
        if args.num_sessions > 1:
            print(f"\n--- Session {s+1}/{args.num_sessions} (seed={args.seed + s}) ---")

        result = train_session(env, benchmarks, args, s, device)

        # Save session
        sess_dir = out_dir / "sessions" / f"session_{s}"
        sess_dir.mkdir(parents=True, exist_ok=True)

        sess_data = {
            "session_id": result["session_id"],
            "seed": result["seed"],
            "converged": result["converged"],
            "convergence_step": result["convergence_step"],
            "final_delta_combined": result["final_delta_combined"],
            "final_avg_step_profits": result["final_avg_step_profits"],
            "metrics": result["metrics"],
            "limit_strategy": result["limit_strategy"],
            "deviation_experiment": result["deviation_experiment"],
        }
        with open(sess_dir / "session.json", "w") as f:
            json.dump(sess_data, f, indent=2)

        for fid, agent in result["agents"].items():
            torch.save(agent.ac.state_dict(), sess_dir / f"agent_{fid}.pt")
            norm = result["obs_normalizers"].get(fid)
            if norm is not None:
                with open(sess_dir / f"normalizer_{fid}.json", "w") as f:
                    json.dump(norm.state_dict(), f)

        all_final_delta_combined.append(result["final_delta_combined"])
        all_convergence_steps.append(result["convergence_step"])

        if args.num_sessions > 1:
            dc = result["final_delta_combined"]
            print(
                f"  Final Δ_combined={dc:.3f}  "
                f"({'converged' if result['converged'] else 'max steps'})"
            )

    # Aggregate
    aggregate = {
        "num_sessions": args.num_sessions,
        "convergence_step_mean": float(np.mean(all_convergence_steps)),
        "convergence_step_std": float(np.std(all_convergence_steps)),
    }
    aggregate["delta_combined_mean"] = float(np.mean(all_final_delta_combined))
    aggregate["delta_combined_std"] = float(np.std(all_final_delta_combined))

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results → {out_dir}")
    print(f"Sessions: {args.num_sessions}")
    print(
        f"  Δ_combined: {aggregate['delta_combined_mean']:.3f} "
        f"± {aggregate['delta_combined_std']:.3f}"
    )
    print(f"  Avg convergence: {aggregate['convergence_step_mean']:.0f} ± {aggregate['convergence_step_std']:.0f} steps")


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Agent PPO — electricity market collusion (Calvano methodology)"
    )

    # Environment
    p.add_argument("--history-len", type=int, default=1)
    p.add_argument("--episode-len", type=int, default=168)

    # PPO
    p.add_argument("--total-timesteps", type=int, default=500_000,
                   help="Max timesteps per session (safety limit)")
    p.add_argument("--rollout-len", type=int, default=2048)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=64)

    # Calvano methodology
    p.add_argument("--num-sessions", type=int, default=1,
                   help="Independent training sessions (paper uses 1000)")
    p.add_argument(
        "--convergence-mode",
        type=str,
        default="delta",
        choices=("delta", "kl", "none"),
        help="delta: stop when |Δ_comb−Δ_comb,prev| < --delta-convergence-threshold for "
        "--convergence-patience updates. kl: same with policy KL vs --kl-threshold. "
        "none: never early-stop on convergence (run full --total-timesteps; patience ignored).",
    )
    p.add_argument(
        "--convergence-patience",
        type=int,
        default=0,
        help="Consecutive PPO updates satisfying the active criterion (0 = no early stop). "
        "Ignored when --convergence-mode none.",
    )
    p.add_argument(
        "--delta-convergence-threshold",
        type=float,
        default=0.01,
        help="delta mode: |Δ_combined−Δ_combined,prev| < this for patience updates",
    )
    p.add_argument(
        "--kl-threshold",
        type=float,
        default=0.01,
        help="kl mode: max_f KL vs this for patience (intra-update KL if policy-kl-lag=0, else lagged KL)",
    )
    p.add_argument(
        "--policy-kl-lag",
        type=int,
        default=0,
        help="If k>0, log KL(π_{t−k}‖π_t) on rollout obs; kl convergence uses this when mode=kl. "
        "k=0 keeps only pre→post update KL within the same iteration.",
    )

    # System
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/default")
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument(
        "--log-format",
        type=str,
        default="structured",
        choices=("structured", "legacy"),
        help="structured: key=value line (conv mode, d_jump, kl_for_conv, streak). "
        "legacy: older bracketed one-line format.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
