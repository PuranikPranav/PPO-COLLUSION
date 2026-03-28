"""
Multi-Agent Independent PPO for studying tacit collusion in electricity markets.

Methodology adapted from Calvano, Calzolari, Denicolò & Pastorello (2021):
  - Normalized collusion metric Δ = (π − πC) / (πM − πC) (logged / analyzed, not the stop rule)
  - Convergence criterion: policy KL below --kl-threshold for N consecutive PPO updates
  - Multiple independent sessions with different seeds, averaged
  - Post-training analysis: limit strategy + impulse response to deviation

Usage
-----
    # Quick single run
    python experiments/ppo.py --history-len 1 --total-timesteps 500000

    # Full Calvano-style experiment (50 sessions, KL-based convergence stopping)
    python experiments/ppo.py --history-len 1 --num-sessions 50 \
        --convergence-patience 100 --kl-threshold 0.01 --total-timesteps 2000000 \
        --output-dir results/h1
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from scipy.optimize import minimize as scipy_minimize

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
# Actor-Critic network
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
        """Return the greedy (mean) action in MW — no exploration noise."""
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
        "profits": {str(k): float(v) for k, v in profits.items()},
    }


def compute_monopoly_benchmark(env: ElectricityMarketEnv):
    
    """Joint profit maximization: all firms act as one monopolist."""
    caps = np.array([p["cap"] for p in PLANTS])

    def neg_total_profit(g_flat):
        g = np.clip(g_flat, 0, caps)
        gen_per_node = np.zeros(NUM_NODES)
        for i, plant in enumerate(PLANTS):
            gen_per_node[plant["node"]] += g[i]

        lmps, demand = env._clear_market(gen_per_node)
        if lmps is None:
            return 1e6

        total = 0.0
        for i, plant in enumerate(PLANTS):
            total += lmps[plant["node"]] * g[i] - plant["mc"] * g[i] - 0.5 * plant["qc"] * g[i] ** 2
        return -total

    best = None
    for _ in range(50):
        x0 = np.random.uniform(0, 1, len(PLANTS)) * caps
        res = scipy_minimize(neg_total_profit, x0, method="L-BFGS-B",
                             bounds=[(0, c) for c in caps])
        if best is None or res.fun < best.fun:
            best = res

    g_opt = np.clip(best.x, 0, caps)
    gen_per_node = np.zeros(NUM_NODES)
    for i, plant in enumerate(PLANTS):
        gen_per_node[plant["node"]] += g_opt[i]
    lmps, demand = env._clear_market(gen_per_node)
    avg_lmp = float(np.sum(lmps * demand) / np.sum(demand)) if np.sum(demand) > 0 else 0

    profits = {}
    for fid in range(NUM_FIRMS):
        p = 0.0
        for pidx in FIRM_PLANT_IDX[fid]:
            plant = PLANTS[pidx]
            g = g_opt[pidx]
            p += lmps[plant["node"]] * g - plant["mc"] * g - 0.5 * plant["qc"] * g ** 2
        profits[fid] = p

    return {
        "avg_lmp": avg_lmp,
        "lmps": lmps.tolist(),
        "gens": [float(g) for g in g_opt],
        "total_gen": float(np.sum(g_opt)),
        "total_profit": float(-best.fun),
        "profits": {str(k): float(v) for k, v in profits.items()},
    }


def compute_delta(avg_step_profit: float, pi_c: float, pi_m: float) -> float:
    """Calvano normalized profit gain: 0 = competitive, 1 = full collusion."""
    denom = pi_m - pi_c
    if abs(denom) < 1e-8:
        return 0.0
    return (avg_step_profit - pi_c) / denom


# ======================================================================
# Post-training analysis
# ======================================================================
def build_reference_obs(env, benchmarks, num_points=20):
    """Grid of observations spanning plausible LMP range (limit-strategy sweep)."""
    comp_lmps = np.array(benchmarks["competitive"]["lmps"])
    comp_avg = np.mean(comp_lmps)
    targets = np.linspace(15, 38, num_points)
    ref_obs = []
    for target in targets:
        scaled = comp_lmps * (target / comp_avg) if comp_avg > 0 else np.full(NUM_NODES, target)
        obs = np.tile(scaled, env.history_len).astype(np.float32)
        ref_obs.append(obs)
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
    comp_lmps = np.array(benchmarks["competitive"]["lmps"])
    comp_avg = np.mean(comp_lmps)
    lmp_grid = np.linspace(15, 38, num_points)

    strategies = {str(fid): [] for fid in range(NUM_FIRMS)}
    for target in lmp_grid:
        scaled = comp_lmps * (target / comp_avg)
        obs = np.tile(scaled, env.history_len).astype(np.float32)
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
            obs, _, done, _ = env.step(actions)
            if done:
                obs = env.reset()

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
        for fid in range(NUM_FIRMS):
            trace_gen[str(fid)].append(float(np.sum(actions[fid])))
        trace_lmp.append(info.get("avg_lmp", 0))

        # --- Post-deviation: both play deterministic ---
        for _ in range(horizon):
            if done:
                obs = env.reset()
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


# ======================================================================
# Single-session training
# ======================================================================
def train_session(env, benchmarks, args, session_id, device):
    seed = args.seed + session_id
    np.random.seed(seed)
    torch.manual_seed(seed)

    pi_c = {fid: benchmarks["competitive"]["profits"][str(fid)] for fid in range(NUM_FIRMS)}
    pi_m = {fid: benchmarks["monopoly"]["profits"][str(fid)] for fid in range(NUM_FIRMS)}

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

    # Convergence tracking — KL divergence between consecutive policies
    use_convergence = args.convergence_patience > 0
    stable_count = 0
    last_agent_kls = {fid: 0.0 for fid in range(NUM_FIRMS)}

    # Logging
    log_rows = []
    obs = env.reset()
    total_steps = 0
    ep_rewards = {f: 0.0 for f in range(NUM_FIRMS)}
    episode_count = 0
    converged = False

    recent_lmps = deque(maxlen=2000)
    recent_profits = {f: deque(maxlen=200) for f in range(NUM_FIRMS)}
    recent_gens = {f: deque(maxlen=2000) for f in range(NUM_FIRMS)}

    num_updates = args.total_timesteps // args.rollout_len
    wall_start = time.time()

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
                ep_rewards[fid] += rewards[fid]

            if "lmps" in info:
                recent_lmps.append(info["avg_lmp"])
                for fid in range(NUM_FIRMS):
                    gen_total = sum(
                        info["gen"].get(pidx, 0) for pidx in FIRM_PLANT_IDX[fid]
                    )
                    recent_gens[fid].append(gen_total)

            obs = obs_next
            total_steps += 1

            if done:
                for fid in range(NUM_FIRMS):
                    recent_profits[fid].append(ep_rewards[fid])
                    ep_rewards[fid] = 0.0
                episode_count += 1
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

        if use_convergence:
            max_kl = max(last_agent_kls.values())
            if max_kl < args.kl_threshold:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= args.convergence_patience:
                converged = True

        # ---------- logging ----------
        if (update + 1) % args.log_interval == 0:
            avg_lmp = np.mean(recent_lmps) if recent_lmps else 0
            row = {
                "step": total_steps,
                "episodes": episode_count,
                "avg_lmp": float(avg_lmp),
                "wall_sec": time.time() - wall_start,
            }
            for fid in range(NUM_FIRMS):
                ep_prof = float(np.mean(recent_profits[fid])) if recent_profits[fid] else 0
                avg_gen = float(np.mean(recent_gens[fid])) if recent_gens[fid] else 0
                avg_step_prof = ep_prof / args.episode_len if args.episode_len > 0 else 0
                delta = compute_delta(avg_step_prof, pi_c[fid], pi_m[fid])
                row[f"firm_{fid}_ep_profit"] = ep_prof
                row[f"firm_{fid}_avg_gen"] = avg_gen
                row[f"firm_{fid}_delta"] = float(delta)
                row[f"firm_{fid}_kl"] = last_agent_kls.get(fid, 0)
            row["max_kl"] = max(last_agent_kls.values())
            row["kl_streak"] = stable_count
            log_rows.append(row)

            if args.num_sessions == 1:
                d0 = row.get("firm_0_delta", 0)
                d1 = row.get("firm_1_delta", 0)
                mk = row.get("max_kl", 0)
                print(
                    f"[{total_steps:>8d}] ep {episode_count:>4d} | "
                    f"LMP ${row['avg_lmp']:.2f} | "
                    f"Δ0={d0:.3f} Δ1={d1:.3f} | "
                    f"g0={row['firm_0_avg_gen']:.1f} g1={row['firm_1_avg_gen']:.1f} | "
                    f"KL={mk:.2e} streak={stable_count}"
                )

        if converged:
            if args.num_sessions == 1:
                print(f"  ✓ Converged at step {total_steps} (KL < {args.kl_threshold} for {args.convergence_patience} consecutive updates)")
            break

    # ---------- post-training analysis ----------
    limit_strat = compute_limit_strategy(agents, obs_normalizers, env, benchmarks)
    deviation_exp = run_deviation_experiment(env, agents, obs_normalizers)

    # Final Δ
    final_delta = {}
    for fid in range(NUM_FIRMS):
        if recent_profits[fid]:
            avg_step = float(np.mean(recent_profits[fid])) / args.episode_len
            final_delta[str(fid)] = compute_delta(avg_step, pi_c[fid], pi_m[fid])
        else:
            final_delta[str(fid)] = 0.0

    return {
        "session_id": session_id,
        "seed": seed,
        "converged": converged,
        "convergence_step": total_steps,
        "final_delta": final_delta,
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

    env = ElectricityMarketEnv(
        history_len=args.history_len,
        episode_len=args.episode_len,
    )

    # Compute benchmarks
    benchmarks = {
        "competitive": compute_competitive_benchmark(env),
        "monopoly": compute_monopoly_benchmark(env),
    }

    comp = benchmarks["competitive"]
    mono = benchmarks["monopoly"]
    print("\n=== Benchmarks ===")
    print(f"  Competitive:  avg LMP ${comp['avg_lmp']:.2f}  |  F0 π={comp['profits']['0']:.1f}/step  F1 π={comp['profits']['1']:.1f}/step")
    print(f"  Monopoly:     avg LMP ${mono['avg_lmp']:.2f}  |  F0 π={mono['profits']['0']:.1f}/step  F1 π={mono['profits']['1']:.1f}/step")
    print(f"  (Δ=0 at competitive, Δ=1 at monopoly)")
    print("=" * 50)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sessions").mkdir(exist_ok=True)

    config = vars(args)
    config["benchmarks"] = benchmarks
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run sessions
    all_final_deltas = {str(fid): [] for fid in range(NUM_FIRMS)}
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
            "final_delta": result["final_delta"],
            "metrics": result["metrics"],
            "limit_strategy": result["limit_strategy"],
            "deviation_experiment": result["deviation_experiment"],
        }
        with open(sess_dir / "session.json", "w") as f:
            json.dump(sess_data, f, indent=2)

        for fid, agent in result["agents"].items():
            torch.save(agent.ac.state_dict(), sess_dir / f"agent_{fid}.pt")

        for fid in range(NUM_FIRMS):
            all_final_deltas[str(fid)].append(result["final_delta"][str(fid)])
        all_convergence_steps.append(result["convergence_step"])

        if args.num_sessions > 1:
            d0 = result["final_delta"]["0"]
            d1 = result["final_delta"]["1"]
            print(f"  Final Δ: F0={d0:.3f}  F1={d1:.3f}  ({'converged' if result['converged'] else 'max steps'})")

    # Aggregate
    aggregate = {
        "num_sessions": args.num_sessions,
        "convergence_step_mean": float(np.mean(all_convergence_steps)),
        "convergence_step_std": float(np.std(all_convergence_steps)),
    }
    for fid in range(NUM_FIRMS):
        vals = all_final_deltas[str(fid)]
        aggregate[f"firm_{fid}_delta_mean"] = float(np.mean(vals))
        aggregate[f"firm_{fid}_delta_std"] = float(np.std(vals))

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results → {out_dir}")
    print(f"Sessions: {args.num_sessions}")
    for fid in range(NUM_FIRMS):
        m = aggregate[f"firm_{fid}_delta_mean"]
        s = aggregate[f"firm_{fid}_delta_std"]
        print(f"  Firm {fid} Δ: {m:.3f} ± {s:.3f}")
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
    p.add_argument("--convergence-patience", type=int, default=0,
                   help="Stop when policy KL < threshold for N consecutive PPO updates (0=disabled)")
    p.add_argument("--kl-threshold", type=float, default=0.01,
                   help="KL divergence threshold for policy convergence")

    # System
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results/default")
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=50)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
