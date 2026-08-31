"""
The Calvano-Calzolari-Denicolo-Pastorello Q-learning algorithm, verbatim.

Learning rule (paper eq. 3):
    Q_{t+1}(s_t, a_t) = (1-alpha) Q_t(s_t, a_t)
                        + alpha [ pi_t + delta * max_{a'} Q_t(s_{t+1}, a') ]
Exploration (paper eq. 4):
    epsilon_t = exp(-beta t),  epsilon-greedy over the finite action set.
Initialisation (paper fn. 12):
    Q_0(s, a) = E_{a_{-i} ~ Uniform} [ pi_i(a, a_{-i}) ] / (1 - delta)
    -- the discounted payoff that would accrue if rivals randomised uniformly,
    identical across states; the initial state is drawn at random.
Tie-breaking (paper fn. 17):
    ties in the argmax are broken by choosing the HIGHER output.
Convergence (paper sec. 4):
    the greedy action does not change for 100,000 consecutive periods.

Everything is vectorised across independent SESSIONS (the paper runs 1,000 and
averages), so one numpy pass advances every session by one period. Sessions that
converge are snapshotted and removed from the working arrays, so the loop gets
cheaper over time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qlearning_collusion.market import DiscreteMarket


# ---------------------------------------------------------------------------
# State spaces
# ---------------------------------------------------------------------------
class PriceState:
    """Imperfect monitoring: s_t = (binned) hub LMP of period t-1."""

    name = "imperfect"

    def __init__(self, market):
        self.mk = market
        self.n_states = market.n_price_states
        self._tab = market.price_state_p        # (n_profiles, h) -> int

    def next_state(self, actions, P, u_idx):
        return self._tab[P, u_idx]

    def initial(self, rng, size):
        # Paper: "the initial state is selected randomly" -- over reachable ones.
        return rng.choice(self.mk.reachable_states, size=size)

    def describe(self):
        return f"price bins (|S| = {self.n_states})"


class ProfileState:
    """Perfect monitoring: s_t = the whole output profile of period t-1.

    Paper (n = 2): S = A x A, so |S| = k^n. The state is public and identical
    across firms.
    """

    name = "perfect"

    def __init__(self, market):
        self.mk = market
        self.n = market.n_agents
        # The profile index IS the perfect-monitoring state: it encodes exactly
        # the previous period's full action profile.
        self.n_states = market.n_profiles

    def next_state(self, actions, P, u_idx):
        return P

    def initial(self, rng, size):
        return rng.integers(0, self.n_states, size=size)

    def describe(self):
        return f"rivals' past output profile (|S| = {self.n_states})"


class RichState:
    """Imperfect monitoring with the full public market signal (the "19-variable"
    state): s_t = discretised (nodal LMPs, transmission congestion / shadow
    prices, realised demand) of period t-1.

    This is the PPO observation vector brought back into the tabular model. It is
    still IMPERFECT monitoring -- rivals' outputs are not in the state, so a low
    price remains consistent with many rival profiles -- but the agent now sees
    realised demand, so an adverse demand shock is no longer perfectly
    confounded with a rival's expansion. `market.rich_state_report()` measures
    exactly how much of the confounding survives.

    See `ProfileAPI.build_rich_states` for what is in the state and why rivals'
    generation is deliberately excluded.
    """

    name = "rich"

    def __init__(self, market):
        self.mk = market
        if not hasattr(market, "rich_state_p"):
            market.build_rich_states()
        self.n_states = market.n_rich_states
        self._tab = market.rich_state_p          # (n_profiles, h) -> int

    def next_state(self, actions, P, u_idx):
        return self._tab[P, u_idx]

    def initial(self, rng, size):
        return rng.choice(self.mk.reachable_rich_states, size=size)

    def describe(self):
        return (f"rich market signal [{', '.join(self.mk.rich_state_components)}] "
                f"(|S| = {self.n_states})")


STATE_SPACES = {"imperfect": PriceState, "perfect": ProfileState,
                "rich": RichState}


# ---------------------------------------------------------------------------
@dataclass
class QLearnConfig:
    alpha: float = 0.15          # learning rate       (paper baseline)
    beta: float = 4e-6           # exploration decay   (paper baseline)
    delta: float = 0.95          # discount factor     (paper baseline)
    monitoring: str = "imperfect"
    n_sessions: int = 1000       # paper runs 1,000
    max_iter: int = 5_000_000
    conv_window: int = 100_000   # paper's stabilisation criterion
    seed: int = 0
    log_every: int = 2_000
    dtype: str = "float32"
    eval_periods: int = 20_000   # post-convergence evaluation horizon
    eval_burn_in: int = 1_000


@dataclass
class QLearnResult:
    cfg: QLearnConfig = None
    greedy: np.ndarray = None            # (S, n, |S|) final greedy policy
    converged: np.ndarray = None         # (S,) bool
    conv_iter: np.ndarray = None         # (S,) iteration of convergence (-1 if none)
    final_state: np.ndarray = None       # (S,) state at convergence
    log_iters: np.ndarray = None
    log_q: np.ndarray = None             # (T, n) mean greedy output
    log_delta: np.ndarray = None         # (T,) normalised profit gain
    log_profit: np.ndarray = None        # (T, n)
    log_price: np.ndarray = None         # (T,)
    log_eps: np.ndarray = None
    log_active: np.ndarray = None        # (T,) sessions still learning
    eval: dict = field(default_factory=dict)
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
def initial_q(mk, delta: float) -> np.ndarray:
    """Q_0(s, a) = E_{rivals uniform, shock}[pi_i(a, .)] / (1 - delta).

    Same value in every state (paper fn. 12). Averaging over every rival action
    profile with agent i's own action pinned to `a` is exactly the paper's
    expectation, and works whatever the profile structure is. Returns
    (n_agents, max_actions).
    """
    out = np.zeros((mk.n_agents, mk.max_actions))
    for i in range(mk.n_agents):
        own = mk.agent_action_of_profile(i)                 # (n_profiles,)
        for a in range(int(mk.n_actions[i])):
            out[i, a] = float(mk.exp_profit_p[i][own == a].mean())
    return out / (1.0 - delta)


# ---------------------------------------------------------------------------
def _argmax_high(Q: np.ndarray) -> np.ndarray:
    """argmax along the LAST axis, ties broken toward the HIGHER index.

    (Paper fn. 17: "ties are broken by choosing the higher output".)
    """
    k = Q.shape[-1]
    return (k - 1) - np.argmax(Q[..., ::-1], axis=-1)


def train(mk, cfg: QLearnConfig, verbose: bool = True) -> QLearnResult:
    rng = np.random.default_rng(cfg.seed)
    dt = np.dtype(cfg.dtype)
    n, k, h = mk.n_agents, mk.max_actions, mk.h
    space = STATE_SPACES[cfg.monitoring](mk)
    nS = space.n_states
    S = cfg.n_sessions

    # Profile-indexed payoffs: prof[i, P, u] (see profile_api).
    prof = np.ascontiguousarray(mk.profit_p.astype(dt))          # (n, P, h)

    grid_bm = mk.grid_benchmarks()
    pi_N = np.array(grid_bm["nash"]["profits"])
    pi_M = np.array(grid_bm["monopoly"]["profits"])
    tot_N, tot_M = pi_N.sum(), pi_M.sum()

    # ---- working arrays ----------------------------------------------------
    # Converged sessions are FROZEN in place (their Q and greedy policy stop
    # updating) rather than removed, so the logged averages below always cover
    # all S sessions -- exactly what the paper's Figures 1-2 average over.
    Q = np.empty((S, n, nS, k), dtype=dt)
    Q[:] = initial_q(mk, cfg.delta).astype(dt)[None, :, None, :]
    greedy = _argmax_high(Q[0, :, 0, :])                          # identical rows
    greedy_all = np.empty((S, n, nS), dtype=np.int8)
    greedy_all[:] = greedy[None, :, None]
    state = space.initial(rng, S).astype(np.int64)
    conv_count = np.zeros(S, dtype=np.int64)
    frozen = np.zeros(S, dtype=bool)

    out_conv_it = np.full(S, -1, dtype=np.int64)
    out_state = np.zeros(S, dtype=np.int64)

    log_i, log_q, log_pi, log_p, log_eps, log_act = [], [], [], [], [], []
    acc_q = np.zeros(n); acc_pi = np.zeros(n); acc_p = 0.0; acc_cnt = 0

    ar_n = np.arange(n)
    rowS = np.arange(S)[:, None]
    t0 = time.time()
    t = 0
    while t < cfg.max_iter and not frozen.all():
        eps = np.exp(-cfg.beta * t)

        # --- greedy profile (what Figures 1-2 track) ------------------------
        a_g = greedy_all[rowS, ar_n[None, :], state[:, None]].astype(np.int64)

        # --- action selection: epsilon-greedy (frozen sessions never explore)
        explore = (rng.random((S, n)) < eps) & ~frozen[:, None]
        a = np.where(explore, rng.integers(0, k, size=(S, n)), a_g) if explore.any() else a_g

        # --- market clears ---------------------------------------------------
        P = mk.pidx(a)
        u_idx = rng.integers(0, h, size=S) if h > 1 else np.zeros(S, dtype=np.int64)
        pi = prof[ar_n[None, :], P[:, None], u_idx[:, None]]          # (S, n)
        s_next = space.next_state(a, P, u_idx)

        # --- Q update (paper eq. 3), skipped for frozen sessions -------------
        maxQ_next = Q[rowS, ar_n[None, :], s_next[:, None], :].max(axis=-1)   # (S,n)
        target = pi + cfg.delta * maxQ_next
        idx = (rowS, ar_n[None, :], state[:, None], a)
        cur = Q[idx]
        Q[idx] = np.where(frozen[:, None], cur,
                          (1.0 - cfg.alpha) * cur + cfg.alpha * target)

        # --- refresh greedy at the visited state, track stabilisation --------
        newg = _argmax_high(Q[rowS, ar_n[None, :], state[:, None], :]).astype(np.int8)
        oldg = greedy_all[rowS, ar_n[None, :], state[:, None]]
        changed = (newg != oldg).any(axis=1) & ~frozen
        greedy_all[rowS, ar_n[None, :], state[:, None]] = newg
        conv_count += 1
        conv_count[changed] = 0

        state = s_next
        t += 1

        # --- freeze newly converged sessions ---------------------------------
        newly = (~frozen) & (conv_count >= cfg.conv_window)
        if newly.any():
            frozen |= newly
            out_conv_it[newly] = t
            out_state[newly] = state[newly]

        # --- logging: GREEDY outputs / profits (paper Figs. 1-2) -------------
        P_g = mk.pidx(a_g)
        acc_q += mk.q_agent[ar_n[None, :], a_g].mean(axis=0)
        acc_pi += mk.exp_profit_p[ar_n[None, :], P_g[:, None]].mean(axis=0)
        acc_p += float(mk.exp_price_p[P_g].mean())
        acc_cnt += 1
        if t % cfg.log_every == 0:
            log_i.append(t); log_eps.append(eps)
            log_q.append(acc_q / acc_cnt)
            log_pi.append(acc_pi / acc_cnt)
            log_p.append(acc_p / acc_cnt)
            log_act.append(int(S - frozen.sum()))
            if verbose and (t % (cfg.log_every * 50) == 0):
                d = (acc_pi.sum() / acc_cnt - tot_N) / (tot_M - tot_N)
                print(f"    t={t:>9,}  eps={eps:.4f}  learning={S-frozen.sum():>4}/{S}  "
                      f"q={np.round(acc_q/acc_cnt,1)}  Delta={d:+.3f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            acc_q[:] = 0; acc_pi[:] = 0; acc_p = 0.0; acc_cnt = 0

    # sessions that never stabilised keep their last policy and state
    out_conv = frozen.copy()
    out_state[~frozen] = state[~frozen]
    out_greedy = greedy_all

    res = QLearnResult(
        cfg=cfg,
        greedy=out_greedy,
        converged=out_conv,
        conv_iter=out_conv_it,
        final_state=out_state,
        log_iters=np.array(log_i),
        log_q=np.array(log_q),
        log_profit=np.array(log_pi),
        log_price=np.array(log_p),
        log_eps=np.array(log_eps),
        wall_time=time.time() - t0,
    )
    res.log_active = np.array(log_act)
    res.log_delta = (res.log_profit.sum(axis=1) - tot_N) / (tot_M - tot_N)
    res.eval = evaluate(mk, cfg, space, out_greedy, out_state, grid_bm)
    return res


# ---------------------------------------------------------------------------
def evaluate(mk: DiscreteMarket, cfg: QLearnConfig, space, greedy, start_state,
             grid_bm=None, seed_offset: int = 12345) -> dict:
    """Play the LEARNED (frozen, epsilon = 0) strategies forward and average.

    This is the paper's "average per-firm profit upon convergence": the limit
    strategies are deterministic maps state -> action, so the only randomness
    left is the demand shock.
    """
    grid_bm = grid_bm or mk.grid_benchmarks()
    pi_N = np.array(grid_bm["nash"]["profits"])
    pi_M = np.array(grid_bm["monopoly"]["profits"])

    rng = np.random.default_rng(cfg.seed + seed_offset)
    S, n, h = greedy.shape[0], mk.n_agents, mk.h
    ar_n = np.arange(n)
    rowS = np.arange(S)[:, None]
    state = start_state.copy()

    sum_q = np.zeros((S, n)); sum_pi = np.zeros((S, n))
    sum_p = np.zeros(S); sum_avg = np.zeros(S); sum_g = np.zeros(S)
    T = cfg.eval_periods
    for step in range(T + cfg.eval_burn_in):
        a = greedy[rowS, ar_n[None, :], state[:, None]].astype(np.int64)
        P = mk.pidx(a)
        u_idx = rng.integers(0, h, size=S) if h > 1 else np.zeros(S, dtype=np.int64)
        if step >= cfg.eval_burn_in:
            sum_q += mk.q_agent[ar_n[None, :], a]
            sum_pi += mk.profit_p[ar_n[None, :], P[:, None], u_idx[:, None]]
            sum_p += mk.price_p[P, u_idx]
            sum_avg += mk.avg_lmp_p[P, u_idx]
            sum_g += mk.total_gen_p[P]
        state = space.next_state(a, P, u_idx)

    q = sum_q / T; pi = sum_pi / T
    delta_ses = (pi.sum(axis=1) - pi_N.sum()) / (pi_M.sum() - pi_N.sum())
    delta_firm = (pi - pi_N[None, :]) / (pi_M - pi_N)[None, :]
    return {
        "delta": float(delta_ses.mean()),
        "delta_se": float(delta_ses.std(ddof=1) / np.sqrt(S)) if S > 1 else 0.0,
        "delta_per_session": delta_ses,
        "delta_per_firm": delta_firm.mean(axis=0).tolist(),
        "gen_per_firm": q.mean(axis=0).tolist(),
        "profit_per_firm": pi.mean(axis=0).tolist(),
        "total_gen": float((sum_g / T).mean()),
        "hub_price": float((sum_p / T).mean()),
        "avg_lmp": float((sum_avg / T).mean()),
        "bench": {
            "nash_profits": pi_N.tolist(), "monopoly_profits": pi_M.tolist(),
            "nash_gen": grid_bm["nash"]["total_gen"],
            "monopoly_gen": grid_bm["monopoly"]["total_gen"],
            "nash_hub_price": grid_bm["nash"]["hub_price"],
            "monopoly_hub_price": grid_bm["monopoly"]["hub_price"],
        },
    }
