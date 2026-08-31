"""Experiment definitions, persistence, and the post-convergence analyses
(limit strategy + forced-deviation impulse response)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from iso_market.node_network import MARKET
from qlearning_collusion.market import DiscreteMarket
from qlearning_collusion.profile_api import attach_hub_profile_api
from qlearning_collusion.qlearn import (
    QLearnConfig, QLearnResult, STATE_SPACES, train, evaluate,
)

# Artifacts are namespaced by market so the two-firm run cannot clobber the
# published three-firm results (which live in the unsuffixed `results/`).
RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results" if MARKET == "three_firm" else f"results_{MARKET}",
)


# ---------------------------------------------------------------------------
# The four cells of the paper's Table I, plus the baseline alias.
#   monitoring   : "imperfect" (state = past price) | "perfect" (state = past
#                  output profile)
#   deterministic: True  -> no demand shock  ("Deterministic Demand" column)
#                  False -> h equiprobable shocks ("Stochastic Demand")
# ---------------------------------------------------------------------------
CELLS = {
    "imperfect_stochastic":    dict(monitoring="imperfect", deterministic=False),
    "imperfect_deterministic": dict(monitoring="imperfect", deterministic=True),
    "perfect_stochastic":      dict(monitoring="perfect",   deterministic=False),
    "perfect_deterministic":   dict(monitoring="perfect",   deterministic=True),
    # The "19-variable" state: the public market signal (nodal LMPs, congestion
    # / shadow prices, realised demand) instead of the single binned price.
    # Still imperfect monitoring -- rivals' outputs are NOT in the state.
    "rich_stochastic":         dict(monitoring="rich",      deterministic=False),
    "rich_deterministic":      dict(monitoring="rich",      deterministic=True),
}
# The four cells that make up the paper's Table I (the rich-state cells are a
# separate information-structure experiment, not part of that 2x2).
TABLE1_CELLS = ["imperfect_stochastic", "imperfect_deterministic",
                "perfect_stochastic", "perfect_deterministic"]
BASELINE = "imperfect_stochastic"


def build_market(deterministic=False, k=15, xi=0.2, m=8.0, h=2, **kw):
    """The discrete game for whichever market MARKET_CONFIG selects.

    three_firm      : one plant per firm, all at the hub -> `DiscreteMarket` (the
        J-lattice fast path). `attach_hub_profile_api` gives it the same
        profile-indexed interface the other markets expose, purely by re-indexing
        tables it already has, so its published numbers are unchanged.
    three_firm_dist : one plant per firm, ONE PER NODE (nodes 1, 2, 3, costliest
        at node 3). Total hub generation no longer determines the clearing, so
        the J-lattice collapse is invalid -> `DiscreteMarketMulti`.
    two_firm        : firm 0 owns two plants at two different nodes -> likewise.
    """
    if MARKET == "three_firm":
        mk = DiscreteMarket(k=k, xi=xi, shock_steps_m=m, h=h,
                            deterministic=deterministic, **kw)
        return attach_hub_profile_api(mk)
    from qlearning_collusion.market_multi import DiscreteMarketMulti
    return DiscreteMarketMulti(k=k, xi=xi, shock_steps_m=m, h=h,
                               deterministic=deterministic, **kw)


# ---------------------------------------------------------------------------
def run(name: str, *, n_sessions=1000, max_iter=4_000_000, k=15, xi=0.2, m=8.0,
        h=2, seed=0, alpha=0.15, beta=4e-6, delta=0.95, conv_window=100_000,
        log_every=2000, dtype="float32", tag: str = "", verbose=True) -> dict:
    if name not in CELLS:
        raise KeyError(f"unknown experiment {name!r}; choose from {list(CELLS)}")
    spec = CELLS[name]
    mk = build_market(deterministic=spec["deterministic"], k=k, xi=xi, m=m, h=h)
    cfg = QLearnConfig(
        alpha=alpha, beta=beta, delta=delta, monitoring=spec["monitoring"],
        n_sessions=n_sessions, max_iter=max_iter, conv_window=conv_window,
        seed=seed, log_every=log_every, dtype=dtype,
    )
    if verbose:
        print(mk.describe(), flush=True)
        space = STATE_SPACES[cfg.monitoring](mk)
        print(f"\n>>> {name}{tag}: monitoring={cfg.monitoring} "
              f"({space.describe()}), sessions={n_sessions}, "
              f"alpha={alpha}, beta={beta:g}, delta={delta}\n", flush=True)

    res = train(mk, cfg, verbose=verbose)
    out = save(name + tag, mk, res)
    if verbose:
        print(f"\n=== {name}{tag} DONE in {res.wall_time/60:.1f} min ===")
        print(summary_text(out))
    return out


# ---------------------------------------------------------------------------
def save(name: str, mk: DiscreteMarket, res: QLearnResult) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    extra = {}
    if res.cfg.monitoring == "rich":
        # the rich state's own index table + a price per state, so the figures
        # can place a rich state on the same price axis as the price-only ones
        extra = dict(rich_state=mk.rich_state_p,
                     rich_state_value=mk.rich_state_value,
                     rich_state_count=mk.rich_state_count)
    np.savez_compressed(
        os.path.join(RESULTS, f"{name}.npz"),
        greedy=res.greedy, converged=res.converged, conv_iter=res.conv_iter,
        final_state=res.final_state, log_iters=res.log_iters, log_q=res.log_q,
        log_delta=res.log_delta, log_profit=res.log_profit,
        log_price=res.log_price, log_eps=res.log_eps, log_active=res.log_active,
        delta_per_session=res.eval["delta_per_session"],
        q_grid=mk.q_agent, price_state_value=mk.price_state_value,
        price=mk.price_p, exp_price=mk.exp_price_p, total_gen_grid=mk.total_gen_p,
        price_state=mk.price_state_p, u_levels=mk.u_levels, **extra,
    )
    meta = {
        "name": name,
        "config": asdict(res.cfg),
        "market": {
            "k": mk.k, "xi": mk.xi, "h": mk.h, "m": mk.shock_steps_m,
            "deterministic": mk.deterministic,
            "v": float(getattr(mk, "v", np.mean(mk.firm_step))),
            "n_firms": mk.n_agents,
            "market_config": MARKET,
            "q_grid": mk.q_agent.tolist(),
            "monitoring_report": mk.monitoring_report(),
            "rich_state_report": (mk.rich_state_report()
                                  if res.cfg.monitoring == "rich" else None),
            "grid_benchmarks": _jsonable(mk.grid_benchmarks()),
            "continuous_benchmarks": mk.bench_continuous,
        },
        "result": {
            "wall_time_min": res.wall_time / 60.0,
            "converged_frac": float(res.converged.mean()),
            "median_conv_iter": float(np.median(res.conv_iter[res.converged]))
                                 if res.converged.any() else None,
            "delta": res.eval["delta"],
            "delta_se": res.eval["delta_se"],
            "delta_per_firm": res.eval["delta_per_firm"],
            "gen_per_firm": res.eval["gen_per_firm"],
            "profit_per_firm": res.eval["profit_per_firm"],
            "total_gen": res.eval["total_gen"],
            "hub_price": res.eval["hub_price"],
            "avg_lmp": res.eval["avg_lmp"],
            "bench": res.eval["bench"],
        },
    }
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def load(name: str):
    with open(os.path.join(RESULTS, f"{name}.json")) as fh:
        meta = json.load(fh)
    z = np.load(os.path.join(RESULTS, f"{name}.npz"))
    return meta, z


def summary_text(meta: dict) -> str:
    r, m = meta["result"], meta["market"]
    b = r["bench"]
    L = [
        f"  sessions converged      : {100*r['converged_frac']:.1f}% "
        f"(median at {r['median_conv_iter']:,.0f} iterations)"
        if r["median_conv_iter"] else
        f"  sessions converged      : {100*r['converged_frac']:.1f}%",
        f"  DELTA (profit gain)     : {100*r['delta']:.2f}%  (+/- {100*r['delta_se']:.2f} s.e.)",
        f"  Delta per firm          : {[round(100*x,1) for x in r['delta_per_firm']]} %",
        f"  total generation        : {r['total_gen']:.1f} MW   "
        f"(Nash {b['nash_gen']:.1f} -> Monopoly {b['monopoly_gen']:.1f})",
        f"  hub LMP                 : ${r['hub_price']:.2f}      "
        f"(Nash ${b['nash_hub_price']:.2f} -> Monopoly ${b['monopoly_hub_price']:.2f})",
        f"  qty-weighted avg LMP    : ${r['avg_lmp']:.2f}",
        f"  non-revealing prices    : {m['monitoring_report']['measured_nonrevealing_fraction']:.3f}",
    ]
    rr = m.get("rich_state_report")
    if rr:
        L.append(f"  rich state |S|          : {rr['n_rich_states']} "
                 f"[{', '.join(rr['components'])}]")
        L.append(f"  non-revealing STATES    : {rr['measured_nonrevealing_fraction']:.3f}"
                 f"   (price-only "
                 f"{m['monitoring_report']['measured_nonrevealing_fraction']:.3f}, "
                 f"perfect 0.000)")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Post-convergence analysis 1: the average LIMIT STRATEGY (paper Figure 3)
# ---------------------------------------------------------------------------
def limit_strategy(name: str):
    """Average greedy output as a function of the previous period's price.

    Only price states that the converged sessions actually VISIT are meaningful
    (Q-values in never-visited states are still at their initialisation), so we
    weight each state by its visit frequency under the learned strategies and
    report that weight alongside the average action.
    """
    meta, z = load(name)
    mk = build_market(deterministic=meta["market"]["deterministic"],
                      k=meta["market"]["k"], xi=meta["market"]["xi"],
                      m=meta["market"]["m"], h=meta["market"]["h"])
    greedy = z["greedy"]                       # (S, n, |S|)
    S, n, nS = greedy.shape
    mon = meta["config"]["monitoring"]
    if mon not in ("imperfect", "rich"):
        raise ValueError("limit_strategy needs a price-indexable state "
                         "('imperfect' or 'rich'), not %r" % mon)

    # visit frequencies under the learned strategies
    space = STATE_SPACES[mon](mk)
    rng = np.random.default_rng(7)
    state = z["final_state"].copy()
    rowS = np.arange(S)[:, None]
    ar_n = np.arange(n)
    visits = np.zeros((S, nS))
    T = 5000
    for _ in range(T):
        a = greedy[rowS, ar_n[None, :], state[:, None]].astype(np.int64)
        J = mk.pidx(a)
        u = rng.integers(0, mk.h, size=S)
        np.add.at(visits, (np.arange(S), state), 1.0)
        state = space.next_state(a, J, u)
    visits /= T

    q_of_state = mk.q_agent[ar_n[None, None, :], greedy.transpose(0, 2, 1)]  # (S,nS,n)
    avg_q = q_of_state.mean(axis=0)                          # (nS, n) unweighted
    w = visits.sum(axis=0)
    w = w / w.sum()
    # Visit-weighted average across sessions, per state: a state that only a
    # few sessions ever reach should be described by THOSE sessions' strategies.
    # States no session ever reaches fall back to the unweighted mean (they are
    # drawn faded in Figure 3 anyway).
    tot = visits.sum(axis=0, keepdims=True)
    vw = np.divide(visits, tot, out=np.zeros_like(visits), where=tot > 0)
    wavg_q = np.einsum("sk,skn->kn", vw, q_of_state)
    dead = (tot[0] == 0)
    wavg_q[dead] = avg_q[dead]

    # For the rich state the x-axis is the mean reference-node price of each
    # state, so Figure 3 stays readable and comparable with the price-only run.
    price_axis = (mk.rich_state_value if mon == "rich" else mk.price_state_value)
    # Price bins are already in price order, but a rich state index is a
    # lexicographic code over (price, pocket LMP, congestion, demand) and is NOT.
    # Everything downstream plots against price, so sort here once and hand back
    # the permutation for callers that carry their own state-indexed arrays.
    order = np.argsort(price_axis, kind="stable")
    return {
        "price": price_axis[order],
        "avg_q": avg_q[order],               # (nS, n)
        "weighted_avg_q": wavg_q[order],     # (nS, n)
        "state_weight": w[order],            # (nS,)
        "order": order,                      # state index -> position on the price axis
        "monitoring": mon,
        "market": mk,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Post-convergence analysis 2: FORCED DEVIATION (paper Figure 4)
# ---------------------------------------------------------------------------
def deviation_experiment(name: str, deviator: int = 0, demand: str = "high",
                         pre: int = 20, post: int = 25, mode: str = "best_response",
                         dev_steps: int = 4, n_phases: int = 12, seed: int = 99):
    """Impulse response to an exogenous output expansion by one algorithm.

    Following the paper: starting from the converged behaviour, ONE algorithm is
    forced to defect by expanding production for a single period while the others
    keep playing their learned strategy; from the next period on the cheater also
    reverts to its learned strategy. Demand is FROZEN in the high or low state
    throughout, so the only thing moving the price is the deviation itself.

    mode = "best_response": the deviator plays its static profit-maximising
           output against the rivals' current outputs (clipped to the grid) --
           this is the genuinely tempting one-shot defection.
    mode = "steps": the deviator adds `dev_steps` grid steps to its output.
    """
    meta, z = load(name)
    mk = build_market(deterministic=meta["market"]["deterministic"],
                      k=meta["market"]["k"], xi=meta["market"]["xi"],
                      m=meta["market"]["m"], h=meta["market"]["h"])
    # Whatever the cell was trained under: the rest of this routine only needs
    # `next_state`, which both state spaces provide.
    space = STATE_SPACES[meta["config"]["monitoring"]](mk)
    greedy = z["greedy"]
    S0, n, _ = greedy.shape
    conv = z["converged"].astype(bool)
    if conv.any():
        greedy, start = greedy[conv], z["final_state"][conv]
    else:
        start = z["final_state"]
    S0 = greedy.shape[0]
    # With demand frozen and the policies deterministic, each session settles
    # into a CYCLE. Sessions sitting at different phases of their own cycle
    # would leave a sawtooth in the averaged plot that has nothing to do with
    # the deviation. So we replicate every session `n_phases` times, offset by
    # 0, 1, ..., n_phases-1 periods, which averages the cycle out EXACTLY and
    # leaves only the common response to the period-0 shock.
    greedy = np.repeat(greedy, n_phases, axis=0)
    start = np.repeat(start, n_phases)
    offset = np.tile(np.arange(n_phases), S0)
    S = greedy.shape[0]
    rowS = np.arange(S)[:, None]
    ar_n = np.arange(n)

    if mk.h > 1:
        u_idx = {"high": mk.h - 1, "low": 0}[demand]
    else:
        u_idx = 0

    def step(state, force=None):
        a = greedy[rowS, ar_n[None, :], state[:, None]].astype(np.int64)
        if force is not None:
            a = a.copy()
            a[:, deviator] = force(a)
        J = mk.pidx(a)
        ui = np.full(S, u_idx)
        pi = mk.profit_p[ar_n[None, :], J[:, None], ui[:, None]]
        return a, J, pi, space.next_state(a, J, ui)

    def forced(a):
        if mode == "steps":
            return np.clip(a[:, deviator] + dev_steps, 0,
                           int(mk.n_actions[deviator]) - 1)
        # static best response of `deviator` to the rivals' current actions
        ka = int(mk.n_actions[deviator])
        dev = mk.dev_pidx(mk.pidx(a)[:, None], deviator, np.arange(ka)[None, :])
        pay = mk.exp_profit_p[deviator][dev]
        return pay.argmax(axis=1)

    # --- settle onto the (frozen-demand) limit path, then spread the phases --
    state = start.copy()
    for _ in range(300):
        _, _, _, state = step(state)
    for p in range(n_phases):
        _, _, _, nxt = step(state)
        state = np.where(offset > p, nxt, state)

    rec = {"q": [], "profit": [], "price": []}
    base = {"q": [], "profit": [], "price": []}

    def record(store, a, J, pi):
        store["q"].append(mk.q_agent[ar_n[None, :], a].mean(axis=0))
        store["profit"].append(pi.mean(axis=0))
        store["price"].append(float(mk.price_p[J, u_idx].mean()))

    # --- run the deviation path and the no-deviation counterfactual side by
    #     side from the IDENTICAL state, so the punishment is isolated ---------
    s_dev, s_base = state.copy(), state.copy()
    for _ in range(pre):
        a, J, pi, s_dev = step(s_dev)
        record(rec, a, J, pi)
        a, J, pi, s_base = step(s_base)
        record(base, a, J, pi)

    a, J, pi, s_dev = step(s_dev, force=forced)      # the forced deviation
    record(rec, a, J, pi)
    a, J, pi, s_base = step(s_base)                  # counterfactual: no deviation
    record(base, a, J, pi)
    dev_period = pre

    for _ in range(post):
        a, J, pi, s_dev = step(s_dev)
        record(rec, a, J, pi)
        a, J, pi, s_base = step(s_base)
        record(base, a, J, pi)

    return {
        "q": np.array(rec["q"]),                 # (pre+1+post, n)  # noqa: E501
        "profit": np.array(rec["profit"]),
        "price": np.array(rec["price"]),
        "q_nodev": np.array(base["q"]),          # counterfactual, same sessions
        "profit_nodev": np.array(base["profit"]),
        "price_nodev": np.array(base["price"]),
        "dev_period": dev_period,
        "deviator": deviator,
        "demand": demand,
        "n_sessions": S0,
        "n_phases": n_phases,
        "market": mk,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Post-convergence analysis 3: IS THE DEVIATION ACTUALLY UNPROFITABLE?
# ---------------------------------------------------------------------------
def deviation_value_test(name: str, deviator: int = 0, demand: str = "high",
                         horizon: int = 200, n_phases: int = 12,
                         mode: str = "best_response", dev_steps: int = 4):
    """The incentive-compatibility check the profit level alone cannot give.

    Supra-Nash profits could mean two very different things: the algorithms are
    genuinely colluding (an output expansion is *deterred* by the punishment it
    triggers), or they simply failed to optimise. This separates them.

    For every converged session we run two paths from the IDENTICAL state --
    one where firm `deviator` is forced to expand output for a single period,
    one where it is not -- and compare the discounted payoff streams:

        V_dev - V_stay = sum_t delta^t (pi_dev,t - pi_stay,t)

    A POSITIVE one-period gain with a NEGATIVE discounted total is exactly the
    signature of collusion sustained by punishment: cheating pays today and
    loses over the punishment phase. Demand is frozen so the comparison is not
    contaminated by shocks.
    """
    meta, z = load(name)
    mk = build_market(deterministic=meta["market"]["deterministic"],
                      k=meta["market"]["k"], xi=meta["market"]["xi"],
                      m=meta["market"]["m"], h=meta["market"]["h"])
    space = STATE_SPACES[meta["config"]["monitoring"]](mk)
    delta = meta["config"]["delta"]

    greedy = z["greedy"]
    conv = z["converged"].astype(bool)
    start = z["final_state"]
    if conv.any():
        greedy, start = greedy[conv], start[conv]
    S0, n, _ = greedy.shape
    greedy = np.repeat(greedy, n_phases, axis=0)
    start = np.repeat(start, n_phases)
    offset = np.tile(np.arange(n_phases), S0)
    S = greedy.shape[0]
    rowS = np.arange(S)[:, None]
    ar_n = np.arange(n)
    u_idx = ({"high": mk.h - 1, "low": 0}[demand]) if mk.h > 1 else 0

    def step(state, force=None):
        a = greedy[rowS, ar_n[None, :], state[:, None]].astype(np.int64)
        if force is not None:
            a = a.copy()
            a[:, deviator] = force(a)
        J = mk.pidx(a)
        ui = np.full(S, u_idx)
        pi = mk.profit_p[ar_n[None, :], J[:, None], ui[:, None]]
        return a, J, pi, space.next_state(a, J, ui)

    def forced(a):
        if mode == "steps":
            return np.clip(a[:, deviator] + dev_steps, 0,
                           int(mk.n_actions[deviator]) - 1)
        ka = int(mk.n_actions[deviator])
        dev = mk.dev_pidx(mk.pidx(a)[:, None], deviator, np.arange(ka)[None, :])
        pay = mk.exp_profit_p[deviator][dev]
        return pay.argmax(axis=1)

    state = start.copy()
    for _ in range(300):
        _, _, _, state = step(state)
    for p in range(n_phases):
        _, _, _, nxt = step(state)
        state = np.where(offset > p, nxt, state)

    s_dev, s_base = state.copy(), state.copy()
    a, J, pi_d, s_dev = step(s_dev, force=forced)
    _, _, pi_b, s_base = step(s_base)
    one_period = (pi_d - pi_b)[:, deviator]
    disc = one_period.copy()
    path_d, path_b = [pi_d[:, deviator]], [pi_b[:, deviator]]
    for t in range(1, horizon):
        _, _, pi_d, s_dev = step(s_dev)
        _, _, pi_b, s_base = step(s_base)
        disc += (delta ** t) * (pi_d - pi_b)[:, deviator]
        path_d.append(pi_d[:, deviator]); path_b.append(pi_b[:, deviator])

    # collapse the phase replicates back to one number per session
    one_s = one_period.reshape(S0, n_phases).mean(axis=1)
    disc_s = disc.reshape(S0, n_phases).mean(axis=1)
    return {
        "deviator": deviator,
        "demand": demand,
        "n_sessions": int(S0),
        "one_period_gain": float(one_s.mean()),
        "one_period_gain_positive_frac": float((one_s > 0).mean()),
        "discounted_gain": float(disc_s.mean()),
        "deterred_frac": float((disc_s < 0).mean()),
        "one_period_per_session": one_s,
        "discounted_per_session": disc_s,
        "profit_path_dev": np.array(path_d).mean(axis=1),
        "profit_path_stay": np.array(path_b).mean(axis=1),
        "delta": delta,
        "market": mk,
    }
