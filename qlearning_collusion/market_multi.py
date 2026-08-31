"""
Discrete Cournot game for the TWO-FIRM / THREE-PLANT market
(MARKET_CONFIG=two_firm): firm 0 owns a cheap base unit at node 0 and an
expensive peaker at the node-1 hub; firm 1 owns one mid-cost plant at the hub.

This is the structural analogue of Calvano-Calzolari-Denicolo-Pastorello (2021)
"Algorithmic collusion with imperfect monitoring", whose baseline is a DUOPOLY --
so this market, not the three-firm one, is the direct counterpart of their
headline cell.


HOW THE TWO-PLANT FIRM'S ACTION SPACE IS DISCRETISED
====================================================
The obvious worry is that firm 0 has two decision variables while the Q-learning
algorithm gives each agent one scalar action. Three options exist: (a) a 2-D
product grid (k^2 = 225 actions), (b) a scalar total with some fixed internal
split rule, (c) two cooperative learners inside the firm. Option (b) is normally
unsafe, because a fixed split rule can put the Nash and monopoly benchmarks out
of reach and so corrupt the Delta denominator.

Here option (b) is not merely safe, it is EXACT, for a reason specific to this
market: **no transmission line congests anywhere on the relevant action range**
(verified: max |shadow price| = 3e-6 over every grid profile x every demand
state). With no congestion the DC-OPF returns a single uniform LMP, so firm 0's
two plants are paid the SAME price. Its internal allocation problem therefore
reduces to minimising its own production cost for a given total -- there is no
price-arbitrage motive left -- and least-cost dispatch is exactly
profit-maximising. Measured directly: the best attainable gain from re-splitting
away from least-cost dispatch is $0.000000 across all profiles and shock states
(`verify_split_optimality`).

So firm 0's action is its TOTAL output on a k-point grid, dispatched internally
by equalising marginal cost across its own plants (water-filling, capacity
clipped). This
  * keeps ONE scalar action per agent, exactly as in the paper;
  * keeps |A_i| = k = 15 for both firms, so |S| = k^2 = 225 under perfect
    monitoring -- identical to the paper's duopoly baseline;
  * loses nothing relative to a 2-D action grid (proved above);
  * preserves Cournot simultaneity, because the split depends only on the firm's
    OWN action, never on the rival's.
It also reproduces both benchmark splits exactly: least-cost dispatch of 115 MW
gives (115, 0) = the monopoly allocation, and of 158 MW gives (115, 43) = the
Nash allocation.


WHICH NASH? (this market breaks the paper's LCP benchmark)
==========================================================
The repo's Nash benchmark is the networked-Cournot LCP of the source paper
(eqs. 39-45), in which each firm's stationarity condition uses the inverse-demand
slope of the node its plant sits on. That is correct when a firm's plants all sit
at one node -- which is why it agrees with a direct best-response calculation on
the three-firm hub market.

It is NOT correct here, because firm 0 straddles two nodes. The LCP has firm 0
withhold at node 0 as though it were a local monopolist there (53.79 MW), but
node 0 is price-coupled to the rest of the network -- the line never binds -- so
that withholding buys nothing. Consequence: at the LCP "Nash" point, firm 0 can
raise its profit by 25.2% ($3,007.85 -> $3,764.81) by unilaterally deviating.
The LCP point is a conjectural-variations equilibrium, not a Nash equilibrium of
the game the algorithms actually play.

Using it as the Delta denominator would manufacture apparent collusion: agents
that merely learn to best-respond, with no coordination whatsoever, would already
score Delta > 0. So Delta here is measured against the TRUE best-response Nash,
computed by iterating exact best responses on the DC-OPF game
(`best_response_nash`):

    plants (115.00, 43.00, 181.50), total 339.50 MW, LMP $39.13,
    profits F0 $3,524.54 / F1 $3,357.80, total $6,882.34.

Both Nash notions are reported side by side in `describe()`. Set nash_mode="lcp"
to reproduce the docx table's denominator instead.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from iso_market.market_env import (
    ElectricityMarketEnv,
    PLANTS,
    NUM_FIRMS,
    NUM_NODES,
    FIRM_PLANT_IDX,
)
from iso_market.node_network import MARKET
from qlearning_collusion.market import continuous_benchmarks, CACHE_DIR
from qlearning_collusion.profile_api import ProfileAPI


# ---------------------------------------------------------------------------
def least_cost_split(total: float, mc: np.ndarray, qc: np.ndarray,
                     cap: np.ndarray) -> np.ndarray:
    """Cheapest way to produce `total` from plants with C_p(g) = mc*g + qc*g^2/2.

    Water-filling on the common marginal cost lambda:
        g_p(lambda) = clip((lambda - mc_p) / qc_p, 0, cap_p)
    solved by bisection on lambda. Profit-maximising too whenever all the plants
    face the same price (see module docstring).
    """
    total = float(np.clip(total, 0.0, cap.sum()))
    lo, hi = float(mc.min()), float((mc + qc * cap).max()) + 1.0
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        g = np.clip((lam - mc) / qc, 0.0, cap)
        if g.sum() < total:
            lo = lam
        else:
            hi = lam
    g = np.clip((0.5 * (lo + hi) - mc) / qc, 0.0, cap)
    s = g.sum()
    if s > 1e-12:                       # kill residual bisection error
        g *= total / s
    return np.clip(g, 0.0, cap)


# ---------------------------------------------------------------------------
@dataclass
class DiscreteMarketMulti(ProfileAPI):
    """The finite Cournot game the Q-learners play on the two-firm market."""

    k: int = 15
    xi: float = 0.2
    shock_steps_m: float = 8.0
    h: int = 2
    deterministic: bool = False
    price_node: Optional[int] = None
    nash_mode: str = "br"                 # "br" (true Nash) | "lcp" (paper's)

    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.nash_mode not in ("br", "lcp"):
            raise ValueError("nash_mode must be 'br' or 'lcp'")
        self.env = ElectricityMarketEnv()
        self.n_plants = len(PLANTS)
        self.n_agents = self.n_firms = NUM_FIRMS
        self.plant_node = np.array([p["node"] for p in PLANTS])
        self.plant_firm = np.array([p["firm"] for p in PLANTS])
        self.mc = np.array([p["mc"] for p in PLANTS])
        self.qc = np.array([p["qc"] for p in PLANTS])
        self.cap = np.array([p["cap"] for p in PLANTS])
        self.firm_plants = [list(FIRM_PLANT_IDX[f]) for f in range(self.n_firms)]
        self.firm_cap = np.array([self.cap[pl].sum() for pl in self.firm_plants])

        if self.deterministic:
            self.h = 1
        if self.price_node is None:
            # The single "market price" the imperfect-monitoring state is built
            # from. Take the GENERATION node with the largest demand intercept,
            # i.e. the main load centre firms actually sell into. On the hub
            # market that is the hub; with one plant per node it is node 2.
            from iso_market.node_network import Q0 as _Q0
            gen_nodes = np.unique(self.plant_node)
            self.price_node = int(gen_nodes[np.argmax(_Q0[gen_nodes])])

        bm = continuous_benchmarks(self.env)
        self.bench_continuous = bm
        self.q_mono_plant = np.array(bm["monopoly"]["gens"])
        self.q_lcp_plant = np.array(bm["nash"]["gens"])
        self.q_mono_firm = np.array([self.q_mono_plant[pl].sum() for pl in self.firm_plants])
        self.q_lcp_firm = np.array([self.q_lcp_plant[pl].sum() for pl in self.firm_plants])

        self._continuous_equilibria()

        self._build_grids()
        self._enumerate_profiles()
        self._calibrate_shock()
        self._build_price_table()
        self._build_price_states()
        self._build_payoffs()

    # ------------------------------------------------------------------
    # market primitives
    # ------------------------------------------------------------------
    def _plant_gen(self, firm_totals) -> np.ndarray:
        g = np.zeros(self.n_plants)
        for f, pl in enumerate(self.firm_plants):
            g[pl] = least_cost_split(firm_totals[f], self.mc[pl], self.qc[pl],
                                     self.cap[pl])
        return g

    def _clear_plants(self, plant_gen: np.ndarray, u: float = 0.0):
        node_gen = np.zeros(NUM_NODES)
        np.add.at(node_gen, self.plant_node, plant_gen)
        self.env._demand_u = float(u)
        lmps, demand, flows, shadow = self.env._clear_market(node_gen)
        if lmps is None:
            raise RuntimeError(f"DC-OPF infeasible at {plant_gen}, u={u}")
        self.env._demand_u = 0.0
        return lmps.copy(), demand.copy(), flows.copy(), shadow.copy()

    def _firm_profits(self, firm_totals, u: float = 0.0):
        g = self._plant_gen(firm_totals)
        lmps, demand, flows, shadow = self._clear_plants(g, u)
        plant_pi = lmps[self.plant_node] * g - (self.mc * g + 0.5 * self.qc * g ** 2)
        firm_pi = np.array([plant_pi[pl].sum() for pl in self.firm_plants])
        return firm_pi, g, lmps, demand, flows, shadow

    # ------------------------------------------------------------------
    def best_response_nash(self, tol: float = 1e-5) -> dict:
        """True Nash of the DC-OPF game, by iterated exact best responses.

        Each firm maximises its OWN TOTAL profit over its total output (with
        least-cost internal dispatch), taking the rival's total as given. This is
        the equilibrium concept the Q-learners' stage game actually has -- unlike
        the LCP, which imposes a node-local conjecture that is wrong for a firm
        whose plants straddle two nodes (see module docstring).
        """
        q = self.q_lcp_firm.copy()
        for sweep in range(200):
            prev = q.copy()
            for f in range(self.n_firms):
                grid = np.linspace(0.0, self.firm_cap[f], 241)
                for _ in range(4):                     # coarse -> fine
                    vals = []
                    for t in grid:
                        cand = q.copy(); cand[f] = t
                        vals.append(self._firm_profits(cand)[0][f])
                    j = int(np.argmax(vals))
                    step = grid[1] - grid[0]
                    grid = np.linspace(max(0.0, grid[j] - step),
                                       min(self.firm_cap[f], grid[j] + step), 41)
                q[f] = grid[int(np.argmax([
                    self._firm_profits(np.where(np.arange(self.n_firms) == f, t, q))[0][f]
                    for t in grid]))]
            if np.abs(q - prev).max() < tol:
                break
        pi, g, lmps, demand, flows, shadow = self._firm_profits(q)
        return {
            "firm_gen": q.tolist(),
            "plant_gens": g.tolist(),
            "total_gen": float(g.sum()),
            "profits": pi.tolist(),
            "total_profit": float(pi.sum()),
            "lmps": lmps.tolist(),
            "avg_lmp": float(np.sum(lmps * demand) / np.sum(demand)),
            "max_shadow": float(np.abs(shadow).max()),
            "sweeps": sweep,
        }

    def _continuous_equilibria(self):
        """Best-response Nash + direct joint monopoly, cached to disk.

        These are properties of the CONTINUOUS market -- they do not depend on
        k, xi, h or whether demand is deterministic -- but each costs tens of
        thousands of DC-OPF solves. Every downstream analysis rebuilds the
        market, so without a cache the benchmark solves, not the learning,
        dominate the runtime.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"equilibria_{MARKET}_{self.nash_mode}.json")
        if os.path.exists(path):
            with open(path) as fh:
                c = json.load(fh)
            self.br_nash = c["br_nash"]
            self.mono_direct = c["mono_direct"]
            self._br_dev_gain = c["br_dev_gain"]
            self._lcp_dev_gain = c["lcp_dev_gain"]
            self.mono_mpec_profit = c["mono_mpec_profit"]
            self.mono_source = c["mono_source"]
        else:
            self.br_nash = self.best_response_nash()
            # Is each candidate actually an equilibrium of THIS game? (max_dev_gain)
            self._br_dev_gain = self.max_dev_gain(np.array(self.br_nash["firm_gen"]))
            self._lcp_dev_gain = self.max_dev_gain(self.q_lcp_firm)
            # Cross-check the MPEC monopoly against a direct solve on this very
            # game and keep the better one -- the action grid is centred on it, so
            # a stalled MPEC would silently mis-centre everything downstream.
            self.mono_direct = self.joint_monopoly_direct()
            self.mono_mpec_profit = float(np.sum(self._firm_profits(self.q_mono_firm)[0]))
            self.mono_source = ("direct"
                                if self.mono_direct["total_profit"] > self.mono_mpec_profit + 1e-6
                                else "mpec")
            with open(path, "w") as fh:
                json.dump(dict(br_nash=self.br_nash, mono_direct=self.mono_direct,
                               br_dev_gain=self._br_dev_gain,
                               lcp_dev_gain=self._lcp_dev_gain,
                               mono_mpec_profit=self.mono_mpec_profit,
                               mono_source=self.mono_source), fh, indent=1)

        self.q_br_firm = np.array(self.br_nash["firm_gen"])
        if self.mono_source == "direct":
            self.q_mono_firm = np.array(self.mono_direct["firm_gen"])
            self.q_mono_plant = np.array(self.mono_direct["plant_gens"])

    def max_dev_gain(self, firm_totals, n: int = 241) -> float:
        """Largest profit ANY firm can win by unilaterally changing its output.

        Zero identifies a genuine Nash equilibrium of the DC-OPF game; a large
        value says the candidate point is an equilibrium of some other game (the
        LCP's node-local conjecture, say) and must not be used as the Delta
        denominator, because agents that merely learn to best-respond would then
        score Delta > 0 with no coordination at all.
        """
        q = np.asarray(firm_totals, float)
        base = self._firm_profits(q)[0]
        worst = 0.0
        for f in range(self.n_firms):
            for t in np.linspace(0.0, self.firm_cap[f], n):
                c = q.copy(); c[f] = t
                worst = max(worst, float(self._firm_profits(c)[0][f] - base[f]))
        return worst

    def joint_monopoly_direct(self, starts=None, tol: float = 1e-4,
                              sweeps: int = 40, n: int = 81) -> dict:
        """Joint monopoly solved directly on the DC-OPF game, not via the MPEC.

        The cartel picks every firm's total output to maximise the SUM of firm
        profits, with prices coming from the same ISO clearing the Q-learners
        face. Coordinate ascent with grid refinement, from several starts.

        This exists as a cross-check on `compute_monopoly_benchmark`, which
        solves the wheeling-fee MPEC with SLSQP multistart. On the hub market the
        two agree to the digit; once plants sit on different nodes the MPEC's
        complementarity relaxation is much harder and can stall at a non-optimal
        point, which would mis-centre the whole action grid. `__post_init__`
        keeps whichever of the two actually earns more.
        """
        cands = list(starts or [])
        cands += [self.q_mono_firm.copy(), self.q_lcp_firm.copy(),
                  0.5 * self.firm_cap, 0.3 * self.firm_cap]
        best_q, best_v = None, -np.inf
        for q0 in cands:
            q = np.clip(np.asarray(q0, float).copy(), 0.0, self.firm_cap)
            for sw in range(sweeps):
                prev = q.copy()
                for f in range(self.n_firms):
                    lo, hi = 0.0, self.firm_cap[f]
                    for _ in range(3):
                        grid = np.linspace(lo, hi, n)
                        vals = []
                        for t in grid:
                            c = q.copy(); c[f] = t
                            vals.append(self._firm_profits(c)[0].sum())
                        j = int(np.argmax(vals))
                        st = grid[1] - grid[0]
                        lo, hi = max(0.0, grid[j] - st), min(self.firm_cap[f], grid[j] + st)
                    q[f] = 0.5 * (lo + hi)
                if np.abs(q - prev).max() < tol:
                    break
            v = float(self._firm_profits(q)[0].sum())
            if v > best_v:
                best_v, best_q = v, q.copy()
        pi, g, lmps, demand, flows, shadow = self._firm_profits(best_q)
        return {
            "firm_gen": best_q.tolist(), "plant_gens": g.tolist(),
            "total_gen": float(g.sum()), "profits": pi.tolist(),
            "total_profit": float(pi.sum()), "lmps": lmps.tolist(),
            "avg_lmp": float(np.sum(lmps * demand) / np.sum(demand)),
            "max_shadow": float(np.abs(shadow).max()),
        }

    def verify_split_optimality(self, n_probe: int = 40) -> dict:
        """Largest gain firm 0 could get by abandoning least-cost dispatch.

        Sweeps every grid profile and every demand state; for each, re-allocates
        the firm's committed total across its own plants on a fine grid. Returns
        the best improvement found -- 0.0 confirms the 1-D reduction is exact.
        """
        worst, where = 0.0, None
        for f, pl in enumerate(self.firm_plants):
            if len(pl) < 2:
                continue
            for a in range(self.k):
                for b in range(self.k):
                    tot = np.array([self.q_firm_grid[0][a], self.q_firm_grid[1][b]])
                    for u in self.u_levels:
                        base = self._firm_profits(tot, u)[0][f]
                        g0 = self._plant_gen(tot)
                        T = g0[pl].sum()
                        for x in np.linspace(max(0.0, T - self.cap[pl[1]]),
                                             min(self.cap[pl[0]], T), n_probe):
                            g = self._plant_gen(tot)
                            g[pl[0]], g[pl[1]] = x, T - x
                            lmps = self._clear_plants(g, u)[0]
                            pp = lmps[self.plant_node] * g - (self.mc * g + 0.5 * self.qc * g ** 2)
                            gain = pp[pl].sum() - base
                            if gain > worst:
                                worst, where = float(gain), (a, b, float(u), float(x))
        return {"max_gain_from_resplit": worst, "at": where}

    # ------------------------------------------------------------------
    def _build_grids(self):
        """Per-FIRM total-output grid spanning [q^M, q^N] extended by xi.

        Paper: A = {70, ..., 105} with q^M = 75, q^C = 100, i.e. the grid spans
        the monopoly-to-Nash range extended by xi = 0.2 of it on each side.
        """
        q_nash = self.q_br_firm if self.nash_mode == "br" else self.q_lcp_firm
        self.q_nash_firm = q_nash
        lo_c = np.minimum(self.q_mono_firm, q_nash)
        hi_c = np.maximum(self.q_mono_firm, q_nash)
        gap = hi_c - lo_c
        if np.any(gap <= 1e-9):
            raise ValueError(f"degenerate monopoly/Nash range per firm: {gap}")
        lo = np.maximum(0.0, lo_c - self.xi * gap)
        hi = np.minimum(self.firm_cap, hi_c + self.xi * gap)
        self.q_firm_grid = np.stack(
            [np.linspace(lo[f], hi[f], self.k) for f in range(self.n_firms)]
        )                                                    # (n_firms, k)
        self.firm_step = (hi - lo) / (self.k - 1)
        self.grid_lo, self.grid_hi = lo, hi
        # per-plant dispatch of every grid action
        self.grid_plant_gen = np.zeros((self.n_firms, self.k, self.n_plants))
        for f in range(self.n_firms):
            for a in range(self.k):
                pl = self.firm_plants[f]
                self.grid_plant_gen[f, a, pl] = least_cost_split(
                    self.q_firm_grid[f, a], self.mc[pl], self.qc[pl], self.cap[pl])

    # ------------------------------------------------------------------
    def _enumerate_profiles(self):
        self.n_actions = np.full(self.n_firms, self.k, dtype=np.int64)
        self.max_actions = self.k
        self.n_profiles = self.k ** self.n_firms
        radix = (self.k ** np.arange(self.n_firms - 1, -1, -1)).astype(np.int64)
        self.agent_profile_weight = np.stack(
            [np.arange(self.k) * radix[f] for f in range(self.n_firms)])
        self.valid_action = np.ones((self.n_firms, self.k), dtype=bool)
        self.q_agent = self.q_firm_grid.copy()

        acts = np.array(list(itertools.product(range(self.k), repeat=self.n_firms)),
                        dtype=np.int64)                       # (P, n_firms)
        self._profile_acts = acts
        self._own = [acts[:, f] for f in range(self.n_firms)]
        self._dev_base = [np.arange(self.n_profiles) - self._own[f] * radix[f]
                          for f in range(self.n_firms)]

        # (P, n_plants) dispatch, and (P,) totals
        self.profile_plant_gen = sum(
            self.grid_plant_gen[f][acts[:, f]] for f in range(self.n_firms))
        self.total_gen_p = self.profile_plant_gen.sum(axis=1)
        self.firm_gen_p = np.stack(
            [self.q_firm_grid[f][acts[:, f]] for f in range(self.n_firms)])

    def agent_action_of_profile(self, i: int) -> np.ndarray:
        return self._own[i]

    # ------------------------------------------------------------------
    def _calibrate_shock(self):
        """Intercept shift u sized so the hub LMP moves by m output steps."""
        if self.deterministic or self.shock_steps_m <= 0:
            self.u_max = 0.0
            self.u_levels = np.zeros(1)
            self.price_step_per_output_step = None
            self.v_ref = float(self.firm_step.mean())
            return
        ref = np.array([self.q_nash_firm[f] for f in range(self.n_firms)])
        v_ref = float(self.firm_step.mean())
        self.v_ref = v_ref
        p0 = self._firm_profits(ref)[2][self.price_node]
        bump = ref.copy(); bump[0] += v_ref
        p1 = self._firm_profits(bump)[2][self.price_node]
        step = abs(p1 - p0)
        self.price_step_per_output_step = step
        target = self.shock_steps_m * step
        lo, hi = 0.0, 80.0
        for _ in range(80):
            u = 0.5 * (lo + hi)
            pu = self._firm_profits(ref, u)[2][self.price_node]
            if (pu - p0) < target:
                lo = u
            else:
                hi = u
        self.u_max = 0.5 * (lo + hi)
        self.u_levels = (np.linspace(-self.u_max, self.u_max, self.h)
                         if self.h > 1 else np.zeros(1))

    # ------------------------------------------------------------------
    def _table_cache_path(self) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        raw = json.dumps({
            "market": MARKET, "k": self.k, "xi": self.xi, "h": self.h,
            "m": self.shock_steps_m, "det": self.deterministic,
            "nash": self.nash_mode, "pn": self.price_node,
            "u": round(float(self.u_max), 9),
            "grid": np.round(self.q_firm_grid, 9).tolist(),
        }, sort_keys=True)
        return os.path.join(CACHE_DIR,
                            f"tables_{hashlib.md5(raw.encode()).hexdigest()[:12]}.npz")

    def _build_price_table(self):
        """Clear the DC-OPF once for every (action profile, demand state).

        With one plant per firm on three different nodes this is k^n * h solves
        (3,375 x 2 at the baseline), which is why the result is cached to disk:
        every downstream analysis (limit strategy, deviation, the discount-factor
        sweep, every figure) rebuilds the market, and re-solving each time would
        dominate the runtime.
        """
        cache = self._table_cache_path()
        if os.path.exists(cache):
            z = np.load(cache)
            for nm in ("nodal_lmps", "avg_lmp_p", "demand_tab", "flows_tab",
                       "shadow_tab", "plant_profit_p"):
                setattr(self, nm, z[nm])
            self.price_p = self.nodal_lmps[:, :, self.price_node].copy()
            self.max_shadow_on_grid = float(np.abs(self.shadow_tab).max())
            return

        P, hn = self.n_profiles, self.h
        self.nodal_lmps = np.zeros((P, hn, NUM_NODES))
        self.avg_lmp_p = np.zeros((P, hn))
        self.demand_tab = np.zeros((P, hn, NUM_NODES))
        self.flows_tab = np.zeros((P, hn, self.env.num_lines))
        self.shadow_tab = np.zeros((P, hn, self.env.num_lines))
        self.plant_profit_p = np.zeros((P, hn, self.n_plants))
        for j in range(P):
            g = self.profile_plant_gen[j]
            for iu, u in enumerate(self.u_levels):
                lmps, dem, fl, sh = self._clear_plants(g, u)
                self.nodal_lmps[j, iu] = lmps
                self.avg_lmp_p[j, iu] = float(np.sum(lmps * dem) / np.sum(dem))
                self.demand_tab[j, iu] = dem
                self.flows_tab[j, iu] = fl
                self.shadow_tab[j, iu] = sh
                self.plant_profit_p[j, iu] = (
                    lmps[self.plant_node] * g - (self.mc * g + 0.5 * self.qc * g ** 2))
        self.price_p = self.nodal_lmps[:, :, self.price_node].copy()
        self.max_shadow_on_grid = float(np.abs(self.shadow_tab).max())
        np.savez_compressed(
            cache, nodal_lmps=self.nodal_lmps, avg_lmp_p=self.avg_lmp_p,
            demand_tab=self.demand_tab, flows_tab=self.flows_tab,
            shadow_tab=self.shadow_tab, plant_profit_p=self.plant_profit_p)

    # ------------------------------------------------------------------
    def _build_price_states(self):
        """Uniform bins on the hub LMP, at the paper's resolution n(k-1)+m+1."""
        m = int(round(self.shock_steps_m)) if not self.deterministic else 0
        self.n_price_states = self.n_firms * (self.k - 1) + m + 1
        p_lo, p_hi = float(self.price_p.min()), float(self.price_p.max())
        self.price_bin_width = (p_hi - p_lo) / (self.n_price_states - 1)
        self.price_lo, self.price_hi = p_lo, p_hi
        idx = np.rint((self.price_p - p_lo) / self.price_bin_width).astype(np.int64)
        self.price_state_p = np.clip(idx, 0, self.n_price_states - 1)
        self.price_state_value = p_lo + self.price_bin_width * np.arange(self.n_price_states)
        self.reachable_states = np.unique(self.price_state_p)

    # ------------------------------------------------------------------
    def _build_payoffs(self):
        self.profit_p = np.stack([
            self.plant_profit_p[:, :, self.firm_plants[f]].sum(axis=2)
            for f in range(self.n_firms)])                    # (n_firms, P, h)
        self.exp_profit_p = self.profit_p.mean(axis=2)
        self.exp_price_p = self.price_p.mean(axis=1)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        gb = self.grid_benchmarks()
        mr = self.monitoring_report()
        L = []
        L.append("=" * 78)
        L.append(f"DISCRETE COURNOT GAME -- NETWORKED MARKET  [MARKET_CONFIG={MARKET}]")
        L.append("=" * 78)
        L.append(f"agents={self.n_firms}  plants={self.n_plants}  k={self.k}  "
                 f"|A_i|={self.k} each  profiles={self.n_profiles}  "
                 f"|S|(perfect)={self.k**self.n_firms}")
        L.append(f"xi={self.xi}  h={self.h}  m={self.shock_steps_m}  "
                 f"deterministic={self.deterministic}  price node={self.price_node}  "
                 f"nash_mode={self.nash_mode}")
        L.append("")
        L.append("Continuous benchmarks (repo solvers, per PLANT):")
        for nm in ("competitive", "nash", "monopoly"):
            b = self.bench_continuous[nm]
            L.append(f"  {nm:12s} gen={b['total_gen']:7.2f}  avgLMP=${b['avg_lmp']:6.2f}  "
                     f"pi=${b['total_profit']:8.2f}  plants={[round(x,2) for x in b['gens']]}")
        n = self.br_nash
        L.append(f"  {'nash(BR)':12s} gen={n['total_gen']:7.2f}  avgLMP=${n['avg_lmp']:6.2f}  "
                 f"pi=${n['total_profit']:8.2f}  plants={[round(x,2) for x in n['plant_gens']]}")
        md = self.mono_direct
        L.append(f"  {'mono(direct)':12s} gen={md['total_gen']:7.2f}  avgLMP=${md['avg_lmp']:6.2f}  "
                 f"pi=${md['total_profit']:8.2f}  plants={[round(x,2) for x in md['plant_gens']]}")
        L.append(f"    (MPEC monopoly earns ${self.mono_mpec_profit:.2f}; "
                 f"grid centred on the {self.mono_source.upper()} solution)")
        L.append("")
        L.append("  ^ the LCP 'nash' is NOT an equilibrium of the game the algorithms play:")
        L.append("    it gives each firm its OWN node's inverse-demand slope, which is only")
        L.append("    right when that node is islanded by congestion. Measured max unilateral")
        L.append(f"    deviation gain at the LCP point: ${self._lcp_dev_gain:,.2f} "
                 f"(at the BR point: ${self._br_dev_gain:,.2f}).")
        L.append(f"    Delta is measured against nash({self.nash_mode.upper()}).")
        L.append("")
        L.append("Per-firm action grid (TOTAL MW; internal split = least-cost dispatch):")
        for f in range(self.n_firms):
            pl = self.firm_plants[f]
            L.append(f"  firm {f} (plants {pl}, nodes {[int(x) for x in self.plant_node[pl]]}, "
                     f"cap {self.firm_cap[f]:.0f}): "
                     f"[{self.q_firm_grid[f,0]:7.2f} ... {self.q_firm_grid[f,-1]:7.2f}] "
                     f"step {self.firm_step[f]:5.2f}   "
                     f"(q^M={self.q_mono_firm[f]:6.2f}, q^N={self.q_nash_firm[f]:6.2f})")
        multi = max(len(pl) for pl in self.firm_plants) > 1
        L.append(f"  max |shadow price| over the whole grid: {self.max_shadow_on_grid:.3g}")
        if multi:
            # The 1-D reduction of a multi-plant firm is only exact when its
            # plants are paid the SAME price, i.e. when nothing congests between
            # them (see the module docstring).
            L.append("    ^ with a multi-plant firm this must be ~0 for the least-cost"
                     " split to be exactly profit-maximising")
        else:
            L.append("    ^ every firm owns ONE plant, so there is no internal split to"
                     " make and congestion is simply part of the payoff")
        L.append("")
        L.append("Imperfect monitoring:")
        L.append(f"  intercept shock u = +/- {mr['shock_u_max']:.3f} $/MWh  "
                 f"(levels {[round(x,2) for x in mr['shock_levels']]})")
        if mr["price_impact_per_output_step"]:
            L.append(f"  price impact of ONE output step ({self.v_ref:.2f} MW) = "
                     f"${mr['price_impact_per_output_step']:.3f}/MWh "
                     f"-> shock = {self.shock_steps_m:g} steps (paper m=8)")
        L.append(f"  hub LMP (node {self.price_node}) ${mr['price_range'][0]:.2f} - "
                 f"${mr['price_range'][1]:.2f} in {mr['n_price_states']} bins of "
                 f"${mr['price_bin_width']:.3f}")
        L.append(f"  non-revealing price states: measured {mr['measured_nonrevealing_fraction']:.3f} "
                 f"(paper closed form: {mr['paper_closed_form_fraction']:.3f})")
        L.append("")
        L.append("Grid benchmarks (these define Delta):")
        for nm in ("nash", "monopoly"):
            b = gb[nm]
            L.append(f"  {nm:9s} a={b['actions']} gen={b['total_gen']:7.2f}  "
                     f"hubP=${b['hub_price']:6.2f}  pi=${b['total_profit']:8.2f}  "
                     f"firms={[round(x,1) for x in b['profits']]}")
            L.append(f"            plants={[round(x,2) for x in b['plant_gens']]}")
        L.append(f"  {gb['nash']['kind']}")
        gapt = gb["monopoly"]["total_profit"] - gb["nash"]["total_profit"]
        L.append(f"  Nash->Monopoly profit gap: ${gapt:.1f} "
                 f"({100*gapt/gb['nash']['total_profit']:.1f}%)")
        ir = [gb["monopoly"]["profits"][i] - gb["nash"]["profits"][i]
              for i in range(self.n_firms)]
        L.append(f"  per-firm cartel IR margins: {[round(x,1) for x in ir]} "
                 f"({'ALL POSITIVE' if min(ir) > 0 else 'NOT all positive!'})")
        L.append("=" * 78)
        return "\n".join(L)

    # ------------------------------------------------------------------
    def signature(self) -> str:
        raw = json.dumps({
            "market": "two_firm", "k": self.k, "xi": self.xi,
            "m": self.shock_steps_m, "h": self.h, "det": self.deterministic,
            "pn": self.price_node, "nash": self.nash_mode,
        }, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:10]


if __name__ == "__main__":
    mk = DiscreteMarketMulti()
    print(mk.describe())
    print("\nsplit-optimality check:", mk.verify_split_optimality())
