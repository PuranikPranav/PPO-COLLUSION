"""
Discrete Cournot market on the networked (DC-OPF) topology, built to be the
exact structural analogue of Calvano-Calzolari-Denicolo-Pastorello (2021),
"Algorithmic collusion with imperfect monitoring".

Mapping from the paper's Green-Porter model to this repo's topology
--------------------------------------------------------------------
Paper                                    Here
-----                                    ----
p_t = d_t - (q_1 + ... + q_n)            p_t = LMP at the generation hub,
                                         produced by the ISO's DC-OPF clear of
                                         the 5-node Liu & Hobbs network given
                                         the firms' committed generation.
d_t i.i.d. over {290, 310}               all nodal inverse-demand intercepts
                                         P0_i shift by u_t, i.i.d. over
                                         {-u, ..., +u} (h equiprobable levels).
                                         The shock realises AFTER firms commit.
A_i = {q^1, ..., q^k}, q^{j+1}-q^j = v   per-firm grid of k outputs with a
                                         COMMON step v, each grid centred on
                                         that firm's own [q^M_i, q^C_i] range
                                         (firms here are asymmetric in MC/QC).
s_t = p_{t-1} (imperfect monitoring)     s_t = binned hub LMP of t-1.
s_t = (q_{1,t-1},...,q_{n,t-1}) (perfect) same.

Why the whole market collapses to a lookup table
------------------------------------------------
In the three-firm hub market every firm owns exactly one plant and all plants
sit at node 2, so the ISO's clearing problem depends on the firms' actions only
through the TOTAL hub generation G. With a common action step v,
G = G_0 + v * J with J = sum_i a_i in {0, ..., n(k-1)}. So there are only
n(k-1)+1 distinct clearings per demand state, which we precompute once. The
Q-learning inner loop is then pure array indexing, which is what makes millions
of iterations x hundreds of sessions feasible.

The code below does NOT assume the hub layout: it enumerates the distinct
per-node generation vectors reachable on the grid and caches a clearing for
each. The hub case just happens to collapse to n(k-1)+1 of them.
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

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "_cache")


# ---------------------------------------------------------------------------
# Continuous benchmarks (competitive / Nash-Cournot LCP / joint monopoly)
# ---------------------------------------------------------------------------
def continuous_benchmarks(env: Optional[ElectricityMarketEnv] = None,
                          use_cache: bool = True) -> dict:
    """Competitive, Nash-Cournot (paper eqs. 39-45 LCP) and joint-monopoly points.

    These are the *continuous* benchmarks of the repo's market. They are used to
    CENTRE the discrete action grid; the Delta denominator itself is computed on
    the grid (see DiscreteMarket.grid_benchmarks), because the grid game is the
    game the algorithms actually play.
    """
    from iso_market.node_network import MARKET

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"continuous_benchmarks_{MARKET}.json")
    if use_cache and os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)

    from experiments.ppo import (
        compute_competitive_benchmark,
        compute_cournot_nash_benchmark,
        compute_monopoly_benchmark,
    )

    env = env or ElectricityMarketEnv()
    out = {
        "competitive": compute_competitive_benchmark(env),
        "nash": compute_cournot_nash_benchmark(env),
        "monopoly": compute_monopoly_benchmark(env),
    }
    with open(cache, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


# ---------------------------------------------------------------------------
@dataclass
class DiscreteMarket:
    """The finite Cournot game the Q-learners play.

    Parameters
    ----------
    k : number of output levels per firm (paper baseline: 15).
    xi : how far the grid extends beyond [q^M_i, q^C_i], as a fraction of that
        range, on each side. Paper: A = {70,...,105} with q^M=75, q^C=100, so
        xi = 5/25 = 0.2.
    shock_steps_m : the demand shock's price impact measured in *output steps*
        (the paper's m: d^{j+1}-d^j = m*v). m=8 in the paper's baseline, which
        is what makes an adverse shock confusable with a rival expanding output
        by up to 8 grid steps. The absolute intercept shift u is solved for
        numerically so that the hub LMP moves by m * (per-output-step price
        impact).
    h : number of equiprobable demand levels (paper baseline: 2).
    deterministic : if True the demand shock is switched off entirely (h forced
        to 1). Used for the two "Deterministic Demand" cells of Table I.
    """

    k: int = 15
    xi: float = 0.2
    shock_steps_m: float = 8.0
    h: int = 2
    deterministic: bool = False
    price_node: Optional[int] = None          # default: the firms' own node

    # --- filled in by __post_init__ ---
    n_firms: int = field(init=False)
    v: float = field(init=False)
    q_grid: np.ndarray = field(init=False)     # (n_firms, k) quantities in MW
    cost: np.ndarray = field(init=False)       # (n_firms, k) production cost
    u_levels: np.ndarray = field(init=False)   # (h,) intercept shifts
    price: np.ndarray = field(init=False)      # (J+1, h) hub LMP
    avg_lmp: np.ndarray = field(init=False)    # (J+1, h) qty-weighted avg LMP
    total_gen_grid: np.ndarray = field(init=False)  # (J+1,) total MW
    price_state: np.ndarray = field(init=False)     # (J+1, h) -> state index
    price_state_value: np.ndarray = field(init=False)  # (n_states,) bin centres
    profit: np.ndarray = field(init=False)     # (n_firms, k, J+1, h)

    # ------------------------------------------------------------------
    def __post_init__(self):
        self.env = ElectricityMarketEnv()
        self.n_firms = NUM_FIRMS
        for f in range(self.n_firms):
            if len(FIRM_PLANT_IDX[f]) != 1:
                raise NotImplementedError(
                    "The Calvano Q-learning setup gives each firm ONE scalar action. "
                    f"Firm {f} owns {len(FIRM_PLANT_IDX[f])} plants; use "
                    "MARKET_CONFIG=three_firm (one plant per firm)."
                )
        self.plant_of_firm = np.array([FIRM_PLANT_IDX[f][0] for f in range(self.n_firms)])
        self.node_of_firm = np.array([PLANTS[p]["node"] for p in self.plant_of_firm])
        self.mc = np.array([PLANTS[p]["mc"] for p in self.plant_of_firm])
        self.qc = np.array([PLANTS[p]["qc"] for p in self.plant_of_firm])
        self.cap = np.array([PLANTS[p]["cap"] for p in self.plant_of_firm])

        if self.deterministic:
            self.h = 1

        bm = continuous_benchmarks(self.env)
        self.bench_continuous = bm
        q_nash = np.array([bm["nash"]["gens"][p] for p in self.plant_of_firm])
        q_mono = np.array([bm["monopoly"]["gens"][p] for p in self.plant_of_firm])
        self.q_nash_cont, self.q_mono_cont = q_nash, q_mono

        # ---- action grid -------------------------------------------------
        # Paper: span = (1 + 2*xi) * (q^C - q^M) over k-1 steps. With asymmetric
        # firms we use ONE common step v (the mean firm's step) so that total
        # generation stays on a lattice, and centre each firm's grid on the
        # midpoint of its own [q^M_i, q^C_i].
        gaps = q_nash - q_mono
        if np.any(gaps <= 0):
            raise ValueError(f"Nash output must exceed monopoly output per firm; got {gaps}")
        self.v = float((1.0 + 2.0 * self.xi) * gaps.mean() / (self.k - 1))
        mid = 0.5 * (q_nash + q_mono)
        offs = (np.arange(self.k) - (self.k - 1) / 2.0) * self.v
        self.q_grid = np.clip(mid[:, None] + offs[None, :], 0.0, self.cap[:, None])
        if not np.allclose(self.q_grid, mid[:, None] + offs[None, :]):
            raise ValueError("Action grid hit a capacity bound; lower k or xi.")

        self.cost = self.mc[:, None] * self.q_grid + 0.5 * self.qc[:, None] * self.q_grid ** 2

        # ---- reachable generation lattice --------------------------------
        # J = sum_i a_i indexes the lattice of per-node generation vectors.
        self.J_max = self.n_firms * (self.k - 1)
        self._build_gen_lattice()

        # ---- demand shock size -------------------------------------------
        self._calibrate_shock()

        # ---- clear the market on the whole lattice -----------------------
        self._build_price_table()

        # ---- price-state (imperfect monitoring signal) discretisation -----
        self._build_price_states()

        # ---- payoff tensor ------------------------------------------------
        # profit[i, a_i, J, u] = p(J,u)*q_i(a_i) - C_i(q_i(a_i))
        self.profit = (
            self.price[None, None, :, :] * self.q_grid[:, :, None, None]
            - self.cost[:, :, None, None]
        )
        # Expected over the (equiprobable) demand states.
        self.exp_profit = self.profit.mean(axis=3)          # (n, k, J+1)
        self.exp_price = self.price.mean(axis=1)            # (J+1,)

    # ------------------------------------------------------------------
    def _build_gen_lattice(self):
        """Per-node generation for every reachable J (needs the hub layout, or
        falls back to enumerating action profiles when firms sit on different
        nodes)."""
        nodes = np.unique(self.node_of_firm)
        self.single_node = len(nodes) == 1
        if self.single_node:
            node = int(nodes[0])
            base = self.q_grid[:, 0].sum()
            self.total_gen_grid = base + self.v * np.arange(self.J_max + 1)
            self.gen_per_node = np.zeros((self.J_max + 1, NUM_NODES))
            self.gen_per_node[:, node] = self.total_gen_grid
        else:
            # General case: J does not pin down the nodal split, so we keep a
            # per-profile table instead. (Not needed for MARKET_CONFIG=three_firm.)
            raise NotImplementedError(
                "Firms sit on different nodes; the J-lattice collapse does not "
                "apply. Use the three-firm hub market."
            )

    # ------------------------------------------------------------------
    def _hub_lmp(self, gen_node_vec: np.ndarray, u: float):
        """Clear the DC-OPF at intercept shift u; return (hub LMP, avg LMP)."""
        self.env._demand_u = float(u)
        lmps, demand, flows, shadow = self.env._clear_market(gen_node_vec.copy())
        if lmps is None:
            raise RuntimeError(f"DC-OPF infeasible at gen={gen_node_vec}, u={u}")
        node = self.price_node if self.price_node is not None else int(self.node_of_firm[0])
        tot = float(np.sum(demand))
        avg = float(np.sum(lmps * demand) / tot) if tot > 0 else 0.0
        return float(lmps[node]), avg, lmps.copy(), demand.copy(), flows.copy(), shadow.copy()

    def _calibrate_shock(self):
        """Pick the intercept shift u so the hub LMP moves by m output steps.

        The paper sets d^{j+1}-d^j = m*v exactly; because p = d - Q there, one
        output step moves the price by v and the shock moves it by m*v. Here the
        price is a DC-OPF output, so we measure the per-output-step price impact
        numerically at the centre of the lattice and solve for u by bisection.
        """
        if self.deterministic or self.shock_steps_m <= 0:
            self.u_max = 0.0
            self.u_levels = np.zeros(1)
            self.price_step_per_output_step = None
            return

        Jm = self.J_max // 2
        p_hi, _, *_ = self._hub_lmp(self.gen_per_node[Jm - 1], 0.0)
        p_lo, _, *_ = self._hub_lmp(self.gen_per_node[Jm + 1], 0.0)
        step = abs(p_hi - p_lo) / 2.0          # |dp| per one-firm one-step change
        self.price_step_per_output_step = step
        target = self.shock_steps_m * step

        p0, _, *_ = self._hub_lmp(self.gen_per_node[Jm], 0.0)
        lo, hi = 0.0, 60.0
        for _ in range(80):
            u = 0.5 * (lo + hi)
            pu, _, *_ = self._hub_lmp(self.gen_per_node[Jm], u)
            if (pu - p0) < target:
                lo = u
            else:
                hi = u
        self.u_max = 0.5 * (lo + hi)
        # h equiprobable, equally spaced levels spanning [-u_max, +u_max]
        self.u_levels = (
            np.linspace(-self.u_max, self.u_max, self.h) if self.h > 1 else np.zeros(1)
        )
        self.env._demand_u = 0.0

    # ------------------------------------------------------------------
    def _build_price_table(self):
        Jn, hn = self.J_max + 1, self.h
        self.price = np.zeros((Jn, hn))
        self.avg_lmp = np.zeros((Jn, hn))
        self.nodal_lmps = np.zeros((Jn, hn, NUM_NODES))
        self.demand_tab = np.zeros((Jn, hn, NUM_NODES))
        self.flows_tab = np.zeros((Jn, hn, self.env.num_lines))
        self.shadow_tab = np.zeros((Jn, hn, self.env.num_lines))
        for j in range(Jn):
            for iu, u in enumerate(self.u_levels):
                p, a, lmps, dem, fl, sh = self._hub_lmp(self.gen_per_node[j], u)
                self.price[j, iu] = p
                self.avg_lmp[j, iu] = a
                self.nodal_lmps[j, iu] = lmps
                self.demand_tab[j, iu] = dem
                self.flows_tab[j, iu] = fl
                self.shadow_tab[j, iu] = sh
        self.env._demand_u = 0.0

    # ------------------------------------------------------------------
    def _build_price_states(self):
        """Bin the hub LMP onto a uniform grid, exactly as the paper's price
        set {80, 82.5, ..., 170} is a uniform grid of step v.

        Paper: with n firms, k actions and a shock of m steps, the price takes
        (n-1)(k-1) + m + 1 values from the point of view of a firm that knows
        its own output... and n(k-1) + m + 1 values in total. We use the total
        count so the resolution matches the paper's, then MEASURE how much
        confounding this actually produces (`monitoring_report`).
        """
        m = int(round(self.shock_steps_m)) if not self.deterministic else 0
        self.n_price_states = self.J_max + m + 1
        p_lo, p_hi = float(self.price.min()), float(self.price.max())
        self.price_bin_width = (p_hi - p_lo) / (self.n_price_states - 1)
        self.price_lo, self.price_hi = p_lo, p_hi
        idx = np.rint((self.price - p_lo) / self.price_bin_width).astype(np.int64)
        self.price_state = np.clip(idx, 0, self.n_price_states - 1)
        self.price_state_value = p_lo + self.price_bin_width * np.arange(self.n_price_states)
        # Only states that are actually reachable matter; keep the full grid so
        # index arithmetic stays trivial, but record which are live.
        self.reachable_states = np.unique(self.price_state)

    # ------------------------------------------------------------------
    def monitoring_report(self) -> dict:
        """How imperfect is monitoring, measured on the actual price table?

        For each (own action a_i, observed price state s) pair we count the
        distinct rival output profiles J_{-i} consistent with it. A price state
        is 'revealing' for firm i when at most one J_{-i} is consistent.
        """
        Jm1 = (self.n_firms - 1) * (self.k - 1)      # max sum of rivals' indices
        rows = []
        for i in range(self.n_firms):
            revealing, total = 0, 0
            for a in range(self.k):
                seen = {}
                for jm in range(Jm1 + 1):
                    J = a + jm
                    for iu in range(self.h):
                        seen.setdefault(int(self.price_state[J, iu]), set()).add(jm)
                total += len(seen)
                revealing += sum(1 for v in seen.values() if len(v) == 1)
            rows.append(1.0 - revealing / total)
        frac = float(np.mean(rows))
        # Paper's closed form for the baseline, for reference.
        m = self.shock_steps_m
        kk, nn, hh = self.k, self.n_firms, self.h
        if self.deterministic or self.h == 1:
            theory = 0.0
        else:
            num = (hh - 3) * m + kk * (nn - 1) - nn + 2
            den = (hh - 1) * m + kk * (nn - 1) - nn + 2
            theory = float(num / den)
        return {
            "measured_nonrevealing_fraction": frac,
            "per_firm": rows,
            "paper_closed_form_fraction": theory,
            "n_price_states": int(self.n_price_states),
            "reachable_price_states": int(len(self.reachable_states)),
            "price_bin_width": float(self.price_bin_width),
            "price_range": [self.price_lo, self.price_hi],
            "shock_u_max": float(self.u_max),
            "shock_levels": self.u_levels.tolist(),
            "price_impact_per_output_step": self.price_step_per_output_step,
        }

    # ------------------------------------------------------------------
    # Benchmarks ON THE GRID -- these define Delta.
    # ------------------------------------------------------------------
    def grid_benchmarks(self) -> dict:
        """Stage-game Nash and joint-monopoly restricted to the action grid.

        Delta = (pi - pi^N) / (pi^M - pi^N) uses these, because they are the
        best the algorithms could possibly do given their discretised actions.
        Expectations are taken over the demand shock, which is exactly what the
        firms face (they commit output before the shock realises).
        """
        n, k = self.n_firms, self.k
        # profile payoff: exp_profit[i, a_i, J]
        # --- joint monopoly: maximise total expected profit over all profiles
        best_tot, best_prof = -np.inf, None
        for prof in itertools.product(range(k), repeat=n):
            J = sum(prof)
            tot = sum(self.exp_profit[i, prof[i], J] for i in range(n))
            if tot > best_tot:
                best_tot, best_prof = tot, prof
        mono = self._profile_summary(best_prof)

        # --- pure-strategy Nash of the one-shot grid game (exhaustive check)
        nash_profiles = []
        for prof in itertools.product(range(k), repeat=n):
            J = sum(prof)
            ok = True
            for i in range(n):
                base = self.exp_profit[i, prof[i], J]
                Jo = J - prof[i]
                dev = self.exp_profit[i, np.arange(k), Jo + np.arange(k)]
                if dev.max() > base + 1e-9:
                    ok = False
                    break
            if ok:
                nash_profiles.append(prof)
        if not nash_profiles:
            # No pure NE on the grid: fall back to the profile minimising the
            # max deviation gain (an epsilon-equilibrium), and report it.
            best_eps, best_p = np.inf, None
            for prof in itertools.product(range(k), repeat=n):
                J = sum(prof)
                eps = 0.0
                for i in range(n):
                    Jo = J - prof[i]
                    dev = self.exp_profit[i, np.arange(k), Jo + np.arange(k)]
                    eps = max(eps, dev.max() - self.exp_profit[i, prof[i], J])
                if eps < best_eps:
                    best_eps, best_p = eps, prof
            nash_profiles = [best_p]
            nash_kind = f"eps-equilibrium (eps=${best_eps:.2f})"
        else:
            nash_kind = f"pure NE ({len(nash_profiles)} found)"
        # If several, take the one with lowest total profit (most competitive).
        nash_prof = min(nash_profiles,
                        key=lambda p: sum(self.exp_profit[i, p[i], sum(p)] for i in range(n)))
        nash = self._profile_summary(nash_prof)
        nash["kind"] = nash_kind
        nash["all_profiles"] = [list(p) for p in nash_profiles]

        return {"nash": nash, "monopoly": mono}

    def _profile_summary(self, prof) -> dict:
        n = self.n_firms
        J = int(sum(prof))
        q = np.array([self.q_grid[i, prof[i]] for i in range(n)])
        pi = np.array([self.exp_profit[i, prof[i], J] for i in range(n)])
        return {
            "actions": [int(a) for a in prof],
            "J": J,
            "gens": q.tolist(),
            "total_gen": float(q.sum()),
            "profits": pi.tolist(),
            "total_profit": float(pi.sum()),
            "hub_price": float(self.exp_price[J]),
            "avg_lmp": float(self.avg_lmp[J].mean()),
        }

    # ------------------------------------------------------------------
    def describe(self) -> str:
        gb = self.grid_benchmarks()
        mr = self.monitoring_report()
        L = []
        L.append("=" * 78)
        L.append("DISCRETE COURNOT GAME ON THE NETWORKED (DC-OPF) MARKET")
        L.append("=" * 78)
        L.append(f"firms={self.n_firms}  k={self.k}  step v={self.v:.3f} MW  "
                 f"xi={self.xi}  h={self.h}  deterministic={self.deterministic}")
        L.append("")
        L.append("Continuous benchmarks (repo solvers):")
        for nm in ("competitive", "nash", "monopoly"):
            b = self.bench_continuous[nm]
            L.append(f"  {nm:12s} gen={b['total_gen']:7.1f} MW  avgLMP=${b['avg_lmp']:6.2f}  "
                     f"pi=${b['total_profit']:8.1f}  per-firm={[round(x,1) for x in b['gens']]}")
        L.append("")
        L.append("Action grid (MW):")
        for i in range(self.n_firms):
            L.append(f"  firm {i}: [{self.q_grid[i,0]:.1f} ... {self.q_grid[i,-1]:.1f}]  "
                     f"(q^M={self.q_mono_cont[i]:.1f}, q^N={self.q_nash_cont[i]:.1f})")
        L.append("")
        L.append("Imperfect monitoring:")
        L.append(f"  intercept shock u = +/- {mr['shock_u_max']:.3f} $/MWh  "
                 f"(levels {[round(x,2) for x in mr['shock_levels']]})")
        if mr["price_impact_per_output_step"]:
            L.append(f"  price impact of ONE output step = ${mr['price_impact_per_output_step']:.3f}/MWh "
                     f"-> shock = {self.shock_steps_m:g} steps (paper m=8)")
        L.append(f"  hub LMP range ${mr['price_range'][0]:.2f} - ${mr['price_range'][1]:.2f} "
                 f"in {mr['n_price_states']} bins of ${mr['price_bin_width']:.3f}")
        L.append(f"  non-revealing price states: measured {mr['measured_nonrevealing_fraction']:.3f} "
                 f"(paper closed form for this n,k,m,h: {mr['paper_closed_form_fraction']:.3f})")
        L.append("")
        L.append("Grid benchmarks (these define Delta):")
        for nm in ("nash", "monopoly"):
            b = gb[nm]
            L.append(f"  {nm:9s} a={b['actions']} gen={b['total_gen']:7.1f} MW  "
                     f"hubP=${b['hub_price']:6.2f}  pi=${b['total_profit']:8.1f}  "
                     f"per-firm={[round(x,1) for x in b['profits']]}")
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
            "k": self.k, "xi": self.xi, "m": self.shock_steps_m, "h": self.h,
            "det": self.deterministic, "n": self.n_firms,
            "v": round(self.v, 6), "u": round(float(self.u_max), 6),
        }, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:10]


if __name__ == "__main__":
    mk = DiscreteMarket()
    print(mk.describe())
