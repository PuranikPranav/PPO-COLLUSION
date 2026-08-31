"""
The unified PROFILE-INDEXED view of a discrete Cournot game.

`qlearn.py` and `experiments.py` were originally written against the three-firm
hub market, where the joint action profile collapses to the scalar lattice
J = sum_i a_i. That collapse is invalid as soon as firms own plants at different
nodes (see `market_multi`). This module replaces J with a general profile index
so one code path serves both markets:

    P = pidx(a)                     joint action profile -> flat index
    dev_pidx(P, i, a_new)           same profile with agent i deviating to a_new
    profit_p[i, P, u]               agent i's profit
    price_p[P, u], avg_lmp_p[P, u], total_gen_p[P], price_state_p[P, u]

For the hub market `pidx` is a mixed-radix encoding of the same profiles the
J-lattice indexed, so every number it produces is identical to before -- the
lattice was only ever a compression of this table.

A market class supplies:
    n_agents, n_actions (n_agents,), max_actions, n_profiles,
    agent_profile_weight (n_agents, max_actions) int,
    valid_action (n_agents, max_actions) bool,
    q_agent (n_agents, max_actions) MW,
    profit_p, exp_profit_p, price_p, exp_price_p, avg_lmp_p, total_gen_p,
    price_state_p, price_state_value, reachable_states, n_price_states,
    _dev_base (list of (n_profiles,) int arrays),
    h, k, shock_steps_m, deterministic, u_levels, u_max
and inherits the methods below.
"""

from __future__ import annotations

import numpy as np


class ProfileAPI:
    # ------------------------------------------------------------------
    def pidx(self, a: np.ndarray) -> np.ndarray:
        """(..., n_agents) action indices -> (...,) flat profile index."""
        a = np.asarray(a)
        w = self.agent_profile_weight[np.arange(self.n_agents)[None, :], a]
        return w.sum(axis=-1)

    def dev_pidx(self, P: np.ndarray, i: int, a_new: np.ndarray) -> np.ndarray:
        """Profile index with agent `i`'s action replaced by `a_new`."""
        return self._dev_base[i][P] + self.agent_profile_weight[i, a_new]

    # ------------------------------------------------------------------
    # THE RICH ("19-variable") STATE
    # ------------------------------------------------------------------
    def build_rich_states(self, pocket_node: int = 4, pocket_bins: int = 8,
                          demand_bins: int = 8, flow_bins: int = 0,
                          shadow_tol: float = 1e-4):
        """Discretise the PPO-style public market observation into a Q-table index.

        WHAT THIS IS. The PPO agents in `iso_market/market_env.py` observe a
        19-dimensional vector per step:

            [ nodal LMPs (5) | line flows (5) | shadow prices (5)
              | per-plant generation (3) | own previous profit (1) ]

        A tabular Q-learner needs a FINITE state, so the continuous block is
        binned. Two of the 19 signals are deliberately left out and it matters
        which:

          * per-plant generation is the rivals' action profile. Putting it in the
            state IS perfect monitoring, which already has its own Table I cell
            (`monitoring="perfect"`). Including it here would silently convert the
            imperfect-monitoring experiment into the perfect-monitoring one.
          * own previous profit is a deterministic function of own action and the
            price, so given the price bins it adds no information the agent does
            not already have.

        What remains is exactly the public post-clearing signal Andrew asked to
        bring back -- previous price, transmission congestion (which lines bind
        and how hard), and REALISED DEMAND -- so monitoring stays imperfect: a
        low price is still consistent with many rival profiles, but the agent can
        now partly tell an adverse demand shock from a rival's expansion, because
        the shock moves realised demand and a rival's expansion does not move it
        the same way.

        COMPONENTS (each a deterministic function of (profile, shock), so the
        Q-learning inner loop stays pure array indexing):

          price      the reference-node LMP bin already used by the price-only
                     state, so the two state spaces are directly comparable
          pocket     the load-pocket (node 5) LMP, binned -- this is the
                     congestion signal that matters on this topology
          congestion the sign pattern of the 5 transmission shadow prices
          demand     total realised system demand, binned
          flows      optional, off by default: line flows are nearly collinear
                     with (demand, generation) and mostly inflate |S|

        The reachable component tuples are then compacted to consecutive indices,
        so |S| is the number of DISTINCT observations the market can actually
        produce, not the product of the bin counts.
        """
        P, h = self.n_profiles, self.h
        comps = [np.asarray(self.price_state_p).reshape(P, h)]
        names = ["price"]

        def _bin(x, nb):
            lo, hi = float(np.min(x)), float(np.max(x))
            if nb <= 1 or hi - lo < 1e-12:
                return np.zeros(x.shape, dtype=np.int64)
            w = (hi - lo) / nb
            return np.clip(((x - lo) / w).astype(np.int64), 0, nb - 1)

        if pocket_bins > 0 and hasattr(self, "nodal_lmps"):
            comps.append(_bin(self.nodal_lmps[:, :, pocket_node], pocket_bins))
            names.append("pocket_lmp")
        if hasattr(self, "shadow_tab"):
            sg = np.sign(np.where(np.abs(self.shadow_tab) > shadow_tol,
                                  self.shadow_tab, 0.0)).astype(np.int64) + 1
            code = np.zeros((P, h), dtype=np.int64)
            for l in range(sg.shape[2]):
                code = code * 3 + sg[:, :, l]
            comps.append(code)
            names.append("congestion")
        if demand_bins > 0 and hasattr(self, "demand_tab"):
            comps.append(_bin(self.demand_tab.sum(axis=2), demand_bins))
            names.append("total_demand")
        if flow_bins > 0 and hasattr(self, "flows_tab"):
            for l in range(self.flows_tab.shape[2]):
                comps.append(_bin(self.flows_tab[:, :, l], flow_bins))
                names.append(f"flow_{l}")

        stack = np.stack(comps, axis=-1).reshape(P * h, len(comps))
        uniq, inv = np.unique(stack, axis=0, return_inverse=True)
        self.rich_state_p = inv.reshape(P, h).astype(np.int64)
        self.n_rich_states = int(uniq.shape[0])
        self.rich_state_components = names
        self.rich_state_key = uniq
        self.reachable_rich_states = np.arange(self.n_rich_states)
        # a representative reference-node price per state, for plotting
        flat_price = np.asarray(self.price_p).reshape(P * h)
        val = np.zeros(self.n_rich_states)
        np.add.at(val, inv, flat_price)
        cnt = np.bincount(inv, minlength=self.n_rich_states).astype(float)
        self.rich_state_value = val / np.maximum(cnt, 1)
        self.rich_state_count = cnt
        return self.n_rich_states

    def rich_state_report(self) -> dict:
        """How much rival information the rich state actually reveals.

        Same measurement as `monitoring_report`, but on the rich state: for each
        (own action, observed state) pair, count the distinct rival profiles
        consistent with it. Comparing the two numbers is the whole point -- it
        says how far the extra signals move the market away from the
        price-only baseline and toward perfect monitoring.
        """
        if not hasattr(self, "rich_state_p"):
            self.build_rich_states()
        rows = []
        for i in range(self.n_agents):
            own = self.agent_action_of_profile(i)
            rival = self._dev_base[i][np.arange(self.n_profiles)]
            revealing, total = 0, 0
            for a in range(int(self.n_actions[i])):
                sel = own == a
                seen = {}
                for Pi, rv in zip(np.where(sel)[0], rival[sel]):
                    for iu in range(self.h):
                        seen.setdefault(int(self.rich_state_p[Pi, iu]), set()).add(int(rv))
                total += len(seen)
                revealing += sum(1 for v in seen.values() if len(v) == 1)
            rows.append(1.0 - revealing / max(total, 1))
        return {
            "measured_nonrevealing_fraction": float(np.mean(rows)),
            "per_agent": rows,
            "n_rich_states": int(self.n_rich_states),
            "components": list(self.rich_state_components),
        }

    # ------------------------------------------------------------------
    def monitoring_report(self) -> dict:
        """How imperfect is monitoring, measured on the actual price table?

        For each (own action a_i, observed price state s) pair, count the
        distinct RIVAL action profiles consistent with it. A state is
        'revealing' for agent i when at most one rival profile is consistent.
        """
        rows = []
        for i in range(self.n_agents):
            own = self.agent_action_of_profile(i)          # (P,)
            rival = self._dev_base[i][np.arange(self.n_profiles)]
            revealing, total = 0, 0
            for a in range(int(self.n_actions[i])):
                sel = own == a
                seen = {}
                for P, rv in zip(np.where(sel)[0], rival[sel]):
                    for iu in range(self.h):
                        seen.setdefault(int(self.price_state_p[P, iu]), set()).add(int(rv))
                total += len(seen)
                revealing += sum(1 for v in seen.values() if len(v) == 1)
            rows.append(1.0 - revealing / max(total, 1))
        frac = float(np.mean(rows))

        # the paper's closed form, evaluated with n = number of agents
        m, kk, nn, hh = self.shock_steps_m, self.k, self.n_agents, self.h
        if self.deterministic or self.h == 1:
            theory = 0.0
        else:
            num = (hh - 3) * m + kk * (nn - 1) - nn + 2
            den = (hh - 1) * m + kk * (nn - 1) - nn + 2
            theory = float(num / den)
        return {
            "measured_nonrevealing_fraction": frac,
            "per_agent": rows,
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
    def grid_benchmarks(self, use_cache: bool = True) -> dict:
        """Stage-game Nash and joint monopoly, restricted to the action grid.

        Delta = (pi - pi^N) / (pi^M - pi^N) uses these, because the grid game is
        the game the algorithms actually play. Expectations are over the demand
        shock, which is what agents face (they commit before it realises).
        """
        if use_cache and getattr(self, "_gb_cache", None) is not None:
            return self._gb_cache

        tot = self.exp_profit_p.sum(axis=0)                   # (P,)
        mono_P = int(np.argmax(tot))

        # --- pure-strategy Nash: no agent gains by unilaterally deviating -----
        ok = np.ones(self.n_profiles, dtype=bool)
        eps_worst = np.zeros(self.n_profiles)
        allP = np.arange(self.n_profiles)
        for i in range(self.n_agents):
            n_a = int(self.n_actions[i])
            dev = self._dev_base[i][:, None] + self.agent_profile_weight[i, :n_a][None, :]
            best = self.exp_profit_p[i][dev].max(axis=1)      # (P,)
            gain = best - self.exp_profit_p[i][allP]
            eps_worst = np.maximum(eps_worst, gain)
            ok &= gain <= 1e-9

        nash_list = np.where(ok)[0]
        if len(nash_list):
            kind = f"pure NE ({len(nash_list)} found)"
            nash_P = int(nash_list[np.argmin(tot[nash_list])])
        else:
            nash_P = int(np.argmin(eps_worst))
            kind = f"eps-equilibrium (eps=${eps_worst[nash_P]:.2f})"

        out = {
            "nash": self._profile_summary(nash_P),
            "monopoly": self._profile_summary(mono_P),
        }
        out["nash"]["kind"] = kind
        out["nash"]["n_pure_ne"] = int(len(nash_list))
        self._gb_cache = out
        return out

    # ------------------------------------------------------------------
    def _profile_summary(self, P: int) -> dict:
        pi = self.exp_profit_p[:, P]
        acts = [int(self.agent_action_of_profile(i)[P]) for i in range(self.n_agents)]
        d = {
            "profile": int(P),
            "actions": acts,
            "gens": [float(self.q_agent[i, acts[i]]) for i in range(self.n_agents)],
            "total_gen": float(self.total_gen_p[P]),
            "profits": pi.tolist(),
            "total_profit": float(pi.sum()),
            "hub_price": float(self.exp_price_p[P]),
            "avg_lmp": float(self.avg_lmp_p[P].mean()),
        }
        if hasattr(self, "profile_plant_gen"):
            d["plant_gens"] = self.profile_plant_gen[P].tolist()
            d["nodal_lmps"] = self.nodal_lmps[P].mean(axis=0).tolist()
        return d


# ---------------------------------------------------------------------------
def attach_hub_profile_api(mk):
    """Give the three-firm hub `DiscreteMarket` the profile-indexed API.

    Pure re-indexing of tables it already has -- every value is identical to
    what the J-lattice produced, so the published three-firm numbers are
    unchanged. Used so `qlearn.py` has a single code path.
    """
    n, k, h = mk.n_firms, mk.k, mk.h
    P = k ** n
    mk.n_agents = n
    mk.n_actions = np.full(n, k, dtype=np.int64)
    mk.max_actions = k
    mk.n_profiles = P

    acts = np.array(list(np.ndindex(*([k] * n))), dtype=np.int64)      # (P, n)
    radix = (k ** np.arange(n - 1, -1, -1)).astype(np.int64)
    mk.agent_profile_weight = np.stack([np.arange(k) * radix[i] for i in range(n)])
    mk.valid_action = np.ones((n, k), dtype=bool)
    mk.q_agent = mk.q_grid.copy()
    mk._profile_acts = acts

    J = acts.sum(axis=1)                                               # (P,)
    mk.price_p = mk.price[J]                                           # (P,h)
    mk.avg_lmp_p = mk.avg_lmp[J]
    mk.total_gen_p = mk.total_gen_grid[J]
    mk.price_state_p = mk.price_state[J]
    mk.exp_price_p = mk.exp_price[J]
    # Network tables, re-indexed by profile so the rich ("19-variable") state can
    # be built on the hub market too. Same values, different index.
    mk.nodal_lmps = mk.nodal_lmps[J]                                   # (P,h,N)
    mk.demand_tab = mk.demand_tab[J]
    mk.flows_tab = mk.flows_tab[J]
    mk.shadow_tab = mk.shadow_tab[J]
    mk.profit_p = np.stack([mk.profit[i, acts[:, i], J] for i in range(n)])   # (n,P,h)
    mk.exp_profit_p = mk.profit_p.mean(axis=2)

    own = [acts[:, i] for i in range(n)]
    mk._dev_base = [np.arange(P) - own[i] * radix[i] for i in range(n)]
    mk.agent_action_of_profile = lambda i, _a=own: _a[i]
    mk._gb_cache = None

    # NOTE: `monitoring_report` is deliberately NOT overridden -- DiscreteMarket
    # has its own J-lattice version and its published number stays untouched.
    for name in ("pidx", "dev_pidx", "grid_benchmarks", "_profile_summary",
                 "build_rich_states", "rich_state_report"):
        setattr(mk, name, getattr(ProfileAPI, name).__get__(mk, type(mk)))
    return mk
