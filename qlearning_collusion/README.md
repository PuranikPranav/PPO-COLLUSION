# Q-learning collusion on the networked electricity market

Replication of **Calvano, Calzolari, Denicolò & Pastorello (2021), "Algorithmic
collusion with imperfect monitoring"** (CEPR DP15738) — and its perfect-monitoring
predecessor, Calvano et al. (2020, *AER* 110(10)) — on **this repo's own
topology**: the 5-node Liu & Hobbs network with three asymmetric firms at the
node-2 generation hub, cleared by the ISO's DC-OPF.

The question this answers: *does the collusion those papers report with tabular
Q-learning and a discretised action space survive when the payoff function is a
real networked market clearing rather than a textbook linear inverse demand?*

## Three market structures

| market | `MARKET_CONFIG` | structure | results |
|---|---|---|---|
| three-firm, **one plant per node** | `three_firm_dist` (default) | 3 firms, 1 plant each at nodes 1 / 2 / 3, costliest at node 3; node 5 an inelastic load pocket | [`RESULTS_DIST.md`](RESULTS_DIST.md) |
| three-firm hub | `three_firm` | 3 firms, 1 plant each, all at the node-2 hub | [`RESULTS.md`](RESULTS.md) — Δ = 68.78% |
| two-firm / three-plant | `two_firm` | firm 0 = base @ node 1 + peaker @ hub; firm 1 = 1 plant @ hub | [`RESULTS_2FIRM.md`](RESULTS_2FIRM.md) — Δ = 64.14% |

`three_firm_dist` is the current default. It implements two changes asked for at
the 27 Jul advisor meeting:

* **the three plants are redistributed one per node**, with the most expensive
  plant at node 3 — the head of the radial 3–4–5 tail;
* **node 5's demand is inelastic** (point elasticity ≈ −0.06 against −0.7…−2.4
  everywhere else), so it is a genuine load pocket rather than just another
  elastic node.

The **two-firm** market is the direct counterpart of the paper's duopoly
baseline (n = 2, k = 15, |S| = k² = 225), and reproduces its Table I most closely.
It also required two structural fixes documented in `market_multi.py`: how to
discretise a firm owning two plants (the 1-D reduction is provably exact there),
and why the LCP Nash benchmark is *not* an equilibrium when a firm straddles two
nodes. Run `python -m qlearning_collusion.compare_markets` for the side-by-side.

### Which Nash, and why it matters here

The repo's Nash benchmark is the source paper's networked-Cournot **LCP**
(eqs. 39–45), in which each firm's stationarity uses the inverse-demand slope of
**its own node**. That is right when the firm's node is islanded by congestion,
and it is exactly right on the hub market (where it agrees with a direct
best-response calculation to the digit).

Once every firm sits alone on its own node it stops being right, because nodes 1
and 3 are steep *locally* but price-coupled to the rest of the network whenever
their lines are slack. On `three_firm_dist` the consequences are stark:

* at the LCP point a firm can gain **$1,288** by unilaterally deviating — it is
  not an equilibrium of the game the algorithms play;
* LCP "Nash" generation (207.6 MW) is **below** joint monopoly (213.4 MW), i.e.
  the benchmark is *more* collusive than the cartel it is supposed to bound.

Using it as the Δ denominator would manufacture apparent collusion. So Δ is
measured against the **iterated best-response Nash of the DC-OPF game**
(max unilateral deviation gain **$0.00**). Both numbers are printed side by side
by `python -m qlearning_collusion.run market`, and the LCP point is still
reported in every table.

---

## Why this exists

The "impossibility" result says collusion cannot emerge with a **continuous**
action space. The economics papers report substantial collusion with a
**discretised** one. So before drawing any conclusion from continuous-action PPO
runs, we discretise each firm's action space and run the *exact* algorithm the
economics papers ran, on the same market the PPO agents face.

---

## How the paper's model maps onto this market

| Calvano et al. (2021) | Here |
|---|---|
| `p_t = d_t − (q_1 + … + q_n)` | `p_t` = **LMP at the reference node** — the generation node with the largest load (node 2 in every structure) — produced by the ISO's DC-OPF clear given committed generation |
| `d_t` i.i.d. over `{290, 310}` | all nodal intercepts `P0_i` shift by `u_t`, i.i.d. over `h` equiprobable levels; the shock realises **after** firms commit |
| `A_i = {q¹,…,q^k}`, step `v` | per-firm grid of `k` outputs on a **common** step `v`, each grid centred on that firm's own `[q^M_i, q^C_i]` (firms here are asymmetric in MC/QC) |
| `s_t = p_{t−1}` (imperfect monitoring) | `s_t` = binned reference-node LMP of `t−1` |
| `s_t = (q_{1,t−1},…,q_{n,t−1})` (perfect) | same |
| — (no counterpart) | **`s_t` = the rich "19-variable" state**: the whole public market signal of `t−1` (see below) |
| `n = 2`, symmetric, MC = 0 | `n = 3`, asymmetric `MC ∈ {15,16,18}`, `QC ∈ {0.09,0.075,0.05}`, capacities, and five thermal limits |

Everything else is the paper verbatim: ε-greedy with `ε_t = e^{−βt}`,
`α = 0.15`, `β = 4×10⁻⁶`, `δ = 0.95`, `k = 15`, `h = 2`, `Q₀` initialised at the
discounted payoff under uniformly-randomising rivals, ties in the argmax broken
toward the **higher** output, and convergence declared when the greedy action is
unchanged for **100,000** consecutive periods.

### The grid lands where the paper's does

The action grid is built as `[q^M_i, q^C_i]` extended by `ξ = 0.2` of that range
on each side — the paper's `A = {70, 72½, …, 105}` around `q^M = 75`,
`q^C = 100`. On this market that puts:

* the **Cournot–Nash** profile at action index **12–13** (paper: 12)
* the **joint monopoly** profile at action index **2** (paper: 2)

so the two benchmarks sit at the same places on the grid as in the paper, and
firms retain headroom both to cheat (expand) and to punish.

### Why the inner loop is a lookup table

On the **hub** market every firm owns one plant and all three sit at node 2, so
the ISO's clearing depends on the actions only through **total hub generation**
`G = G₀ + v·J`, `J = Σᵢ aᵢ ∈ {0,…,n(k−1)}`. There are therefore only
`n(k−1)+1 = 43` distinct DC-OPF clears per demand state. `market.py` does not
*assume* this — it detects the single-node layout and raises if firms sit on
different nodes.

Once the plants are **redistributed one per node** that collapse is gone: `J` no
longer pins down the nodal split, and two profiles with the same total can clear
at completely different prices. `market_multi.py` therefore enumerates all
`k^n = 3,375` profiles and clears each one under each demand state. That is
6,750 solves instead of 86, paid once and cached, after which the Q-learning
inner loop is still pure array indexing — which is what keeps millions of
iterations × 1,000 sessions tractable.

This is also why monitoring is *structurally* worse on the distributed market:
one nodal price can no longer summarise what the rivals did, because it no
longer depends on them only through a scalar.

### The rich "19-variable" state

The PPO agents in `iso_market/market_env.py` see a **19-dimensional** observation
per step:

```
[ nodal LMPs (5) | line flows (5) | shadow prices (5) | per-plant gen (3) | own previous profit (1) ]
```

`monitoring="rich"` brings that observation back into the tabular model, binned
into a finite state (`ProfileAPI.build_rich_states`). Two of the 19 signals are
deliberately left out, and it matters which:

* **per-plant generation is the rivals' action profile.** Putting it in the state
  *is* perfect monitoring, which already has its own Table I cell. Including it
  would silently convert the experiment into the one next to it.
* **own previous profit** is a deterministic function of own action and the
  price, so given the price bins it adds nothing.

What remains is exactly the list Andrew asked for — previous price, transmission
congestion / shadow prices, and **realised demand**. Monitoring stays imperfect,
but the agent can now partly tell an adverse demand shock from a rival's
expansion, because the shock moves realised demand and a rival's expansion does
not move it the same way. Measured on `three_firm_dist`:

| state | \|S\| | fraction of observations that do **not** pin down rivals |
|---|---|---|
| price only (`imperfect`) | 51 | 0.962 |
| **rich 19-variable** (`rich`) | **192** | **0.801** |
| full profile (`perfect`) | 3,375 | 0.000 |

so the rich state sits where it should — strictly more informative than a single
price, still a long way from observing rivals.

### How imperfect is the monitoring?

The demand shock is calibrated **numerically** so that it moves the reference-node
LMP by `m = 8` output steps — the paper's baseline `m`, the thing that makes an
adverse shock confusable with a rival expanding output by up to 8 grid steps.
That works out to an intercept shift of about ±$2.89/MWh on the hub market and
±$4.28/MWh on `three_firm_dist`.

Because the DC-OPF price is piecewise-linear in generation (congestion regimes
switch), the price lattice does not collide *exactly* the way the paper's
`p = d − Q` does, so we **measure** the confounding rather than assume it:
`monitoring_report()` counts, for each (own action, observed price state), how
many rival output profiles are consistent with it.

| market | measured non-revealing fraction |
|---|---|
| three-firm hub | **0.384** — close to the paper's "roughly a third" |
| three-firm, one plant per node | **0.962** |

The distributed market is far more opaque, and structurally so: with all firms at
one node the price is a function of the scalar `J`, so a price bin nearly pins
down the rivals' aggregate. With firms on three different nodes the rivals'
*profile* matters, not just its sum, and a single nodal price cannot identify a
two-dimensional object. This is the reason the rich 19-variable state is worth
having here in a way it would not be on the hub.

---

## Layout

```
market.py         the hub game (J-lattice fast path): benchmarks, action grid,
                  DC-OPF lookup tables, price-state binning, diagnostics
market_multi.py   the general game (firms on different nodes): profile
                  enumeration, best-response Nash, direct joint monopoly
profile_api.py    the profile-indexed view shared by both, and the rich
                  ("19-variable") state construction
qlearn.py         the Calvano Q-learning algorithm, vectorised across sessions;
                  the three state spaces (price / rich / full profile)
experiments.py    experiment definitions, persistence, limit-strategy and
                  forced-deviation analyses
delta_sweep.py    the discount-factor sweep and its Figure 4b panel
network_report.py which lines bind, in every outcome incl. learned collusion
figures.py        Figures 1–5 replications
fig_demand.py     nodal demand curves / the node-5 load pocket
fig0_network.py   the topology, one panel per ownership structure
fig4b_…, fig6_…, fig_lmp_learning.py, fig_strategy.py   further figures
run.py            CLI
make_all_figures.sh  regenerate everything for one MARKET_CONFIG
results_<cfg>/    .npz (policies + trajectories) and .json (summary) per run
figures_<cfg>/    rendered PNGs
```

## Running it

`MARKET_CONFIG` selects the structure; it defaults to `three_firm_dist`.

```bash
# describe the discretised game and its benchmarks
python -m qlearning_collusion.run market

# the baseline: imperfect monitoring + stochastic demand
python -m qlearning_collusion.run train imperfect_stochastic --sessions 1000 --iters 4000000

# all four cells of the paper's Table I
python -m qlearning_collusion.run table1 --sessions 1000

# the rich 19-variable state
python -m qlearning_collusion.run train rich_stochastic --sessions 1000
python -m qlearning_collusion.run train rich_deterministic --sessions 1000

# the discount-factor sweep (0.99 / 0.95 / 0.9 / 0.8 / 0.7 / 0.5)
python -m qlearning_collusion.delta_sweep train --sessions 500

# which transmission lines bind, in every outcome incl. learned collusion
python -m qlearning_collusion.network_report imperfect_stochastic

# every figure and report for the current MARKET_CONFIG
./qlearning_collusion/make_all_figures.sh
```

Runtime: roughly 15 minutes per million iterations at 1,000 sessions with six
runs sharing an 8-core laptop (|S| = 51 price bins; the |S| = 3,375
perfect-monitoring cell is about 25% slower).

**Caches.** Building the game on a multi-node market costs `k^n · h` = 6,750
DC-OPF solves plus the continuous Nash / monopoly searches — about 4 minutes.
Every analysis and figure rebuilds the market, so both are cached under
`results*/_cache/`. Delete that directory after changing any market parameter.

---

## Results

See [`RESULTS.md`](RESULTS.md).
