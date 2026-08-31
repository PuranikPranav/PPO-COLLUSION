# Update on the four changes from the 27 Jul meeting

Everything below is on the new default market, `MARKET_CONFIG=three_firm_dist`.
The old hub market is still available as `MARKET_CONFIG=three_firm` and its
results are untouched in `results/`.

Learned numbers (Δ, the Table I cells, the discount sweep, the congestion report)
are generated from the run artefacts by

```bash
MARKET_CONFIG=three_firm_dist python -m qlearning_collusion.summarize > RESULTS_DIST.md
```

so this document covers **what changed and why**; `RESULTS_DIST.md` covers
**what came out**.

---

## 1. Redistribute the three plants — one per node, most expensive at node 3

| firm | node | MC | QC | cap (MW) |
|---|---|---|---|---|
| 0 | **1** | 15.0 | 0.090 | 135 |
| 1 | **2** | 16.0 | 0.075 | 145 |
| 2 | **3** | **18.0** | 0.050 | 155 |

Node 3 is the head of the radial 3–4–5 tail, so the most expensive unit is now
the one that feeds the load pocket. Marginal-cost ordering 15 < 16 < 18, and
average cost at the collusive dispatch $18.23 < $18.73 < **$19.73**/MWh.

**Why the cost spread is only 15/16/18.** Widening it (14/16/20, 13/16/22) does
make node 3 "more expensive", but it also makes the cartel cut that plant from
69 MW to 41 MW (or 13 MW), at which point **the node-3 firm earns less under
joint monopoly than at Nash** and has no reason to collude at all. Per-firm
individual rationality fails and the whole collusion index becomes meaningless.
Verified across the spread; 15/16/18 is the widest spread that keeps all three
firms' cartel margins positive (+$617 / +$609 / +$428).

## 2. Node 5's demand made inelastic

| node | P⁰ | Q⁰ | slope ($/MWh per MW) | elasticity at $30 / $45 / $60 |
|---|---|---|---|---|
| 1 | 85 | 140 | 0.61 | −0.55 / −1.13 / −2.40 |
| 2 | 95 | 300 | 0.32 | −0.46 / −0.90 / −1.71 |
| 3 | 90 | 90 | 1.00 | −0.50 / −1.00 / −2.00 |
| 4 | 88 | 14 | 6.29 | −0.52 / −1.05 / −2.14 |
| **5** | **900** | **11** | **81.8** | **−0.034 / −0.053 / −0.071** |

With linear demand `d = Q⁰(1 − p/P⁰)` the elasticity is `−(Q⁰/P⁰)·p/d`, so an
inelastic node needs a **large P⁰ relative to Q⁰** — a near-vertical
inverse-demand curve whose intercept is effectively the value of lost load.
Node 5 now takes ~10.3–10.4 MW whatever the price does, an order of magnitude
less responsive than anywhere else. (Before this change it was −0.71 to −1.13,
i.e. no more inelastic than the rest of the network.)

Both orientations are drawn in `figures_three_firm_dist/fig_demand_curves.png` —
quantity-against-price, where inelastic looks **flat**, and the textbook
inverse-demand picture, where the same curve is **steep**.

### The load pocket now actually does something

| | competitive | Nash | monopoly |
|---|---|---|---|
| LMP nodes 1–3 | $25–26 | $41 | $57.16 |
| **LMP nodes 4–5** | **$52.54** | **$52.54** | $57.16 |
| binding lines | 2–3, **3–4** | **3–4** | none |

Under competition and Nash the 3–4 corridor is congested, so the pocket pays a
large premium over the rest of the system. When the cartel withholds, the
*system* price rises above the pocket's congestion price, the corridor
un-congests, and the premium disappears. **Collusion de-congests the network.**

## 3. The 19-variable state is back

The PPO observation in `iso_market/market_env.py` is 19-dimensional:

```
[ nodal LMPs (5) | line flows (5) | shadow prices (5) | per-plant gen (3) | own previous profit (1) ]
```

`monitoring="rich"` bins that into a finite Q-learning state. Two signals are
deliberately excluded:

* **per-plant generation is the rivals' action profile** — putting it in the
  state *is* perfect monitoring, which already has its own Table I cell;
* **own previous profit** is a deterministic function of own action and price.

What remains is exactly the list from the meeting: previous price, transmission
congestion / shadow prices, and **realised demand**. Measured informativeness:

| state | \|S\| | fraction of observations that do **not** identify rivals |
|---|---|---|
| price only | 51 | 0.962 |
| **rich 19-variable** | **192** | **0.801** |
| full profile (perfect) | 3,375 | 0.000 |

so it sits strictly between the two, which is the point: the agent can now
partly tell an adverse demand shock from a rival's expansion, because the shock
moves realised demand and a rival's expansion does not move it the same way.

## 4. Discount-factor sweep

δ ∈ {0.99, 0.95, 0.9, 0.8, 0.7, 0.5}, 500 sessions each, everything else fixed.

| δ | Δ | deviation at t=0 (MW) | rivals at t+1 (MW) | peak punishment (MW) | periods above 10% of peak | punishment area (MW·periods) |
|---|---|---|---|---|---|---|
| 0.99 | 79.36% | 17.41 | **6.18** | 6.18 | 3 | 12.30 |
| 0.95 | 68.32% | 14.91 | **4.19** | 4.19 | 3 | 6.83 |
| 0.90 | 63.62% | 13.29 | **2.57** | 2.57 | 2 | 3.61 |
| 0.80 | 62.02% | 12.28 | **1.84** | 1.84 | 1 | 1.74 |
| 0.70 | 62.81% | 11.77 | **1.03** | 1.03 | 1 | −0.59 |
| 0.50 | 64.93% | 11.59 | **0.76** | 0.76 | 1 | −0.22 |

**The expectation holds, and monotonically.** The post-deviation response
flattens all the way down: the rivals' retaliation falls from 6.18 MW to
0.76 MW — a factor of eight — and its lifetime collapses from three periods to
one. By δ = 0.7 the punishment area is already negative, i.e. there is no
retaliation left at all.

**Δ itself is NOT monotone**, and that is worth a sentence rather than hiding:
it falls 79.4 → 62.0% as δ goes 0.99 → 0.8, then edges *back up* to 64.9% at
δ = 0.5. Read alongside the punishment column, the interpretation is clean —
below δ ≈ 0.8 the supra-Nash profit is no longer punishment-supported. It is the
floor tabular Q-learning reaches on its own, which is exactly why the profit
level alone cannot separate collusion from failure to optimise, and why
`fig5_deviation_value.png` tests deterrence directly instead.

Figures: `fig7_delta_sweep_deviation.png` (Figure 4b once per δ) and
`fig7b_delta_sweep_summary.png` (the three columns above against δ).

---

## What the algorithms learned

| cell | Δ | converged | total gen (MW) | ref-node LMP |
|---|---|---|---|---|
| imperfect (price only) · stochastic | **68.37%** | 100% | 261.5 | $49.10 |
| imperfect (price only) · deterministic | 83.57% | 100% | 247.8 | $51.38 |
| perfect (full profile) · stochastic | 46.46% | **5.5%** ⚠ | 277.0 | $46.43 |
| perfect (full profile) · deterministic | 79.06% | 96% | 253.3 | $50.43 |
| **rich 19-variable** · stochastic | **67.93%** | 100% | 261.3 | $49.17 |
| **rich 19-variable** · deterministic | 80.68% | 100% | 251.1 | $50.80 |

Three things to note:

1. **The redistribution barely moved the headline.** Δ = 68.37% here against
   68.78% on the old hub market — the collusion result is robust to the siting.
2. **The extra information did not help the cartel.** The rich 19-variable state
   lands at 67.93% against the price-only 68.37% (and 80.68 vs 83.57 without the
   shock). Giving the agents congestion and realised demand slightly *reduces*
   collusion, which is the same direction as the paper's counterintuitive
   finding that coarser information sustains more of it.
3. **The perfect-monitoring stochastic cell did not converge** (5.5% of sessions
   at the 4M-iteration cap; |S| = k³ = 3,375 with a demand shock on top). Its
   46.46% is a snapshot, not a limit strategy, so the difference-in-differences
   that uses it (+17.41 pp) should not be quoted as a result yet. That cell
   needs a longer run.

**Is the collusion punishment-supported?** Yes. Forcing a firm into its static
best response for one period earns it **+$218** (high demand) / **+$100** (low
demand) immediately, but the discounted stream comes to **−$116** / **−$711** —
deterred in 42% / 73% of sessions.

---

## Which Nash — the one thing that had to change

The LCP benchmark (paper eqs. 39–45) gives each firm the inverse-demand slope of
**its own node**. That is right when the node is islanded by congestion, and it
is right on the hub market. Once every firm sits alone on its own node it is not,
because nodes 1 and 3 are steep locally but price-coupled to the rest of the
network whenever their lines are slack. On this market:

* a firm gains **$1,288** by unilaterally deviating from the LCP point — it is
  not an equilibrium of the game the algorithms play;
* LCP "Nash" generation is **207.6 MW**, *below* joint monopoly's **213.4 MW** —
  the benchmark is more collusive than the cartel it is meant to bound.

Using it as the Δ denominator would manufacture apparent collusion: agents that
merely learn to best-respond would score Δ > 0 with no coordination at all. So Δ
is measured against the **iterated best-response Nash of the DC-OPF game**
(max unilateral deviation gain **$0.00**). The LCP point is still computed and
reported in every table, and `python -m qlearning_collusion.run market` prints
both side by side.

## Benchmarks

| outcome | gen (MW) | avg LMP | profit ($/period) | binding lines (rent $/MWh) |
|---|---|---|---|---|
| perfect competition | 395.4 | $27.03 | 1,831 | 2–3 (−2.24), 3–4 (+27.62) |
| Nash-Cournot (LCP) | 207.6 | $58.13 | 8,089 | none |
| **Nash-Cournot (best response)** | **306.6** | **$41.83** | **6,515** | 3–4 (+11.30) |
| joint monopoly | 213.4 | $57.16 | 8,169 | none |

* generation Competitive > Nash > Monopoly ✓
* LMP and profit Competitive < Nash < Monopoly ✓
* per-firm cartel IR margins +$617 / +$609 / +$428, all positive ✓
* Nash → monopoly profit gap **+25.4%** (the hub market's was +22.5%)
* caps sit 35–48% above Nash output, so there is headroom to cheat and to punish
* on the action grid the Cournot–Nash profile lands at index **12** and joint
  monopoly at index **2** — the same places as in the paper

`python -m qlearning_collusion.network_report` gives the full per-line table for
all of these plus the **learned collusive** outcome, where each line's binding
*frequency* is reported because the strategies cycle and demand keeps moving.
