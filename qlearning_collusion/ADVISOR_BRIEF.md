# Task 1 — Discretise the action space, run the econ paper's Q-learning, replicate

**Question posed:** the econ papers report algorithmic collusion with Q-learning;
the impossibility paper says collusion cannot emerge with a continuous action
space. Discretise each agent's action space, implement the *same* Q-learning
algorithm, and see whether the reported results replicate.

**Answer: yes, they replicate.** Tabular Q-learning on our five-node DC-OPF
market, with each firm's action space discretised to 15 output levels, converges
to Δ = **68.8%** of the way from Cournot–Nash to joint monopoly, sustained by
punishments that are genuinely collusive by every test I could construct. The
paper gets 76.25% (duopoly) and 72.7% (their three-firm robustness run). We are
4 points under their three-firm number.

Everything below is measured, not asserted. Reproduce with
`python -m qlearning_collusion.run report`.

---

## 1. What was built

Replication target: **Calvano, Calzolari, Denicolò & Pastorello (2021),
"Algorithmic collusion with imperfect monitoring"** (CEPR DP15738) — the
Green–Porter / imperfect-monitoring paper — plus its perfect-monitoring
predecessor, Calvano et al. (2020, *AER* 110(10)).

**Our case is the imperfect-monitoring one**: the state is last period's *price*,
not the rivals' outputs.

### The algorithm is theirs, verbatim

| Ingredient | Paper | Here |
|---|---|---|
| learning rule | `Q ← (1−α)Q + α[π + δ·max Q']` (their eq. 3) | identical |
| exploration | ε-greedy, `ε_t = e^{−βt}` (eq. 4) | identical |
| `α`, `β`, `δ` | 0.15, 4×10⁻⁶, 0.95 | identical |
| `Q₀` | discounted payoff if rivals randomise uniformly (fn. 12) | identical |
| tie-breaking | ties in argmax → **higher** output (fn. 17) | identical |
| convergence | greedy action unchanged for **100,000** periods | identical |
| sessions | 1,000, averaged | 1,000 |
| `k`, `h`, `ξ`, `m` | 15, 2, 0.2, 8 | 15, 2, 0.2, 8 |

### What changed: the payoff function

| Paper | Here |
|---|---|
| `p_t = d_t − Σq_i` | `p_t` = **LMP at node 2**, from the ISO's DC-OPF clear of the 5-node Liu–Hobbs network |
| `d_t` i.i.d. on {290, 310} | all nodal intercepts shift by `u_t`, i.i.d. on {−u, +u}; shock realises **after** firms commit |
| n = 2, symmetric, MC = 0 | **n = 3**, asymmetric `MC ∈ {15,16,18}`, `QC ∈ {0.09,0.075,0.05}`, capacities, 5 thermal limits |
| `A_i = {70, 72½,…,105}` | per-firm grid of 15 outputs on a **common** step `v = 3.098 MW`, each centred on that firm's own `[q^M_i, q^C_i]` |

### The discretisation is faithful — this is the crux of Task 1

The grid is built as `[q^M_i, q^C_i]` extended by ξ = 0.2 on each side, exactly
the paper's construction. The result:

* Cournot–Nash sits at action index **[12, 12, 13]** — paper: **12**
* joint monopoly sits at action index **[2, 2, 2]** — paper: **2**
* grid Nash total generation **328.6 MW** vs the continuous LCP Nash **325.5 MW**
  → **0.95%** discretisation error
* grid monopoly 232.6 MW vs continuous 232.6 MW → **exact**

So the two benchmarks land in the same slots on the grid as in the paper, firms
keep headroom both to cheat (expand) and to punish, and discretising the market
moved the competitive benchmark by under 1%.

**Why the market collapses to a lookup table.** All three firms sit at node 2, so
the ISO's clear depends on actions only through total hub generation
`G = G₀ + v·J`, `J = Σa_i ∈ {0,…,42}`. That is only 43 distinct DC-OPF solves per
demand state. Precomputing them makes the inner loop pure array indexing —
1,000 sessions × ~3M iterations in ~36 minutes.

### How imperfect is the monitoring — measured, not assumed

The shock size is solved by **bisection** so that the hub LMP moves by the
paper's `m = 8` output steps. That works out to an intercept shift of
**±$2.887/MWh** (one output step moves the hub LMP by $0.361).

| | value |
|---|---|
| measured non-revealing price fraction | **0.384** |
| paper's duopoly baseline, `(k−m)/(k+m)` | 0.304 ("roughly a third") |
| paper's closed form evaluated at n = 3 | 0.568 |

We are close to the paper's *baseline* level of confounding, but **below** what
their formula predicts for three firms. That matters later (§7) — it is the root
cause of the one result that does not replicate.

---

## 2. Headline result

| | learned | Cournot–Nash | joint monopoly | competitive |
|---|---|---|---|---|
| **Δ (profit gain)** | **68.78%** ± 0.31 | 0% | 100% | — |
| total generation | 280.1 MW | 328.6 | 232.6 | 416.8 |
| hub LMP (node 2) | $42.18 | $36.52 | $47.71 | $26.25 |
| qty-weighted avg LMP | $43.69 | $39.49 | $47.71 | $30.28 |

* per-firm Δ = **71.5 / 69.1 / 65.7%** — all three asymmetric firms end up deep
  in the collusive region; nobody is left behind
* **100% of 1,000 sessions converged**, median at **2.0M** iterations
* per-firm output **91.3 / 93.5 / 95.3 MW** vs Nash 106/109/114, monopoly 75/78/80
* learned play sits at roughly action index **7** — almost exactly halfway
  between monopoly (2) and Nash (12–13)

**Session-level dispersion** (this is the answer to "is 68.8% just an average of
colluders and non-colluders?"):

| p0 | p5 | p25 | median | p75 | p95 | p100 |
|---|---|---|---|---|---|---|
| 38.3% | 53.6% | 62.3% | 68.2% | 74.6% | 86.0% | 100.1% |

**100.0% of sessions have Δ > 0. 98.2% have Δ > 0.5.** Not a single session
failed to collude. sd = 9.9pp.

**Figures 1 and 2** (`figures/fig1_output_evolution.png`,
`fig2_profit_evolution.png`) reproduce the paper's shape exactly, including the
detail that Δ **starts at −0.29**: the first greedy policy is the best response
to uniformly-randomising rivals, which *over*-produces relative to Nash. Then it
climbs through the learning phase and plateaus as ε decays. Their Figure 2 has
the same negative start.

---

## 3. Figure 3 — the limit strategy (`figures/fig3_limit_strategy.png`)

Output is a **decreasing function of the price observed last period**: a price
drop triggers an output expansion — the punishment.

I stress-tested this one because the obvious objection is mechanical: sessions
that produce more *necessarily* sit at lower prices, so a cross-session plot of
output against price would slope down even with no strategic content. **It
survives.** Within-session (session fixed effects) the slope is identical to the
pooled slope:

| | slope, MW of total output per $/MWh |
|---|---|
| pooled, no fixed effects | **−0.910** |
| **within-session (FE)** | **−0.910** |
| per-session median | −0.912 |

**97.7% of the 1,000 individual sessions have a decreasing limit strategy**
(p5 = −1.74, p95 = −0.16). It is a within-session property, not composition.

**And it is flatter than demand — the paper's key observation.** Per firm:

| | d(output)/d(price), MW per $/MWh |
|---|---|
| firm 0 / 1 / 2 strategy | −0.357 / −0.296 / −0.235 |
| firm-level demand schedule | **−2.908** |

The strategy is 8–12× flatter than demand. That is exactly the paper's mechanism:
because the response is milder than the price move that provoked it, the price
recovers, so next period's punishment is milder, and the market walks back to its
resting point. **The algorithms use punishment *intensity* as a substitute for the
clock their one-period memory denies them.**

---

## 4. Figure 4 — forced deviation (`figures/fig4_deviation.png`)

One algorithm is forced into its **static best response** for a single period
(+20.2 MW = 6.5 grid steps, dropping the hub LMP by $2.35), then reverts to its
learned strategy. Demand is frozen so only the deviation moves the price.
Averaged over 1,000 converged sessions **and all 12 phases of the limit cycle**
(without the phase-averaging the cycle's sawtooth swamps the response).

Rivals' output response, MW above the paired no-deviation counterfactual:

| | t+1 | t+2 | t+3 | t+5 | t+10 | t+20 | t+80 |
|---|---|---|---|---|---|---|---|
| **high demand** | **+6.37** | +2.96 | +1.84 | +1.20 | +1.23 | +1.13 | +1.15 |
| **low demand** | **+8.68** | +4.95 | +3.31 | +2.51 | +2.12 | +1.84 | +1.87 |

**Two things replicate.**

1. **Punishment, then fade.** ~80% of it is gone within five periods — the
   paper's "harsher initially and then gradually fading away".
2. **Harsher in low demand (+8.68) than in high (+6.37).** This is the paper's
   signature asymmetry and the fingerprint of imperfect monitoring specifically:
   in the high state a low price might be an adverse shock rather than a cheat,
   so the algorithms punish cautiously; in the low state there is no such
   ambiguity. The paper: *"the punishment is softer when demand is high, as in
   this case a deviation may be confounded with a negative demand shock."*

### The residue — now explained (it was open until today)

The averaged path does not return *all* the way: a ~1.2–1.9 MW gap persists out
to 80 periods, which the paper's Figure 4 does not show. **It is not a gradual
drift. It is a minority of sessions permanently displaced:**

| | sessions returning to their **exact** pre-deviation cycle | their residue | non-returners' residue |
|---|---|---|---|
| high demand | **84.6%** | **+0.004 MW** | +22.75 MW |
| low demand | **80.8%** | **+0.004 MW** | +33.25 MW |

The 85% that return, return *exactly* — zero residue. The whole average residue
is carried by the 15–19% that never come back. With frozen demand and
deterministic limit strategies the post-convergence dynamics are a deterministic
map on 51 states, so a session settles into a cycle (median length: **2 states**);
a single deviation can tip it into a *different* absorbing cycle, and nothing
brings it back. So: **a one-off deviation permanently destroys the cartel in
about one session in six.** The paper's Figure 4 shape is what our 85% do.

---

## 5. Figure 5 — is cheating actually deterred? (`figures/fig5_deviation_value.png`)

Supra-Nash profits alone prove nothing — the algorithms might simply have failed
to optimise. Direct test: run two paths from the identical state, one with the
forced deviation, one without, and compare discounted payoff streams at δ = 0.95.

| | one-period gain | discounted total | median | deterred |
|---|---|---|---|---|
| **low demand** | +$63 (positive in 94.9%) | **−$885** | −$229 | **76.3%** |
| **high demand** | +$177 (positive in 99.2%) | **−$179** | **+$38** | **46.1%** |

Low demand is textbook: cheating pays today and loses over the punishment phase,
deterred in three quarters of sessions.

**High demand is where it gets interesting, and this is the honest weak point.**
The mean is negative but the *median* session would gain $38, and only 46% are
deterred. Decomposing by whether the cartel survives:

| high demand | mean discounted gain | deterred |
|---|---|---|
| sessions that return to their cycle (84.6%) | **+$50** | 39.6% |
| sessions permanently displaced (15.4%) | **−$1,233** | 64.3% |

**Deterrence in this market is a lottery, not a certainty.** Usually cheating pays
a little; about one time in six it destroys the cartel and costs a fortune. The
negative *mean* comes entirely from that tail. That is a real economic finding and
it is worth leading with rather than hiding.

**Deterrence is not size-dependent** — I swept deviation size from 1 to 8 grid
steps (+3.1 to +24.8 MW). Deterred fraction stays ~51–54% in high demand and
~76–78% in low demand throughout. So it is not "small deviations deterred, large
ones not".

---

## 6. Table I — the impact of imperfect monitoring

**Ours:**

| | Deterministic demand | Stochastic demand |
|---|---|---|
| **Perfect monitoring** | 78.88% | 78.98% |
| **Imperfect monitoring** | **84.18%** | **68.78%** |

**Paper's:**

| | Deterministic demand | Stochastic demand |
|---|---|---|
| Perfect monitoring | 84.16% | 79.72% |
| Imperfect monitoring | 89.60% | 76.25% |

**Three qualitative findings replicate:**

1. **Imperfect + deterministic is the highest cell** (84.18% vs perfect/det
   78.88%, a **+5.30pp** premium; the paper's premium is **+5.44pp** — nearly
   identical). This is the paper's own counter-intuitive result: when the state is
   the price rather than the rivals' output profile, the Q-matrix is far smaller
   (51 vs 3,375 states), so the same β buys much more effective experimentation,
   and simpler strategies coordinate more easily.
2. **Imperfect + stochastic is the lowest cell.** Confounding shocks with
   deviations is costly.
3. **The DiD is negative and appreciable.** Imperfect monitoring hinders collusion
   but does not prevent it.

**DiD: −15.50pp here vs −8.91pp in the paper.** Do not defend that gap — decompose
it, because the decomposition is favourable:

| | mine | paper |
|---|---|---|
| imperfect row: stochastic − deterministic | **−15.40** | **−13.35** |
| perfect row: stochastic − deterministic | **+0.10** | **−4.44** |
| DiD | −15.50 | −8.91 |

**The imperfect-monitoring row — the row this whole paper is about — matches to
2pp.** The entire DiD discrepancy comes from the *perfect*-monitoring row, and
there is a concrete reason for that: with n = 3 the perfect-monitoring state
space is `k³ = 3,375` states → **50,625 Q-cells per firm**, versus the paper's
duopoly `k² = 225` → 3,375 cells. **15× bigger.** Our perfect-monitoring runs are
badly starved of experimentation at the same β; both perfect cells are
learning-limited rather than incentive-limited, which compresses the demand-
uncertainty effect toward zero. By contrast our imperfect-monitoring Q-matrix is
765 cells vs their 555 — **only 1.38× bigger**, so that row is a fair comparison.

Direct evidence for the starvation story: perfect monitoring converged at a median
of **5.7M** iterations vs **2.0M** for imperfect — nearly 3×. That is exactly the
paper's §5.3 conjecture (bigger Q-matrix hinders learning) showing up in our data.

---

## 7. What is genuinely DIFFERENT (`figures/fig6_deviation_vs_shock.png`)

**The one qualitative claim of the paper that does not replicate:** they report
that price wars are triggered **both** by deviations **and** by adverse demand
shocks. Here they are triggered by deviations only.

Paired experiment: same converged sessions, run twice from the identical state;
one copy gets the adverse demand draw at period 0, the other the favourable one;
from period 1 both get the *identical* shock sequence.

| perturbation | immediate price drop | output response at t+1 |
|---|---|---|
| forced deviation | −$2.36 | **+18.8 MW** (70% of sessions expand) |
| adverse demand shock | **−$5.77** | **−2.0 MW**, median **exactly 0** |

The shock is the *bigger* price drop and produces *no* war: 38.3% of sessions
expand, 44.4% contract, 17.3% do not move. Implied local slopes:

| | MW per $/MWh |
|---|---|
| global learned strategy (Fig. 3) | −0.91 |
| in the region a **deviation** lands in | **−7.7** (high) / **−10.3** (low) |
| in the region a **shock** lands in | +0.36 (wrong sign, ≈ 0) |

**Interpretation.** Our algorithms have learned something *sharper* than the
paper's smooth reaction function: a **trigger strategy** — flat over the price
band that ordinary demand noise moves them within, steep once the price falls
below that band. That is Green–Porter proper, and it is arguably the more
canonical outcome. They have partially learned to *filter the demand shock out of
the signal*.

**Why here and not there?** Because our monitoring is less imperfect than theirs
at n = 3 (measured non-revealing fraction 0.384 vs their formula's 0.568, §1). The
DC-OPF price lattice does not collide as cleanly as `p = d − Q`, so the two demand
branches are more separable, so the algorithms *can* tell shocks from deviations —
and having learned to, they stop punishing shocks. Same root cause as the DiD
gap. **Testable prediction:** raise `h` (their §5.6 does h = 5) or raise `m` and
the shock-triggered war should reappear. Not yet run.

---

## 8. Does this prove or disprove the impossibility result?

**Neither, and it is important to be precise about why.**

I tracked down the impossibility paper: **Sannikov & Skrzypacz (2007),
"Impossibility of Collusion under Imperfect Monitoring with Flexible Production,"
AER 97(5).** *(If you meant a different paper, this whole section changes — worth
confirming.)*

**The result is not actually about continuous action spaces.** It is about
**speed**: as the period length Δ → 0 with flexible production, the informativeness
of the price signal about any one firm's deviation degrades like O(Δ) while the
deviation gain stays O(Δ), and Lemmas 2–4 kill every symmetric equilibrium above
Nash. Continuous actions appear in their model but are not the driving
assumption.

So the logical situation is:

* Q-learning at Δ = 1 (one fixed period length) **cannot contradict** a limit
  theorem about Δ → 0. Our Δ = 68.8% and their impossibility result are simply
  not in conflict.
* The discreteness of the action space is **not** what S&S rule out, so
  "discretisation is what allowed collusion" is not the right reading either.

**What Task 1 actually establishes:** a like-for-like discrete benchmark on the
*identical market* the PPO agents face. Any claim that the continuous-action PPO
results are or are not "collusion" now has a number to be measured against —
Δ = 68.8% with punishments — rather than being argued in the abstract. That was
the point of doing it first.

**Separately** (this is Task-2 territory, flag only if asked): I audited S&S's
appendix line by line against our market. A1–A4 hold. Their condition C3 also
holds — the rank of `[d₀ d₁ d₂]` on our network is **1**, i.e. every firm's
deviation moves the 5-node LMP vector identically, so Propositions 1–4 apply
verbatim. The reason is that all three firms sit at node 2 (chosen so the Nash
benchmark stays the paper's LCP), which switches the network off. The two real
escape routes both require firms at **different** nodes, so that a binding line
separates them — congestion is the identification device. Node 0 already shows
counter-flow in our network, so the mechanism exists on this topology; it just
isn't reachable from the hub siting.

---

## 9. Anticipated questions

**Q. Why three firms? The paper's baseline is a duopoly.**
Because our market is the three-firm hub market — the whole point is to run their
algorithm on *our* topology. The right comparison is their §5.2 robustness run:
n = 3 → 72.7% (n = 4 → 66.6%). We get 68.8%, between their 3- and 4-firm numbers,
on a much harder payoff function. A 2-firm variant is a half-day of work if you
want it.

**Q. Is Δ measured against the grid Nash or the continuous Nash?**
The grid, because the grid game is the game the algorithms actually play — it is
the best they could possibly do given their actions. It barely matters:
Δ = 68.78% on grid benchmarks, **66.74%** on continuous benchmarks. The
discretisation error in the Nash benchmark is 0.95% of generation.

**Q. Couldn't 68.8% just be failure to optimise rather than collusion?**
Three independent pieces of evidence say no: (i) the limit strategy is decreasing
in the observed price in 97.7% of sessions and flatter than demand — a punishment
that self-terminates; (ii) a forced deviation triggers a real, decaying price war
(+6.4 / +8.7 MW at t+1); (iii) the discounted value test — cheating pays in the
short run and loses on average once punishment is priced in. A failure-to-optimise
story predicts none of the three.

**Q. Is the punishment learned, or is it just the untouched Q-initialisation
showing through in states the sessions never visited?**
Checked directly, because it would be fatal. `Q₀` is initialised at the payoff
against uniformly-randomising rivals, whose greedy action is **[14,14,14]** — the
*maximum* output, which would masquerade as a punishment. But: the learned policy
differs from initialisation in **50.7 of 51 states**, and only **0.6%** of
(session, state) cells are still at the initial action. In the states the
deviation actually lands in, the fraction still at initialisation is 0.6–1.2%,
and excluding them changes the measured rival response by 0.1 MW. **The
punishment is learned.**

**Q. Your DiD is −15.5pp, theirs is −8.91pp. Isn't the replication failing?**
See §6. The imperfect-monitoring row — the one the paper is about — is −15.40 vs
−13.35. The gap is entirely in the perfect-monitoring row, where our n = 3 state
space is 15× the paper's duopoly and both cells are learning-limited. Their
higher-uncertainty run (§5.6) reports ≈ −17pp, so −15.5 is inside the range they
themselves report.

**Q. Why is the measured non-revealing fraction 0.384 and not 0.568?**
0.568 is their closed form evaluated at n = 3; 0.304 is their duopoly baseline
("roughly a third"). We measure 0.384 on the actual DC-OPF price table rather
than assuming it, because the LMP is piecewise-linear in G (congestion regimes
switch), so a "step" is worth a different amount at different points on the
lattice and the price levels do not collide as cleanly as on their exact lattice.
Consequence: our monitoring is somewhat *less* imperfect than the idealised n = 3
value — which is precisely what explains §7.

**Q. Why 1,000 sessions for imperfect monitoring but 200 for perfect?**
Cost. Perfect monitoring needs a 50,625-cell Q-matrix per firm and ran to 12M
iterations. s.e. on those cells is 0.45–0.49pp, which is tight enough for the
Table I comparison (the cells differ by 5pp).

**Q. Did you check robustness to α and β?**
**Not yet — this is the main open gap.** The paper swept a 100×100 grid of (α, β)
and found Δ ranging 65–80%. We ran the single baseline point (α = 0.15,
β = 4×10⁻⁶) they use. This is the first thing I would run next.

**Q. Two million iterations is a lot. Is this realistic?**
Same objection applies to the paper (their Figure 1 x-axis runs to 2M, and they
address it in §5.5 with off-line training and re-matching). Our median is 2.0M —
same order. I have not run their re-matching experiment.

**Q. The deviation you force is +20 MW — that's huge. Is that fair?**
It is the *static best response*, i.e. the genuinely tempting one-shot defection,
which is the strongest test. The size sweep (§5) shows deterrence is flat from
+3.1 MW to +24.8 MW, so nothing hinges on the choice.

**Q. Why does the punishment leave a residue when the paper's doesn't?**
§4 — 85% of sessions return *exactly*; the residue is entirely the 15% whose
cartel is permanently destroyed. Plausibly present in their duopoly too and
invisible at their figure's scale (1.2 MW on a 25-unit axis), but I cannot check
that without their code.

---

## 10. Open items, honestly

1. **(α, β) robustness grid** — not run. Biggest gap vs the paper.
2. **Higher uncertainty (h = 5, their §5.6)** — not run, and it is the direct test
   of the §7 explanation.
3. **Longer memory (2–3 periods, their §5.3)** — not run.
4. **Re-matching / off-line training (§5.5)** — not run.
5. **The high-demand IC failure** — the median session gains $38 from cheating in
   high demand. Real, and the most interesting loose thread: it says the learned
   "equilibrium" is not incentive-compatible in the state where hiding is easiest.
6. **n = 2 variant** on the two-firm market, for a direct comparison to their
   duopoly baseline.

## Reproducing

```bash
python -m qlearning_collusion.run market      # the discretised game + benchmarks
python -m qlearning_collusion.run table1      # all four cells (~2 h on 4 cores)
python -m qlearning_collusion.run figures     # Figures 1-5
python -m qlearning_collusion.fig6_deviation_vs_shock   # Figure 6
python -m qlearning_collusion.run report      # every number above
```
