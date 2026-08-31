# Results — does Q-learning collude on this topology?

**Yes.** Tabular Q-learning with a discretised action space converges to
substantially supra-competitive profits on the three-firm hub market, and the
strategies it converges to are genuinely collusive: deviations trigger price
wars that fade, and the price wars are harsher exactly where theory says they
should be.

Baseline experiment: **imperfect monitoring** (state = last period's hub LMP)
with **stochastic demand**, 1,000 independent sessions, `k = 15`, `α = 0.15`,
`β = 4×10⁻⁶`, `δ = 0.95`. **100% of sessions converged**, median at **2.0M
iterations** — the same order of magnitude the paper reports.

---

## 1. Headline number

| | learned | Cournot–Nash | joint monopoly | competitive |
|---|---|---|---|---|
| **Δ (profit gain)** | **68.78%** ± 0.31 | 0% | 100% | — |
| total generation | 280.1 MW | 328.6 | 232.6 | 416.8 |
| hub LMP (node 2) | $42.18 | $36.52 | $47.71 | $26.25 |
| qty-weighted avg LMP | $43.69 | $39.49 | $47.71 | $30.28 |

Per-firm Δ = **71.5% / 69.1% / 65.7%** — all three asymmetric firms end up deep
in the collusive region, none is left behind.

**Against the paper:** 68.8% here vs **76.25%** in their duopoly baseline and
**72.7%** in their three-firm robustness run (§5.2). Given that this market
replaces a textbook `p = d − Σq` with a five-node DC-OPF clear, congestion
regimes, quadratic costs and asymmetric firms, landing 4 points under their
three-firm figure is a close replication.

![Figure 2](figures/fig2_profit_evolution.png)

The trajectory has the paper's exact shape: Δ starts at **−0.29** — the
algorithms' first greedy policy is the best response to uniformly-randomising
rivals, which *over*-produces relative to Nash — then climbs through the
learning phase and plateaus once ε decays.

![Figure 1](figures/fig1_output_evolution.png)

Greedy outputs start above the Cournot–Nash band (~110–116 MW), fall through it,
and settle at **91.3 / 93.5 / 95.3 MW** — far below Nash (105/108/112) and about
a fifth of the way from monopoly (76/78/78).

---

## 2. The strategies are genuinely collusive

Supra-Nash profits alone prove nothing — the algorithms might simply have failed
to optimise. Three pieces of evidence say otherwise.

### 2a. The limit strategy is a Green–Porter price war

![Figure 3](figures/fig3_limit_strategy.png)

Output is a **decreasing function of the price observed last period**: a price
drop triggers an output expansion — the punishment. And, exactly as in the
paper, **the strategy is much flatter than the demand schedules**. That is what
makes the punishment self-terminating: because the response is milder than the
price move that provoked it, the price recovers, so next period's punishment is
milder still, and the market walks back to its resting point. The algorithms use
punishment *intensity* as a substitute for the clock their one-period memory
denies them.

### 2b. Forced deviations get punished — harder when they can't hide

![Figure 4](figures/fig4_deviation.png)

One algorithm is forced into its static best response for a single period, then
reverts to its learned strategy; demand is frozen so only the deviation moves
the price. Averaged over 1,000 converged sessions and all 12 phases of the limit
cycle:

| punishment (MW above the no-deviation counterfactual) | t+1 | t+2 | t+3 | t+5 | t+10 | t+20 | t+80 |
|---|---|---|---|---|---|---|---|
| **high demand** | +6.36 | +2.93 | +1.83 | +1.19 | +1.23 | +1.16 | +1.15 |
| **low demand** | +8.69 | +4.92 | +3.34 | +2.50 | +2.11 | +1.86 | +1.86 |

Two things to read off this.

**The punishment is harsher in low demand (+8.69 MW) than in high demand
(+6.36 MW).** This is the paper's signature asymmetry, and it is the fingerprint
of *imperfect monitoring specifically*: in the high state a low price might be
an adverse demand shock rather than a cheat, so the algorithms punish more
cautiously; in the low state there is no such ambiguity.

**The price war decays fast, then leaves a residue.** ~80% of the punishment is
gone within five periods — the "harsher initially, then gradually fading" shape
the paper describes. But unlike the paper's figure it does not return all the
way: a ~1.2–1.9 MW gap persists out to 80 periods.

That residue is **not** a gradual drift, and it is now explained. Decomposing by
whether a session returns to its exact pre-deviation cycle:

| | returns to its exact pre-deviation cycle | residue of those | residue of the rest |
|---|---|---|---|
| high demand | **84.6%** | **+0.004 MW** | +22.75 MW |
| low demand | **80.8%** | **+0.004 MW** | +33.25 MW |

The 85% that return, return *exactly*. The entire average residue is carried by
the 15–19% that never come back. With demand frozen and the limit strategies
deterministic, post-convergence play is a deterministic map on 51 states, so each
session settles into a cycle (median length 2 states); a single deviation can tip
it into a *different* absorbing cycle and nothing brings it back. **A one-off
deviation permanently destroys the cartel in about one session in six.** The
paper's Figure 4 shape is what the other 85% do.

### 2c. Cheating is unprofitable once the punishment is priced in

![Figure 5](figures/fig5_deviation_value.png)

The direct incentive-compatibility test. For each converged session, run two
paths from the identical state — one with the forced deviation, one without —
and compare discounted payoff streams at δ = 0.95:

| | one-period gain | discounted total | median | deterred |
|---|---|---|---|---|
| **low demand** | +$63 | **−$885** | −$229 | **76.3%** of sessions |
| **high demand** | +$177 | **−$179** | +$38 | **46.1%** of sessions |

In the low state the pattern is textbook collusion: cheating pays today
(+$63, positive in 94.9% of sessions) and loses over the punishment phase
(−$885), and it is deterred in three quarters of sessions.

In the high state deterrence is **much weaker** — the mean is still negative but
the *median* session would gain $38 from cheating, and only 46% are deterred.
That is not a defect of the replication; it is imperfect monitoring doing
exactly what the theory says it does. High demand is precisely the state where a
deviation can hide behind a possible adverse shock, so the equilibrium punishment
is softer and the deviation constraint nearly binds. It is the microfoundation of
the Table I result below.

**Deterrence here is a lottery, not a certainty.** Splitting the high-demand test
by whether the cartel survives the deviation (see the cycle-return table above):

| high demand | mean discounted gain | deterred |
|---|---|---|
| sessions that return to their cycle (84.6%) | **+$50** | 39.6% |
| sessions permanently displaced (15.4%) | **−$1,233** | 64.3% |

Usually cheating pays a little; about one time in six it destroys the cartel and
costs a fortune. The negative *mean* comes entirely from that tail.

Deterrence is also **not size-dependent**: sweeping the forced deviation from 1 to
8 grid steps (+3.1 to +24.8 MW) leaves the deterred fraction at ~51–54% in high
demand and ~76–78% in low demand throughout.

### 2d. But shocks do NOT trigger price wars — the one thing that differs

![Figure 6](figures/fig6_deviation_vs_shock.png)

The paper reports that price wars are triggered **both** by deviations **and** by
adverse demand shocks. Here they are triggered by deviations only.

Paired experiment: the same converged sessions run twice from the identical
state; one copy gets the adverse demand draw at period 0, the other the
favourable one; from period 1 on both get the *identical* shock sequence.

| perturbation | immediate price drop | output response at t+1 |
|---|---|---|
| forced deviation | −$2.36 | **+18.8 MW** (70% of sessions expand) |
| adverse demand shock | **−$5.77** | **−2.0 MW**, median **exactly 0** |

The shock is the *bigger* price drop and produces no war: 38.3% of sessions
expand, 44.4% contract, 17.3% do not move. Implied local slopes:

| | MW per $/MWh |
|---|---|
| global learned strategy (Figure 3) | −0.91 |
| region a **deviation** lands in | **−7.7** (high) / **−10.3** (low) |
| region a **shock** lands in | +0.36 (≈ 0, wrong sign) |

So the learned strategy is sharper than the paper's smooth reaction function: a
**trigger** — flat over the price band ordinary demand noise moves it within,
steep once the price falls below that band. That is Green–Porter proper, and the
algorithms have partially learned to filter the shock out of the signal.

**Why here and not there:** our monitoring is less imperfect than theirs at
n = 3 (measured non-revealing fraction 0.384 vs their formula's 0.568), because
the DC-OPF price lattice does not collide as cleanly as `p = d − Q`. The two
demand branches are more separable, so the algorithms *can* tell shocks from
deviations. **Testable prediction:** raise `h` (their §5.6 uses h = 5) or raise
`m` and the shock-triggered war should reappear. Not yet run.

---

## 3. Table I — the impact of imperfect monitoring

| | Deterministic demand | Stochastic demand |
|---|---|---|
| **Perfect monitoring** | 78.88% | 78.98% |
| **Imperfect monitoring** | **84.18%** | **68.78%** |

Difference-in-differences (the pure imperfect-monitoring effect):
**−15.50 pp** (paper: −8.91 pp).

Paper's Table I for comparison: 84.16 / 79.72 / 89.60 / 76.25.

Three qualitative findings replicate:

1. **Imperfect + deterministic is the highest cell (84.18%)** — higher than
   perfect monitoring. This is the paper's own counter-intuitive result: when
   the state is the price rather than the rivals' output profile the Q-matrix is
   far smaller (51 vs 3,375 states here), so the same β buys much more effective
   experimentation, and simpler strategies coordinate more easily.
2. **Imperfect + stochastic is the lowest cell (68.78%).** Confounding shocks
   with deviations is costly.
3. **The DiD is negative and appreciable.** Imperfect monitoring hinders
   collusion, but does not prevent it — 68.8% is still deep collusion.

Two honest caveats on this table:

- The effect here (−15.5 pp) is larger than the paper's baseline (−8.91 pp) but
  in line with their higher-uncertainty run (§5.6, ≈ −17 pp). Plausible reason:
  the DC-OPF price is not exactly the paper's lattice, so the confounding does
  not switch off as cleanly at the extremes.
- Under perfect monitoring, demand uncertainty costs essentially nothing here
  (78.88 → 78.98) whereas the paper loses ~4.4 pp. That is expected: with the
  state defined as rivals' past outputs, the shock is pure payoff noise, and
  with near-linear demand certainty-equivalence makes it almost irrelevant.
  The DiD *differences within each monitoring regime*, so this shows up as a
  larger DiD rather than as a bias in the comparison.

**On the perfect-monitoring cells specifically:** with `n = 3` the state space is
`k³ = 3,375` (vs `k² = 225` for the paper's duopoly), a 15× larger Q-matrix.
Those runs used 200 sessions and up to 12M iterations, and reached 100%
convergence at a median of 5.7M iterations — nearly 3× the imperfect-monitoring
median, which is itself direct evidence for the paper's §5.3 conjecture that a
bigger Q-matrix hinders learning.

---

## 4. Bearing on the impossibility result

The impossibility argument is about a **continuous** action space. This
experiment is the discrete counterpart on the same market, and it does not
contradict that argument — it brackets it. On this topology, with actions
restricted to 15 grid points per firm, Q-learning finds and sustains collusion
with punishments (Δ ≈ 69%). Any claim that the continuous-action PPO results
are or are not "collusion" now has a like-for-like discrete benchmark on the
identical market to be measured against.

Worth noting for the comparison: the grid is not doing the work by accident.
The action grid is centred on `[q^M_i, q^C_i]` extended by ξ = 0.2, which places
Cournot–Nash at action index 12–13 and joint monopoly at index 2 — the same
positions they occupy in the paper's `{70, 72½, …, 105}` grid. And the unique
pure-strategy stage-game Nash of the discretised game sits at total generation
328.6 MW against the continuous LCP Nash's 325.5 MW, so discretisation moves the
competitive benchmark by less than 1%.

---

## Reproducing

```bash
python -m qlearning_collusion.run market      # the discretised game + benchmarks
python -m qlearning_collusion.run table1      # all four cells (~2 h on 4 cores)
python -m qlearning_collusion.run figures     # Figures 1-5
python -m qlearning_collusion.run report      # every number in this file
```

Raw artefacts are in `results/` (`.npz` = learned policies + trajectories,
`.json` = summary, `.log` = training console output).
