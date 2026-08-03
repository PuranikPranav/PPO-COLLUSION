# Slide plan — Task 1: Q-learning replication

**14 core slides + 7 backups.** Layout convention: figure on the **left ~60%**,
bullets stacked on the **right ~40%**. Bullets below are written to be pasted
verbatim — they are short on purpose. The *"Say:"* line is what you talk over the
slide; don't put it on the slide.

Figures live in `qlearning_collusion/figures/`.

---

## 1 — Title

> ## Can Q-learning algorithms learn to collude on a real electricity network?
> Replicating Calvano, Calzolari, Denicolò & Pastorello (2021)
> on the five-node DC-OPF market
>
> *Task 1 of 2 — the discrete benchmark for the PPO work*

**Say:** "The assignment was: don't take either side on faith. Take the economics
papers' exact algorithm, discretise our action space, run it on our market, and
see whether their result actually shows up. Short answer: it does."

---

## 2 — The question I'm refereeing

No figure. Two boxes side by side.

**Left box — the economics papers claim:**
- Simple learning algorithms teach themselves to collude
- No communication, nobody programmed it
- Calvano et al. 2021: **Δ = 76% (duopoly), 72.7% (3 firms)**

**Right box — the impossibility paper claims:**
- Collusion under imperfect monitoring is impossible
- Sannikov & Skrzypacz (2007)

**Bottom strip:**
> **My job:** rebuild their algorithm exactly, run it on **our** market, measure.

**Say:** "These can't both be right in the same setting — so the first thing to
establish is whether the empirical claim survives on a realistic market at all.
That's what gives us a yardstick for the PPO work later."

---

## 3 — What I built: their algorithm, our market

No figure. One two-column table — this is your credibility slide.

| Kept **identical** to the paper | Replaced with **our** market |
|---|---|
| Learning rule (their eq. 3) | `p = d − Σq` → **LMP from the ISO's DC-OPF** |
| ε-greedy, `ε_t = e^(−βt)` | 1 price → **5 nodal prices, congestion, line limits** |
| α = 0.15, β = 4×10⁻⁶, δ = 0.95 | 2 identical firms → **3 asymmetric firms** |
| `Q₀` = payoff vs. uniform rivals | MC = 0 → **MC ∈ {15,16,18}, quadratic costs** |
| Ties → higher output | |
| Convergence = greedy stable 100k periods | |
| k = 15, h = 2, ξ = 0.2, m = 8 | |
| 1,000 sessions | |

**Say:** "Everything on the left is theirs, verbatim, including the awkward
choices like breaking ties toward higher output — which biases *against* finding
collusion. Only the payoff function changed."

---

## 4 — The actual assignment: discretising the action space

No figure — use this table.

| rung | 0 | **2** | 7 | **12** | **13** | 14 |
|---|---|---|---|---|---|---|
| firm 0 | 69.1 | **75.3** | 90.8 | **106.3** | 109.4 | 112.5 |
| firm 1 | 71.6 | **77.8** | 93.3 | **108.8** | 111.9 | 115.0 |
| firm 2 | 73.3 | **79.5** | 95.0 | 110.5 | **113.6** | 116.7 |

**Bullets:**
- 15 output levels per firm, step **3.098 MW**
- Grid spans [monopoly, Nash] **+ 20% on each side**
- **Monopoly lands on rung 2** — paper: rung 2 ✓
- **Nash lands on rung 12–13** — paper: rung 12 ✓
- Discretisation error in the Nash benchmark: **0.95%**

**Say:** "This is the slide that says I didn't rig the game. Room above Nash so
firms *can* cheat and *can* punish; room below monopoly so they could over-collude.
Same slots as the paper, so the game has the same shape."

---

## 5 — What makes monitoring imperfect

No figure. Big centred arithmetic.

> **One firm moves one rung (3.1 MW)** → hub price moves **$0.36**
> **A demand shock** → hub price moves **$2.89 = 8 rungs**

**Bullets:**
- Firms **never** see rivals' output — only the price
- Price drops $2.89. Two explanations fit **equally well**:
  - demand came in low, nobody cheated
  - rivals collectively produced 8 rungs more
- **Measured** non-revealing fraction: **0.384** (paper's baseline: 0.304)
- Kill the shock → non-revealing fraction **0.007**. Monitoring becomes perfect.

**Say:** "This is the whole game. And the last line is the proof that demand
uncertainty is doing the work — without it, the price gives everything away and
Green–Porter has nothing to explain."

---

## 6 — Headline: it replicates

**Figure: `fig2_profit_evolution.png`**

**Bullets:**
- **Δ = 68.78% ± 0.31**
- Paper: 76.25% (2 firms), **72.7% (3 firms)**
- **100% of 1,000 sessions converged**, median 2.0M iterations
- Output **328.6 → 280.1 MW**; hub price **$36.52 → $42.18**
- Δ starts at **−0.29** — same as the paper's Figure 2

**Say — point at the negative start first:** "That dip below zero looks like a bug
and isn't. Early on, rivals behave randomly, and the best response to randomness
is to *over*-produce relative to Nash. Their Figure 2 has exactly the same
signature — it's a fingerprint that the implementation is right."

Then: "Then it climbs as exploration decays and flattens at 69%. Two million
iterations isn't arbitrary — it's when `e^(−βt)` switches exploration off."

---

## 7 — Not an average of colluders and non-colluders

**Figure: `fig_delta_distribution.png`**

**Bullets:**
- **Every single one of 1,000 sessions had Δ > 0**
- 98.2% above Δ = 0.5
- median 68.2%, 5th pct 53.6%, 95th pct 86.0%
- Worst session: 38.3%. Best: 100.1%

**Say:** "The obvious worry with a 69% average is that it's half perfect cartels
and half nothing. It isn't — the whole distribution sits in the collusive region."

---

## 8 — *(section divider)*

> # Is this genuine collusion, or just failure to optimise?
> High profits alone prove nothing. Three independent tests.

**Say:** "This is the distinction that matters for antitrust. If the algorithms
simply never discovered that cheating was profitable, that's incompetence, not
collusion. So I ran three tests that a failure-to-optimise story cannot pass."

---

## 9 — Test 1: the learned strategy IS a punishment

**Figure: `fig3_limit_strategy.png`**

**Bullets:**
- Axis: price seen **last** period → output **this** period
- **Line slopes down** = a price drop triggers an output expansion
- **97.7% of the 1,000 sessions** learned a downward-sloping rule
- Strategy is **8–12× flatter than demand** (−0.30 vs −2.91)
- Flatness ⟹ price recovers ⟹ punishment fades **by itself**

**Say — the flatness is the subtle bit, spend time here:** "Sloping down is the
punishment. But the *flatness* is the clever part. Because they retaliate less
than the price drop that provoked them, the price partly recovers, so next
period's retaliation is smaller, and the war dies out on its own. These
algorithms have one period of memory — no clock, no way to count 'punish for 7
rounds.' They use punishment *intensity* as a substitute for a clock. That's the
paper's most elegant finding and it reproduces here."

---

## 10 — Test 2: deviations get punished, harder when they can't hide

**Figure: `fig4_deviation.png`**

**Bullets:**
- One firm forced to cheat for **one period** (+20.2 MW), then reverts
- Paired design: identical twin session with no cheat, subtracted

| rivals' extra output | t+1 | t+2 | t+3 | t+5 |
|---|---|---|---|---|
| **high demand** | +6.4 | +3.0 | +1.8 | +1.2 |
| **low demand** | **+8.7** | +4.9 | +3.3 | +2.5 |

- **~80% of the punishment gone within 5 periods**
- **Harsher in LOW demand** — the fingerprint of imperfect monitoring

**Say — the asymmetry is the money line:** "Look at the two rows. In high demand
they punish *less*. Why? Because in the high state, a low price might genuinely
be a bad demand draw — punishing a possibly-innocent rival is expensive. In the
low state there's no such excuse, so they hammer it. **They learned to calibrate
punishment to how ambiguous the evidence is.** Nobody programmed that. The paper
reports the same asymmetry in the same direction."

---

## 11 — Test 3: cheating doesn't pay — but it's a lottery

**Figure: `fig5_deviation_value.png`**

**Bullets:**
- Clone the session; force one copy to cheat; compare **discounted** profit

| | day-1 gain | discounted total | deterred |
|---|---|---|---|
| **low demand** | +$63 | **−$885** | 76.3% |
| **high demand** | +$177 | **−$179** (median **+$38**) | 46.1% |

**Then the decomposition box — this is your best original finding:**

| high demand | mean gain from cheating |
|---|---|
| cartel survives (84.6%) | **+$50** |
| cartel destroyed (15.4%) | **−$1,233** |

**Say:** "Low demand is textbook — pays today, loses over the punishment phase.
High demand looks weak: only 46% deterred, and the median session would actually
*gain* by cheating. But splitting it explains everything. **Deterrence here is a
lottery, not a certainty.** Usually cheating pays you a little. About one time in
six it knocks the system permanently onto a worse equilibrium and costs a
fortune. On average that risk is enough — but it's a risk premium, not a
guaranteed punishment. I think that's a more realistic picture of an algorithmic
cartel than the paper's."

---

## 12 — Table I: the cost of being blind

No figure. Two tables side by side.

**Mine:**

| | certain demand | random demand |
|---|---|---|
| perfect monitoring | 78.88% | 78.98% |
| imperfect monitoring | **84.18%** | **68.78%** |

**Paper's:**

| | certain demand | random demand |
|---|---|---|
| perfect monitoring | 84.16% | 79.72% |
| imperfect monitoring | 89.60% | 76.25% |

**Bullets:**
- **The counter-intuitive cell replicates**: imperfect + certain demand is the
  *highest*. Smaller Q-matrix (51 vs 3,375 states) ⟹ easier to learn.
  Premium **+5.30pp** vs their **+5.44pp**
- Imperfect + random is the lowest cell in both
- **DiD: −15.5pp (mine) vs −8.91pp (theirs)** — decompose, don't defend:

| | mine | paper |
|---|---|---|
| **imperfect row** | **−15.40** | **−13.35** |
| perfect row | +0.10 | −4.44 |

**Say:** "The row this paper is actually about matches to two points. The whole
discrepancy is the *perfect*-monitoring row — and there's a concrete reason. With
3 firms the perfect-monitoring Q-matrix is 50,625 cells, **15× the paper's
duopoly**. Both perfect cells are limited by learning difficulty, not economics.
Evidence: they needed 5.7M iterations to converge versus 2.0M. My imperfect
matrix is only 1.38× theirs, which is why that row is the fair comparison."

---

## 13 — What's different: they punish cheating, not bad luck

**Figure: `fig6_deviation_vs_shock.png`**

**Bullets:**
- Paper: price wars triggered by deviations **and** adverse shocks
- Here: **only by deviations**

| | price fall | response next period |
|---|---|---|
| forced cheat | −$2.36 | **+18.8 MW** |
| adverse shock | **−$5.77** | **−2.0 MW, median exactly 0** |

- Reaction steepness: **−7.7 to −10.3** where a cheat lands; **≈ 0** where a shock lands
- ⟹ a **trigger strategy**, not a smooth reaction function

**Say:** "The shock is more than twice the price drop and produces no reaction at
all. So they are not following 'low price → produce more.' They ignore price
moves inside the range everyday demand noise creates, and retaliate only past a
threshold. **They learned to filter the noise out of the signal.** That's
Green–Porter's original trigger-price construction, arguably more canonical than
what the paper found. And I can explain why we get it and they don't: our
monitoring is *less* imperfect — 0.384 versus their 0.568 at three firms — so the
signal is clean enough to separate the two causes."

**End with the prediction:** "That's falsifiable. Raise the number of demand
states to 5, as their §5.6 does, and the shock-triggered war should come back.
One day of compute."

---

## 14 — What this does and does not settle

No figure. Three lines, well spaced.

> **Replication: ✓** Q-learning with a discretised action space collides on our
> network market. Δ = 68.8%, sustained by punishments.
>
> **Impossibility result: not contradicted.** Sannikov–Skrzypacz is about
> **speed** (period length → 0), not action-space continuity. We ran at one fixed
> period length. Different questions.
>
> **What we now have:** a like-for-like discrete benchmark on the *identical*
> market the PPO agents face. **Δ = 68.8% is the number to measure PPO against.**

**Say:** "I want to be careful here rather than overclaim. I read the impossibility
paper and its driving assumption is the ability to react arbitrarily fast, not
discreteness. So these two literatures aren't actually in conflict — and Task 1
doesn't adjudicate between them. What it does is give us a calibrated yardstick."

**Ask him directly:** "Before I go further — is Sannikov–Skrzypacz the paper you
had in mind? If it's a different one, my reading of section 14 changes."

---

## 15 — Next

- **(α, β) robustness grid** — the paper sweeps 100×100 and gets 65–80%. I ran
  their single baseline point. ← biggest gap
- **h = 5 higher uncertainty** — directly tests the slide-13 explanation
- Longer memory (2–3 periods); re-matching / off-line training
- **Then: PPO with continuous actions, measured against 68.8%**

---

# BACKUP SLIDES

Keep these after the "Next" slide. Each answers one specific attack.

### B1 — "Is the punishment real, or leftover initialisation?"
**The most dangerous question.** The blank Q-matrix's default action is **maximum
output** — which would masquerade as a punishment.
- Learned policy differs from init in **50.7 of 51** states
- Only **0.6%** of (session, state) cells untouched
- Excluding them changes the measured punishment by **0.1 MW**

### B2 — "Isn't Figure 3's slope mechanical?"
High-output sessions automatically sit at low prices, so a cross-session plot
would slope down regardless of strategy.
- Pooled slope: **−0.910**
- **Within-session (fixed effects): −0.910** — identical
- 97.7% of individual sessions slope down

### B3 — "Grid Nash or continuous Nash?"
- Δ = **68.78%** on grid benchmarks
- Δ = **66.74%** on continuous benchmarks
- Grid Nash 328.6 MW vs continuous 325.5 MW → 0.95%

### B4 — "Why 3 firms?"
Our market is the three-firm hub market. Paper's §5.2: n=3 → 72.7%, n=4 → 66.6%.
**72.7% is the comparison number.**

### B5 — Where the residue in Figure 4 comes from

| | return to their exact original cycle | their residue | everyone else |
|---|---|---|---|
| high demand | **84.6%** | **+0.004 MW** | +22.8 MW |
| low demand | **80.8%** | **+0.004 MW** | +33.2 MW |

Frozen demand + deterministic policies ⟹ a 51-state machine that must end in a
loop. There is more than one loop, and nothing random to knock it back.

### B6 — "Maybe they only deter *small* cheating"
Swept the forced deviation from +3.1 MW to +24.8 MW. Deterred fraction: 51–54%
(high), 76–78% (low). **Flat.**

### B7 — Output evolution
**Figure: `fig1_output_evolution.png`** — greedy outputs start above the Nash
band, fall through it, settle at 91/94/95 MW.

---

# Delivery notes

- **Slides 9–11 are the heart.** If you're short on time, cut 7 and 15, never
  those three.
- **Volunteer your weaknesses** — the 46% deterrence in high demand, the missing
  (α,β) grid, the shock result that doesn't replicate. Every one of them has a
  worked explanation behind it, and volunteering them is what makes the numbers
  you *do* claim credible.
- **One framing to hold:** every result after slide 8 comes from the same paired
  twin-study design — clone the session, poke one copy, subtract. Say that once,
  early, and the rest of the deck reads as one method rather than four.
- If he asks **"is 69% a lot?"**: output down 15%, hub price up $5.66/MWh (+15.5%),
  profit up $896 per period. And note **profit runs ahead of quantity** — only
  50.5% of the way to monopoly in output, but 68.8% in profit, because profit is
  flat near the monopoly point. **The profit damage exceeds the visible quantity
  distortion**, which is exactly what makes it hard for a regulator to spot.
