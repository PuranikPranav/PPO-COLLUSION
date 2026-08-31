# Slides 6 onward — results, both structures paired

**Layout rule, applied to every slide: Structure 1 (2 firms) on the LEFT,
Structure 2 (3 firms) on the RIGHT. Never swap.** After two slides the audience
stops reading labels and just knows which side is which. That consistency is
what makes a paired deck fast instead of confusing.

**Narration rule: state the finding ONCE, then show it holds twice.** Do not
walk through Structure 1 and then walk through Structure 2 — that doubles the
time and halves the impact. Say "output falls in the observed price — here it is
in the duopoly, and again with three firms."

Figure paths:
`figures_two_firm/…` = Structure 1 · `figures/…` = Structure 2

---

## 6 — Both structures collude

**FIGURES**
- left `figures_two_firm/fig2_profit_evolution.png`
- right `figures/fig2_profit_evolution.png`

| | Structure 1 | Structure 2 | paper |
|---|---|---|---|
| **Δ** | **64.14% ± 0.34** | **68.78% ± 0.31** | 76.25% (n=2) · 72.7% (n=3) |
| converged | 100% of 1,000 | 100% of 1,000 | all |
| median iterations | 2.10M | 2.00M | ~10⁶ |

**Takeaway strip:** *Both curves start negative, climb through learning, and
plateau deep in the collusive region.*

**Say:** "Same shape in both, and the same shape as the paper's Figure 2 —
including the detail that Δ *starts negative*. That looks like a bug and isn't:
early on the rivals are behaving randomly, and the best response to randomness is
to over-produce relative to Nash. Their figure has the identical signature. It's
the cleanest evidence the implementation is right."

Then: "It flattens around two million iterations, which isn't arbitrary — that's
when the exploration rate `e^(−βt)` has decayed to essentially zero."

---

## 7 — Not an average of colluders and non-colluders

**FIGURES**
- left `figures_two_firm/fig_delta_distribution.png`
- right `figures/fig_delta_distribution.png`

| | Structure 1 | Structure 2 |
|---|---|---|
| sessions with Δ > 0 | **100.0%** | **100.0%** |
| sessions with Δ > 0.5 | 91.8% | 98.2% |
| median | 63.7% | 68.2% |
| worst / best session | 30.1% / 96.1% | 38.3% / 100.1% |

**Takeaway strip:** *Not one session out of 2,000 failed to collude.*

**Say:** "The obvious worry about a 64% or 69% average is that it's half perfect
cartels and half nothing. It isn't. Both entire distributions sit in the
collusive region, and the worst single session out of two thousand is still 30%
of the way to a cartel."

*Cut this slide first if you're short on time — it's insurance, not argument.*

---

## 8 — DIVIDER (no figure)

> # Is this genuine collusion, or just failure to optimise?
> High profits alone prove nothing.
> **Three independent tests. Both structures.**

**Say:** "This is the distinction that actually matters for antitrust. If the
algorithms simply never discovered that cheating was profitable, that's
incompetence, not collusion. So here are three tests that a
failure-to-optimise story cannot pass — and I ran all of them on both markets."

**This is the highest-value slide in the deck.** It reframes everything after it
from "more graphs" into "here is why you should believe me."

---

## 9 — Test 1: the learned strategy IS a punishment

**FIGURES**
- left `figures_two_firm/fig3_limit_strategy.png`
- right `figures/fig3_limit_strategy.png`

Axis on both: **price seen last period → output produced this period.**

| slope, MW per $/MWh | Structure 1 | Structure 2 |
|---|---|---|
| learned strategy | −0.24 / −0.33 | −0.36 / −0.30 / −0.24 |
| demand schedule | **−6.1** | **−2.91** |
| flatter than demand by | **~20×** | **8–12×** |

**Takeaway strip:** *Output falls in the observed price — that is the punishment.
And it is far flatter than demand, which is what makes the punishment stop.*

**Say — spend time on the flatness, it's the subtle part:** "Sloping down is the
punishment: see a low price, retaliate by flooding. But the *flatness* is the
clever bit. Because they retaliate less than the price drop that provoked them,
the price partly recovers, so next period's retaliation is smaller, and the war
dies out on its own. These algorithms have one period of memory — no clock, no
way to count 'punish for seven rounds.' They use punishment *intensity* as a
substitute for a clock. That's the paper's central mechanism and it reproduces
in both markets."

**Have ready (backup B2):** the composition objection — "isn't that slope
mechanical, since high-output sessions automatically sit at low prices?" Answer
for Structure 2: within-session slope = pooled slope = −0.910, and **97.7% of
individual sessions** slope down. It survives.

---

## 10 — Test 2: deviations get punished, harder when they can't hide

**FIGURES**
- left `figures_two_firm/fig4_deviation.png`
- right `figures/fig4_deviation.png`

Rival's extra output above the paired no-deviation twin:

| | t+1 | t+2 | t+3 | t+5 |
|---|---|---|---|---|
| **S1 high demand** | +6.65 | +2.91 | +1.85 | +1.65 |
| **S1 low demand** | **+10.23** | +4.38 | +2.86 | +2.14 |
| **S2 high demand** | +6.37 | +2.96 | +1.84 | +1.20 |
| **S2 low demand** | **+8.68** | +4.95 | +3.31 | +2.51 |

**Takeaway strip:** *A real price war that fades — and it is harsher in LOW
demand in both markets.*

**Say — the asymmetry is the money line:** "Compare the two rows within each
market. In high demand they punish *less*. Why? Because in the high state a low
price might genuinely be a bad demand draw, and punishing a possibly-innocent
rival is expensive. In the low state there's no such excuse, so they hammer it.
**They learned to calibrate punishment to how ambiguous the evidence is.** Nobody
programmed that, it appears in both markets independently, and the paper reports
the same asymmetry in the same direction."

**The residue, if he notices the lines don't fully return:**

| | sessions returning to their exact original cycle |
|---|---|
| Structure 1 | 78.7% (high) / 86.7% (low) |
| Structure 2 | 84.6% (high) / 80.8% (low) |

Those return with essentially zero residue; the whole average gap is the ~15–21%
that never return. **A one-off deviation permanently destroys the cartel in
roughly one session in six.**

---

## 11 — Test 3: cheating doesn't pay — but it's a lottery

**FIGURE**
- `figures/fig5_deviation_value.png` (Structure 2)
- **Structure 1 has no Figure 5 yet** — see note below

| | one-period gain | discounted total | deterred |
|---|---|---|---|
| **S1 low demand** | +$23 | **−$457** | **76.1%** |
| **S1 high demand** | +$122 | **−$91** | 42.2% |
| **S2 low demand** | +$63 | **−$885** | **76.3%** |
| **S2 high demand** | +$177 | **−$179** (median **+$38**) | 46.1% |

**Takeaway strip:** *Cheating pays today and loses once the punishment is priced
in — in both markets, in both demand states.*

Then the decomposition (Structure 2):

| high demand | mean gain from cheating |
|---|---|
| cartel survives (84.6%) | **+$50** |
| cartel destroyed (15.4%) | **−$1,233** |

**Say:** "Low demand is textbook in both markets — deterred in about three
quarters of sessions. High demand looks weak: only 42–46% deterred. But splitting
by whether the cartel survives explains it. **Deterrence here is a lottery, not a
certainty.** Usually cheating pays you a little; about one time in six it knocks
the system permanently onto a worse equilibrium and costs a fortune. On average
that risk is enough — but it's a risk premium, not a guaranteed punishment. I
think that's a more realistic picture of an algorithmic cartel than the paper's."

---

## 12 — Table I: the cost of being blind (no figure)

Three small tables across the slide.

**Structure 1**

| | certain | random |
|---|---|---|
| perfect | 78.51% | 70.03% |
| imperfect | **85.41%** | **64.14%** |

**Structure 2**

| | certain | random |
|---|---|---|
| perfect | 78.88% | 78.98% |
| imperfect | **84.18%** | **68.78%** |

**Paper**

| | certain | random |
|---|---|---|
| perfect | 84.16% | 79.72% |
| imperfect | **89.60%** | **76.25%** |

| | S1 | S2 | paper |
|---|---|---|---|
| imperfect row | −21.27 | −15.39 | −13.35 |
| perfect row | **−8.48** | +0.10 | **−4.44** |
| **DiD** | **−12.79** | −15.50 | **−8.91** |

**Takeaway strip:** *All four orderings replicate in both structures, including
the counter-intuitive one.*

**Say:** "Three things replicate. First, imperfect-plus-certain-demand is the
*highest* cell in both markets — better than seeing everything. That's the
paper's own counter-intuitive result: with the price as the state the Q-matrix
is 51 rows instead of 3,375, and a smaller table is easier to learn. Second,
imperfect-plus-random is the worst cell in both. Third, the DiD is negative:
blindness hinders collusion but doesn't prevent it."

**Then, importantly:** "Structure 1 is the closer replication, and not just on
the DiD. Look at the *perfect*-monitoring row. Structure 1 shows a real
demand-uncertainty penalty of −8.48, like the paper's −4.44. Structure 2 shows
zero. That's because with three firms the perfect-monitoring Q-matrix is 50,625
cells — fifteen times the paper's — so both of those cells are limited by
learning difficulty rather than economics. The duopoly doesn't have that problem,
which is exactly why it's the right primary comparison."

---

## 13 — What does NOT replicate

**FIGURES**
- left `figures_two_firm/fig6_deviation_vs_shock.png`
- right `figures/fig6_deviation_vs_shock.png`

| | price fall | rivals' response next period |
|---|---|---|
| forced cheat | −$2.36 | **+18.8 MW** |
| adverse demand shock | **−$5.77** | **−2.0 MW, median exactly 0** |

**Takeaway strip:** *The paper reports price wars triggered by deviations AND by
bad demand draws. Here, only deviations — in both structures.*

**Say:** "The shock is more than twice the price drop and produces no reaction at
all. So they are not following a simple 'low price → produce more' rule. They
ignore price moves inside the range everyday demand noise creates, and retaliate
only past a threshold. **They learned to filter the noise out of the signal.**
That's a trigger strategy — Green–Porter's original construction, arguably more
canonical than what the paper found. And it appears in both markets
independently, so it isn't a fluke of one configuration."

**Go straight into slide 14 — do not leave this hanging as an unexplained gap.**

---

## 14 — …and I think I know why (no figure)

> **The demand shock is calibrated to 2× the paper's size.**
> The paper's `m = 8` is the gap from a hot day *to* a mild day.
> Ours is 8 rungs from average *in each direction* → **16 rungs**.

Measured consequence — how often a firm *cannot* tell which demand state it was:

| | Structure 1 | Structure 2 |
|---|---|---|
| demand state ambiguous | **3.5%** | 28.6% |

**Takeaway strip:** *The shock is so large the two demand states barely overlap —
so the algorithms can identify the weather, so they don't punish it.*

**Say:** "Bigger shock is not monotonically more confusing. If the shock exceeds
the rival's whole output range, the two demand states stop overlapping and you
can just read the weather off the price — monitoring goes back to being
effectively perfect. That's the regime we're in, especially in the duopoly, where
firms identify the weather 96.5% of the time. Which is precisely the thing `m = 8`
was chosen to prevent."

**The prediction:** "So this is falsifiable. Fix the calibration and the
shock-triggered price war should appear. That would recover the paper's result.
It's a one-line fix plus a rerun."

**Say plainly:** "I found this while preparing these slides. It doesn't touch Δ,
the punishments, or the Table I orderings — those are all about responses to
*cheating*. But it does mean I can't claim my monitoring is as imperfect as
theirs, and I think it explains the one thing that didn't replicate."

*Volunteering this is worth more than the result it complicates. It's the
difference between a student who ran the code and one who audited it.*

---

## 15 — What this settles (no figure)

> **Replication: ✓** — Q-learning with a discretised action space colludes on a
> real network market. **Δ = 64.1% and 68.8%**, sustained by punishments, in two
> independent ownership structures.
>
> **Impossibility result: not contradicted.** Sannikov–Skrzypacz is about
> **speed** (period length → 0), not action-space continuity. We ran at one fixed
> period length. Different questions.
>
> **What we now have:** a like-for-like discrete benchmark on the identical market
> the PPO agents face. **These are the numbers to measure PPO against.**

**Ask him directly:** "Is Sannikov–Skrzypacz the impossibility paper you had in
mind? If it's a different one, my reading of this slide changes."

---

## 16 — Next (no figure)

- **Fix the shock calibration and rerun** — tests the slide-14 prediction
- **(α, β) robustness grid** — the paper sweeps 100×100 and gets 65–80%; I ran
  their single baseline point
- **h = 5 higher uncertainty** (their §5.6)
- Longer memory; re-matching / off-line training
- **Then: PPO with continuous actions, measured against 64.1% / 68.8%**

---

# Backups (keep after slide 16)

| # | Answers | Content |
|---|---|---|
| B1 | "Is the punishment real or leftover initialisation?" | Blank Q-matrix's default action is *max output* — would masquerade as punishment. Learned policy differs from init in **50.7 of 51** states; only **0.6%** of cells untouched |
| B2 | "Isn't Figure 3's slope mechanical?" | Pooled −0.910, **within-session −0.910**, 97.7% of sessions slope down |
| B3 | "Grid Nash or continuous Nash?" | Δ = 68.78% on grid, 66.74% on continuous; discretisation error 0.95% |
| B4 | "Why is S1's per-firm Δ = [222%, 39%]?" | Firm 0's Nash→monopoly gap is only $73, so the ratio explodes. **Report absolute gains: +$161 / +$177** — they split the cartel's gains equally in dollars rather than aiming at joint monopoly |
| B5 | "Why does S1's Nash total exceed the LCP's?" | The paper's LCP Nash isn't an equilibrium there — Firm 0 gains +25.2% by best-responding, because its plants straddle two nodes. Δ uses the BR-Nash |
| B6 | "Maybe only small cheating is deterred?" | Swept +3.1 → +24.8 MW; deterred fraction flat (51–54% high, 76–78% low) |
| B7 | Output evolution | `figures_two_firm/fig1_output_evolution.png` + `figures/fig1_output_evolution.png` |

---

# Gap to close

**Structure 1 has no `fig5_deviation_value.png`.** Slide 11 is the only slide
that breaks the paired layout. Two options:

1. **Generate it** (~10 min) so slide 11 matches the rest of the deck.
2. **Present slide 11 with Structure 2's figure and a paired numbers table** —
   defensible, since the S1 numbers exist and are in the table above.

Option 1 is better if there's time.
