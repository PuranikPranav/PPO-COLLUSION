# Plain-English companion to ADVISOR_BRIEF.md

Read this first, then the brief. Nothing here is new science — it is the same
results, explained from scratch. Every technical term in the brief appears below
in **bold** with a plain explanation.

---

# PART 0 — The whole project in one page

**The disagreement we are refereeing.**

Some economics papers claim: *if you let simple learning algorithms set prices or
quantities against each other, they teach themselves to collude — no
communication, nobody programmed them to.* That is alarming for antitrust.

Another paper (the "impossibility" paper) claims: *no, under certain conditions
collusion is mathematically impossible.*

Your advisor's instruction was: **don't take either side on faith.** Take the
economics papers' exact algorithm, run it on *our* market (an electricity
network), and see whether their result actually shows up.

**What we did.** We rebuilt the algorithm from Calvano et al. (2021) line by
line, plugged it into our five-node power grid model, ran it 1,000 separate
times, and measured how collusive the outcome was.

**What we found.** It collides — sorry, it *colludes*. The algorithms end up
about **69% of the way** from full competition to a perfect cartel, and they do
it by punishing each other, exactly the way the theory says a cartel must.

**Why this matters for your thesis.** Your main work uses PPO (a modern deep-RL
algorithm) with *continuous* actions. Before you can say anything about whether
PPO colludes, you need a yardstick: what does the *classic* algorithm do on the
*same* market? Now you have that yardstick. It is 69%.

---

# PART 1 — The vocabulary

## 1.1 What is Q-learning?

Imagine a robot running one power plant. Every day it must choose how much
electricity to produce. It keeps a **scorecard** — a big table:

|                          | produce 69 MW | produce 72 MW | ... | produce 112 MW |
|--------------------------|---------------|---------------|-----|----------------|
| yesterday's price was $36 | 41,200        | 43,900        | ... | 38,100         |
| yesterday's price was $37 | 40,800        | 44,300        | ... | 37,600         |
| ... | | | | |

Each cell answers: *"If I'm in this situation and I take this action, how much
money will I make **in total, from now until forever**?"*

That table is called the **Q-matrix**. `Q(s, a)` = the value of doing action `a`
in situation `s`.

- The **rows** are called **states** — the situation the robot finds itself in.
- The **columns** are called **actions** — the choices available.
- The robot does not know the true numbers at the start. It **learns** them by
  trying things and watching what happens.

**The learning rule** (the brief calls it "their eq. 3"):

```
Q(s,a)  ←  (1−α)·Q(s,a)  +  α·[ π  +  δ·max Q(s′,a′) ]
           └── old belief ──┘     └── what just happened ──┘
```

In words: *"Keep 85% of what I used to believe, and nudge 15% toward what I just
learned. What I just learned = the profit I actually earned this round, plus the
value of the best thing I can do from wherever I ended up."*

That is the entire algorithm. It is genuinely simple — which is exactly why
economists like it: there are only three knobs, and each one has a clear economic
meaning.

## 1.2 The three knobs

**α = 0.15 — the learning rate.** How fast the robot changes its mind. 0.15 means
"move 15% of the way toward the new evidence each time." Too high and it
overreacts to noise; too low and it never learns.

**δ = 0.95 — the discount factor.** How much the robot cares about the future. A
dollar next period is worth 95 cents today. **This is the single most important
number for collusion.** If δ = 0, the robot only cares about today, so it always
cheats, and collusion is impossible. Collusion only works when the future
punishment outweighs today's gain. δ = 0.95 means the future matters a lot.

**β = 4×10⁻⁶ — the exploration decay.** See next.

## 1.3 Exploration: ε-greedy and `ε_t = e^(−βt)`

If the robot always does what its scorecard says is best, it will never discover
anything better. So it sometimes acts randomly. That is **ε-greedy**:

> With probability ε, ignore the scorecard and pick a random action.
> Otherwise, do whatever the scorecard rates highest (the **greedy** action).

ε shrinks over time as `ε_t = e^(−βt)`:

| iteration | ε (chance of acting randomly) |
|---|---|
| 0 | 100% — pure trial and error |
| 173,000 | 50% |
| 1,000,000 | 1.8% |
| **2,000,000** | **0.03% — exploration is essentially over** |

**This is why our runs converge at ~2 million iterations.** It is not arbitrary
and it is not a bug. It is baked into β: `t ≈ 8/β = 2,000,000`. When your advisor
asks "why does it take 2 million rounds?", the answer is "because β = 4×10⁻⁶ is
the paper's value, and that is when exploration switches off."

The **greedy action** is a term you will use constantly: it just means "the action
the scorecard currently rates highest." Figures 1 and 2 track greedy actions, not
actions actually played, because during training the played action is often a
random experiment.

## 1.4 `Q₀` — where the scorecard starts

The scorecard has to start *somewhere*. The paper's choice (their footnote 12),
which we copied:

> Start every cell at: *"what would I earn if my rivals just picked randomly
> forever?"*, converted into a lifetime value by dividing by (1−δ).

Why you should care: against randomly-behaving rivals, **producing a lot is
good**. So the initial scorecard's favourite action is the **maximum output**
(action index 14). This matters enormously — see §2.6, the "is the punishment
real?" check.

## 1.5 Discretisation — the actual assignment

Q-learning needs a **finite** list of actions, because the scorecard needs a
finite number of columns. Real output is a continuous number (91.3 MW,
91.31 MW, ...). So we chop it into a **menu**:

> Each firm chooses from **k = 15** output levels, spaced **v = 3.098 MW** apart.

That is the "discretise each agent's action space" part of your task. It is the
hinge of the whole exercise, because the impossibility argument was (allegedly)
about continuous actions.

**Where do we put the 15 rungs?** This is the one design choice that could ruin
everything, so it is worth understanding.

We anchor on two reference points:
- **q^M** — each firm's output if all three firms merged into one monopolist and
  maximised joint profit. (Low output, high price.)
- **q^C** — each firm's output at **Cournot–Nash**: everyone acting selfishly,
  nobody colluding. (High output, low price.)

The grid spans `[q^M, q^C]` **plus 20% extra on each side** (that 20% is the
**ξ = 0.2** in the brief). The extra room is essential:
- room *above* Nash so a firm **can cheat** (flood the market) and **can punish**
- room *below* monopoly so they could in principle over-collude

**The check that this is honest.** In the paper's grid, Nash lands on rung 12 and
monopoly on rung 2. In ours:

| | paper | ours |
|---|---|---|
| Cournot–Nash sits at rung | 12 | **12, 12, 13** |
| joint monopoly sits at rung | 2 | **2, 2, 2** |

Same slots. So we did not accidentally build a grid that makes collusion easy or
hard — the game has the same shape as theirs.

**And the discretisation barely distorts the economics:** the true (continuous)
Nash output is 325.5 MW; the best the 15-rung grid can do is 328.6 MW. That is a
**0.95%** error. Monopoly matches exactly.

## 1.6 Perfect vs imperfect monitoring — this is our case

**Perfect monitoring:** at the end of each round you see exactly how much each
rival produced. Cheating is instantly visible. (This is the older 2020 paper.)

**Imperfect monitoring:** *you never see what your rivals produced.* All you see
is **the market price**. And the price also moves because of random demand.

So when the price comes in low, you face a genuine puzzle:

> *Did a rival cheat and flood the market? Or was demand just weak today?*

**You cannot tell.** That is what "imperfect monitoring" means, and **that is our
case** — it is realistic for electricity markets, where you see the clearing
price but not each competitor's dispatch.

This is why the **state** (the row of the scorecard) is **last period's price**.
That is literally all the information the algorithm has.

## 1.7 Green–Porter — the theory being tested

Green & Porter (1984) is the classic answer to: *how can a cartel survive if you
can't see cheating?*

Their answer, in plain terms:

> Agree on a rule: **"as long as the price stays above threshold X, everybody
> keeps output low. The moment the price drops below X, everybody floods the
> market for a while — a price war. Then we go back to cooperating."**

Two features to understand:

1. **The punishment must be temporary.** If it were permanent ("grim trigger"),
   then the first time demand happened to be weak, the cartel would collapse
   forever by accident. So price wars have to end.
2. **You must sometimes punish when nobody actually cheated.** If you only
   punished provable cheating, a cheater could always hide behind "must have been
   weak demand." So innocent price drops get punished too. That is the *cost* of
   imperfect monitoring — it is why cartels earn less when monitoring is poor.

**The clock problem, and the paper's cleverest finding.** Green–Porter firms have
infinite memory, so they can count: "punish for exactly 7 rounds." Our algorithms
remember only **one period**. They have no clock. So how do they end a punishment?

Answer: **they use the *intensity* of the punishment as a substitute for a clock.**
Their reaction is *milder* than the price drop that triggered it. So the price
partially recovers → next period's punishment is milder → the price recovers more
→ ... and the war fades out on its own. No counting required. That is what
Figure 3 shows, and it is the paper's most elegant result.

## 1.8 The market: LMP, DC-OPF, "the hub"

The paper uses a toy demand curve: `price = d − (total output)`. One price, one
number.

We use a real electricity market model instead:

- **DC-OPF** = "DC Optimal Power Flow." The grid operator solves an optimisation
  every round: meet demand at every location at the lowest cost, while respecting
  the physical limits of the transmission lines.
- **LMP** = "Locational Marginal Price." Because power has to flow over limited
  lines, **each location gets its own price**. If a line is full ("congested"),
  prices on the two sides diverge.
- **Node** = a location on the grid. Our network has 5.
- **The hub** = node index 1 in the code, which the papers call "node 2." All
  three firms' plants sit there, so they all face the same price. (That siting
  was chosen for a separate reason — so our Nash benchmark matches the analytical
  formulation in the source paper.)

**Why the market collapses into a lookup table** (the sentence in §1 of the
brief). Because all three firms are at the same node, the grid operator's answer
depends only on the **total** MW they produce, not on who produced what. With a
common step size, total output can only take **43 distinct values**. So we solve
the DC-OPF 43 times up front, store the answers, and then the learning loop is
just table lookups. That is what makes 1,000 sessions × 3 million rounds finish
in 36 minutes instead of weeks. **This is an engineering trick, not an economic
assumption** — the code checks the assumption holds and refuses to run otherwise.

## 1.9 The demand shock: `h = 2` and `m = 8`

Demand is random. **h = 2** means two equally likely states: high demand and low
demand. Firms commit their output **before** seeing which one happens.

**m = 8** is the important one. It sets *how big* the shock is, measured in
**output steps**:

> The shock moves the price by as much as **8 rungs** of output would.

So if you see the price fall by "5 rungs worth," you genuinely cannot tell
whether a rival expanded by 5 rungs or demand came in weak. **That is the
confounding, and m controls how bad it is.** m = 8 is the paper's baseline.

In the paper's toy model, price = d − Q, so the shock size is trivially
`8 × v`. In our market the price comes out of an optimisation, so we had to
**solve for it numerically by bisection**: try a shock size, clear the market, see
how far the price moved, adjust, repeat. Answer: **±$2.887/MWh** on the demand
intercepts. (**Bisection** = binary search: guess halfway, check if you overshot,
halve the interval, repeat.)

## 1.10 "Non-revealing price fraction" = how blind are they?

A number between 0 and 1: **what fraction of the time does the price you observe
fail to tell you exactly what your rivals did?**

- 0 = you can always deduce rivals' output → effectively perfect monitoring
- 1 = the price tells you nothing

| | value |
|---|---|
| ours, **measured** on the real DC-OPF price table | **0.384** |
| paper's duopoly baseline | 0.304 ("roughly a third") |
| paper's formula plugged in at 3 firms | 0.568 |

**Remember 0.384 vs 0.568.** It comes back twice: it explains why our
difference-in-differences is bigger than theirs, *and* it explains the one
result that doesn't replicate. Our algorithms are somewhat **less** blind than
the paper's idealised three-firm case.

We **measured** it rather than using the formula because our price comes from an
optimisation with congestion, so the price levels don't line up on a perfect
lattice the way `p = d − Q` does.

## 1.11 Δ — the scoreboard

**This is the single number your advisor will focus on.** It is a ruler:

```
Δ  =  (learned profit − Nash profit) / (monopoly profit − Nash profit)

Δ = 0   →  fully competitive (Cournot–Nash). No collusion at all.
Δ = 1   →  perfect cartel. As if all three firms merged.
Δ = 0.688 → about 69% of the way to a perfect cartel.
```

Three benchmarks appear in the brief:
- **Competitive** (416.8 MW, cheapest price) — price = marginal cost, textbook
  perfect competition. Not used in Δ; shown for context.
- **Cournot–Nash** (328.6 MW) — everyone selfish but strategic. **Δ = 0 here.**
- **Joint monopoly** (232.6 MW) — the cartel's dream. **Δ = 1 here.**

**"± 0.31"** is the **standard error**: the uncertainty on the average across
1,000 sessions. It is tiny, meaning 68.8% is precisely estimated.

## 1.12 "Sessions"

One **session** = one complete independent experiment: three fresh algorithms
with blank scorecards, learning against each other for millions of rounds until
they settle.

We run **1,000** of them because each session's outcome is random (random
exploration, random demand). The paper does the same.

**Why the brief reports the whole distribution, not just the mean:** an average of
69% could hide "half collude perfectly, half don't collude at all." It doesn't:

| worst session | 5th pct | median | 95th pct | best |
|---|---|---|---|---|
| 38.3% | 53.6% | 68.2% | 86.0% | 100.1% |

**Every single one of the 1,000 sessions had Δ > 0.** 98.2% were above 50%. The
result is not driven by outliers.

## 1.13 Convergence

> "the greedy action does not change for 100,000 consecutive periods"

Meaning: for 100,000 rounds in a row, the scorecard's favourite action in every
situation the algorithms actually visit has not changed. At that point we declare
learning finished and freeze them.

Ours: **100% of sessions converged**, median at 2.0M rounds. No session failed.

## 1.14 Limit strategy, limit cycle, phases

After freezing, each algorithm is just a fixed rule: *"if I see price X, I produce
Y."* That rule is the **limit strategy** — what Figure 3 plots.

Because the rule is deterministic, once demand is frozen the system settles into a
repeating loop: state A → state B → state A → ... That loop is the **limit cycle**.
Median length: **2 states**.

**Why "averaged over all 12 phases" appears in the brief.** Different sessions are
at different points in their own loop when you start measuring. Averaging them
naively produces a zigzag that has nothing to do with the experiment. So we
replicate each session 12 times, each one offset by one extra period, which
cancels the loop exactly and leaves only the real response. Without this, the
Figure 4 result is unreadable.

## 1.15 Paired counterfactual / impulse response

This is the workhorse experimental design, used for Figures 4, 5 and 6.

> Take a converged session. **Clone it.** Run both copies from the *identical*
> state. In one copy, do something (force a firm to cheat). In the other, do
> nothing. Give both copies identical random draws afterwards. **Subtract.**

Because everything else is identical, the difference *is* the causal effect of the
thing you did. It is a twin study. **Impulse response** just means "poke it once,
then watch the following periods."

## 1.16 Static best response

The output that maximises your profit **this period only**, taking rivals' current
output as given, ignoring all future consequences.

It is the most tempting possible one-shot cheat. We use it as the forced deviation
because it is the **strongest test** — if even the greediest possible cheat gets
deterred, the collusion is real.

For us it is **+20.2 MW (6.5 rungs)**, which drops the price by $2.36.

---

# PART 2 — Walking through the results

## 2.1 The headline

| | learned | Nash | monopoly |
|---|---|---|---|
| Δ | **68.78%** | 0% | 100% |
| total generation | 280.1 MW | 328.6 | 232.6 |
| hub price | $42.18 | $36.52 | $47.71 |

**Read it like this:** left to themselves, three algorithms that were never told
about each other, never communicated, and were only trying to maximise their own
profit, ended up **withholding 48 MW of generation** and pushing the price up
**$5.66/MWh** above the competitive-strategic benchmark. That is the finding.

The paper gets 76.25% with two firms and 72.7% with three. We get 68.8% with three
firms on a much harder market. **That is a successful replication** — you expect
some loss from adding congestion, quadratic costs, and asymmetric firms.

**"Δ starts at −0.29"** — worth mentioning, because it looks like a bug and isn't.
At the very beginning the algorithms are best-responding to rivals who are
behaving completely randomly. The best response to randomness is to produce *more*
than Nash. So they start *below* the competitive benchmark (negative Δ), then
learn their way up. **The paper's Figure 2 has exactly the same negative start.**
It is a fingerprint of a correct implementation.

## 2.2 Figure 3 — the limit strategy

**What it shows:** the horizontal axis is the price you saw last period; the
vertical axis is how much you produce now. **The line slopes down.**

**What that means in words:** *"the lower the price I saw, the more I produce."*

That sounds economically backwards until you realise it is the **punishment**. A
low price is the signal that someone may have cheated, so you retaliate by
flooding the market.

**The second, subtler feature — and the important one.** The dashed lines are the
demand curves. **Our strategy line is much flatter than them:**

| | slope (MW per $/MWh) |
|---|---|
| strategy, per firm | −0.24 to −0.36 |
| demand curve, per firm | −2.91 |

8 to 12 times flatter. **Why that matters:** it means the retaliation is *smaller*
than the price move that provoked it. So the price bounces back a bit, so next
period's retaliation is smaller still, and the war dies out by itself. That is the
"intensity as a substitute for a clock" mechanism from §1.7. **Without this
flatness, the price war would never end.**

**The objection I stress-tested (and you should raise before he does).** Someone
could say: *"that downward slope is mechanical, not strategic. A session that
happens to produce a lot will automatically sit at low prices. So plotting output
against price across 1,000 sessions must slope down regardless of strategy."*

This is called a **composition** problem — you're mixing across sessions that
differ in level, instead of measuring the effect within one session.

The fix is **session fixed effects**: instead of comparing session A to session B,
compare each session *to itself* at different prices. Result:

| | slope |
|---|---|
| naive (pooled across sessions) | −0.910 |
| **within-session (fixed effects)** | **−0.910** |
| **fraction of the 1,000 sessions with a downward-sloping strategy** | **97.7%** |

Identical. **The objection does not hold.** 97.7% of individual algorithms
learned a punishing strategy on their own. Good line to have ready.

## 2.3 Figure 4 — the forced deviation

**The experiment:** freeze demand. Take converged algorithms. Force one firm to
cheat maximally for exactly one period (+20.2 MW). Then let it go back to its
learned rule. Watch what the other two do.

**Result** — extra MW the rivals produce, versus the identical no-cheat twin:

| | t+1 | t+2 | t+3 | t+5 | t+20 |
|---|---|---|---|---|---|
| **high demand** | **+6.37** | +2.96 | +1.84 | +1.20 | +1.13 |
| **low demand** | **+8.68** | +4.95 | +3.31 | +2.51 | +1.84 |

**Two things to say about this.**

**(a) It's a real, fading price war.** Rivals retaliate hard immediately, then back
off; ~80% of the punishment is gone within five periods. That is precisely the
paper's description: *"harsher initially and then gradually fading away."*

**(b) The punishment is harsher in LOW demand (+8.68) than in HIGH (+6.37).**
This is the most important single number in the whole replication, because it is
the **fingerprint of imperfect monitoring specifically**.

Think about why. In the **high**-demand state, a low price *might* be an adverse
demand shock rather than cheating — so the algorithms hold back, because
punishing an innocent rival is costly. In the **low**-demand state there is no
such excuse available, so they punish hard. **They learned to calibrate punishment
to how ambiguous the evidence is.** Nobody programmed that. The paper reports the
same asymmetry, in the same direction.

**The residue** — and how to handle it if he notices. The lines don't return
*all* the way to where they started; a ~1.2–1.9 MW gap persists forever. The
paper's figure shows a full return. Until today this was unexplained. Now:

| | sessions that return to their *exact* original cycle | their residue | everyone else's residue |
|---|---|---|---|
| high demand | **84.6%** | **+0.004 MW** | +22.75 MW |
| low demand | **80.8%** | **+0.004 MW** | +33.25 MW |

**In plain words:** 85% of sessions recover *perfectly* — zero residue. The
entire average gap comes from the ~15% that never recover at all.

**Why some never recover.** After freezing, the algorithms are deterministic rules
and demand is frozen, so the system is a machine with 51 possible states that must
end up in a repeating loop. There is more than one such loop. A big enough shove
can knock the system out of the good loop and into a worse one — **and since
nothing is random any more, nothing ever knocks it back.**

**So the honest headline is: a single act of cheating permanently destroys the
cartel in about one session in six.** That is a *better* story than the paper's,
not a failure to replicate.

## 2.4 Figure 5 — is cheating actually deterred?

**Why this test is necessary.** High profits alone prove nothing. Maybe the
algorithms just never figured out that cheating was profitable — that would be
*incompetence*, not collusion. Antitrust cares about the difference.

**The test** (this is what economists call an **incentive-compatibility** check):
clone the session, force one copy to cheat, and add up the *discounted* profit
difference over 250 periods: `Σ δ^t (profit_cheat − profit_no-cheat)`. If that
total is **negative**, cheating genuinely doesn't pay, and the high profits are
being held up by fear of punishment. That is real collusion.

| | gain on day 1 | discounted total | deterred in |
|---|---|---|---|
| **low demand** | +$63 | **−$885** | **76.3%** of sessions |
| **high demand** | +$177 | **−$179** (median **+$38**) | **46.1%** of sessions |

**Low demand: textbook.** Cheating pays today, loses over the punishment phase,
deterred three quarters of the time.

**High demand is weaker, and you should present it honestly** — it is your most
interesting result. Only 46% deterred, and the *median* session would actually
*gain* $38 by cheating. But the decomposition rescues it:

| high demand | average gain from cheating | |
|---|---|---|
| sessions whose cartel survives (84.6%) | **+$50** | cheating pays a bit |
| sessions whose cartel is destroyed (15.4%) | **−$1,233** | catastrophic |

**Say it like this:** *"Deterrence in this market is a lottery, not a certainty.
Usually cheating pays you a little. But about one time in six it destroys the
cartel permanently and costs you a fortune. On average, that risk is enough to
make cheating unprofitable — but it is a risk premium, not a guaranteed
punishment."*

That is a genuinely novel characterisation and it is defensible because you have
the decomposition.

**"Not size-dependent."** The natural follow-up is *"maybe they only deter small
cheating."* We swept the cheat from +3.1 MW to +24.8 MW. The deterred fraction
barely moves (51–54% high, 76–78% low). So no — nothing hinges on how big a cheat
we chose.

## 2.5 Table I and "difference-in-differences"

**The 2×2 design.** We want to isolate *the effect of not being able to see your
rivals*. But the imperfect-monitoring setup also has random demand, and random
demand hurts profits by itself. So we run all four combinations:

|  | demand is certain | demand is random |
|---|---|---|
| **you see rivals' output** (perfect) | 78.88% | 78.98% |
| **you only see the price** (imperfect) | **84.18%** | **68.78%** |

**Difference-in-differences (DiD)** is the standard trick for separating two
tangled effects. Take how much random demand hurts you when monitoring is
imperfect, then subtract how much it hurts you when monitoring is perfect.
Whatever's left is the pure cost of being blind:

```
(68.78 − 84.18) − (78.98 − 78.88)  =  −15.40 − 0.10  =  −15.50 pp
```

**Three things replicate:**

1. **The weird cell replicates.** Imperfect + certain demand (84.18%) is the
   *highest* of the four — *better* than perfect monitoring. That seems backwards
   until you see why: when the state is just "the price," the scorecard has 51
   rows; when the state is "everyone's exact output," it has 3,375 rows. **A
   smaller scorecard is easier to learn.** Less information → simpler problem →
   better coordination. The paper finds the same thing (+5.44pp premium; ours is
   +5.30pp — nearly identical).
2. Imperfect + random demand is the worst cell. Confusion is costly.
3. The DiD is negative: **blindness hurts collusion but does not prevent it.**

**How to handle "your DiD is −15.5, theirs is −8.91."** Don't defend it —
decompose it:

| | ours | paper |
|---|---|---|
| **imperfect row** (the row the paper is about) | **−15.40** | **−13.35** |
| perfect row | +0.10 | −4.44 |

**The row that matters matches to 2 points.** The entire discrepancy is in the
*perfect*-monitoring row, and there is a concrete reason: with 3 firms, the
perfect-monitoring scorecard has 15³ = 3,375 rows → **50,625 cells**, which is
**15× bigger** than the paper's two-firm version. At the same exploration rate,
our algorithms simply don't get to visit every cell enough. Both perfect cells are
limited by *learning difficulty*, not by *economics*, which flattens the demand-
uncertainty effect to zero.

**Hard evidence for that story:** perfect-monitoring runs needed a median of
**5.7 million** rounds to converge versus **2.0 million** for imperfect. Nearly
3× slower. The paper conjectured exactly this (their §5.3) but our numbers
demonstrate it.

And note our *imperfect* scorecard is 765 cells vs their 555 — only 1.38× bigger.
**That is why the imperfect row is a fair comparison and the perfect row isn't.**

## 2.6 The check that the punishment is real, not leftover initialisation

**This is the most dangerous question your advisor could ask, so know it cold.**

The worry: remember from §1.4 that the blank scorecard's favourite action is
**maximum output**. Now, the forced deviation pushes the price into an unusual
region. If the algorithms never visited that region during training, the scorecard
there would still be blank — and blank means *maximum output*. So what looks like
"they learned to punish" could just be "they never learned anything there, and the
default happens to look like punishment."

That would invalidate Figure 4 entirely.

**We checked. It's fine:**

- the learned policy differs from the blank initialisation in **50.7 of the 51**
  price states
- only **0.6%** of all (session, state) cells are still at the initial value
- in the exact states the deviation lands in, only 0.6–1.2% are untouched, and
  removing them changes the measured punishment by 0.1 MW

**The punishment is learned.** Say that with confidence.

## 2.7 Figure 6 — the one thing that does NOT replicate

**The paper's claim:** price wars are triggered by cheating **and** by innocent
bad-demand draws. (Per §1.7, that's theoretically necessary — otherwise cheaters
hide behind bad demand.)

**Our finding:** only cheating triggers a war.

The experiment is the clone-and-subtract design again, but poking demand instead
of behaviour:

| what we did | how far the price fell | what rivals did next period |
|---|---|---|
| forced one firm to cheat | −$2.36 | **+18.8 MW** — war |
| forced an adverse demand draw | **−$5.77** | **−2.0 MW, median exactly 0** — nothing |

**Read that twice.** The demand shock is *more than twice the price drop*, and it
produces **no reaction at all** (38% of sessions expand, 44% contract, 17% don't
move — a coin flip).

**What that means.** If the algorithms were simply following "low price → produce
more," the bigger price drop would produce the bigger war. They aren't. They have
learned something sharper: a **trigger strategy**.

- **smooth reaction function** (what the paper found): react a little to every
  price wiggle
- **trigger strategy** (what we found): *ignore* price moves inside the normal
  range that everyday demand noise creates; retaliate hard only once the price
  falls **below** that range

The numbers show this directly:

| where the price lands | how steeply they react (MW per $/MWh) |
|---|---|
| overall average | −0.91 |
| where a **cheat** puts it | **−7.7 to −10.3** |
| where a **shock** puts it | +0.36 ≈ nothing |

**They learned to filter the noise out of the signal.** They punish only price
drops too large to be blamed on demand.

**This is arguably more canonical than the paper's result** — an explicit trigger
price is exactly Green–Porter's original construction.

**Why we get this and they don't** — and it's a number you already have:
**our monitoring is less imperfect than theirs (0.384 vs 0.568, §1.10).** Our
price signal is cleaner, so the two demand states are more distinguishable, so the
algorithms *can* tell shocks from cheating — and once they can, they stop
punishing shocks.

**This gives you a concrete prediction to offer, which advisors love:** *raise the
number of demand states (their §5.6 uses h = 5) or raise m, and the shock-triggered
war should reappear.* Not yet run — it's a one-day experiment.

## 2.8 The impossibility paper

**What the brief says, in plain terms.** The impossibility paper is
**Sannikov & Skrzypacz (2007)**. Two things you need to know:

1. **Its result is not really about continuous action spaces.** It is about
   **speed**. Their theorem says: as the time between decisions shrinks toward
   zero (firms can react faster and faster), collusion becomes impossible. The
   intuition: in a very short interval, a cheater barely moves the price, so the
   price signal contains almost no information about cheating, while the profit
   from cheating stays proportional. Signal degrades faster than temptation →
   punishment can't be calibrated → collusion dies.
2. **Therefore our result does not contradict it, and cannot.** We ran at *one
   fixed* period length. Their theorem is about a *limit* as period length → 0.
   Δ = 68.8% at one speed says nothing about what happens as speed → ∞.

**How to say this out loud:** *"I read the impossibility paper carefully. Its
driving assumption is the ability to react arbitrarily fast, not the continuity of
the action space. So the Q-learning result and the impossibility theorem aren't
actually in conflict — they're answering different questions. Which means Task 1
doesn't adjudicate between them; what it does is give us a calibrated benchmark on
our own market to measure the PPO results against."*

**One caveat to raise early:** if your advisor had a *different* paper in mind,
this section changes. Ask him in the first five minutes.

---

# PART 3 — Cheat sheet

| Term | Plain meaning |
|---|---|
| **Q-learning** | Robot keeps a scorecard of "how good is action A in situation S", updates it from experience |
| **Q-matrix** | That scorecard. Rows = situations, columns = actions |
| **State** | The situation. For us: **last period's price** |
| **Action** | How much to produce, from a menu of 15 |
| **Greedy action** | Whatever the scorecard currently rates highest |
| **α = 0.15** | Learning rate — move 15% toward new evidence |
| **δ = 0.95** | How much the future matters. Collusion is impossible without this |
| **β = 4×10⁻⁶** | How fast random experimentation switches off (→ ~2M rounds) |
| **ε-greedy** | Act randomly with probability ε, otherwise follow the scorecard |
| **Discretisation** | Chopping continuous output into 15 rungs, 3.098 MW apart |
| **ξ = 0.2** | The grid extends 20% past monopoly and Nash, leaving room to cheat and punish |
| **k = 15** | Number of rungs |
| **h = 2** | Two demand states, high and low |
| **m = 8** | The demand shock moves the price by 8 rungs' worth — the confusion |
| **Imperfect monitoring** | You see only the price, never rivals' output |
| **Non-revealing fraction** | How often the price fails to reveal what rivals did. Ours: 0.384 |
| **LMP / DC-OPF** | Real grid pricing: each location gets its own price after an optimisation |
| **Node / hub** | Grid location; all 3 firms sit at the same one |
| **Δ** | 0 = competition, 1 = perfect cartel. **Ours = 0.688** |
| **Cournot–Nash** | Everyone selfish. Δ = 0 |
| **Joint monopoly** | Cartel's dream. Δ = 1 |
| **Session** | One complete independent experiment. We run 1,000 |
| **Convergence** | Favourite action unchanged for 100,000 rounds |
| **Limit strategy** | The frozen rule "if I see price X, I produce Y" |
| **Limit cycle** | The repeating loop play settles into. Median length 2 |
| **Paired counterfactual** | Clone the session, poke one copy, subtract. A twin study |
| **Impulse response** | Poke once, watch the following periods |
| **Static best response** | The most profitable one-period cheat, ignoring the future |
| **Incentive compatibility** | Does cheating actually lose money once punishment is counted? |
| **Session fixed effects** | Compare each session to itself, not to other sessions |
| **DiD** | 2×2 trick to separate "can't see rivals" from "demand is random" |
| **Green–Porter** | Classic theory: cartels survive blindness via temporary price wars |
| **Trigger strategy** | Ignore small price moves; retaliate hard past a threshold |
| **Sannikov–Skrzypacz** | The impossibility paper. Really about **speed**, not continuity |
