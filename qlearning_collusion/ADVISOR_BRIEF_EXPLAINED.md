# The advisor brief, explained in plain English

A companion to [`ADVISOR_BRIEF.md`](ADVISOR_BRIEF.md). Same section numbers, so
you can read them side by side. Nothing here is new work — it is the same
results, unpacked.

---

## Part A — The 60-second version

Three sentences, no jargon:

1. Some economists found that when you let simple learning robots run a market,
   they teach themselves to keep prices high — without ever communicating. That
   looks like a cartel, except nobody agreed to anything.
2. Another paper argues that this cannot really happen, and that the earlier
   result is an artefact of how the experiment was set up.
3. Your advisor asked: rebuild the economists' experiment *exactly*, but on
   **our** electricity market, and see whether the robots still form a cartel.
   **They do.**

Everything else in the brief is either (a) proving that "they do" is real and
not an illusion, or (b) being honest about the places it does not match.

---

## Part B — The vocabulary you need first

Read this once and the whole brief becomes readable.

### The market words

**Firm / plant.** A *firm* is a company. A *plant* is one power station. A firm
can own several plants. In the three-firm market each firm owns one plant. In
the two-firm market, firm 0 owns two plants and firm 1 owns one.

**Node.** A location on the electricity grid. Our network has 5 nodes connected
by 5 wires. Power is generated at some nodes and consumed at others.

**MC and QC (cost).** A plant's cost of producing `g` megawatts is
`MC·g + ½·QC·g²`. `MC` is the cost of the first megawatt. The `QC` term means
**each extra megawatt costs more than the last** — like a car burning more fuel
per mile the faster you drive. So a plant with MC = 10 and QC = 0.065 pays
$10/MWh at the very first megawatt but ~$17.5/MWh at its 115th.

**Cap.** The physical maximum a plant can produce. It cannot exceed this no
matter what.

**LMP (Locational Marginal Price).** The price of electricity **at one specific
node**. The key oddity of electricity markets: the same electricity can have
different prices in different places at the same instant, because the wires
between them have limited capacity. "$/MWh" = dollars per megawatt-hour.

**Thermal limit.** The maximum power a wire can carry before it overheats.

**Congestion.** When a wire is *maxed out*. This is the single most important
concept for understanding the two-firm result. If no wire is maxed out,
electricity flows freely and **every node has the same price**. If a wire is
maxed out, cheap power physically cannot reach the far side, so the far side
pays more — **prices split apart**.

**Shadow price.** A number the solver produces for each wire, meaning "how much
money would we save if this wire were one MW bigger?" If the wire is not full,
making it bigger saves nothing, so the shadow price is **0**. So *"max shadow
price = 0.0000029"* is just a precise way of saying **"no wire is full
anywhere — the grid is completely uncongested."** That single fact drives §0a of
the two-firm results.

**DC-OPF (DC Optimal Power Flow).** The computer program the grid operator runs.
You hand it "here is how much each plant is producing"; it works out how power
flows through the wires, how much each node consumes, and what the price is at
each node. It is the *rulebook of the market*. In the economists' paper this
whole thing was replaced by one line of algebra (`price = demand − total
output`). Ours is a realistic simulation instead. **That substitution is the
entire scientific contribution of the exercise.**

### The economics words

**Cournot competition.** A model where firms compete by choosing **how much to
produce** (not what price to charge). All firms decide at the same moment, then
the market determines the price from the total. Produce more → price falls.

**Three benchmark outcomes.** Everything is measured against these three points:

| name | what it means | outcome |
|---|---|---|
| **Competitive** | firms behave as if they have no market power; produce until price = cost | most output, lowest price, lowest profit |
| **Nash equilibrium** | each firm selfishly maximises its own profit, taking rivals' output as given | middle |
| **Joint monopoly** | all firms act as ONE company maximising combined profit | least output, highest price, highest profit |

**Nash equilibrium**, more carefully: a set of choices where **no single firm can
improve its own profit by unilaterally changing its own choice**. That is the
definition — and it is the test used in §0b of the two-firm write-up to prove the
paper's Nash was wrong. It is the "honest competition" reference point: selfish,
non-cooperative, but not stupid.

**Joint monopoly** is what a cartel would do if it were legal and enforceable.

**Δ (Delta) — the collusion score.** This is the single most important number in
the whole project.

```
        (what the robots actually earned)  −  (Nash profit)
Δ  =   ─────────────────────────────────────────────────────
        (joint monopoly profit)            −  (Nash profit)
```

Read it as a percentage on a ruler:

```
  Nash                                              Joint monopoly
 (honest competition)                                  (full cartel)
   |──────────────────────────────────────────────────────|
   0%                        64%                        100%
                              ▲
                     the robots ended up here
```

- **Δ = 0%** → the robots merely competed honestly. No collusion.
- **Δ = 100%** → the robots behaved exactly like a single monopolist. Perfect cartel.
- **Δ = 64%** → they got 64% of the way from honest competition to a full cartel.

This is why choosing the right Nash point matters so much: **Nash is the zero
mark on the ruler.** Put the zero in the wrong place and every reading is wrong.

**Individual rationality (IR).** A cartel only holds together if **every** member
is better off inside it than outside. If one firm earns more by competing
honestly, it will simply compete. "IR margin" = how much a firm gains by joining
the cartel. If negative, that firm should walk away.

**δ (delta, discount factor) = 0.95.** How much a firm cares about the future.
$1 next period is worth $0.95 now, the period after $0.90, and so on. Near 1 =
patient. This matters because **collusion is a trade**: give up profit today
(produce less), gain profit tomorrow (higher prices sustained). Only a patient
firm takes that trade. *(Unfortunately the literature uses δ for the discount
factor and Δ for the collusion score. They are unrelated.)*

**δ\* (delta-star), the critical discount factor.** The minimum patience needed
to make a cartel stick. Computed as

```
           (one-shot gain from cheating)
δ*  =   ───────────────────────────────────────
        (one-shot gain from cheating) + (cartel rent)   ... roughly
```

If δ\* = 0.31, the cartel holds easily (0.95 ≫ 0.31 — lots of slack). If
δ\* = 0.81, it *only just* holds. This one number explains the whole two-firm
result.

### The learning words

**Q-learning.** A basic reinforcement-learning method from the 1980s. Deliberately
simple — it does not use neural networks. The robot keeps a big spreadsheet:

|                     | produce 106 MW | produce 110 MW | … | produce 167 MW |
|---------------------|---|---|---|---|
| **saw price $38 last period** | 3421.7 | 3502.1 | … | 3399.8 |
| **saw price $39 last period** | 3455.2 | 3488.6 | … | 3402.5 |
| **…**               | | | | |

Each cell holds "if I'm in *this* situation and take *this* action, how much
total future profit do I expect?" The robot plays, sees what it actually earned,
and nudges the number in that cell toward the truth. Repeat millions of times and
the spreadsheet becomes accurate. **This spreadsheet is the "Q-matrix" or
"Q-table".**

**State.** The "situation" — the row label. All the robot knows about the world
right now. Ours only remembers **one period back** (see *monitoring* below).

**Action.** The "choice" — the column label. Here: how many MW to produce.

**Action space / discretising it.** The *set of choices allowed*. In real life a
firm could produce any amount — 141.7 MW, 141.71 MW, 141.712 MW — infinitely
many options (**continuous**). Q-learning's spreadsheet needs a finite number of
columns, so we restrict it to 15 specific allowed values (**discrete**).
**"Discretising the action space" just means: replace the infinite dial with a
15-position switch.** This is Task 1's whole premise, because the impossibility
paper's claim is specifically about the infinite-dial version.

**k = 15.** The number of positions on that switch.

**The grid, and ξ (xi) = 0.2.** *Where* to put the 15 positions. We span from the
monopoly output to the Nash output, then extend 20% further at each end so the
firm has room to cheat (produce more than Nash) and to punish. This is copied
exactly from the paper. Checking that "Nash sits at position 12 and monopoly at
position 2" — the same slots as the paper — is how we prove the copy is faithful.

**α (alpha) = 0.15, the learning rate.** How hard to nudge a cell toward the new
information. 0.15 = move 15% of the way. Too high → jumpy; too low → never learns.

**ε-greedy exploration and β = 4×10⁻⁶.** A robot that always picks its current
best guess never discovers anything better. So with probability ε it picks a
**random** action instead ("explore"); otherwise it picks its best ("exploit").
ε starts at 1 (100% random) and decays as `ε = e^(−βt)`. With β = 4×10⁻⁶, after
1 million periods ε ≈ 1.8%; after 2 million ≈ 0.03%. So the robot explores wildly
early and settles down late. **β controls how long the exploration phase lasts.**

**Session.** One complete experiment: two (or three) fresh robots with blank
spreadsheets, run until they settle. Because exploration is random, every session
turns out differently — so we run **1,000 sessions** and report the average.

**Convergence.** A session is "converged" when every robot's best action stops
changing for **100,000 periods in a row**. That is the paper's own criterion.
"100% of sessions converged, median 2.10M iterations" = every experiment settled
down, typically after ~2.1 million rounds.

**Greedy action / greedy policy.** What the robot would do if it stopped
exploring — its current best guess. This is what we measure.

**Limit strategy.** The final rulebook a robot ends up with: "if last period's
price was X, produce Y." Figure 3 is a plot of that rule.

### The monitoring words (this paper's core idea)

**Perfect monitoring.** The robot sees **exactly what each rival produced** last
period. If a rival cheats, you know instantly and unambiguously.

**Imperfect monitoring.** The robot only sees **last period's price**. It cannot
see rivals' outputs. This is realistic — real firms see market prices, not their
competitors' internal production logs.

Why that is hard: a **low price** has two possible causes.
1. A rival cheated and flooded the market. → should punish
2. Demand happened to be weak this period. → should not punish

**You cannot tell which.** That ambiguity is what the whole paper is about.

**Stochastic vs deterministic demand.** *Stochastic* = demand randomly wobbles
each period (there are `h = 2` possible levels, high and low). *Deterministic* =
no wobble. The ambiguity above only exists when demand is stochastic — with steady
demand, a low price can *only* mean cheating.

**Non-revealing price fraction (0.292, 0.304, 0.384…).** A measurement of *how
ambiguous* the price signal is: the fraction of situations where seeing the price
does **not** tell you what rivals did. 0.30 = "in about 30% of cases the price is
genuinely ambiguous." We compute this on our actual market and compare it to the
paper's formula, to confirm our version is about as ambiguous as theirs.

**m = 8.** The size of the demand wobble, expressed in "output steps" — a demand
shock moves the price about as much as a rival changing output by 8 grid
positions would. Copied from the paper. This is what makes shocks and cheating
confusable.

**Green–Porter.** A famous 1984 theory paper. Its idea: when firms can't observe
each other, a cartel is sustained by "if the price drops below a threshold, we
all flood the market for a while, then go back to cooperating." Occasional price
wars are the *cost* of policing a cartel you can't directly monitor. The brief
asks whether the robots reinvented this. Roughly, yes — with a twist (§3).

---

## Part C — Walking through the brief, section by section

### §1 "What was built"

**"The algorithm is theirs, verbatim."** The table lists every knob in the
economists' algorithm and confirms ours is set identically. This matters because
if we got a different answer with a different algorithm, we'd have learned
nothing. **Same algorithm, different market — so any difference in results is
caused by the market.** That is the experiment.

- *learning rule `Q ← (1−α)Q + α[π + δ·max Q']`* — the nudge described above. In
  words: "new estimate = 85% of my old estimate + 15% of (what I just earned +
  0.95 × the best I think I can do next)."
- *`Q₀`, the starting spreadsheet* — you have to fill the blank spreadsheet with
  *something*. The paper fills it with "what I'd earn if my rivals picked
  randomly." We copy that. (§9 checks this starting guess isn't secretly causing
  the result — an important audit, explained below.)
- *tie-breaking → higher output* — if two actions look equally good, pick the one
  producing more. This is deliberately the *anti*-collusive tiebreak, so it can't
  be accused of nudging robots toward cartels.

**"What changed: the payoff function."** The honest statement of what is
different: they had `price = demand − total output`; we have a 5-node grid
simulation with capacity limits and rising costs. Everything else is identical.

**"The discretisation is faithful — this is the crux of Task 1."**
Because the whole task is "discretise the action space properly," this is proof we
did it right rather than in a way that accidentally manufactures the answer. The
evidence: Nash lands at switch-position 12 and monopoly at position 2 — **the same
positions as in the paper's grid**. And restricting to 15 options moved the Nash
benchmark by only **0.95%**, so the 15-position switch is a good stand-in for the
infinite dial.

**"Why the market collapses to a lookup table."** Pure computational speed. In
the three-firm market all plants sit at the *same* node, so the grid simulation
only cares about the **total** MW injected there — not who produced it. There are
only 43 possible totals, so we run the expensive simulation 43 times up front and
then just look answers up. That turns months of compute into 36 minutes. *(This
shortcut breaks in the two-firm market, because its plants sit at two different
nodes — which is why that case needed new code.)*

### §2 "Headline result"

Δ = 68.8% — the robots got 69% of the way from honest competition to a full
cartel. Supporting details:

- *"± 0.31"* — the **standard error**, the uncertainty on the average. 68.8 ± 0.31
  means the true value is almost certainly between about 68.2 and 69.4. Tiny
  uncertainty = a solid number.
- *"per-firm Δ = 71.5 / 69.1 / 65.7%"* — all three firms colluded to a similar
  degree. Nobody was exploited. This matters: a high average could hide "two
  firms colluded and one got crushed."
- *"Session-level dispersion"* and the percentile table — answers "is 68.8% just
  an average of colluders and non-colluders?" **p5 = 53.6%** means the worst 5% of
  sessions still hit 53.6%. **100% of sessions have Δ > 0** — not a single one of
  the 1,000 failed to collude. So 68.8% is *typical*, not an average of extremes.
- *"Δ starts at −0.29"* — a lovely detail. At the very beginning, robots
  best-respond to rivals who are still flailing randomly, and the right answer to
  "my rival is behaving unpredictably" is to produce *more* than Nash. So the
  score starts **below zero** (worse than honest competition), then climbs.
  **The paper's Figure 2 has the same negative dip** — reproducing an incidental
  wrinkle like that is strong evidence the replication is faithful.

### §3 "Figure 3 — the limit strategy"

The finding: **the robots' final rule is "if the price fell, produce more."**
Producing more pushes the price down further — that is a punishment. So the
robots invented retaliation on their own.

**The objection this section defends against** is subtle and worth understanding,
because it's the kind of thing an examiner will raise. The plot has price on one
axis and output on the other. But price and output are *mechanically* linked —
more output always means a lower price. So a downward-sloping cloud of points
might be pure arithmetic, with no strategy in it at all.

The fix is **within-session (fixed-effects) analysis**: instead of comparing
different sessions to each other, look *inside each individual session* and ask
whether that one robot produces more after seeing a low price. The slope is
identical (−0.910 both ways), and **97.7% of individual sessions slope downward**.
So it is a real behavioural rule, not an artefact of mixing sessions together.

**"Flatter than demand" — the key mechanism.** Compare two slopes:

- The demand schedule slopes at −2.9: if the price drops $1, that *corresponds to*
  2.9 MW more output in the market.
- The robots' rule slopes at −0.3: if the price drops $1, they add only 0.3 MW.

Because the robots retaliate **more mildly than the price drop that provoked
them**, the price partially recovers next period. A milder price drop then
triggers an even milder punishment, and so on — the price war **shrinks itself
out of existence**.

This solves a real puzzle. The robots only remember one period. They have **no
clock** — no way to count "punish for exactly 5 rounds then stop." Green–Porter
strategies need that clock. So instead the robots make the punishment *fade*
automatically. **They use intensity as a substitute for a timer.** That's the
paper's cleverest observation, and we reproduce it.

### §4 "Figure 4 — forced deviation"

The experiment: take converged robots, **force** one to cheat (produce a lot more)
for exactly one period, then let it go back to normal. Watch what the others do.

- *"static best response"* — we make it cheat by the *most tempting* amount, the
  amount that maximises its profit this period ignoring consequences. The
  strongest possible test.
- *"demand is frozen"* — we switch off random demand during the test so the only
  thing moving the price is the cheating. Clean experiment.
- *"paired no-deviation counterfactual"* — we run the *same* robots from the
  *same* starting point *without* the cheat, and subtract. So "+6.37 MW" means
  "6.37 MW more than they would have produced anyway." Without this control you
  couldn't tell punishment from normal fluctuation.
- *"all 12 phases of the limit cycle"* — after settling, robots often cycle
  through a repeating pattern (produce 141, 143, 141, 143…). If different sessions
  sit at different points in their cycle, averaging produces a meaningless
  zigzag. So we run each session from every possible starting point in its cycle
  and average, which cancels the cycle out exactly.

**Result 1: punishment, then fade.** +6.37 MW at t+1, decaying to ~+1.2. About
80% of the punishment is gone within five periods. Matches the paper.

**Result 2 — the important one: punishment is harsher in low demand (+8.68) than
in high demand (+6.37).** Here is why that is the smoking gun. In **high** demand,
a low price *might* just be weak demand — so punishing risks starting a war over
nothing, and the robots hold back. In **low** demand there's no such excuse, so
they hit hard. **The robots learned to be forgiving exactly when evidence is
ambiguous.** No one programmed that. It is the fingerprint of imperfect
monitoring, and it is the paper's own signature finding.

**"The residue."** The averaged line doesn't return *all* the way to normal — a
~1.2 MW gap persists forever, which the paper's figure doesn't show. The
explanation is the nice bit: it is not that everyone drifts slightly. It's that
**85% of sessions return *exactly* (residue +0.004 MW ≈ 0) and 15% never return at
all** (residue +23 MW). Once demand is frozen and strategies are fixed, the system
is a deterministic loop; a single shock can knock it into a *different* permanent
loop with nothing to pull it back. So: **a single act of cheating permanently
destroys the cartel about one time in six.** The average was hiding two completely
different behaviours.

### §5 "Figure 5 — is cheating actually deterred?"

**The most important sceptical check in the document.** High profits alone prove
nothing — maybe the robots are just *bad at their jobs* and accidentally
under-producing. That would look identical to collusion in the profit numbers.

The distinguishing test: **is cheating actually punished enough to not be worth
it?** Run the cheat and the no-cheat path, add up profits over the long run
(discounting the future by δ = 0.95 each period), and compare.

The collusion signature is: **positive today, negative overall.**
- Low demand: +$63 today, **−$885** overall → cheating is a bad idea. **76.3%
  deterred.** Textbook.
- High demand: +$177 today, −$179 overall on average — but the **median** session
  would *gain* $38, and only **46% are deterred.**

**The brief is honest that high demand is a weak point**, and correctly refuses to
bury it. Then it explains it: splitting by whether the cartel survives, sessions
that recover *gain* $50 from cheating, while the 15% permanently destroyed lose
$1,233. So **deterrence is a lottery** — usually cheating pays a little,
occasionally it's catastrophic, and the *average* is negative only because of that
rare disaster. That is a genuine economic finding, not a failure.

*(Note this is exactly what theory predicts: high demand is precisely when
cheating is easiest to disguise, so that is where the cartel is weakest.)*

**"Deterrence is not size-dependent"** — we re-ran it forcing cheats of every size
from tiny to huge; the deterred fraction barely moves. So the result isn't an
artefact of choosing one particular cheat size.

### §6 "Table I — the impact of imperfect monitoring"

Four experiments in a 2×2 grid, crossing:
- perfect vs imperfect monitoring (see rivals' outputs, or only the price)
- steady vs wobbly demand

**Finding 1 (counter-intuitive, and it replicates).** *Imperfect* monitoring with
steady demand scores **higher** (84.2%) than *perfect* monitoring (78.9%). Seeing
**less** helps the robots collude more! Why: with perfect monitoring the
spreadsheet has 3,375 rows (every possible combination of rivals' outputs); with
price-only it has 51 rows. **A smaller spreadsheet is far easier to learn.** The
robots coordinate better on a simpler rule. The paper found the same, and our
premium (+5.30pp) nearly exactly matches theirs (+5.44pp).

**Finding 2.** Imperfect monitoring + wobbly demand is the *worst* cell — that's
where cheating hides behind demand shocks.

**Finding 3: "DiD" = difference-in-differences.** A standard technique for
isolating one cause when two things change together. Going from the best cell to
the worst changes *both* monitoring and demand wobble, so you can't attribute the
drop to either alone. So you take a difference of differences:

```
DiD = (effect of demand wobble WITH imperfect monitoring)
    − (effect of demand wobble WITH perfect monitoring)
```

The demand-wobble effect that is *common to both* cancels out, leaving only the
part attributable specifically to **not being able to see your rivals**. Ours is
−15.5pp; the paper's is −8.91pp. Both negative and sizeable — imperfect
monitoring hurts collusion, but nowhere near destroys it.

**The brief's handling of the gap is the model answer.** Rather than defending
−15.5 vs −8.91, it *decomposes* it and shows the mismatch is entirely in the
**perfect-monitoring** row, not the imperfect one. Our imperfect row is −15.40 vs
their −13.35 — a 2pp match, and **that is the row the entire paper is about.**
The perfect-monitoring row is off because with 3 firms the spreadsheet is 50,625
cells per firm (15× the paper's duopoly), so those runs are limited by *learning
difficulty* rather than by economics. Direct evidence: they took 5.7M rounds to
converge vs 2.0M — nearly 3× longer. **The problem is diagnosed, not excused.**

### §7 "What is genuinely DIFFERENT"

Every honest replication has a part that doesn't match. Here it is.

The paper says price wars break out for **two** reasons: someone cheated, *or*
demand happened to be bad. That second one matters theoretically — if the robots
*never* punished after a bad demand draw, a clever cheater could always hide
behind "must have been demand," and the cartel would collapse.

**In our market, only cheating triggers a war.** A bad demand draw causes a
*bigger* price drop (−$5.77 vs −$2.36) and yet provokes **no** retaliation
(median response: exactly zero).

The "implied local slopes" table explains what the robots actually learned. Their
overall rule looks like a gentle −0.91 slope, but locally it is two different
things: **flat (+0.36 ≈ 0)** in the price range where ordinary demand noise lives,
and **very steep (−7.7 to −10.3)** in the lower range a real cheat pushes it into.

That is not a smooth reaction curve — it is a **trigger strategy**: *ignore
ordinary noise; retaliate hard once the price falls past a threshold.* Which is
arguably a **purer** form of Green–Porter than the paper found. The robots learned
to **tell demand shocks apart from cheating**, and once they could, they stopped
punishing the innocent ones.

**Why here and not there:** our price signal is *less* ambiguous than theirs
(non-revealing fraction 0.384 vs their predicted 0.568), because a realistic grid
simulation produces messier, more distinguishable prices than a clean algebraic
formula. Same root cause as the DiD gap in §6 — one explanation covering both
discrepancies, which is a good sign. And it comes with a **falsifiable
prediction**: crank up the demand uncertainty and the shock-triggered wars should
come back. Not yet run — that's the honest bit.

### §8 "Does this prove or disprove the impossibility result?"

**Neither — and the section explains why the question is slightly mis-posed.**

The impossibility paper (Sannikov & Skrzypacz 2007) is usually described as
"collusion is impossible with continuous actions." Reading it carefully, **that's
not what it says.** Its result is about **speed**: as firms are allowed to react
faster and faster (period length → 0), the price signal becomes uninformative
faster than the temptation to cheat shrinks, and collusion dies.

Continuous actions appear in their setup but are **not what does the work.**

So:
- Our experiment runs at one fixed period length. A theorem about "as period
  length → 0" **cannot be contradicted** by an experiment that never takes that
  limit. The two results simply do not meet.
- And "discretising is what let them collude" is **also not right**, since
  discreteness isn't what their theorem rules out.

**What the exercise actually delivers:** a like-for-like number on *our own
market*. Now, when the continuous-action PPO agents do something, there is a
concrete benchmark — Δ = 68.8% with real punishments — to compare against, instead
of arguing in the abstract. That is why this had to be done first.

The last paragraph is a deeper audit: their theorem's conditions genuinely **do**
apply to our three-firm market, because all three firms sit at the same node,
which effectively switches the network off (every firm's deviation moves prices
identically — "rank 1"). The escape routes require firms at **different** nodes
separated by a congested wire, so that you can tell *who* deviated from *where*
prices moved. **Congestion is the identification device.** Our network can do that
— just not with all firms parked on one node.

### §9 "Anticipated questions"

Two are worth highlighting because they are the strongest audits in the document.

**"Is the punishment learned, or just the starting spreadsheet showing through?"**
This is the sharpest possible attack and it would have been fatal. Remember the
blank spreadsheet is filled with "what I'd earn against random rivals" — and the
best action under *that* guess is **maximum output**. So if the robots simply
never visited some situations, those cells would still say "produce maximum," and
in a rare situation (like being cheated on) the robot would produce a lot — which
would **look exactly like a punishment but be nothing but the untouched initial
guess.**

Checked directly: the learned policy differs from the initial guess in **50.7 of
51 states**, only **0.6%** of cells are still untouched, and specifically in the
states a deviation lands in, removing the untouched ones changes the measured
punishment by 0.1 MW. **The punishment is genuinely learned.** Being able to
answer this is what separates a real replication from a plot that looks right.

**"Did you check robustness to α and β?"** — *"Not yet — this is the main open
gap."* The paper swept 10,000 combinations and got Δ between 65–80%. We ran one
setting. Stating this plainly, unprompted, is the right call.

### §10 "Open items, honestly"

A list of what hasn't been done. Note **item 6 — "n = 2 variant on the two-firm
market" — is now complete**, and is written up in
[`RESULTS_2FIRM.md`](RESULTS_2FIRM.md).

---

## Part D — The two-firm results, and its two extra ideas

The two-firm case (`RESULTS_2FIRM.md`) got **Δ = 64.14%** — same story, deep
collusion. But it needed two ideas the three-firm case didn't. Both are worth
understanding because they're the parts most likely to be questioned.

### Idea 1: how do you discretise a firm that owns TWO plants?

Q-learning gives each robot **one** number to choose. But firm 0 owns two plants,
so it has **two** numbers to choose. Mismatch.

The obvious fix — "let it choose a total, then split that total between its plants
by some fixed rule" — is normally **dangerous**, because if your fixed splitting
rule can't produce the Nash split *and* the monopoly split, those benchmarks
become unreachable and the Δ ruler breaks.

Here it turned out to be **exactly right**, for a reason specific to this market:
**no wire is ever congested** (that's the "shadow price = 0.0000029" measurement).
No congestion → one uniform price everywhere → **both of firm 0's plants get paid
the same price.** And once both plants earn the same price, deciding how to split
production between them is no longer a strategic question at all — it's just
"produce it as cheaply as possible."

We didn't take that on faith. We tried **every** alternative split in **every**
situation and measured what the firm could gain by deviating from cheapest-first:

> **$0.000000000012** — i.e. zero.

So the two-plant firm's choice really does collapse to a single number **with
provably nothing lost.** And the cheapest-first rule happens to reproduce both
benchmark splits exactly. Bonus: this leaves us with 2 robots × 15 actions —
**exactly the paper's own duopoly setup**, making it the *closest* of the two
replications.

### Idea 2: the benchmark Nash in the docx is wrong for this market

This is the one to be most careful explaining, because it contradicts a number in
your own benchmark table.

The Nash point in the docx comes from a formula (the "LCP") where each firm is
assumed to exercise market power **against its own node's local demand.** That's
correct when a firm's plants are all at one location — true for all three firms in
the three-firm market, which is why it worked there.

But firm 0 here has plants at **two** locations. The formula assumes it withholds
production at node 0 to push up node 0's local price. **But node 0's price isn't
local** — no wire is congested, so node 0 is fused to the rest of the grid and its
price is set globally. Withholding there achieves nothing.

The test is just the definition of Nash — *can anyone improve by changing only
their own choice?*

> At the docx's "Nash" point, firm 0 can raise its profit **25%**
> ($3,007.85 → $3,764.81) by simply changing its own output.

Since somebody can improve unilaterally, **it is not a Nash equilibrium.**

**Why this really matters:** Nash is the **zero mark on the Δ ruler.** If the zero
is set at a point that isn't actually equilibrium, then a robot that learns
nothing but plain selfish best-responding — zero coordination, zero collusion —
would still score Δ > 0. **You would "discover" collusion that isn't there.**
Given the whole project is a referee on whether reported algorithmic collusion is
real, publishing that would be exactly the error we're supposed to be catching.

So we computed the real Nash by brute force (let each firm best-respond over and
over until nobody wants to move) and used that instead: **(115.0, 43.19, 181.41)**.
The docx number is still reported alongside for continuity.

### The economics finding that came out of it

In the two-firm market, firm 0 is a **reluctant cartel member.** Its cheap plant is
already maxed out at capacity in *both* the competitive and the cartel outcome, so
the only thing it can cut is its *expensive* plant. Result:

- firm 0 must cut **43 MW** and gains only **+$72 (+2.1%)**
- firm 1 cuts only **29 MW** and gains **+$455 (+13.6%)**

Firm 0 does the most work for the least reward. Its δ\* is **0.81** — meaning it
only *barely* has enough patience to bother colluding (vs ≈0.49 for every firm in
the three-firm market, which is comfortable).

And the robots responded to this in a way nobody programmed: **they didn't aim for
joint monopoly at all.** Joint monopoly would split the gains +$72/+$455 —
wildly unfair to firm 0. Instead they landed on **+$161/+$177 — an almost
perfectly equal split of the spoils**, which is exactly what makes firm 0 willing
to participate.

**The robots solved the cartel's fairness problem, not just its profit problem.**
The three-firm market couldn't have revealed this, because its firms were nearly
identical to begin with. This is the genuinely new economics in the two-firm case.

*(A caution that follows from this: the per-firm Δ of "222% / 39%" does **not**
mean firm 0 was "more than fully collusive." Δ divides by that firm's own
Nash-to-monopoly gap, and firm 0's gap is a tiny $72, so dividing by it produces a
wild number. Always read the dollar gains for this market, not the per-firm
percentages.)*

---

## Part E — If you only remember five things

1. **Δ is a ruler** from honest competition (0%) to a perfect cartel (100%). The
   robots scored **68.8%** (three firms) and **64.1%** (two firms). That is deep
   collusion, and it replicates the economics papers.
2. **The algorithm is copied exactly; only the market is ours.** So any difference
   in results is caused by the market, which is the whole point of the exercise.
3. **High profit alone proves nothing** — bad robots look the same as colluding
   robots in the profit column. What proves collusion is the *behaviour*: they
   retaliate when the price falls, they retaliate **more gently when the evidence
   is ambiguous**, and cheating loses money once retaliation is priced in.
4. **One thing doesn't replicate**, and it's stated plainly: our robots don't
   start price wars after bad demand luck, because our realistic market gives a
   clearer signal than the paper's toy formula. One explanation covers this *and*
   the DiD gap, and it comes with a testable prediction.
5. **This doesn't refute the impossibility paper** — that result is about reaction
   *speed*, not discrete vs continuous actions, so the two never actually collide.
   What we built is a **benchmark**: a real number, on our own market, that the
   continuous-action PPO results can now be measured against.
