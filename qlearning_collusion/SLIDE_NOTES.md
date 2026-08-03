# Speaking notes — figure-by-figure

Keep this open during the meeting. Each block is ~20–40 seconds spoken.
Bold = the number to say out loud. "If pushed" = only say it if asked.

---

## The market — `fig0_network`
### "One network, two ownership structures"

> "Same physical grid in both experiments — five nodes, five wires, same demand,
> same line limits. The only thing that changes is **who owns which power
> station**. In one version, three firms with one plant each, all sitting at the
> same hub. In the other, two firms — one owns two plants at *different*
> locations, the other owns one. So any difference in the results comes from
> ownership structure alone, not from the market."

**If pushed — why two structures?** The paper's own baseline is a *duopoly*, so
the two-firm market is the direct like-for-like comparison. The three-firm market
is our repo's main topology. Running both means we can separate "does it
replicate" from "does it depend on structure."

---

## Slide 6 — `fig2_profit_evolution` ×2
### "Both structures learn to collude"

> "This is the collusion score over training time. Zero on this axis means the
> firms are competing honestly — the Cournot–Nash point. One hundred means
> they're behaving as a single monopoly. Both structures climb to about two
> thirds of the way up: **64% with two firms, 69% with three**. The paper reports
> 76% and 73%, so we land a few points under, on a far harder market."

> "One detail worth pointing out — the curve **starts below zero**. That isn't
> noise. At the very start each robot is best-responding to a rival that's still
> choosing at random, and the right answer to a randomising rival is to
> *over*-produce. So they begin worse than competitive and learn their way up.
> The paper's figure has the same dip, which is a good sign the replication is
> faithful."

The flat section at the end = exploration has decayed to zero and strategies have
locked in.

**If pushed — why does it plateau there and not at 100?** §4 of the write-up:
in the two-firm market the cartel is barely worth joining for firm 0, so they
settle short of full monopoly. That's the economics, not a training failure.

---

## Slide 7 — `fig_delta_distribution` ×2
### "Not one session in 2,000 failed to collude"

> "Each experiment is **1,000 completely independent runs** — fresh robots, new
> random seeds, no shared history. This is the histogram of where each one ended
> up. **Every single one of the 2,000 runs finished above the Cournot–Nash
> benchmark.** The weakest run in the two-firm market still reached **30%**, the
> weakest three-firm run **38%**. Medians are 64% and 68%."

> "So the headline number isn't an average of some sessions colluding and others
> failing. Collusion is the **typical** outcome, not the lucky one."

⚠️ **Fix your subtitle before the call.** It currently says *"ends above the
**competitive** benchmark."* Δ > 0 means above the **Cournot–Nash** benchmark,
which is a much *stronger* claim — competitive is far below Nash. Say
"Cournot–Nash," or you're understating your own result and using the wrong
benchmark name.

---

## Slide 9 — `fig3_limit_strategy` ×2
### "The learned strategy is a punishment rule"

> "This is the rule each robot ended up with. Horizontal axis is the price it
> observed last period; vertical is what it produces now. **The line slopes
> down — when the price falls, it produces more.** Producing more pushes the
> price down further, so that is retaliation. Nobody programmed that in."

> "The critical detail is the comparison with the two grey lines, which are the
> demand curves. **Our strategy line is far flatter** — about −0.3 MW per dollar
> against −6 for demand. That matters, because it means the retaliation is
> *milder* than the price drop that provoked it. So the price partly recovers,
> next period's retaliation is milder still, and **the price war puts itself
> out**."

> "And here's why that's clever: these robots only remember **one period**. They
> have no clock — no way to count 'punish for five rounds, then stop.' So instead
> they make the punishment *fade automatically*. They use punishment **intensity**
> as a substitute for a timer."

**If pushed — isn't a downward slope just arithmetic?** Good objection, and we
tested it. More output mechanically means a lower price, so a cross-session plot
would slope down with no strategy in it at all. We redid it *within* each
session — same slope (−0.910 both ways in the three-firm market), and **97.7% of
individual sessions slope down on their own**. It's a behavioural rule, not a
composition artefact.

---

## Slide 10 — `fig4_deviation` ×2
### "A deviation triggers a price war that fades"

> "We take converged robots and **force one to cheat** — produce a lot more — for
> exactly one period, then let it go back to normal. Demand is frozen so nothing
> else moves the price. The grey line is the *same* robots from the *same* state
> with no cheating, so the gap between the lines is purely the punishment."

> "The rival immediately floods the market — about **+6.6 MW** — and that decays
> away over roughly five periods."

> "But the important thing is the **comparison between the two panels**. The
> punishment is **harsher in low demand, +10.2 MW, than in high demand, +6.6**.
> That's the signature result. In high demand, a low price might genuinely just
> be weak demand — so the robots hold back rather than start a war over nothing.
> In low demand there's no such excuse, so they hit hard. **They learned to be
> forgiving exactly when the evidence is ambiguous.**"

**If pushed — why don't the lines return all the way?** About **80% of sessions
return exactly** to their pre-deviation behaviour — zero residue. The remaining
20% get knocked into a *different* permanent cycle and never come back. So the
visible gap is entirely that minority: **a single act of cheating permanently
destroys the cartel about one time in five.** The paper's figure shape is what
our 80% do.

**If pushed — is the punishment really *learned*?** Yes, and we checked it
specifically because it would have been fatal. The blank Q-table is initialised
so that its best action is *maximum output* — so an unvisited cell would look
exactly like a punishment while being nothing but the untouched starting guess.
Measured: only **0.6%** of cells are still at initialisation, and removing them
changes the punishment by 0.1 MW.

---

## Slide 11 — `fig5_deviation_value` ×2
### "Cheating pays today and loses over the punishment"

> "High profits on their own prove nothing — robots that are simply *bad at
> optimising* would look identical in the profit column. So this is the direct
> test: **is cheating actually deterred?**"

> "We run the same robots twice from the same state, once cheating and once not,
> and add up profits over time, discounting the future. In low demand, cheating
> earns about **+$23 immediately, then loses $457** once the punishment is priced
> in — **deterred in 76% of sessions**. That's textbook collusion: pays today,
> loses tomorrow."

> "I want to be straight about high demand though. The average is still negative,
> but only **−$91**, and only **42% of sessions are deterred**. That's not a flaw
> in the replication — it's exactly what the theory predicts. High demand is
> precisely the state where a cheat can hide behind bad luck, so that's where the
> cartel is weakest."

> "There's also a nice detail: **deterrence is a lottery, not a certainty.**
> Sessions where the cartel survives actually gain a little from cheating; the
> ones permanently destroyed lose a fortune. The negative average comes entirely
> from that tail."

**If pushed — did you pick a convenient cheat size?** No — we forced the *static
best response*, i.e. the most tempting one-shot defection, which is the hardest
test. And we swept sizes from small to large: the deterred fraction barely moves.

---

## Slide 13 — `fig6_deviation_vs_shock` ×2
### "They punish cheating, not bad luck"

Lead with the honesty — it lands much better than being asked.

> "This is the **one result that does not match the paper**, and I want to put it
> up front."

> "The paper says price wars are triggered by two things: someone cheating, *or*
> just a bad demand draw. In our market, **only cheating triggers one.** Left
> panel: the demand shock is actually the **bigger** price drop — **$4.72 versus
> $1.77** — and yet the right panel shows it provokes essentially **no
> response**, while the deviation triggers **+13 MW**."

> "Our reading is that the robots learned something *sharper* than the paper's
> smooth reaction curve. They're **flat** across the price range that ordinary
> demand noise moves them within, and **steep** once the price falls past that.
> That's a **trigger strategy** — arguably a purer form of Green–Porter than the
> paper found. They learned to filter the demand noise out of the signal."

> "As for why: our price signal is **less ambiguous** than theirs, because a
> realistic grid simulation produces messier, more distinguishable prices than
> their one-line demand formula. The same explanation covers the one other place
> we diverge. And it's **testable** — turn the demand uncertainty up and the
> shock-triggered wars should reappear. We haven't run that yet."

That last sentence is the strongest thing on the slide: a divergence with a single
cause and a falsifiable prediction is a *finding*, not a failure.

---

## Slide B7 (backup) — `fig1_output_evolution` ×2
### "Output falls through the Cournot–Nash level and keeps going"

> "Same training run as the profit chart, but in physical quantities. The
> horizontal bands are the benchmarks. Output **starts above the Cournot–Nash
> band, falls straight through it, and settles well below** — **142 and 164 MW**
> against Nash of 158 and 181, heading toward the monopoly point of 115 and 153."

> "So the collusion isn't a pricing artefact. The firms are **physically
> withholding electricity**. That's what the profit gain is actually made of."

⚠️ **Fix this title before the call.** It currently says *"falls through the
**competitive** level."* It does not. Competitive output is far higher — firm 0
would produce 239 MW and firm 1 210 MW. The curve **starts below competitive**
and falls through the **Cournot–Nash** level. Say "Cournot–Nash," or you'll be
corrected.

---

## The three sentences to have ready for any question

1. **"Is it really collusion?"** — Three independent tests, and a
   failure-to-optimise story predicts none of them: they retaliate when the price
   falls; they retaliate *more gently when the evidence is ambiguous*; and
   cheating loses money once retaliation is priced in.
2. **"Does this refute the impossibility paper?"** — No, and it can't. That
   result is about reaction *speed* — the limit as the period length goes to
   zero — not discrete versus continuous actions. We run at one fixed period
   length, so the two never meet. What we've built is a **benchmark** the
   continuous-action PPO results can now be measured against.
3. **"What's the number?"** — Δ = **64.1%** (two firms) and **68.8%** (three),
   against the paper's 76.3% and 72.7%, with 100% of 2,000 sessions colluding.

---

## Appendix — which figures actually PROVE collusion, and which don't

**The rival hypothesis you are arguing against.** There is only one serious
alternative to collusion, and it must be named explicitly:

> *"The algorithms didn't collude. They just **failed to optimise** — they
> under-produce out of incompetence, and that happens to look like restraint."*

This story explains high profits and low output **just as well as collusion
does**. So every profit-level figure is powerless against it. Only figures about
*behaviour* can discriminate. That is why the deck has so many of them.

| figure | proves collusion? | what it actually kills |
|---|---|---|
| `fig0_network` | ✗ no | nothing — setup only |
| `fig2_profit_evolution` | ✗ **no** | states the puzzle; does not solve it |
| `fig1_output_evolution` | ✗ no | kills "it's an accounting/price artefact" — withholding is physical |
| `fig_delta_distribution` | ✗ no | kills "the mean hides failures" — it's universal |
| `fig3_limit_strategy` | ✓ **yes** | kills "unconditional under-production" |
| `fig4_deviation` | ✓✓ **strongest** | kills "no causal retaliation" |
| `fig4b_punishment_split` | ✓ supporting | kills "the residue means it never really recovers" |
| `fig5_deviation_value` | ✓✓ **yes** | kills "punishment exists but is too weak to matter" |
| `fig6_deviation_vs_shock` | ✓ **placebo** | kills "it's a mechanical price-reflex" |

### Tier 1 — figures that establish the puzzle but prove nothing

`fig2`, `fig1`, `fig_delta_distribution`. Profits above Nash, output below Nash,
in every session. **Necessary but not sufficient.** If challenged here, concede
immediately — "agreed, this alone doesn't prove collusion, which is why the next
four slides exist." Conceding this makes the rest far more credible.

One detail *does* bite, though, and is worth deploying: **Δ starts at −0.23.**
Early in training the algorithms find and play the aggressive best response —
they *over*-produce relative to Nash. So they are demonstrably **capable of
optimising**. Then they move away from it. An incompetence story has to explain
why they first optimise correctly and then abandon it.

### Tier 2 — `fig3`: the behaviour is *conditional*

Output is a **systematic function of an observed signal** (last period's price).
An algorithm that merely under-produces has no reason for its output to depend on
anything. **Conditional response to a signal is what "strategy" means.**

Stronger still: the *shape* was a prediction that could have failed. Theory says
the rule must be **flatter than demand** or the punishment never ends. Measured:
−0.24/−0.33 against demand's −6.1. It came out right.

### Tier 3 — `fig4`: the retaliation is *causal*, not correlational

This is the only **interventional** figure. We reach in, force a deviation, and
compare against the *same sessions from the same state* without it. Everything
else is held fixed, so the gap is caused by the deviation. Incompetence predicts
a flat line; we see **+6.6 MW**.

And the single most persuasive number in the deck: **punishment is harsher in
low demand (+10.2) than high (+6.6).** That is a **signed prediction from
theory** about a quantity nobody tuned — punish cautiously when a cheat is
confusable with weak demand. Getting the sign right on a free parameter you never
fitted is what separates a replication from a curve-fit.

### Tier 4 — `fig5`: it satisfies the *definition* of an equilibrium

Tacit collusion isn't just "they retaliate" — it's "retaliation is **severe
enough that cheating doesn't pay**." That is the defining condition, and this is
the only figure that tests it directly: **+$23 today, −$457 discounted, deterred
in 76% of sessions.**

Be honest that high demand only partially passes (42% deterred). Frame it
correctly: **that is where theory says the constraint should be weakest**, so a
partial failure exactly there is confirmation, not refutation.

### The placebo test — `fig6` is doing more work than it looks

This is subtle and worth understanding, because it's the smartest thing in the deck.

Suppose a sceptic says: *"Your 'punishment' in fig4 isn't strategic at all. The
algorithms just have a reflex — price goes down, output goes up. Fig3 shows
exactly that reflex. Nothing intentional about it."*

That objection would explain fig3 **and** fig4 without any collusion. `fig6`
kills it. A demand shock produces an **even bigger price drop** (−$4.72 vs
−$1.77) — and provokes **no expansion at all**.

> Same-sized stimulus, opposite response, depending only on **what caused** the
> price drop.

A mechanical price-reflex cannot do that. The algorithms are responding to the
*strategically relevant event*, not to the price. **So the one result that
diverges from the paper is simultaneously the cleanest proof that the punishment
is targeted rather than reflexive.** Lead with that framing and a weakness
becomes a strength.

### What none of the figures prove — say these before you're asked

- **Not an agreement.** No communication exists in the model. This is *tacit*,
  emergent coordination — which is precisely what makes it interesting and hard
  to regulate.
- **Not full monopoly.** Δ is 64–69%, never 100%. Partial collusion.
- **Not universally incentive-compatible.** High demand only ~42% deterred.
- **Not proof against the impossibility theorem** — different regime entirely
  (see the three-sentence answers above).

### The one-sentence version

> "No single figure proves collusion. **The profit figures state the puzzle; the
> behavioural figures solve it.** Fig 3 shows the response is conditional, fig 4
> shows it's causal, fig 5 shows it's strong enough to deter, and fig 6 shows
> it's targeted at cheating rather than at low prices. The 'they just failed to
> optimise' story predicts none of those four."

---

## Two things to *volunteer* rather than wait to be asked

Both are in the two-firm write-up and both are the kind of thing an advisor is
pleased to hear you found yourself:

- **"The Nash benchmark in the benchmark doc isn't a Nash equilibrium for the
  two-firm market."** At that point firm 0 can raise its own profit **25%** by
  changing only its own output — so by definition it isn't equilibrium. It's fine
  for the three-firm market, where every firm's plants sit at one node; it breaks
  when firm 0 straddles two. This matters because **Nash is the zero on the Δ
  ruler** — using the wrong point would have scored plain selfish best-responding
  as "collusion." We recomputed the true Nash and used that.
- **"The algorithms didn't aim at joint monopoly — they split the gains evenly."**
  Joint monopoly would hand firm 0 only +$72 and firm 1 +$455, which firm 0 has
  almost no reason to accept. The robots landed on **+$161 / +$177 — a nearly
  equal split.** They solved the cartel's *fairness* problem, not just its profit
  problem. The three-firm market couldn't reveal this because its firms were
  nearly identical to start with.
