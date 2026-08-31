# Does the Sannikov–Skrzypacz impossibility result survive on a congested network?

**Reference.** Yuliy Sannikov and Andrzej Skrzypacz, "Impossibility of Collusion
under Imperfect Monitoring with Flexible Production," *AER* 97(5), 2007,
1794–1823. Referred to below as S&S.

**Verification code.** `theory/fingerprint_geometry.py` (this audit),
`theory/signal_rank.py` (earlier rank-only version). All numbers quoted here are
reproduced by `PYTHONPATH=. python theory/fingerprint_geometry.py` on
`MARKET_CONFIG=three_firm_dist`.

---

## 0. Yes — there is a proof, and it is worth being precise about what it proves

S&S prove **five propositions and a corollary**, all of the form "as Δ → 0, the
best equilibrium payoff converges to static Nash." Δ is the **length of a
period**, i.e. how often firms can change output. The result is about **speed of
response**, *not* about continuous action spaces, and not about any fixed Δ.

| | statement |
|---|---|
| Prop. 1 | symmetric PPE, constant MC: v̄(Δ) → v^N |
| Prop. 2 | **asymmetric** PPE: v̄^a(Δ) → 2v^N |
| Cor. 1 | + monetary transfers, money-burning bounded: same |
| Prop. 3 | symmetric, convex costs (A3): same |
| Prop. 4 | asymmetric + transfers, convex costs with **A4** (c‴ ≤ 0): same |
| Prop. 5 | mean-reverting (non-stationary) prices: f̄ → f (MPE) |

The three hypotheses the whole thing rests on, in the authors' own words:
information arrives **continuously without sudden events**; firms can **react
quickly**; and **public signals depend on total market supply only, not on
individual decisions**.

The proof chain:

- **Lemma 1** — A1/A2 ⟹ static Nash unique and symmetric.
- **Lemma 2** — a Gaussian tail test with likelihood difference O(Δ) has type-I
  error > O(Δ^{0.5+ε}). *This is the engine.* Deviation gain is O(Δ); false
  positives cost O(Δ^{1/2}); the punishment swamps the gain.
- **Lemma 3** — the optimal test is bang-bang with a tail critical region (uses
  MLRP of the normal family).
- **Lemma 4** — on the compact collusive set, the best static deviation gain is
  bounded below by ε_π > 0.
- **Prop. 1/3** — Lemmas 2–4 ⟹ symmetric collusion dies.
- **Prop. 2/4** — the *only* new step: relax the N incentive constraints to their
  **sum**. This is licensed by one sentence:

  > "Each deviation has the same effect on the distribution of prices,
  > decreasing the mean of the observed price to p(Q + ε)."

  Summing makes balanced transfers cancel identically (their eqs. (2)→(3),
  (4)→(3)), reducing the asymmetric problem to the symmetric one, where Prop. 1
  applies.

S&S also supply their **own escape hatch** in §V.C: capacity constraints violate
A4 (marginal cost jumps to ∞), so the sum of derivatives is −∞, and asymmetric
equilibria **with enforceable monetary transfers** achieve first best. And §VII
supplies the crucial warning: with two *distinct but correlated* signals
(correlation parameter α → ½), asymmetric equilibria still collapse to Nash.
Distinguishability alone is not enough — it must be strong relative to the noise.

---

## 1. The network game, and what is structurally different

Nodes 𝒩 (|𝒩| = n), lines ℒ, PTDF matrix Φ ∈ ℝ^{|ℒ|×n}. Firm *i* injects q_i at
node ν(i). The ISO clears a DC-OPF; with strictly decreasing nodal inverse
demands P_n(·; u),

    max_{d ≥ 0}  Σ_n ∫₀^{d_n} P_n(x; u) dx
    s.t.  1ᵀd = 1ᵀg      [λ]
          −F ≤ Φ(g − d) ≤ F   [μ]

giving the standard decomposition, with B the active-line set:

    LMP = λ·1 + Φ_Bᵀ μ_B                                    (★)

Firm i's flow profit is π_i = q_i · LMP_{ν(i)} − c_i(q_i).

Three structural departures from S&S:

1. **The public signal is a vector** (n nodal prices), not a scalar.
2. **Each firm is paid a different component of it.** π_i depends on
   LMP_{ν(i)}(q), which is *not* a function of total output Q alone. The stage
   game stops being aggregative.
3. **Line limits are shared constraints**, coupling firms through a constraint
   set rather than only through the price.

Define firm i's **deviation fingerprint**

    d_i := ∂LMP/∂q_i ∈ ℝⁿ.

S&S's Prop.-2 sentence is *exactly* the assertion **d_i = d_j for all i, j**.

---

## 2. Step-by-step audit

### 2.1 Lemma 2 and Lemma 3 survive a vector signal — with a caveat

**Proposition N1 (dimension invariance).** Let the period-average price vector be
Y ~ N(λ(q), Σ/Δ) with Σ ≻ 0, and let a deviation shift the mean by δ ≠ 0. The
family {N(λ + tδ, Σ/Δ)} has monotone likelihood ratio in the scalar

    T = δᵀ Σ⁻¹ Y,     T ~ N(δᵀΣ⁻¹λ, ‖δ‖²_{Σ⁻¹}/Δ),

so Lemmas 2 and 3 apply verbatim with (μ − μ′)/σ replaced by ‖δ‖_{Σ⁻¹}.

*Consequence, and it is a negative one:* **observing n nodal prices instead of one
changes only the constant in Lemma 2, never the Δ-order.** More prices can never
rescue symmetric collusion. Anyone hoping the network helps merely because the
ISO publishes a richer price vector is wrong.

**Caveat (non-degenerate monitoring, N0).** Prop. N1 needs range(δ) ⊆ range(Σ).
If a deviation moves the price vector in a direction the noise cannot produce,
the two measures are mutually singular, the likelihood ratio is unbounded, and
monitoring is **perfect** — collusion is then trivially sustainable for an
uninteresting reason. §5.4 shows the current simulator sits in exactly this
degenerate case whenever a line binds.

### 2.2 Congestion does **not** create AMP-style jumps

A natural hope is that congestion makes prices jump, converting Brownian
monitoring into AMP Poisson monitoring (where collusion survives Δ → 0). **It
does not.** In a strictly convex DC-OPF the shadow price μ_ℓ rises *continuously*
from zero as a line approaches its limit; the primal is unique and continuous in
the parameters, so LMPs are continuous — **kinked, not discontinuous** — in the
shock. S&S Remark 1's "no jumps" premise survives, and Lemma 2's Gaussian tail
argument is intact.

The AMP channel needs genuine discreteness: LP dual degeneracy under
piecewise-linear offers (i.e. real ISO dispatch, where the marginal unit switches
discretely), unit commitment, forced outages, scarcity-pricing steps. That is a
real and probably easier route to breaking S&S, but it is **not congestion**, and
should not be conflated with the network question.

### 2.3 Lemma 1 fails as a theorem, but is repairable

S&S prove symmetry by subtracting the two first-order conditions to get
p′(Q)(q_i − q_j) = 0. On the network the FOCs are

    ∂LMP_{ν(i)}/∂q_i · q_i + LMP_{ν(i)} − c_i′(q_i) = 0

and subtracting cancels nothing. Symmetry is simply false here (heterogeneous
costs, heterogeneous nodes). This is not fatal: **Props. 1–4 need only uniqueness
of the static Nash**, not symmetry. But uniqueness is now a genuine open
hypothesis, because LMP_{ν(i)} is kinked and possibly non-concave in q_i once a
line binds. This repo has already hit the live version of this problem: the
paper's LCP "Nash" is *not* an equilibrium of the redistributed market (max
unilateral deviation gain $1,288), which is why the benchmark is the iterated
best-response Nash. **Uniqueness must be assumed or verified numerically; it
cannot be cited from Lemma 1.**

### 2.4 The main break: an exact criterion for when the summing step dies

Differentiate the KKT system of (★) with respect to an injection at node m.
Assume strict complementarity and LICQ (Φ_B full row rank), so B is locally
constant. With H = diag(1/P_n′(d_n)) ≺ 0, writing dd = H(1·dλ + Φ_Bᵀ dμ_B),
balance 1ᵀdd = 1 and Φ_B(e_m − dd) = 0 give

    ⎡ 1ᵀH1      1ᵀHΦ_Bᵀ  ⎤ ⎡dλ ⎤   ⎡ 1       ⎤
    ⎢                     ⎥ ⎢    ⎥ = ⎢         ⎥          (★★)
    ⎣ Φ_B H1    Φ_BHΦ_Bᵀ ⎦ ⎣dμ_B⎦   ⎣ Φ_B e_m ⎦

The coefficient matrix K = [1 Φ_Bᵀ]ᵀ H [1 Φ_Bᵀ] is negative definite (hence
invertible) whenever [1, Φ_Bᵀ] has full column rank. **The only dependence on the
injection node m is through Φ_B e_m — the column of the binding-line PTDF at that
node.** Hence d_m = a + L·Φ_B e_m with L = [1 Φ_Bᵀ]K⁻¹[0; I] injective, and:

> **Proposition N2 (separation criterion).**
> d_i = d_j  ⟺  Φ_B e_{ν(i)} = Φ_B e_{ν(j)}.
> More generally, rank{d_i − d_1}ᵢ = rank{Φ_B(e_{ν(i)} − e_{ν(1)})}ᵢ, and
> rank{d_1,…,d_N} ≤ 1 + |B|.

Two firms are statistically indistinguishable **iff every binding line has the
same shift factor at their two nodes.** This is purely topological — it does not
depend on costs, demand levels, or the shock. Immediate corollaries:

- **B = ∅ ⟹ d_i = d_j.** With no congestion every LMP equals λ, which depends on
  total injection only. S&S applies verbatim. *Congestion is necessary for any
  break whatsoever.*
- **ν(i) = ν(j) ⟹ d_i = d_j.** Co-located firms are never separable. (This is why
  the old three-firm hub market could never break S&S: the siting chosen for the
  Nash-benchmark reason switched the network off for monitoring purposes.)
- **A radial cut with all firms on one side ⟹ d_i = d_j.** Every node upstream of
  a radial line has the same shift factor onto it. **A load-pocket corridor can
  never separate the firms that feed it,** no matter how hard it is congested.
- Therefore separation requires a binding constraint **inside a mesh/loop**, whose
  shift factors genuinely discriminate between the firms' nodes.

### 2.5 What replaces the summing step: the incentive geometry

Work at the Δ → 0 (continuous-time) limit, where the continuation value of firm i
evolves as dW_i = r(W_i − π_i)dt + r βᵢᵀ(dY − λ dt). Firm i's local incentive
constraint against expanding output is

    βᵢᵀ d_i = −a_i,     a_i := ∂π_i/∂q_i > 0 at any collusive profile.

**Balanced transfers** — value moved between firms, never burnt — means Σᵢβᵢ = 0.

> **Proposition N4 (the positive half, N = 2).** Balanced β₁ = −β₂ satisfying both
> ICs exist **iff d₁ and d₂ are linearly independent** (or the knife-edge
> d₂ = −(a₂/a₁)d₁). The minimum incentive volatility is
>
>     C = min{Σᵢ βᵢᵀΣβᵢ : Σβᵢ = 0, βᵢᵀd_i = −a_i} = cᵀG⁻¹c,
>     G_{ij} = d_iᵀΣ⁻¹d_j,   c = (−a₁, +a₂),
>
> and in the symmetric case (a_i = a, ‖d_i‖_{Σ⁻¹} = s, cosine ρ),
>
>     **C = 2a² / (s²(1 − ρ)).**

This single formula contains the whole story.

- **ρ = 1** — S&S's hypothesis. Then β₁ᵀd₁ < 0 and β₁ᵀd₂ = −β₂ᵀd₂ > 0 are
  contradictory: **no balanced β exists, C = ∞.** Incentives *must* destroy value,
  and Lemma 2 makes the destruction rate O(Δ^{1/2}) against an O(Δ) gain. This is
  a one-line re-derivation of Props. 2 and 4.
- **ρ < 1** — balanced β exists with finite volatility. The value-destruction term
  vanishes to first order. **The S&S obstruction is gone.**
- **ρ → 1** — C ↑ ∞ like 1/(1−ρ). This is precisely S&S §VII's α → ½ warning,
  now exact rather than conjectural. *Congestion existing is not enough; it must
  separate the firms strongly relative to the price noise.*

Two things are worth emphasising about the positive case. First, β is a transfer
of **continuation value**, not money — so unlike S&S §V.C, **no contractible side
payments are required**, which matters because tacit collusion cannot use them.
Second, this is exactly Fudenberg–Levine–Maskin pairwise full rank, restored by
the network.

### 2.6 A second, independent break: Lemma 4 fails under a saturated export path

Lemma 4 asserts the best deviation gain is bounded below by ε_π > 0 uniformly on
the collusive set. On a network, when firm i's export path saturates, extra MW
cannot reach load and its own nodal price collapses, so **a_i = ∂π_i/∂q_i ≤ 0**:
firm i has *no profitable upward deviation*, its IC is slack, and the entire
Prop. 1 machinery has nothing to bite on. Measured at the profit-maximising
separating profile of §5: **min_i a_i = −3.61**.

This is the endogenous, state-dependent analogue of S&S §V.C — but the constraint
is a **shared line**, not a firm-level capacity primitive, so *which* firm is
capped shifts with the shock and with rivals' output. Honest caveat: for that firm
the constraint, not the punishment, is doing the work. Its real content is that a
congestion-capped firm is a **credible non-deviator**, which is exactly what §V.C
needs in order to free up the other firms' incentives.

---

## 3. What can actually be proved

> **Theorem (dichotomy).** Consider the network game with Δ → 0, non-degenerate
> monitoring (N0), unique static Nash, and LICQ + strict complementarity at the
> relevant profiles.
>
> **(i) Impossibility survives.** If Φ_B e_{ν(i)} = Φ_B e_{ν(j)} for all i, j at
> every profile in the collusive region — in particular if no line binds, or all
> firms share a node, or the only binding constraints are radial cuts with the
> firms on one side — then all fingerprints coincide, the Prop.-2 summing step is
> valid, and **S&S Props. 1–4 and Cor. 1 hold verbatim** with p(Q) replaced by
> LMP_{ν(i)}(q) and Lemma 2 applied to the sufficient statistic δᵀΣ⁻¹Y. Balanced
> transfers still cancel; bounded money burning still does not help.
>
> **(ii) Impossibility fails.** If some binding loop constraint separates a pair
> (ρ_{ij} < 1), the summing step is invalid, balanced continuation-value transfers
> satisfying every local IC exist, and the required incentive volatility
> 2a²/(s²(1−ρ)) is **finite** — so the mechanism that drives collusion out as
> Δ → 0 is absent.
>
> **(iii) Quantitative.** The scope of collusion is governed by (1 − ρ), not by
> whether congestion exists at all. As ρ → 1 the required volatility diverges and
> collusion collapses again (S&S §VII).

**Do not assume the result fails — it demonstrably holds on a large part of the
state space, including this market's own first-best collusive point** (§5).

### What is *not* proved

1. **(ii) is a local, first-order statement.** It establishes that the S&S
   obstruction disappears and that pairwise identifiability is restored. Turning
   "finite incentive volatility" into "PPE payoffs bounded away from Nash"
   invokes the continuous-time characterisation of Sannikov (2007) plus the
   Δ → 0 convergence results, and holds for r small enough. **I have not computed
   the r-dependent payoff frontier, so I can say the obstruction is gone, not how
   much collusion is sustainable.**
2. **Local ICs suffice only under own-output concavity.** With congestion the
   profit function is kinked and can be locally non-concave, so a *large* jump
   deviation across a congestion boundary may be profitable even when every local
   IC holds. This cuts **against** the positive result and needs checking directly.
3. **Uniqueness of the static Nash** is a hypothesis, not a lemma (§2.3).
4. The analysis assumes only prices are public. Real ISOs publish generator-level
   output with a lag — which is S&S's own "delay of information" / "additional
   signals at fixed Δ_y" escape, and by their §VII that restores collusion
   independently of anything about networks.

---

## 4. The self-limiting tension, and why it resolves in the interesting direction

There is an obvious objection to the whole positive story: **collusion means
producing less, and producing less relieves congestion.** If the cartel's target
de-congests the network, then B = ∅, ρ = 1, and impossibility returns. On this
market that objection is *literally correct at the first-best point*: at joint
monopoly no line binds and all five LMPs equal $57.16.

The resolution is that **line flows are driven by the dispersion of net
injections, not by their total.** An asymmetric dispatch can saturate a loop line
while cutting total output. And asymmetric dispatch is exactly the class of
schemes — rotation, asymmetric quotas, market-sharing — that Prop. 2 exists to
rule out. So the mechanism is **self-enabling on precisely the schemes S&S needs
to kill**, and the cartel faces a genuine trade-off between the productive
efficiency of symmetric dispatch and the enforceability of asymmetric dispatch.
§5.3 measures that trade-off, and it is remarkably cheap.

---

## 5. Numerical verification — Liu–Hobbs 5-node market, `three_firm_dist`

Firms 0, 1, 2 at nodes 0, 1, 2. Lines 0:(0,1), 1:(1,2), 2:(2,0) form the loop;
3:(2,3) and 4:(3,4) are the radial tail to the node-4 load pocket.

### 5.1 Prop. N2 confirmed exactly

PTDF columns at the three firm nodes:

| line | Φ at nodes (0,1,2) | verdict |
|---|---|---|
| 0 (0–1) | (+0.333, −0.333, 0) | distinct → **separates** |
| 1 (1–2) | (+0.333, +0.667, 0) | distinct → **separates** |
| 2 (2–0) | (−0.667, −0.333, 0) | distinct → **separates** |
| 3 (2–3) | (1, 1, 1) | identical → **cannot separate** |
| 4 (3–4) | (1, 1, 1) | identical → **cannot separate** |

rank of the fingerprint differences equals rank of the PTDF-column differences at
every tested profile (0 = 0 with B={3}; 1 = 1 with B={0}; 2 = 2 with B={1,2}).

### 5.2 At both benchmarks, S&S holds

| | q (MW) | Π | binding | rank{d_i} | ρ | balanced-transfer cost |
|---|---|---|---|---|---|---|
| Joint monopoly | (71.7, 72.7, 69.0) | 8169 | **none** | 1 | 1.000000 | **∞ — S&S bites** |
| Static Nash (BR) | (100.0, 102.1, 104.5) | 6515 | {2–3} radial | 1 | 1.000000 | **∞ — S&S bites** |

At monopoly all five LMPs equal $57.16 and every d_i = −0.1673·**1**. At Nash the
load-pocket corridor is congested but, being radial with all three firms upstream,
it separates nothing: d_i = (−0.1723, −0.1723, −0.1723, 0, 0) for **all three
firms**. Along the whole Nash → monopoly segment, ρ = 1 at every point.

**On this market as currently configured, the impossibility result holds at the
cartel's own target.**

### 5.3 The enforceability–efficiency frontier

Maximising joint profit subject to 1 − ρ ≥ τ over a 25³ grid of profiles:

| τ (min separation) | max Π | % of monopoly | vs Nash | binding | q |
|---|---|---|---|---|---|
| 0 | 8167 | 99.98% | +25.4% | none | (67.5, 72.5, 71.0) |
| 0.05 | 8134 | 99.58% | +24.9% | {0–1} | (90.0, 54.4, 64.6) |
| 0.10 | 7966 | 97.51% | +22.3% | {0–1, 2–3} | (90.0, 60.4, 90.4) |
| 0.20 | 7965 | 97.51% | +22.3% | {1–2} | (73.1, 36.2, 103.3) |
| 0.99 | 7827 | 95.82% | +20.1% | {1–2, 2–0} | (11.2, 60.4, 129.2) |

**Buying ρ ≈ 0.94 costs 0.4% of cartel profit. Buying full orthogonality (ρ ≈ 0)
costs 4.2%, and still leaves +20.1% over Nash.** Statistical identification is
close to free on this topology, and the profiles that deliver it are exactly the
asymmetric ones (at τ = 0.99, firm 0 is nearly shut down and two loop lines bind,
cutting the mesh so that each firm becomes its own island: d₀ hits only node 0,
d₁ only node 1, d₂ only the tail).

### 5.4 A modelling problem that must be fixed first

The simulator's demand shock is a **single scalar** u added to every nodal
intercept, so ∂LMP/∂u = **1** exactly and the price-noise covariance has **rank 1**.
Consequence: the moment *any* line binds, a deviation moves the LMP vector off
span{**1**} and is **perfectly detectable** — the measured residual off span{**1**}
is 0.20–0.89 at every congested profile tested, versus 8×10⁻⁹ (i.e. zero) at the
uncongested monopoly point.

So as specified, the model has genuine imperfect monitoring **only in the
uncongested region — exactly the region where ρ = 1 and S&S bites** — and the
congested region where the theory predicts a break is trivially perfect-monitoring.
That is assumption N0 failing, and it would make any positive simulation result
an artifact.

**Fix:** independent per-node demand shocks (or LMP measurement/estimation noise)
so that Σ has full rank. Then ρ < 1 delivers genuine *partial* identification and
C = 2a²/(s²(1−ρ)) is the operative quantity.

---

## 6. Summary for the advisor

1. **The impossibility result is real and it is about Δ → 0, not continuous
   actions.** Lemma 2 is the engine; the summing step in Prop. 2 is the only place
   the network can touch.
2. **Congestion is necessary but nowhere near sufficient.** The exact condition
   (Prop. N2) is that binding lines have **different shift factors at the firms'
   nodes** — a purely topological criterion. Co-located firms and radial
   load-pocket corridors never separate anything.
3. **Two independent breaks exist, both requiring firms at electrically distinct
   nodes:** the Prop.-2 summing step (via ρ < 1), and Lemma 4's ε_π > 0 (via a
   saturated export path — the endogenous version of S&S §V.C).
4. **What is proved is a dichotomy, and the negative half applies to this market
   as currently built** — at both the Nash and the joint-monopoly benchmarks the
   fingerprints are rank 1 and S&S holds verbatim.
5. **The positive half is a local/first-order result.** It removes the obstruction
   and restores pairwise identifiability; it does not by itself quantify how much
   collusion survives. Two things could still overturn it: non-concavity of profit
   across congestion kinks (large deviations), and non-uniqueness of static Nash.
6. **Congestion does not create Poisson jumps** in a convex DC-OPF. The AMP route
   exists but needs LP dual degeneracy / unit commitment / scarcity pricing, not
   congestion.

### Next steps, in priority order

1. **Fix the shock structure** (§5.4) — independent nodal shocks. Nothing
   downstream is trustworthy until Σ has full rank.
2. **Check global ICs across congestion kinks** at the separating profiles: is the
   profit function concave enough in own output that local ICs suffice?
3. **Compute the r-dependent payoff frontier** at a separating profile
   (Sannikov 2007 ODE) to turn "obstruction removed" into a payoff bound.
4. **Verify static-Nash uniqueness** numerically on the redistributed market, since
   Lemma 1 is unavailable.
5. Only then re-run the learning experiments, with firms sited so that a **loop**
   constraint binds at the collusive profile.
