# Results — `MARKET_CONFIG=three_firm_dist`

Everything below is generated from the saved run artefacts by
`python -m qlearning_collusion.summarize`.

## 1. The market

| node | P⁰ ($/MWh) | Q⁰ (MW) | slope ($/MWh per MW) | elasticity at $45 |
|---|---|---|---|---|
| 1 | 85 | 140 | 0.607 | -1.1250 |
| 2 | 95 | 300 | 0.317 | -0.9000 |
| 3 | 90 | 90 | 1.000 | -1.0000 |
| 4 | 88 | 14 | 6.286 | -1.0465 |
| 5 | 900 | 11 | 81.818 | -0.0526 |

| firm | plant node | MC | QC | cap (MW) |
|---|---|---|---|---|
| 0 | 1 | 15.0 | 0.090 | 135 |
| 1 | 2 | 16.0 | 0.075 | 145 |
| 2 | 3 | 18.0 | 0.050 | 155 |

Line limits (MW): **1-2** 36, **2-3** 45, **3-1** 38, **3-4** 16, **4-5** 12

## 2. Benchmarks

| outcome | total gen (MW) | avg LMP ($/MWh) | total profit ($/period) |
|---|---|---|---|
| perfect competition | 395.4 | 27.03 | 1831 |
| Nash-Cournot — LCP (paper eqs. 39-45) | 207.6 | 58.13 | 8089 |
| joint monopoly | 213.4 | 57.16 | 8169 |
| **Nash-Cournot — best response (Δ denominator)** | 306.6 | 41.83 | 6515 |
| **joint monopoly on the action grid (Δ = 1)** | 213.4 | 57.16 | 8169 |

Nash → monopoly profit gap **$1,654 (25.4%)**; per-firm cartel IR margins [617, 609, 428] (all positive).

Grid action indices: Nash at `[12, 12, 12]`, monopoly at `[2, 2, 2]` (paper: 12 and 2).

## 3. What the algorithms learn

| cell | Δ | s.e. | converged | total gen (MW) | ref-node LMP | avg LMP | \|S\| |
|---|---|---|---|---|---|---|---|
| imperfect (price only) · stochastic demand | **68.37%** | 0.28 | 100% | 261.5 | $49.10 | $49.24 | 51 |
| imperfect (price only) · deterministic demand | **83.57%** | 0.24 | 100% | 247.8 | $51.38 | $51.46 | 43 |
| perfect (full profile) · stochastic demand | **46.46%** | 0.41 | 6% | 277.0 | $46.43 | $46.69 | k^n |
| perfect (full profile) · deterministic demand | **79.06%** | 0.26 | 96% | 253.3 | $50.43 | $50.56 | k^n |
| **rich 19-variable state** · stochastic demand | **67.93%** | 0.26 | 100% | 261.3 | $49.17 | $49.27 | 192 |
| **rich 19-variable state** · deterministic demand | **80.68%** | 0.26 | 100% | 251.1 | $50.80 | $50.91 | 116 |

(Nash → monopoly reference: generation 306.6 → 213.4 MW, reference-node LMP $41.24 → $57.16.)

### Per-firm profit gain

| cell | firm 0 | firm 1 | firm 2 |
|---|---|---|---|
| imperfect (price only) · stochastic demand | 54.7% | 59.4% | 101.0% |
| imperfect (price only) · deterministic demand | 69.6% | 72.9% | 118.8% |
| perfect (full profile) · stochastic demand | 32.1% | 40.0% | 76.3% |
| perfect (full profile) · deterministic demand | 66.6% | 74.6% | 103.4% |
| **rich 19-variable state** · stochastic demand | 55.8% | 56.2% | 102.0% |
| **rich 19-variable state** · deterministic demand | 66.7% | 70.0% | 115.9% |

### How much the state reveals

Fraction of (own action, observed state) pairs consistent with more than one rival profile — 1.0 means the state says nothing about rivals, 0.0 means it identifies them exactly.

| state | \|S\| | non-revealing fraction |
|---|---|---|
| price only | 51 | 0.962 |
| rich 19-variable [price, pocket_lmp, congestion, total_demand] | 192 | 0.801 |
| full profile (perfect) | 15^3 | 0.000 |

## 4. Table I

```

TABLE I - the impact of imperfect monitoring (profit gain Delta)
--------------------------------------------------------------
                      Deterministic Demand   Stochastic Demand
Perfect Monitoring                  79.06%              46.46%
Imperfect Monitoring                83.57%              68.37%
--------------------------------------------------------------
Rich 19-var state                   80.68%              67.93%
--------------------------------------------------------------
(rich = the PPO public observation — nodal LMPs, congestion /
 shadow prices, realised demand — instead of one binned price;
 still imperfect monitoring: rivals' outputs are NOT in the state)
difference-in-differences (pure imperfect-monitoring effect): +17.41 pp
(paper baseline: -8.91 pp; 76.25 / 89.60 / 79.72 / 84.16)

CAUTION - these cells did NOT converge for most sessions, so their
Delta is a snapshot at the iteration cap, not a limit strategy:
    perfect_stochastic           5.5% of sessions converged
Any difference-in-differences involving them inherits that caveat.
```

## 5. Transmission congestion

```
================================================================================================
TRANSMISSION CONGESTION ACROSS MARKET OUTCOMES  (market: three_firm_dist)
================================================================================================
line limits (MW): 1-2=36.0  2-3=45.0  3-1=38.0  3-4=16.0  4-5=12.0

outcome                                      gen   avgLMP    profit   binding lines (rent $/MWh)
------------------------------------------------------------------------------------------------
perfect competition                        395.4    27.03    1831.0   2-3 (-2.24), 3-4 (+27.62)
Nash-Cournot (LCP, paper eqs. 39-45)       207.6    58.13    8089.3   NONE
Nash-Cournot (iterated best response)      306.6    41.83    6514.8   3-4 (+11.30)
joint monopoly                             213.4    57.16    8168.6   NONE
Nash on the action grid (Delta = 0)        306.6    41.83    6514.8   3-4 (+11.30)
monopoly on the action grid (Delta = 1)    213.4    57.16    8168.6   NONE
LEARNED COLLUSION (imperfect_stochastic)   261.5    49.25             1-2 (9% of periods, mean rent +0.38), 3-4 (88% of periods, mean rent +3.78)
------------------------------------------------------------------------------------------------

Per-line detail (flow / limit, utilisation, congestion rent):
  perfect competition
       1-2  flow   +32.85 /  36.0 MW  ( 91.3%)   rent   +0.000   slack
       2-3  flow   -45.00 /  45.0 MW  (100.0%)   rent   -2.243   BINDING
       3-1  flow   +12.15 /  38.0 MW  ( 32.0%)   rent   +0.000   slack
       3-4  flow   +16.00 /  16.0 MW  (100.0%)   rent  +27.623   BINDING
       4-5  flow   +10.36 /  12.0 MW  ( 86.3%)   rent   +0.000   slack
  Nash-Cournot (LCP, paper eqs. 39-45)
       1-2  flow    +8.83 /  36.0 MW  ( 24.5%)   rent   +0.000   slack
       2-3  flow    -0.06 /  45.0 MW  (  0.1%)   rent   -0.000   slack
       3-1  flow    -8.77 /  38.0 MW  ( 23.1%)   rent   -0.000   slack
       3-4  flow   +15.04 /  16.0 MW  ( 94.0%)   rent   +0.000   slack
       4-5  flow   +10.29 /  12.0 MW  ( 85.7%)   rent   +0.000   slack
  Nash-Cournot (iterated best response)
       1-2  flow   +31.89 /  36.0 MW  ( 88.6%)   rent   +0.000   slack
       2-3  flow   -35.83 /  45.0 MW  ( 79.6%)   rent   -0.000   slack
       3-1  flow    +3.94 /  38.0 MW  ( 10.4%)   rent   +0.000   slack
       3-4  flow   +16.00 /  16.0 MW  (100.0%)   rent  +11.300   BINDING
       4-5  flow   +10.36 /  12.0 MW  ( 86.3%)   rent   +0.000   slack
  joint monopoly
       1-2  flow   +24.21 /  36.0 MW  ( 67.3%)   rent   +0.000   slack
       2-3  flow   -22.61 /  45.0 MW  ( 50.2%)   rent   -0.000   slack
       3-1  flow    -1.61 /  38.0 MW  (  4.2%)   rent   -0.000   slack
       3-4  flow   +15.21 /  16.0 MW  ( 95.0%)   rent   +0.000   slack
       4-5  flow   +10.30 /  12.0 MW  ( 85.8%)   rent   +0.000   slack
  Nash on the action grid (Delta = 0)
       1-2  flow   +31.89 /  36.0 MW  ( 88.6%)   rent   +0.000   slack
       2-3  flow   -35.83 /  45.0 MW  ( 79.6%)   rent   -0.000   slack
       3-1  flow    +3.94 /  38.0 MW  ( 10.4%)   rent   +0.000   slack
       3-4  flow   +16.00 /  16.0 MW  (100.0%)   rent  +11.300   BINDING
       4-5  flow   +10.36 /  12.0 MW  ( 86.3%)   rent   +0.000   slack
  monopoly on the action grid (Delta = 1)
       1-2  flow   +24.21 /  36.0 MW  ( 67.3%)   rent   +0.000   slack
       2-3  flow   -22.60 /  45.0 MW  ( 50.2%)   rent   -0.000   slack
       3-1  flow    -1.62 /  38.0 MW  (  4.3%)   rent   -0.000   slack
       3-4  flow   +15.21 /  16.0 MW  ( 95.0%)   rent   +0.000   slack
       4-5  flow   +10.30 /  12.0 MW  ( 85.8%)   rent   +0.000   slack
  LEARNED COLLUSION (imperfect_stochastic, 1000 converged sessions)
       1-2  mean flow   +27.90 /  36.0 MW  ( 77.5%)   mean rent   +0.382   binds   8.6% of periods
       2-3  mean flow   -30.97 /  45.0 MW  ( 68.8%)   mean rent   -0.000   binds   0.0% of periods
       3-1  mean flow    +3.07 /  38.0 MW  (  8.1%)   mean rent   +0.000   binds   0.0% of periods
       3-4  mean flow   +15.96 /  16.0 MW  ( 99.8%)   mean rent   +3.775   binds  87.9% of periods
       4-5  mean flow   +10.36 /  12.0 MW  ( 86.3%)   mean rent   +0.000   binds   0.0% of periods

Nodal LMPs ($/MWh) and demand (MW):
  perfect competition                      LMP=[25.66, 26.41, 24.91, 52.54, 52.54]
                                           d  =[97.74, 216.61, 65.09, 5.64, 10.36]
  Nash-Cournot (LCP, paper eqs. 39-45)     LMP=[58.13, 58.13, 58.13, 58.13, 58.13]
                                           d  =[44.26, 116.45, 31.87, 4.75, 10.29]
  Nash-Cournot (iterated best response)    LMP=[41.24, 41.24, 41.24, 52.54, 52.54]
                                           d  =[72.08, 169.78, 48.76, 5.64, 10.36]
  joint monopoly                           LMP=[57.16, 57.16, 57.16, 57.16, 57.16]
                                           d  =[45.86, 119.5, 32.84, 4.91, 10.3]
  Nash on the action grid (Delta = 0)      LMP=[41.24, 41.24, 41.24, 52.54, 52.54]
                                           d  =[72.08, 169.78, 48.76, 5.64, 10.36]
  monopoly on the action grid (Delta = 1)  LMP=[57.16, 57.16, 57.16, 57.16, 57.16]
                                           d  =[45.86, 119.5, 32.84, 4.91, 10.3]
  LEARNED COLLUSION                        LMP=[48.85, 49.11, 48.98, 52.75, 52.75]
                                           d  =[59.54, 144.93, 41.02, 5.61, 10.36]
================================================================================================
```

## 6. Discount-factor sweep

```
DISCOUNT-FACTOR SWEEP  (cell: imperfect_stochastic, deviator 0, high demand)
--------------------------------------------------------------------------------------------------------
  delta   Delta %   dev MW@t0  rival t+1  peak rival  periods>10%  area MW.per  cartel survives
--------------------------------------------------------------------------------------------------------
   0.99     79.36       17.41       6.18        6.18            3        12.30              83%
   0.95     68.32       14.91       4.19        4.19            3         6.83              87%
    0.9     63.62       13.29       2.57        2.57            2         3.61              88%
    0.8     62.02       12.28       1.84        1.84            1         1.74              88%
    0.7     62.81       11.77       1.03        1.03            1        -0.59              86%
    0.5     64.93       11.59       0.76        0.76            1        -0.22              85%
--------------------------------------------------------------------------------------------------------
dev MW@t0   = the forced deviator's output above its no-deviation twin
rival t+1   = the non-deviating firms' expansion one period later (the punishment)
area        = total rival expansion summed over the post-deviation window
```

