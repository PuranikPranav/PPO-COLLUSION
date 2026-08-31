# nc.mod  —  Nash-Cournot wheeling-fee equilibrium as a square LCP/MCP.
# Solver: PATH.   Data: d.dat.
#
# Reduced complementarity system (reference hub = node N, w_N = 0, balance
# dual = 0, y_N = -sum_{spokes} y).  Each firm receives the nodal price
# (p_hub + w_i); the ISO sets w_i through its wheeling auction.
# Firm FOCs use the FIRM-LEVEL portfolio conjecture of paper eq. (31):
# multi-plant firms internalize the hub-price impact across all their plants.

param N > 0 integer;
set NODES  := 1..N;
set SPOKES := 1..N-1;                 # non-reference nodes (hub = node N)

set G within {1..20, NODES};          # (firm, node) generators; firms numeric
set FIRMS := setof {(f,i) in G} f;

param CAP {G} >= 0;
param MC  {G} >= 0;
param QC  {G} >= 0;

param P0 {NODES} > 0;
param Q0 {NODES} > 0;

param L > 0 integer;
set LINES := 1..L;
param T {LINES} >= 0;
param PTDF {LINES, SPOKES};

var g    {(f,i) in G} >= 0;           # generation
var rho  {(f,i) in G} >= 0;           # capacity multiplier
var y    {SPOKES};                    # ISO net injection  (y_N = -sum y)
var p_hub;                            # hub (energy) price
var w    {SPOKES};                    # wheeling fee        (w_N = 0)
var lam_plus  {LINES} >= 0;           # thermal dual, upper limit
var lam_minus {LINES} >= 0;           # thermal dual, lower limit

# Firm stationarity (paper eq. (31)): firm f maximizes
#   sum_i (p_hub + w_i) g_{f,i} - (MC g_{f,i} + 1/2 QC g_{f,i}^2)
# with the price set by the inverse-demand / market-clearing identity
#   p_hub + w_i = P0_i - (P0_i/Q0_i)(g_{f,i} + G_{-f,i} + y_i),
# taking w, y and rivals' output G_{-f} as given. Substituting the identity and
# differentiating f's Lagrangian w.r.t. g_{f,i} yields
#   0 <= g_{f,i}  _|_  -(p_hub+w_i) + (P0_i/Q0_i) g_{f,i}
#                      + MC + QC g_{f,i} + rho_{f,i} >= 0
s.t. Firm_FOC {(f,i) in G}:
    -(p_hub + (if i == N then 0 else w[i]))
    + (P0[i]/Q0[i])*g[f,i] + MC[f,i] + QC[f,i]*g[f,i] + rho[f,i] >= 0
    complements g[f,i] >= 0;

# Capacity:  0 <= rho  _|_  CAP - g >= 0
s.t. Firm_Cap {(f,i) in G}:
    CAP[f,i] - g[f,i] >= 0
    complements rho[f,i] >= 0;

# Market clearing (inverse demand):  P0 - (P0/Q0)(G_i + y_i) = p_hub + w_i
s.t. Market_Clearing {i in NODES}:
    P0[i] - (P0[i]/Q0[i]) * (
        (sum {(m,bus) in G : bus == i} g[m,bus])
        + (if i == N then -sum {j in SPOKES} y[j] else y[i])
    ) - (p_hub + (if i == N then 0 else w[i])) = 0;

# ISO stationarity (wheeling pricing):  w_i = sum_l PTDF[l,i] (lam+ - lam-)
s.t. Wheeling_Pricing {i in SPOKES}:
    w[i] - sum {l in LINES} PTDF[l,i] * (lam_plus[l] - lam_minus[l]) = 0;

# Thermal limits:  0 <= lam+ _|_ T - flow >= 0 ;  0 <= lam- _|_ T + flow >= 0
s.t. Line_Limit_Plus {l in LINES}:
    T[l] - sum {i in SPOKES} PTDF[l,i] * y[i] >= 0
    complements lam_plus[l] >= 0;

s.t. Line_Limit_Minus {l in LINES}:
    T[l] + sum {i in SPOKES} PTDF[l,i] * y[i] >= 0
    complements lam_minus[l] >= 0;
