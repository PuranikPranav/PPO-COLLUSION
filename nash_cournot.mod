# Nash-Cournot MCP (hub-spoke / Liu & Hobbs style)
# Matches iso_market/node_network.py parameters (P0=55/50/..., MC=22/25, etc.)

param N > 0 integer;
set NODES := 1..N;
set SPOKES := 1..N-1;

set G within {1..20, NODES};
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

var g {(f,i) in G} >= 0;
var rho {(f,i) in G} >= 0;
var y {SPOKES};
var p_hub;
var w {SPOKES};
var lam_plus  {LINES} >= 0;
var lam_minus {LINES} >= 0;

# Firm FOC per paper eq. (31): objective price is (p_hub + w_i), pinned by the
# identity p_hub + w_i = P0_i - (P0_i/Q0_i)(g_{f,i} + G_{-f,i} + y_i); w, y and
# G_{-f} exogenous. Substituting and differentiating gives markup (P0_i/Q0_i) g_{f,i}.
s.t. Firm_FOC {(f,i) in G}:
    -(p_hub + (if i == N then 0 else w[i]))
    + (P0[i]/Q0[i])*g[f,i] + MC[f,i] + QC[f,i]*g[f,i] + rho[f,i] >= 0
    complements g[f,i] >= 0;

s.t. Firm_Cap {(f,i) in G}:
    CAP[f,i] - g[f,i] >= 0 complements rho[f,i] >= 0;

s.t. Market_Clearing {i in NODES}:
    P0[i] - (P0[i]/Q0[i]) * (
        (sum {(m,bus) in G : bus == i} g[m,bus])
        + (if i == N then -sum{j in SPOKES} y[j] else y[i])
    ) - (p_hub + (if i == N then 0 else w[i])) = 0;

s.t. Wheeling_Pricing {i in SPOKES}:
    w[i] - sum {l in LINES} PTDF[l,i] * (lam_plus[l] - lam_minus[l]) = 0;

s.t. Line_Limit_Plus {l in LINES}:
    T[l] - sum {i in SPOKES} PTDF[l,i] * y[i] >= 0 complements lam_plus[l] >= 0;

s.t. Line_Limit_Minus {l in LINES}:
    T[l] + sum {i in SPOKES} PTDF[l,i] * y[i] >= 0 complements lam_minus[l] >= 0;

# Post-solve LMP / profits: computed in .run (params cannot use var expressions here).
