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

var g {(f,i) in G} >= 0, <= CAP[f,i];  
var y {SPOKES};
var p_hub;
var w {SPOKES};
var lam_plus  {LINES} >= 0;
var lam_minus {LINES} >= 0;

maximize Cartel_Profit:
    sum {i in NODES} (p_hub + (if i == N then 0 else w[i])) * (sum {(f,bus) in G : bus == i} g[f,bus])
    - sum {(f,i) in G} (MC[f,i]*g[f,i] + 0.5 * QC[f,i] * (g[f,i]^2));

s.t. Market_Clearing {i in NODES}:
    P0[i] - (P0[i]/Q0[i]) * ( (sum {(m,bus) in G : bus == i} g[m,bus]) + (if i == N then -sum{j in SPOKES} y[j] else y[i]) ) 
    - (p_hub + (if i == N then 0 else w[i])) = 0;

s.t. Wheeling_Pricing {i in SPOKES}:
    w[i] - sum {l in LINES} PTDF[l,i] * (lam_plus[l] - lam_minus[l]) = 0;

s.t. Line_Limit_Plus {l in LINES}:
    T[l] - sum {i in SPOKES} PTDF[l,i] * y[i] >= 0 complements lam_plus[l] >= 0;

s.t. Line_Limit_Minus {l in LINES}:
    T[l] + sum {i in SPOKES} PTDF[l,i] * y[i] >= 0 complements lam_minus[l] >= 0;

# Post-solve LMP / profits: computed in .run (params cannot use var expressions here).