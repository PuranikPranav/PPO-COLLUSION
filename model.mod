# Nash-Cournot + Exogenous ISO — stacked KKT as MCP/LCP (Liu & Hobbs 2012)
# Per-node supply:  sum_{ff: (ff,i) in PLANTS} g[ff,i]  (NOT sum{(ft,i) in PLANTS} — that shadows i)

set NODES;
set FIRMS;
set PLANTS within {FIRMS, NODES};
set LINKS;

param P0 {NODES};
param Q0 {NODES};
param MC {PLANTS};
param QC {PLANTS};
param CAP {PLANTS};
param PTDF {LINKS, NODES};
param T {LINKS};
param line_name {LINKS} symbolic;

var g {PLANTS} >= 0;
var rho {PLANTS} >= 0;
var y {NODES};
var mu;
var lambda_plus {LINKS} >= 0;
var lambda_minus {LINKS} >= 0;

subject to Stationarity {(f,i) in PLANTS}:
   g[f,i] >= 0 complements
   - (P0[i] - (P0[i]/Q0[i]) * (sum {ff in FIRMS: (ff,i) in PLANTS} g[ff,i] + y[i]))
   + (P0[i]/Q0[i])*g[f,i] + (MC[f,i] + QC[f,i]*g[f,i]) + rho[f,i] >= 0;

subject to Capacity {(f,i) in PLANTS}:
   rho[f,i] >= 0 complements
   CAP[f,i] - g[f,i] >= 0;

subject to ISO_Dispatch {i in NODES}:
   P0[i] - (P0[i]/Q0[i]) * (sum {ff in FIRMS: (ff,i) in PLANTS} g[ff,i] + y[i])
   - mu - sum {k in LINKS} PTDF[k,i]*(lambda_minus[k] - lambda_plus[k]) = 0;

subject to Energy_Balance:
   sum {i in NODES} y[i] = 0;

subject to Trans_Upper {k in LINKS}:
   lambda_plus[k] >= 0 complements
   T[k] + sum {i in NODES} PTDF[k,i]*y[i] >= 0;

subject to Trans_Lower {k in LINKS}:
   lambda_minus[k] >= 0 complements
   T[k] - sum {i in NODES} PTDF[k,i]*y[i] >= 0;
