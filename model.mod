# Nash-Cournot with Transmission Constraints (Liu & Hobbs 2012)
# Stacked KKT / Linear Complementarity Problem (Exogenous-ISO model)

# SETS
set NODES;
set FIRMS;
set PLANTS within {FIRMS, NODES};
set LINKS;

# PARAMETERS
param P0 {NODES};
param Q0 {NODES};
param MC {PLANTS};
param QC {PLANTS};
param CAP {PLANTS};
param PTDF {LINKS, NODES};
param T {LINKS};

# VARIABLES
var g {PLANTS} >= 0;
var rho {PLANTS} >= 0;
var y {NODES};
var mu;
var lambda_plus {LINKS} >= 0;
var lambda_minus {LINKS} >= 0;

# Firm stationarity (KKT): g[f,i] >= 0 complements -MR + MC + QC*g + rho >= 0
# Cournot: MR = p_i - (P0_i/Q0_i)*g_fi, where p_i = P0[i] - (P0[i]/Q0[i])*(G_i + y_i)
subject to Stationarity {(f,i) in PLANTS}:
   g[f,i] >= 0 complements 
   - (P0[i] - (P0[i]/Q0[i]) * (sum{(ft,i) in PLANTS} g[ft,i] + y[i])) 
   + (P0[i]/Q0[i])*g[f,i] + (MC[f,i] + QC[f,i]*g[f,i]) + rho[f,i] >= 0;

# Firm capacity: rho complements (CAP - g) >= 0
subject to Capacity {(f,i) in PLANTS}:
   rho[f,i] >= 0 complements 
   CAP[f,i] - g[f,i] >= 0;

# ISO stationarity: price_i = mu + congestion_rent_i
# P0[i] - (P0[i]/Q0[i])*(G[i]+y[i]) = mu + sum_k PTDF[k,i]*(lambda_plus[k] - lambda_minus[k])
subject to ISO_Dispatch {i in NODES}:
   P0[i] - (P0[i]/Q0[i]) * (sum{(f,i) in PLANTS} g[f,i] + y[i]) 
   - mu - sum{k in LINKS} PTDF[k,i]*(lambda_plus[k] - lambda_minus[k]) = 0;

# Energy balance: sum of ISO dispatch = 0
subject to Energy_Balance:
   sum {i in NODES} y[i] = 0;

# Transmission: flow_k = -sum_i PTDF[k,i]*y[i] (injection = -y in hub-spoke)
# Upper bound flow_k <= T[k]: lambda_plus complements (T[k] + sum_i PTDF[k,i]*y[i]) >= 0
subject to Trans_Upper {k in LINKS}:
   lambda_plus[k] >= 0 complements 
   T[k] + sum{i in NODES} PTDF[k,i]*y[i] >= 0;

# Lower bound flow_k >= -T[k]: lambda_minus complements (T[k] - sum_i PTDF[k,i]*y[i]) >= 0
subject to Trans_Lower {k in LINKS}:
   lambda_minus[k] >= 0 complements 
   T[k] - sum{i in NODES} PTDF[k,i]*y[i] >= 0;
