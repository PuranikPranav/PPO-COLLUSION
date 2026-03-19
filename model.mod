# SETS
set NODES;
set FIRMS;
set PLANTS within {FIRMS, NODES};

# PARAMETERS
param P0 {NODES};
param Q0 {NODES};
param MC {PLANTS};
param QC {PLANTS};
param CAP {PLANTS};

# VARIABLES
var g {PLANTS} >= 0;
var rho {PLANTS} >= 0;
var y {NODES};
var mu;

# EQUATIONS
subject to Stationarity {(f,i) in PLANTS}:
   g[f,i] >= 0 complements 
   - (P0[i] - (P0[i]/Q0[i]) * (sum{(ft,i) in PLANTS} g[ft,i] + g[f,i] + y[i])) 
   + (MC[f,i] + QC[f,i]*g[f,i]) + rho[f,i] >= 0;

subject to Capacity {(f,i) in PLANTS}:
   rho[f,i] >= 0 complements 
   CAP[f,i] - g[f,i] >= 0;

subject to ISO_Dispatch {i in NODES}:
   P0[i] - (P0[i]/Q0[i]) * (sum{(f,i) in PLANTS} g[f,i] + y[i]) - mu = 0;

subject to Energy_Balance:
   sum {i in NODES} y[i] = 0;