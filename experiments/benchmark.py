import sys
import os

# Add the parent directory (PPO-COLLUSION) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.iso_solver import DCOPF
import numpy as np

def run_benchmarks():
    iso = DCOPF()
    
    # 1. Nash-Cournot Case (From Table 2)
    # Total Gen = 231.0 MW
    nash_gen = {
        'Firm1_Node1': 125,
        'Firm1_Node2': 45,
        'Firm2_Node2': 80
    }
    
    # 2. Collusion Case (From Table 2)
    # Total Gen = 191.0 MW (Significant withholding)
    collusion_gen = {
        'Firm1_Node1': 115,
        'Firm1_Node2': 21,
        'Firm2_Node2': 59
    }
    
    print(f"{'Metric':<25} | {'Nash-Cournot':<15} | {'Collusion':<15}")
    print("-" * 60)
    
    cases = [("Competitive", nash_gen), ("Collusive", collusion_gen)]
    results_list = []
    
    for name, gen in cases:
        res = iso.solve_market(gen)
        results_list.append(res)
    
    nash, coll = results_list
    
    # Validation Points
    print(f"{'Avg LMP ($/MWh)':<25} | {np.mean(nash['lmps']):>15.2f} | {np.mean(coll['lmps']):>15.2f}")
    print(f"{'Shadow Price Arc 2-3':<25} | {nash['shadow_price_23']:>15.2f} | {coll['shadow_price_23']:>15.2f}")
    print(f"{'Flow on Arc 2-3 (MW)':<25} | {nash['flows'][1]:>15.2f} | {coll['flows'][1]:>15.2f}")
    print(f"{'Total Production Cost':<25} | {nash['production_cost']:>15.2f} | {coll['production_cost']:>15.2f}")
    
    print("\n" + "="*60)
    if np.isclose(np.mean(coll['lmps']), 29.29, atol=0.1):
        print("✅ SUCCESS: Phase 1 Benchmarking matches Liu & Hobbs (2013).")
    else:
        print("❌ WARNING: Results differ from benchmark. Check PTDF or Demand parameters.")
    print("="*60)

if __name__ == "__main__":
    run_benchmarks()


#Bid Qty - build in some flexibility

