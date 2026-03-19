import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.solve_equilibrium import EquilibriumSolver
import numpy as np

def run_benchmarks():
    solver = EquilibriumSolver()
    results = solver.solve_competitive()

    print("\n" + "="*60)
    print(f"  Avg LMP ($/MWh):       {np.mean(results['lmps']):.2f}")
    print(f"  Shadow Price Arc 2-3:  {results['shadow_price_23']:.2f}")
    print(f"  Flow on Arc 2-3 (MW):  {results['flows'][1]:.2f}")
    print("="*60)

if __name__ == "__main__":
    run_benchmarks()
