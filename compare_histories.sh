#!/usr/bin/env bash
# Run from repo root after results/h1, h2, h3 exist (e.g. all three parallel Slurm jobs finished).
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"
if [ ! -f experiments/plot_results.py ]; then
  echo "Run from ppo-collusion repo root." >&2
  exit 1
fi
if [ -d "$HOME/envs/ppo-collusion" ]; then
  # shellcheck source=/dev/null
  source "$HOME/envs/ppo-collusion/bin/activate"
fi
mkdir -p figures
python experiments/plot_results.py \
  --compare-calvano results/h1 results/h2 results/h3 \
  --save figures/
python experiments/plot_results.py \
  --compare results/h1 results/h2 results/h3 \
  --save figures/
echo "Calvano-style cross-H: figures/calvano_compare_quantities_h1_2_3.png"
echo "                     : figures/calvano_compare_profit_h1_2_3.png"
echo "6-panel dashboard    : figures/comparison_h1_2_3.png"
