#!/usr/bin/env bash
# Cross-history plots after training. Run from repo root.
#
# Primary local H=1 run (renamed from h1/):
#   ./compare_histories.sh latest   → latest_results only (Calvano + dashboard need h2/h3 for cross-H)
#
# Cluster layout from run_gilbreth.sh (delta_cont / kl):
#   ./compare_histories.sh delta_cont → results/delta_cont/h{1,2,3} → figures/delta_cont/
#   ./compare_histories.sh kl         → results/kl/h{1,2,3}       → figures/kl/
#   ./compare_histories.sh both       → runs delta_cont then kl
#
# Legacy flat layout (results/h1, h2, h3):
#   ./compare_histories.sh legacy
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

CRIT="${1:-latest}"

run_single() {
  local out="$1"
  mkdir -p "$out"
  python experiments/plot_results.py latest_results --save "$out" --calvano-paper
  echo "Single-run Calvano figures → ${out}/"
}

run_compare() {
  local _label="$1"
  local r1="$2"
  local r2="$3"
  local r3="$4"
  local out="$5"
  mkdir -p "$out"
  python experiments/plot_results.py \
    --compare-calvano "$r1" "$r2" "$r3" \
    --save "$out"
  python experiments/plot_results.py \
    --compare "$r1" "$r2" "$r3" \
    --save "$out"
  echo "Calvano cross-H → ${out}/calvano_compare_quantities_*.png , calvano_compare_profit_*.png"
  echo "6-panel dashboard → ${out}/comparison_*.png"
}

case "$CRIT" in
  latest)
    run_single "latest_results/figures"
    ;;
  both)
    "$0" delta_cont
    "$0" kl
    ;;
  delta_cont|delta)
    run_compare "$CRIT" \
      "results/delta_cont/h1" \
      "results/delta_cont/h2" \
      "results/delta_cont/h3" \
      "figures/delta_cont"
    ;;
  kl)
    run_compare "$CRIT" \
      "results/kl/h1" \
      "results/kl/h2" \
      "results/kl/h3" \
      "figures/kl"
    ;;
  legacy)
    run_compare "legacy" \
      "results/h1" \
      "results/h2" \
      "results/h3" \
      "figures"
    ;;
  *)
    echo "Usage: $0 [latest|legacy|delta_cont|delta|kl|both]" >&2
    exit 1
    ;;
esac
