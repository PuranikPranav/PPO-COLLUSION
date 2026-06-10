#!/usr/bin/env bash
# Cross-history plots after training. Run from repo root.
#
# Primary local H=1 continuing-task run:
#   ./compare_histories.sh            → latest_results → latest_results/figures/
#   ./compare_histories.sh latest     → same
#
# Archived episodic cross-H runs (old_results/delta_crosshistory/):
#   ./compare_histories.sh crosshistory → h1,h2,h3 → old_results/figures/crosshistory/
#
# Optional: after rsync from Gilbreth recreates results/delta_cont/ locally:
#   ./compare_histories.sh delta_cont → results/delta_cont/h{1,2,3} → figures/delta_cont/
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

MODE="${1:-latest}"

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

case "$MODE" in
  latest|"")
    run_single "latest_results/figures"
    ;;
  crosshistory|archive)
    run_compare "$MODE" \
      "old_results/delta_crosshistory/h1" \
      "old_results/delta_crosshistory/h2" \
      "old_results/delta_crosshistory/h3" \
      "old_results/figures/crosshistory"
    ;;
  delta_cont|delta)
    run_compare "$MODE" \
      "results/delta_cont/h1" \
      "results/delta_cont/h2" \
      "results/delta_cont/h3" \
      "figures/delta_cont"
    ;;
  *)
    echo "Usage: $0 [latest|crosshistory|delta_cont]" >&2
    exit 1
    ;;
esac
