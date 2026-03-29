#!/usr/bin/env bash
# Cross-history plots after training. Run from repo root.
#
# Layout from run_gilbreth.sh (delta / kl):
#   ./compare_histories.sh delta    → results/delta/h{1,2,3} → figures/delta/
#   ./compare_histories.sh kl       → results/kl/h{1,2,3}    → figures/kl/
#   ./compare_histories.sh both     → runs delta then kl
#
# Legacy flat layout (results/h1, h2, h3):
#   ./compare_histories.sh
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

CRIT="${1:-legacy}"

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
  both)
    "$0" delta
    "$0" kl
    ;;
  delta|kl)
    run_compare "$CRIT" \
      "results/${CRIT}/h1" \
      "results/${CRIT}/h2" \
      "results/${CRIT}/h3" \
      "figures/${CRIT}"
    ;;
  legacy|"")
    run_compare "legacy" \
      "results/h1" \
      "results/h2" \
      "results/h3" \
      "figures"
    ;;
  *)
    echo "Usage: $0 [legacy|delta|kl|both]" >&2
    exit 1
    ;;
esac
