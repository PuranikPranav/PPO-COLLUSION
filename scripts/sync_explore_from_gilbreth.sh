#!/usr/bin/env bash
# Pull job 11116655 explore results + deviation figures from Gilbreth.
# Requires interactive BoilerKey auth (run this in your own terminal).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
REMOTE="${GILBRETH_REMOTE:-ppuranik@gilbreth.rcac.purdue.edu}"
REMOTE_DIR="${GILBRETH_REMOTE_DIR:-~/ppo-collusion}"
mkdir -p results/explore/h1 figures/explore
rsync -avz --progress \
  "${REMOTE}:${REMOTE_DIR}/results/explore/h1/" \
  "${REPO_ROOT}/results/explore/h1/"
rsync -avz --progress \
  "${REMOTE}:${REMOTE_DIR}/figures/explore/" \
  "${REPO_ROOT}/figures/explore/"
echo "Synced. Replot deviation with:"
echo "  python experiments/plot_results.py results/explore/h1 --save figures/explore --deviation-explainer"
