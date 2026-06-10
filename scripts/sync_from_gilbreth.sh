#!/usr/bin/env bash
# Pull training results and figures from Gilbreth into this repo.
# Requires interactive Purdue auth (BoilerKey) unless SSH keys are configured.
#
# Usage (from repo root or anywhere):
#   bash scripts/sync_from_gilbreth.sh
#   bash scripts/sync_from_gilbreth.sh --all   # also slurm logs from repo root

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${GILBRETH_REMOTE:-ppuranik@gilbreth.rcac.purdue.edu}"
REMOTE_DIR="${GILBRETH_REMOTE_DIR:-~/ppo-collusion}"

RSYNC=(rsync -avz --partial --progress)
# Skip partial transfer temp files
RSYNC+=(--exclude='.rsync-partial/')

SYNC_ALL=false
if [[ "${1:-}" == "--all" ]]; then
    SYNC_ALL=true
fi

echo "Syncing from ${REMOTE}:${REMOTE_DIR}/ → ${REPO_ROOT}/"

mkdir -p results figures latest_results

"${RSYNC[@]}" \
    "${REMOTE}:${REMOTE_DIR}/results/delta_cont/" \
    "${REPO_ROOT}/results/delta_cont/"

"${RSYNC[@]}" \
    "${REMOTE}:${REMOTE_DIR}/results/delta_cont/h1/" \
    "${REPO_ROOT}/latest_results/"

"${RSYNC[@]}" \
    "${REMOTE}:${REMOTE_DIR}/figures/delta_cont/" \
    "${REPO_ROOT}/figures/delta_cont/"

if $SYNC_ALL; then
    "${RSYNC[@]}" \
        --include='slurm-h1-delta-*.out' \
        --include='slurm-h1-delta-*.err' \
        --exclude='*' \
        "${REMOTE}:${REMOTE_DIR}/" \
        "${REPO_ROOT}/"
fi
echo "Done. Local paths:"
echo "  latest_results/              (primary H=1 run — sessions, config)"
echo "  latest_results/deviation_experiment/  (Calvano / impulse plots)"
echo "  results/delta_cont/h1/       (cluster mirror)"
echo "  figures/delta_cont/"

