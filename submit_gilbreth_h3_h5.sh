#!/usr/bin/env bash
# Submit H=3 and H=5 training in parallel (one GPU each on liu334 / a100-40gb).
#
# Defaults for this launcher:
#   --delta-convergence-threshold 0.005  (stricter than H=1's 0.01)
#   --time 8 days
#
# On Gilbreth login node:
#   cd ~/ppo-collusion
#   git fetch origin && git checkout without-episodes && git pull
#   bash submit_gilbreth_h3_h5.sh
#
# Monitor:
#   squeue -u "$USER"
#   tail -f slurm-h3-delta-<jobid>.out
#   tail -f slurm-h5-delta-<jobid>.out
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT="run_gilbreth_h_delta.sh"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing $SCRIPT — run from repo root." >&2
  exit 1
fi

export DELTA_CONV_THRESH="${DELTA_CONV_THRESH:-0.005}"
SLURM_TIME="${SLURM_TIME:-8-00:00:00}"

submit_h() {
  local h="$1"
  sbatch \
    --job-name="ppo-h${h}-delta" \
    --account=liu334 \
    --partition=a100-40gb \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=50G \
    --time="$SLURM_TIME" \
    --output="slurm-h${h}-delta-%j.out" \
    --error="slurm-h${h}-delta-%j.err" \
    "$SCRIPT" "$h"
}

echo "Submitting H=3 and H=5 (1 GPU each)"
echo "  obs_dim: 45 (H=3), 75 (H=5)"
echo "  delta_convergence_threshold=${DELTA_CONV_THRESH}"
echo "  time_limit=${SLURM_TIME}"
J3=$(submit_h 3)
J5=$(submit_h 5)
echo "$J3"
echo "$J5"
echo "Check queue: squeue -u $USER"
