#!/usr/bin/env bash
# Submit H=3 and H=5 training in parallel (one GPU each on liu334 / a100-40gb).
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

submit_h() {
  local h="$1"
  sbatch \
    --job-name="ppo-h${h}-delta" \
    --account=liu334 \
    --partition=a100-40gb \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=50G \
    --time=4-00:00:00 \
    --output="slurm-h${h}-delta-%j.out" \
    --error="slurm-h${h}-delta-%j.err" \
    "$SCRIPT" "$h"
}

echo "Submitting two jobs (1 GPU each): history_len=3 (obs_dim=45), history_len=5 (obs_dim=75)"
J3=$(submit_h 3)
J5=$(submit_h 5)
echo "$J3"
echo "$J5"
echo "Check queue: squeue -u $USER"
