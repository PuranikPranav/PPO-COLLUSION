#!/usr/bin/env bash
# Convenience wrapper: submit H=1 continuing-task run (same as before).
#
# On Gilbreth login node:
#   cd ~/ppo-collusion
#   git fetch origin && git checkout without-episodes && git pull
#   bash run_gilbreth_h1_delta.sh
#
# Or: sbatch -J ppo-h1-delta -o slurm-h1-delta-%j.out -e slurm-h1-delta-%j.err run_gilbreth_h_delta.sh 1
set -euo pipefail
cd "$(dirname "$0")"

exec sbatch \
  --job-name=ppo-h1-delta \
  --account=liu334 \
  --partition=a100-40gb \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=50G \
  --time=4-00:00:00 \
  --output=slurm-h1-delta-%j.out \
  --error=slurm-h1-delta-%j.err \
  run_gilbreth_h_delta.sh 1
