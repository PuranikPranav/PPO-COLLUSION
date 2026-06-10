#!/bin/bash
# H=1 only, combined-Δ convergence (without-episodes / continuing task).
#
# On Gilbreth login node:
#   cd ~/ppo-collusion
#   git fetch origin && git checkout without-episodes && git pull
#   sbatch run_gilbreth_h1_delta.sh
#
# Outputs (cluster paths; rsync maps h1 → latest_results/, figures → latest_results/figures/):
#   results/delta_cont/h1/
#   figures/delta_cont/
#
#SBATCH --job-name=ppo-h1-delta
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-h1-delta-%j.out
#SBATCH --error=slurm-h1-delta-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

module purge
module load cuda 2>/dev/null || true
if [ -n "${PYTHON_MODULE:-}" ]; then
    module load "$PYTHON_MODULE"
fi

ENV_DIR="$HOME/envs/ppo-collusion"
PY=python3
command -v "$PY" >/dev/null || PY=python
if [ ! -d "$ENV_DIR" ]; then
    "$PY" -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$ENV_DIR/bin/activate"
fi

H=1
MODE="delta"
RESULTS_ROOT="results/delta_cont"
FIGURES_ROOT="figures/delta_cont"
SESSIONS=100
TIMESTEPS=2000000
PATIENCE=100
DELTA_CONV_THRESH=0.01
EPISODE_LEN=168
LOG_FORMAT=structured

mkdir -p "$FIGURES_ROOT"

echo "####################################################################"
echo "#  history_len=${H}  (15 obs features)  convergence=${MODE}"
echo "#  continuing task (without-episodes)"
echo "#  results → ${RESULTS_ROOT}/h${H}"
echo "####################################################################"

python experiments/ppo.py \
    --history-len "$H" \
    --num-sessions "$SESSIONS" \
    --total-timesteps "$TIMESTEPS" \
    --convergence-mode "$MODE" \
    --convergence-patience "$PATIENCE" \
    --delta-convergence-threshold "$DELTA_CONV_THRESH" \
    --log-format "$LOG_FORMAT" \
    --episode-len "$EPISODE_LEN" \
    --rollout-len 2048 \
    --hidden-dim 64 \
    --lr 3e-4 \
    --seed 42 \
    --cuda \
    --output-dir "${RESULTS_ROOT}/h${H}"

python experiments/plot_results.py "${RESULTS_ROOT}/h${H}" \
    --save "$FIGURES_ROOT" --calvano-paper

python experiments/plot_results.py --compare-delta "${RESULTS_ROOT}/h${H}" \
    --save "$FIGURES_ROOT"

echo "Done → ${RESULTS_ROOT}/h${H}"
