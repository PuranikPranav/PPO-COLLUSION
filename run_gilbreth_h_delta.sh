#!/bin/bash
# Single history-length run: combined-Δ convergence (without-episodes / continuing task).
#
# Submit from repo root (one GPU per job):
#   sbatch -J ppo-h3-delta -o slurm-h3-delta-%j.out -e slurm-h3-delta-%j.err run_gilbreth_h_delta.sh 3
#   sbatch -J ppo-h5-delta -o slurm-h5-delta-%j.out -e slurm-h5-delta-%j.err run_gilbreth_h_delta.sh 5
#
# Or use submit_gilbreth_h3_h5.sh to launch H=3 and H=5 in parallel on two GPUs.
#
# Cluster outputs:
#   results/delta_cont/h<H>/
#   figures/delta_cont/   (per-H Calvano PNGs; compare plots need multiple H dirs)
#
# Obs dim = history_len × 15 (5 nodal LMPs + 10 line shadow/flow features per step).
#
#SBATCH --job-name=ppo-h-delta
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-h-delta-%j.out
#SBATCH --error=slurm-h-delta-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

H="${1:-1}"
OBS_PER_STEP=15
OBS_DIM=$((H * OBS_PER_STEP))

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
echo "#  history_len=${H}  (obs_dim=${OBS_DIM} = ${H} × ${OBS_PER_STEP})"
echo "#  convergence=${MODE}  continuing task (without-episodes)"
echo "#  GPU: ${CUDA_VISIBLE_DEVICES:-(Slurm-assigned)}  job=${SLURM_JOB_ID:-local}"
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
