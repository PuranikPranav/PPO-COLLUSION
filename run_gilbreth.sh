#!/bin/bash
#SBATCH --job-name=ppo-collusion
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# History length H is the first argument: sbatch run_gilbreth.sh 1
H=${1:?"Usage: sbatch run_gilbreth.sh <H>  (e.g. sbatch run_gilbreth.sh 1)"}

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

# ── Modules ──────────────────────────────────────────────────────────
# xalt may stay loaded after purge; harmless. Use "module --force purge" only if needed.
module purge
module load cuda 2>/dev/null || true
# Gilbreth: there is no generic "anaconda" module—names are versioned. Run once on a login node:
#   module spider anaconda
# If you want Conda instead of system python3, set before sbatch, e.g.:
#   export PYTHON_MODULE='anaconda/2024.06-py311'   # example; use spider output
if [ -n "${PYTHON_MODULE:-}" ]; then
    module load "$PYTHON_MODULE"
fi

# ── Virtual environment (first run creates it) ───────────────────────
ENV_DIR="$HOME/envs/ppo-collusion"
PY=python3
command -v "$PY" >/dev/null || PY=python
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment with $PY ..."
    "$PY" -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$ENV_DIR/bin/activate"
fi

# ── Experiment parameters ────────────────────────────────────────────
SESSIONS=50               # Calvano uses 1000; 50 is practical for PPO
TIMESTEPS=5000000          # Max steps per session — agents need ~3-5M to plateau
PATIENCE=50                # Converge if Δ stable for 50 consecutive PPO updates
CONV_THRESH=0.02           # Δ must change < 0.02 between updates to count as stable
EPISODE_LEN=168            # 1 week of hourly intervals

# ── Run experiment ───────────────────────────────────────────────────
echo "=========================================="
echo "  history_len = $H  |  $SESSIONS sessions"
echo "=========================================="
python experiments/ppo.py \
    --history-len "$H" \
    --num-sessions "$SESSIONS" \
    --total-timesteps "$TIMESTEPS" \
    --convergence-patience "$PATIENCE" \
    --convergence-threshold "$CONV_THRESH" \
    --episode-len "$EPISODE_LEN" \
    --rollout-len 2048 \
    --hidden-dim 64 \
    --lr 3e-4 \
    --seed 42 \
    --cuda \
    --output-dir "results/h${H}"

# ── Generate figure for this H ───────────────────────────────────────
python experiments/plot_results.py "results/h${H}" --save "figures/"

echo "Run complete. Results in results/h${H}, figures in figures/"
