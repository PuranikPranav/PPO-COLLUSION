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

# History lengths to run (override with first arg, e.g. sbatch run_gilbreth.sh "1 2 3")
H_LIST="${1:-1 2 3}"

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
PATIENCE=100               # Converge if KL < threshold for 100 consecutive PPO updates
KL_THRESH=0.01             # KL divergence threshold for convergence
EPISODE_LEN=168            # 1 week of hourly intervals

# ── Run experiments for each history length ──────────────────────────
for H in $H_LIST; do
    echo "=========================================="
    echo "  history_len = $H  |  $SESSIONS sessions"
    echo "=========================================="
    python experiments/ppo.py \
        --history-len "$H" \
        --num-sessions "$SESSIONS" \
        --total-timesteps "$TIMESTEPS" \
        --convergence-patience "$PATIENCE" \
        --kl-threshold "$KL_THRESH" \
        --episode-len "$EPISODE_LEN" \
        --rollout-len 2048 \
        --hidden-dim 64 \
        --lr 3e-4 \
        --seed 42 \
        --cuda \
        --output-dir "results/h${H}"

    # Per-H figure
    python experiments/plot_results.py "results/h${H}" --save "figures/"
    echo "Finished H=$H. Results in results/h${H}"
done

# ── Cross-history comparison figure ──────────────────────────────────
RUN_DIRS=""
for H in $H_LIST; do
    if [ -d "results/h${H}" ]; then
        RUN_DIRS="$RUN_DIRS results/h${H}"
    fi
done
if [ -n "$RUN_DIRS" ]; then
    python experiments/plot_results.py --compare $RUN_DIRS --save "figures/"
    echo "Comparison figure saved to figures/"
fi

echo "All runs complete."
