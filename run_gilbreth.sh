#!/bin/bash
# Run from the repository root (so experiments/ and results/ resolve correctly):
#   cd /path/to/ppo-collusion && sbatch run_gilbreth.sh
# Optional: sbatch run_gilbreth.sh "1 2"   — default is H = 1 2 3
#
# GPU partition (default a30 — often shorter queue than a100-40gb). Override without editing:
#   sbatch --partition=a100-40gb run_gilbreth.sh
# Other Gilbreth partitions (if your account allows): a30, a100-40gb, gilbreth-nodes, …
#   module spider cuda   # match CUDA module to GPU generation if needed
#SBATCH --job-name=ppo-collusion
#SBATCH --account=liu334
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
# Long jobs: 1000 sessions × 2M steps each is weeks of GPU time in practice—raise if the scheduler allows.
#SBATCH --time=7-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# History lengths to run (override with first arg, e.g. sbatch run_gilbreth.sh "1 2 3")
H_LIST="${1:-1 2 3}"

set -euo pipefail
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

# ── Experiment parameters (Calvano-scale paper settings) ─────────────
SESSIONS=1000              # Average metrics over 1000 independent seeds (very long run)
TIMESTEPS=2000000          # Max env steps per session (Calvano-style x-axis cap at 2M)
PATIENCE=100000            # Stop if max KL < threshold for 100k consecutive PPO updates
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

    # Calvano-style Fig 1–2 only (greedy quantities + greedy Δ); full grid is heavy with 1000 sessions
    python experiments/plot_results.py "results/h${H}" --save "figures/" --calvano-paper
    echo "Finished H=$H. Results in results/h${H}"
done

# ── Cross-history comparison figure ──────────────────────────────────
# Only when this job runs multiple H values in one allocation. For parallel
#   sbatch run_gilbreth.sh "1" / "2" / "3", skip here; when all are done run:
#   python experiments/plot_results.py --compare results/h1 results/h2 results/h3 --save figures/
RUN_DIRS=()
for H in $H_LIST; do
    if [ -d "results/h${H}" ]; then
        RUN_DIRS+=("results/h${H}")
    fi
done
N_H=0
for _ in $H_LIST; do N_H=$((N_H + 1)); done
if [ "$N_H" -gt 1 ] && [ "${#RUN_DIRS[@]}" -gt 1 ]; then
    python experiments/plot_results.py --compare "${RUN_DIRS[@]}" --save "figures/"
    echo "Comparison figure saved to figures/"
fi

echo "All runs complete."
