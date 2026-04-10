#!/bin/bash
# Run from the repository root:
#   cd ~/ppo-collusion && sbatch run_gilbreth.sh [H_LIST] [CRIT]
#
# Cancel your running / pending jobs first (Gilbreth login node):
#   squeue -u $USER
#   scancel -u $USER                    # all your jobs
#   scancel <JOBID>                     # one job
#
# Arguments:
#   $1  History lengths (default "1 2 3")
#   $2  Criterion: delta | kl | both (default both)
#       - delta: Δ-stability early stop, logs --log-format structured
#       - kl:    KL early stop,        logs --log-format structured
#       - both:  runs delta sweep then kl sweep (one allocation; long)
#
# Parallel GPUs (submit two jobs):
#   sbatch run_gilbreth.sh "1 2 3" delta
#   sbatch run_gilbreth.sh "1 2 3" kl
#
# Outputs:
#   results/{delta,kl}/h{1,2,3}/     — training outputs per H and criterion
#   figures/{delta,kl}/              — per-H Calvano PNGs + cross-H compare PNGs
#
# GPU partition (default a100-40gb). Same experiment as before: default CRIT=both → delta sweep, then KL.
#   sbatch run_gilbreth.sh                    # H=1,2,3  delta then kl
#   sbatch run_gilbreth.sh "1 2 3" both       # explicit
#   sbatch --partition=a100-80gb run_gilbreth.sh "1 2 3" both
#SBATCH --job-name=ppo-collusion
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Gilbreth expects GPU via GRES (--gpus-per-task alone can yield "No GPUs requested").
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

H_LIST="${1:-1 2 3}"
CRIT="${2:-both}"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

# ── Modules ──────────────────────────────────────────────────────────
module purge
module load cuda 2>/dev/null || true
if [ -n "${PYTHON_MODULE:-}" ]; then
    module load "$PYTHON_MODULE"
fi

# ── Virtual environment ────────────────────────────────────────────────
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

# ── Experiment parameters ─────────────────────────────────────────────
SESSIONS=100
TIMESTEPS=2000000
PATIENCE=100
DELTA_CONV_THRESH=0.01
KL_THRESH=0.01
POLICY_KL_LAG=0
EPISODE_LEN=168
# Structured logs: delta lines emphasize d_jump/streak; kl lines kl_for_conv (see experiments/ppo.py).
LOG_FORMAT=structured

run_sweep() {
    local mode="$1"
    local fig_root="figures/${mode}"
    mkdir -p "$fig_root"

    echo ""
    echo "####################################################################"
    echo "#  Convergence mode: ${mode}"
    echo "#  history lengths: ${H_LIST}"
    echo "#  results → results/${mode}/h<H>   figures → ${fig_root}/"
    echo "####################################################################"

    local -a run_dirs=()

    for H in $H_LIST; do
        echo "=========================================="
        echo "  ${mode}  |  history_len = $H  |  $SESSIONS sessions"
        echo "=========================================="
        python experiments/ppo.py \
            --history-len "$H" \
            --num-sessions "$SESSIONS" \
            --total-timesteps "$TIMESTEPS" \
            --convergence-mode "$mode" \
            --convergence-patience "$PATIENCE" \
            --delta-convergence-threshold "$DELTA_CONV_THRESH" \
            --kl-threshold "$KL_THRESH" \
            --policy-kl-lag "$POLICY_KL_LAG" \
            --log-format "$LOG_FORMAT" \
            --episode-len "$EPISODE_LEN" \
            --rollout-len 2048 \
            --hidden-dim 64 \
            --lr 3e-4 \
            --seed 42 \
            --cuda \
            --output-dir "results/${mode}/h${H}"

        python experiments/plot_results.py "results/${mode}/h${H}" --save "$fig_root" --calvano-paper
        echo "Finished ${mode} H=$H → results/${mode}/h${H}"
        run_dirs+=("results/${mode}/h${H}")
    done

    local n_h=0
    for _ in $H_LIST; do n_h=$((n_h + 1)); done
    if [ "$n_h" -gt 1 ] && [ "${#run_dirs[@]}" -gt 1 ]; then
        python experiments/plot_results.py --compare-calvano "${run_dirs[@]}" --save "$fig_root"
        python experiments/plot_results.py --compare "${run_dirs[@]}" --save "$fig_root"
        echo "Cross-H comparison figures → ${fig_root}/"
    fi
}

case "$CRIT" in
    delta) run_sweep delta ;;
    kl) run_sweep kl ;;
    both)
        run_sweep delta
        run_sweep kl
        ;;
    *)
        echo "Unknown criterion: $CRIT  (use delta, kl, or both)" >&2
        exit 1
        ;;
esac

echo "All runs complete. CRIT=${CRIT}  H_LIST=${H_LIST}"
