#!/bin/bash
# Re-run training with competitive-LMP initial state (new ElectricityMarketEnv
# behavior in iso_market/market_env.py) — delta convergence criterion only.
#
# Submit from the repo root:
#   cd ~/ppo-collusion && sbatch run_gilbreth_compinit.sh [H_LIST]
#
# Arguments:
#   $1  History lengths (default "1 2 3")
#
# Outputs (kept separate from the original 50%-capacity runs):
#   results/delta_compinit/h{1,2,3}/   — training outputs per H
#   figures/delta_compinit/            — per-H Calvano PNGs + cross-H compare PNGs
#                                        (uses --compare-delta which omits KL)
#
# Cancel running / pending jobs from the Gilbreth login node:
#   squeue -u $USER
#   scancel -u $USER                  # all your jobs
#   scancel <JOBID>                   # one job
#
#SBATCH --job-name=ppo-compinit
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

H_LIST="${1:-1 2 3}"
MODE="delta"
RESULTS_ROOT="results/delta_compinit"
FIGURES_ROOT="figures/delta_compinit"

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

# ── Experiment parameters (must match results/delta_cont/h*/config.json) ───
SESSIONS=100
TIMESTEPS=2000000
PATIENCE=100
DELTA_CONV_THRESH=0.01
KL_THRESH=0.01
POLICY_KL_LAG=0
EPISODE_LEN=168
LOG_FORMAT=structured

mkdir -p "$FIGURES_ROOT"

echo ""
echo "####################################################################"
echo "#  Competitive-init re-run  (env now seeds lmp_history from"
echo "#  perfect-competition LMPs — see ElectricityMarketEnv)"
echo "#  Convergence mode: ${MODE}"
echo "#  history lengths : ${H_LIST}"
echo "#  results → ${RESULTS_ROOT}/h<H>"
echo "#  figures → ${FIGURES_ROOT}/"
echo "####################################################################"

run_dirs=()
for H in $H_LIST; do
    echo "=========================================="
    echo "  ${MODE} (compinit)  |  H = $H  |  $SESSIONS sessions"
    echo "=========================================="
    python experiments/ppo.py \
        --history-len "$H" \
        --num-sessions "$SESSIONS" \
        --total-timesteps "$TIMESTEPS" \
        --convergence-mode "$MODE" \
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
        --output-dir "${RESULTS_ROOT}/h${H}"

    python experiments/plot_results.py "${RESULTS_ROOT}/h${H}" --save "$FIGURES_ROOT" --calvano-paper
    echo "Finished ${MODE} compinit H=$H → ${RESULTS_ROOT}/h${H}"
    run_dirs+=("${RESULTS_ROOT}/h${H}")
done

# Cross-H plots (skip KL panel — use --compare-delta)
n_h=0
for _ in $H_LIST; do n_h=$((n_h + 1)); done
if [ "$n_h" -gt 1 ] && [ "${#run_dirs[@]}" -gt 1 ]; then
    python experiments/plot_results.py --compare-calvano "${run_dirs[@]}" --save "$FIGURES_ROOT"
    python experiments/plot_results.py --compare-delta "${run_dirs[@]}" --save "$FIGURES_ROOT"
    python experiments/plot_results.py --compare-generation-profit "${run_dirs[@]}" --save "$FIGURES_ROOT"
    python experiments/plot_results.py --deviation-explainer "${run_dirs[@]}" --save "$FIGURES_ROOT"
    echo "Cross-H comparison figures → ${FIGURES_ROOT}/"
fi

echo "Competitive-init delta sweep complete. H_LIST=${H_LIST}"
