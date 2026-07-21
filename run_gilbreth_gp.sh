#!/bin/bash
#SBATCH --job-name=ppo-gp-collusion
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Gilbreth rejects jobs with "No GPUs requested" unless GRES is set
# (CPU-only partitions are not the default here).
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-gp-%j.out
#SBATCH --error=slurm-gp-%j.err
#
# Live logs: without this, Python fully-buffers stdout when writing to a Slurm
# .out file, so the banner appears and then nothing until the buffer fills.
export PYTHONUNBUFFERED=1
#
# GREEN-PORTER / IMPERFECT-MONITORING run (Calvano et al. 2021 adaptation):
# two-firm market, unobserved i.i.d. demand shocks, price(+own-output) state.
# The deviation experiment runs automatically per frozen demand state and the
# Fig-3/Fig-4-style figures are generated from the session average.
#
# Usage (ARRAY MODE — recommended; sessions run in PARALLEL across tasks):
#   sbatch --array=0-23 run_gilbreth_gp.sh            # 24 tasks x 2 sessions = 48 sessions
#   SESSIONS=3 sbatch --array=0-15 run_gilbreth_gp.sh # 16 tasks x 3 = 48 sessions
# then, after ALL tasks finish, consolidate + plot:
#   python scripts/consolidate_runs.py results/gp_twofirm/merged results/gp_twofirm/task_*
#   python experiments/plot_results.py results/gp_twofirm/merged --save figures/gp_twofirm --fig34
#
# Single-job mode (no --array): runs $SESSIONS sequentially in one task — only
# sensible for small SESSIONS (each 2M-step session takes hours).
#
# The paper runs 1000 sessions x ~2M periods; averaging across sessions is what
# denoises the heterogeneous limit strategies (their Fig 3/4). Aim for
# >=25 total sessions and TIMESTEPS=2000000 for publishable figures.

set -e
export MARKET_CONFIG=two_firm

H="${H:-1}"
SESSIONS="${SESSIONS:-2}"            # sessions PER TASK in array mode
TIMESTEPS="${TIMESTEPS:-2000000}"
OBS_MODE="${OBS_MODE:-price_own}"        # price_own | price_only
DEMAND_SHOCK="${DEMAND_SHOCK:-3.0}"      # +/- $/MWh intercept shock (unobserved)
SHOCK_PERSIST="${SHOCK_PERSIST:-0.5}"    # 0.5 = i.i.d. (paper); 0.9 = persistent load
ENT_COEF="${ENT_COEF:-0.02}"
ENT_COEF_FINAL="${ENT_COEF_FINAL:-0.003}"  # entropy FLOOR: keep off-path states sampled
GAMMA="${GAMMA:-0.99}"
DEV_LEN="${DEV_LEN:-1}"                  # 1 = paper's one-period forced cheat
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results/gp_twofirm}"
FIGURES_ROOT="${FIGURES_ROOT:-figures/gp_twofirm}"
ENV_DIR="${ENV_DIR:-$HOME/envs/ppo-collusion}"

# ---- SLURM array support: each task trains its own seeds into its own dir ----
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEED=$((SEED + SLURM_ARRAY_TASK_ID * SESSIONS))
    RESULTS_ROOT="${RESULTS_ROOT}/task_${SLURM_ARRAY_TASK_ID}"
    ARRAY_MODE=1
else
    ARRAY_MODE=0
fi

module load anaconda 2>/dev/null || true
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$ENV_DIR/bin/activate"
fi

mkdir -p "$FIGURES_ROOT"

echo "####################################################################"
echo "#  GP / imperfect monitoring: obs=${OBS_MODE} shock=±\$${DEMAND_SHOCK} rho=${SHOCK_PERSIST}"
echo "#  H=${H} sessions=${SESSIONS} timesteps=${TIMESTEPS} ent=${ENT_COEF}->${ENT_COEF_FINAL}"
echo "#  results -> ${RESULTS_ROOT}   figures -> ${FIGURES_ROOT}"
echo "####################################################################"

python -u experiments/ppo.py \
    --history-len "$H" \
    --num-sessions "$SESSIONS" \
    --total-timesteps "$TIMESTEPS" \
    --obs-mode "$OBS_MODE" \
    --demand-shock "$DEMAND_SHOCK" \
    --shock-persistence "$SHOCK_PERSIST" \
    --init-policy neutral \
    --init-fraction 0.5 \
    --init-concentration 1.2 \
    --init-weight-scale 0.3 \
    --ent-coef "$ENT_COEF" \
    --ent-coef-final "$ENT_COEF_FINAL" \
    --anneal-lr \
    --gamma "$GAMMA" \
    --convergence-mode delta \
    --convergence-patience 100 \
    --delta-convergence-threshold 0.01 \
    --log-format structured \
    --episode-len 168 \
    --rollout-len 2048 \
    --hidden-dim 64 \
    --lr 3e-4 \
    --deviation-warmup 20 --deviation-horizon 40 --deviation-pre 4 \
    --deviation-len "$DEV_LEN" \
    --seed "$SEED" \
    --cuda \
    --output-dir "$RESULTS_ROOT"

if [ "$ARRAY_MODE" = "1" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} done -> ${RESULTS_ROOT}."
    echo "After ALL tasks finish, consolidate + plot:"
    echo "  python scripts/consolidate_runs.py ${RESULTS_ROOT%/task_*}/merged ${RESULTS_ROOT%/task_*}/task_*"
    echo "  python experiments/plot_results.py ${RESULTS_ROOT%/task_*}/merged --save ${FIGURES_ROOT} --fig34"
    exit 0
fi

# Paper-style figures: Fig 3 (session-averaged limit strategy vs demand lines)
# and Fig 4 (session-averaged deviation response per frozen demand state), plus
# the standard training-evolution and deviation-explainer sets.
python experiments/plot_results.py "$RESULTS_ROOT" --save "$FIGURES_ROOT" --fig34
python experiments/plot_results.py "$RESULTS_ROOT" --save "$FIGURES_ROOT" --calvano-paper || true
python experiments/plot_results.py "$RESULTS_ROOT" --save "$FIGURES_ROOT" --deviation-explainer || true
echo "Done. Figures in ${FIGURES_ROOT}/"
