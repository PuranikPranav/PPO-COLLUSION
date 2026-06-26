#!/bin/bash
# PPO collusion run for the NEW market parameters (iso_market/node_network.py:
# P0=(55,50,...), MC=22/22/25, QC=0.05/0.05/0.025) with NEUTRAL policy init so the
# agents EXPLORE both up and down before settling — instead of being pinned at the
# competitive (max-output) ceiling and only descending.
#
# Why neutral init: competitive output is (near) full capacity, i.e. the top edge of
# the action space. Initializing the policy there forces a one-way descent (the
# "monotone graph" artifact). Starting the policy in the INTERIOR lets each firm sample
# the whole [0, cap] range early, fluctuate up and down, and converge on its own — the
# story the generation plot should tell.
#
# Submit from the repo root (one A100):
#   cd ~/ppo-collusion && git pull
#   sbatch run_gilbreth_explore.sh                 # H=1, 100 sessions, defaults below
#   SESSIONS=1000 sbatch run_gilbreth_explore.sh   # paper-scale
#   H=1 ENT_COEF=0.02 sbatch run_gilbreth_explore.sh
#
# Monitor:  squeue -u $USER ;  tail -f slurm-explore-*.out
# Outputs:  results/explore/h<H>/   figures/explore/
#
# PUNISHMENT EXPERIMENT NOTE: the deviation/punishment figure reveals whether the
# learned equilibrium is sustained by retaliation (genuine tacit collusion). H=1
# (one-period memory) is SUFFICIENT for multi-period punishment -- this is exactly the
# Calvano et al. (2020) / Calzolari et al. (2021) setup. H=1 only needs to DETECT the
# cheat (low price / high rival output last period); the price war then self-perpetuates
# through the state and fades gradually (the price level acts as a surrogate clock), so
# punishment lasts many periods then returns to the collusive resting point. H>1 just
# adds state info (e.g. distinguishing a deviation from a demand shock) and is NOT
# required. Default H=1 matches the canonical setup.
#
#SBATCH --job-name=ppo-explore
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mem=50G
#SBATCH --output=slurm-explore-%j.out
#SBATCH --error=slurm-explore-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

# ---- Tunables (override via env) ----
H="${H:-1}"
SESSIONS="${SESSIONS:-100}"
TIMESTEPS="${TIMESTEPS:-2000000}"
# Exploration / settling knobs (the heart of this run):
INIT_POLICY="${INIT_POLICY:-neutral}"   # neutral = interior start (two-sided exploration)
INIT_FRACTION="${INIT_FRACTION:-0.5}"   # start each plant at 50% of capacity
INIT_CONC="${INIT_CONC:-1.2}"           # low Beta concentration => broad initial sampling
INIT_WSCALE="${INIT_WSCALE:-0.3}"       # per-seed spread around the interior start
ENT_COEF="${ENT_COEF:-0.02}"            # entropy bonus at the START (high exploration)
ENT_COEF_FINAL="${ENT_COEF_FINAL:-0.0}" # anneal entropy -> 0: explore early, exploit late
ANNEAL_LR="${ANNEAL_LR:-1}"             # also decay the learning rate (sharpen late exploitation)
# Convergence (so it SETTLES and early-stops once Δ is stable):
PATIENCE="${PATIENCE:-100}"
DELTA_CONV_THRESH="${DELTA_CONV_THRESH:-0.01}"
EPISODE_LEN="${EPISODE_LEN:-168}"
SEED="${SEED:-42}"
RESULTS_ROOT="${RESULTS_ROOT:-results/explore}"
FIGURES_ROOT="${FIGURES_ROOT:-figures/explore}"

module purge
module load cuda 2>/dev/null || true
if [ -n "${PYTHON_MODULE:-}" ]; then
    module load "$PYTHON_MODULE"
fi

ENV_DIR="${ENV_DIR:-$HOME/envs/ppo-collusion}"
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

mkdir -p "$FIGURES_ROOT"

echo "####################################################################"
echo "#  PPO collusion — NEW market params + exploratory (neutral) init"
echo "#  H=${H}  sessions=${SESSIONS}  timesteps=${TIMESTEPS}"
echo "#  init=${INIT_POLICY} frac=${INIT_FRACTION} conc=${INIT_CONC} ent=${ENT_COEF}"
echo "#  convergence=delta thresh=${DELTA_CONV_THRESH} patience=${PATIENCE}"
echo "#  GPU: ${CUDA_VISIBLE_DEVICES:-(Slurm-assigned)}  job=${SLURM_JOB_ID:-local}"
echo "#  results -> ${RESULTS_ROOT}/h${H}   figures -> ${FIGURES_ROOT}/"
echo "####################################################################"

python experiments/ppo.py \
    --history-len "$H" \
    --num-sessions "$SESSIONS" \
    --total-timesteps "$TIMESTEPS" \
    --init-policy "$INIT_POLICY" \
    --init-fraction "$INIT_FRACTION" \
    --init-concentration "$INIT_CONC" \
    --init-weight-scale "$INIT_WSCALE" \
    --ent-coef "$ENT_COEF" \
    --ent-coef-final "$ENT_COEF_FINAL" \
    $([ "$ANNEAL_LR" = "1" ] && echo --anneal-lr) \
    --convergence-mode delta \
    --convergence-patience "$PATIENCE" \
    --delta-convergence-threshold "$DELTA_CONV_THRESH" \
    --log-format structured \
    --episode-len "$EPISODE_LEN" \
    --rollout-len 2048 \
    --hidden-dim 64 \
    --lr 3e-4 \
    --seed "$SEED" \
    --cuda \
    --output-dir "${RESULTS_ROOT}/h${H}"

# Plots that tell the story: quantity-vs-time (with Competitive/Nash/Monopoly lines),
# Δ-vs-time (0=Nash, 1=Monopoly), per-firm profit, variance funnel, and the
# deviation/punishment figure.
python experiments/plot_results.py "${RESULTS_ROOT}/h${H}" --save "$FIGURES_ROOT" --calvano-paper
python experiments/plot_results.py "${RESULTS_ROOT}/h${H}" --save "$FIGURES_ROOT" --per-firm-profit     || true
python experiments/plot_results.py "${RESULTS_ROOT}/h${H}" --save "$FIGURES_ROOT" --variance-funnel     || true
python experiments/plot_results.py "${RESULTS_ROOT}/h${H}" --save "$FIGURES_ROOT" --deviation-explainer || true

echo "Done -> ${RESULTS_ROOT}/h${H}"
