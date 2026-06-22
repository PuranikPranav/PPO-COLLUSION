#!/bin/bash
# IBM Granite LLM-agent electricity-market game on the Gilbreth A100.
#
# Submit from repo root:
#   sbatch -J llm-granite -o slurm-llm-%j.out -e slurm-llm-%j.err run_gilbreth_llm.sh
#
# Default: Granite 4.1 8B (instruct) in bf16 on a single A100-40GB.
# Fits comfortably (~18 GB), full precision, no quantization caveats.
#
# Alternatives (override via environment variables):
#   # Smaller / faster (3B):
#   MODEL=ibm-granite/granite-4.1-3b-instruct sbatch run_gilbreth_llm.sh
#   # Larger (30B) in bf16 across two A100-40GB GPUs:
#   MODEL=ibm-granite/granite-4.1-30b-instruct TP=2 sbatch -G2 -J llm-granite30 run_gilbreth_llm.sh
#
#SBATCH --job-name=llm-granite8
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm-llm-%j.out
#SBATCH --error=slurm-llm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

# ---- Tunables (override via env) ----
MODEL="${MODEL:-ibm-granite/granite-4.1-8b-instruct}"
BACKEND="${BACKEND:-vllm}"
SESSIONS="${SESSIONS:-20}"
PERIODS="${PERIODS:-300}"
WINDOW="${WINDOW:-1}"               # last-N periods in the prompt; 1 mimics PPO H=1
TP="${TP:-1}"                       # tensor-parallel GPUs (set 2 for the 30B model)
QUANT="${QUANT:-}"                  # empty = bf16 (no quantization); 'fp8' only for big models
TEMPERATURE="${TEMPERATURE:-0.7}"
MAXTOK="${MAXTOK:-256}"
MAXLEN="${MAXLEN:-8192}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/llm_granite8/h1}"
FIGURES_DIR="${FIGURES_DIR:-figures/llm_granite8}"

module purge
module load cuda 2>/dev/null || true

# vLLM 0.11+ requires Python >= 3.10 (uses PEP-604 `X | Y` type syntax).
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
if ! module load "$PYTHON_MODULE" 2>/dev/null; then
    for alt in python/3.10 python3/3.11 python3/3.10; do
        if module load "$alt" 2>/dev/null; then
            PYTHON_MODULE="$alt"
            break
        fi
    done
fi

# IMPORTANT: home has a small quota (~25 GB). vLLM+torch+CUDA (~10 GB) and the
# model cache must live on scratch, which has a large quota. Resolve scratch.
SCRATCH="${RCAC_SCRATCH:-/scratch/gilbreth/$USER}"
[ -d "$SCRATCH" ] || SCRATCH="$HOME/scratch"
mkdir -p "$SCRATCH"

# Keep pip's build/cache off home, too.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/pip_cache}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Separate env for vLLM (it pins its own torch; keep it apart from the PPO env).
ENV_DIR="${ENV_DIR:-$SCRATCH/envs/ppo-llm}"
PY=python3
command -v "$PY" >/dev/null || PY=python

PY_VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: need Python >= 3.10 for vLLM (found ${PY_VER})."
    echo "  Try: module avail python   then  PYTHON_MODULE=python/3.11 sbatch ..."
    exit 1
fi

# Drop a stale venv built with an older interpreter (e.g. system python3.9).
if [ -d "$ENV_DIR" ] && ! "$ENV_DIR/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Removing stale venv at ${ENV_DIR} (Python < 3.10)."
    rm -rf "$ENV_DIR"
fi

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating venv with ${PY} (${PY_VER}) at ${ENV_DIR}"
    "$PY" -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements_llm.txt
else
    source "$ENV_DIR/bin/activate"
fi

# Cache HF weights on scratch so they persist and aren't re-downloaded.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
mkdir -p "$HF_HOME"

QUANT_ARG=""
[ -n "$QUANT" ] && QUANT_ARG="--quantization $QUANT"

echo "####################################################################"
echo "#  IBM Granite LLM agents — electricity market"
echo "#  model=${MODEL} backend=${BACKEND} TP=${TP} quant='${QUANT}'"
echo "#  sessions=${SESSIONS} periods=${PERIODS} window=${WINDOW}"
echo "#  GPU: ${CUDA_VISIBLE_DEVICES:-(Slurm-assigned)}  job=${SLURM_JOB_ID:-local}"
echo "#  results -> ${OUTPUT_DIR}   HF_HOME=${HF_HOME}"
echo "#  python=$(python --version 2>&1)  module=${PYTHON_MODULE:-none}"
echo "#  venv=${ENV_DIR}  pip_cache=${PIP_CACHE_DIR}"
echo "####################################################################"

python llm_market/run_llm_market.py \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --num-sessions "$SESSIONS" \
    --num-periods "$PERIODS" \
    --history-window "$WINDOW" \
    --tensor-parallel-size "$TP" \
    $QUANT_ARG \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAXTOK" \
    --max-model-len "$MAXLEN" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"

# Plot exactly like the PPO runs.
python experiments/plot_results.py "$OUTPUT_DIR" --calvano-paper   --save "$FIGURES_DIR" || true
python experiments/plot_results.py "$OUTPUT_DIR" --per-firm-profit --save "$FIGURES_DIR" || true
python experiments/plot_results.py "$OUTPUT_DIR" --variance-funnel --save "$FIGURES_DIR" || true

echo "Done -> ${OUTPUT_DIR}"
