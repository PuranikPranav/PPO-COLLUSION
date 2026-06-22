#!/bin/bash
# IBM Granite LLM-agent electricity-market game on the Gilbreth A100.
#
# Submit from repo root:
#   sbatch -J llm-granite -o slurm-llm-%j.out -e slurm-llm-%j.err run_gilbreth_llm.sh
#
# Default: Granite 3.3 8B (instruct) in bf16 on a single A100-40GB.
# Fits comfortably (~18 GB), full precision, no quantization caveats.
# (3.3-8b-instruct is a public, dense model proven with vLLM 0.11.)
#
# Alternatives (override via environment variables):
#   # Smaller / faster (2B):
#   MODEL=ibm-granite/granite-3.3-2b-instruct sbatch run_gilbreth_llm.sh
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
MODEL="${MODEL:-ibm-granite/granite-3.3-8b-instruct}"
BACKEND="${BACKEND:-vllm}"
SESSIONS="${SESSIONS:-20}"
PERIODS="${PERIODS:-300}"           # rounds of the repeated game (LLM x-axis is rounds)
#
# MODE selects how the FROZEN model gets the memory it needs to (possibly) collude:
#   memory  -> narrative sliding-window of joint history + carry-forward strategy note
#              (the collusion TREATMENT; the model writes its strategy in words).
#   parity  -> the strict 19xH observation vector; set HISTORY_LEN>1 (e.g. 8) so the
#              obs itself carries a multi-period window (directly comparable to PPO H).
#              HISTORY_LEN=1 here is the deliberately-memoryless BASELINE.
MODE="${MODE:-memory}"
HISTORY_LEN="${HISTORY_LEN:-8}"       # parity mode: obs_dim = 19 x HISTORY_LEN
WINDOW="${WINDOW:-10}"                # memory mode: sliding-window depth in the prompt
GOAL="${GOAL:-own_profit}"            # own_profit | joint_profit
TP="${TP:-1}"                       # tensor-parallel GPUs
QUANT="${QUANT:-}"                  # empty = bf16
TEMPERATURE="${TEMPERATURE:-0.4}"   # lower = steadier choices near the profit peak
MAXTOK="${MAXTOK:-512}"               # reasoning + generation JSON (needs headroom to reason)
MAXLEN="${MAXLEN:-8192}"
SEED="${SEED:-42}"
WARMUP_COMP="${WARMUP_COMP:-0}"       # force competitive baseline for first N rounds (clean start)
# Punishment / impulse-response experiment (Calvano-style retaliation figure).
DEV_FRAC="${DEV_FRAC:-0.2}"
DEV_WARMUP="${DEV_WARMUP:-8}"
DEV_HORIZON="${DEV_HORIZON:-20}"
DEV_SESSIONS="${DEV_SESSIONS:-5}"     # run the deviation/limit experiments on the first N sessions
LIMIT_STRATEGY="${LIMIT_STRATEGY:-0}" # 1 = also sweep the reaction function (parity mode only)
OUTPUT_DIR="${OUTPUT_DIR:-results/llm_granite8/${MODE}}"
FIGURES_DIR="${FIGURES_DIR:-figures/llm_granite8/${MODE}}"

module purge
module load cuda 2>/dev/null || true

# vLLM 0.11+ requires Python >= 3.10 (uses PEP-604 `X | Y` type syntax).
# Gilbreth ships Python via Anaconda, which lives under the `external` module
# tree, so `external` must be loaded first. py312 is the sweet spot: satisfies
# vLLM and has wheels for every dep (torch, numba, etc.); py313 is avoided
# because some pinned deps lack 3.13 wheels.
module load external 2>/dev/null || true
PYTHON_MODULE="${PYTHON_MODULE:-anaconda/2024.10-py312}"
if ! module load "$PYTHON_MODULE" 2>/dev/null; then
    for alt in anaconda/2025.06-py313 anaconda python/3.12 python/3.11 python/3.10; do
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

# Pick the newest interpreter that is >= 3.10. On Gilbreth, Anaconda's `python`
# is 3.12 while the bare `python3` is still the system 3.9, so search explicitly.
PY=""
for cand in python3.12 python3.11 python3.10 python python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$cand"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: no Python >= 3.10 found (needed for vLLM 0.11)."
    echo "  Loaded module: ${PYTHON_MODULE:-none}"
    echo "  python  -> $(command -v python  2>/dev/null) $(python  --version 2>&1)"
    echo "  python3 -> $(command -v python3 2>/dev/null) $(python3 --version 2>&1)"
    echo "  Try: module avail anaconda   then  PYTHON_MODULE=anaconda/2024.10-py312 sbatch ..."
    exit 1
fi
PY_VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# Pinned vLLM version this run expects (kept in sync with requirements_llm.txt).
PIN_VLLM="${PIN_VLLM:-0.11.0}"

# Drop a stale venv built with an older interpreter (e.g. system python3.9).
if [ -d "$ENV_DIR" ] && ! "$ENV_DIR/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Removing stale venv at ${ENV_DIR} (Python < 3.10)."
    rm -rf "$ENV_DIR"
fi

# Drop a venv whose vLLM doesn't match the pin (e.g. an old 0.23 CUDA-13 build).
# This makes the script self-healing after a requirements change.
if [ -d "$ENV_DIR" ]; then
    HAVE_VLLM="$("$ENV_DIR/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)"
    if [ "$HAVE_VLLM" != "$PIN_VLLM" ]; then
        echo "Rebuilding venv: have vLLM '${HAVE_VLLM:-none}', want '${PIN_VLLM}'."
        rm -rf "$ENV_DIR"
    fi
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

# memory mode uses narrative memory (--legacy-memory); parity is the default.
MODE_ARG="--ppo-parity"
[ "$MODE" = "memory" ] && MODE_ARG="--legacy-memory"
LIMIT_ARG=""
[ "$LIMIT_STRATEGY" = "1" ] && LIMIT_ARG="--limit-strategy"

echo "####################################################################"
echo "#  IBM Granite LLM agents — electricity market"
echo "#  model=${MODEL} backend=${BACKEND} TP=${TP} quant='${QUANT}'"
echo "#  mode=${MODE} goal=${GOAL} sessions=${SESSIONS} periods=${PERIODS}"
echo "#  history_len=${HISTORY_LEN} (parity obs_dim=$((HISTORY_LEN * 19))) window=${WINDOW} (memory)"
echo "#  deviation: frac=${DEV_FRAC} warmup=${DEV_WARMUP} horizon=${DEV_HORIZON} on first ${DEV_SESSIONS} sessions"
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
    --warmup-competitive "$WARMUP_COMP" \
    --history-len "$HISTORY_LEN" \
    --history-window "$WINDOW" \
    --goal "$GOAL" \
    $MODE_ARG \
    --tensor-parallel-size "$TP" \
    $QUANT_ARG \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAXTOK" \
    --max-model-len "$MAXLEN" \
    --seed "$SEED" \
    --deviation-frac "$DEV_FRAC" \
    --deviation-warmup "$DEV_WARMUP" \
    --deviation-horizon "$DEV_HORIZON" \
    --deviation-max-sessions "$DEV_SESSIONS" \
    $LIMIT_ARG \
    --output-dir "$OUTPUT_DIR"

# Plot exactly like the PPO runs.
python experiments/plot_results.py "$OUTPUT_DIR" --calvano-paper       --save "$FIGURES_DIR" || true
python experiments/plot_results.py "$OUTPUT_DIR" --per-firm-profit     --save "$FIGURES_DIR" || true
python experiments/plot_results.py "$OUTPUT_DIR" --variance-funnel     --save "$FIGURES_DIR" || true
# The punishment / retaliation figure (now populated for the LLM agents).
python experiments/plot_results.py "$OUTPUT_DIR" --deviation-explainer --save "$FIGURES_DIR" || true

echo "Done -> ${OUTPUT_DIR}"
