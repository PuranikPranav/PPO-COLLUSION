#!/bin/bash
# IBM Granite LLM-agent electricity-market game on the Gilbreth A100.
#
# INFERENCE ONLY — Granite is a frozen, pretrained model served through vLLM.
# Nothing here trains or fine-tunes anything: each period both firms' prompts
# go to the model in one batched call, the JSON replies are parsed into MW
# offers, and the ISO clears the market. "Sessions" are repeated GAMES with
# independent seeds (for error bars), not training runs.
#
# One submission produces the complete paper-reference package (MODE=all):
#   results/llm_granite8/memory/     collusion TREATMENT (narrative memory + strategy note)
#   results/llm_granite8/parity_h8/  information-matched arm (PPO obs vector, H=8)
#   results/llm_granite8/parity_h1/  memoryless BASELINE (PPO obs vector, H=1)
#   figures/llm_granite8/<arm>/      all standard figures for each arm
#   figures/llm_granite8/paper_table.{tex,md}  the table to paste in the paper
# Each session also writes sessions/session_*/transcripts.jsonl — the model's own
# per-period reasoning/strategy text (qualitative collusion evidence to quote).
#
# Submit from repo root:
#   sbatch run_gilbreth_llm.sh
#
# Default: Granite 3.3 8B (instruct) in bf16 on a single A100-40GB.
# Fits comfortably (~18 GB), full precision, no quantization caveats.
# (3.3-8b-instruct is a public, dense model proven with vLLM 0.11.)
#
# Overrides via environment variables:
#   MODEL=ibm-granite/granite-3.3-2b-instruct sbatch run_gilbreth_llm.sh
#   MODE=memory sbatch run_gilbreth_llm.sh     # a single treatment only
#
#SBATCH --job-name=llm-granite8
#SBATCH --account=liu334
#SBATCH --partition=a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm-llm-%j.out
#SBATCH --error=slurm-llm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONUNBUFFERED=1

# ---- Tunables (override via env) ----
MODEL="${MODEL:-ibm-granite/granite-3.3-8b-instruct}"
BACKEND="${BACKEND:-vllm}"
#
# MODE selects how the FROZEN model gets the memory it needs to (possibly) collude:
#   pilot   -> CHEAP first look: memory treatment only, few sessions/periods, still
#              runs the deviation/punishment figure. Use this before MODE=all.
#   memory  -> narrative sliding-window + strategy note (collusion TREATMENT).
#   parity  -> strict 19xH obs vector (HISTORY_LEN=8 info-matched; =1 memoryless baseline).
#   both    -> memory THEN parity + combined paper table.
#   all     -> memory/ + parity_h8/ + parity_h1/ + paper table (full package).
MODE="${MODE:-all}"

# Scale: pilot shrinks unless you override SESSIONS/PERIODS explicitly.
if [ "$MODE" = "pilot" ]; then
    SESSIONS="${SESSIONS:-3}"
    PERIODS="${PERIODS:-80}"
    DEV_SESSIONS="${DEV_SESSIONS:-3}"
else
    SESSIONS="${SESSIONS:-20}"
    PERIODS="${PERIODS:-300}"
    DEV_SESSIONS="${DEV_SESSIONS:-5}"
fi

HISTORY_LEN="${HISTORY_LEN:-8}"       # parity mode: obs_dim = 19 x HISTORY_LEN
WINDOW="${WINDOW:-10}"                # memory mode: sliding-window depth in the prompt
GOAL="${GOAL:-own_profit}"            # own_profit | joint_profit (both firms)
# Optional asymmetric probe: "collude,own_profit" = firm0 is a seeded colluder
# (explicit price-leader prompt), firm1 stays selfish — the "what happens if ONE
# agent colludes" treatment. joint_profit is the softer variant. Empty = GOAL for both.
GOALS="${GOALS:-}"
# Trial phase for the asymmetric probe: both firms play the symmetric GOAL for the
# first N periods, then GOALS switches on (within-session before/after contrast).
GOALS_START="${GOALS_START:-0}"
TP="${TP:-1}"                       # tensor-parallel GPUs
QUANT="${QUANT:-}"                  # empty = bf16
TEMPERATURE="${TEMPERATURE:-0.4}"   # lower = steadier choices near the profit peak
MAXTOK="${MAXTOK:-512}"               # reasoning + generation JSON (needs headroom to reason)
MAXLEN="${MAXLEN:-8192}"
SEED="${SEED:-42}"
# Short competitive forced-start so Δ plots begin at the competitive anchor and the
# LLM's first real choices already have a price history. 0 = LLM chooses from period 1.
WARMUP_COMP="${WARMUP_COMP:-10}"
# Punishment / impulse-response (Calvano-style). DEV_WARMUP ≥ WINDOW so memory is full.
DEV_FRAC="${DEV_FRAC:-0.2}"
DEV_WARMUP="${DEV_WARMUP:-20}"
DEV_HORIZON="${DEV_HORIZON:-20}"
LIMIT_STRATEGY="${LIMIT_STRATEGY:-0}" # 1 = also sweep the reaction function (parity mode only)
# Base dirs; each treatment writes into <base>/<mode>/.
RESULTS_BASE="${RESULTS_BASE:-results/llm_granite8}"
FIGURES_BASE="${FIGURES_BASE:-figures/llm_granite8}"

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
LIMIT_ARG=""
[ "$LIMIT_STRATEGY" = "1" ] && LIMIT_ARG="--limit-strategy"

echo "####################################################################"
echo "#  IBM Granite LLM agents — electricity market (INFERENCE ONLY)"
echo "#  model=${MODEL} backend=${BACKEND} TP=${TP} quant='${QUANT}'"
echo "#  mode=${MODE} goal=${GOAL} goals='${GOALS:-}' goals_start=${GOALS_START} sessions=${SESSIONS} periods=${PERIODS}"
echo "#  history_len=${HISTORY_LEN} (parity obs_dim=$((HISTORY_LEN * 19))) window=${WINDOW} (memory)"
echo "#  warmup_comp=${WARMUP_COMP}  deviation: frac=${DEV_FRAC} warmup=${DEV_WARMUP} horizon=${DEV_HORIZON} on first ${DEV_SESSIONS} sessions"
echo "#  GPU: ${CUDA_VISIBLE_DEVICES:-(Slurm-assigned)}  job=${SLURM_JOB_ID:-local}"
echo "#  results -> ${RESULTS_BASE}/<mode>   HF_HOME=${HF_HOME}"
echo "#  python=$(python --version 2>&1)  module=${PYTHON_MODULE:-none}"
echo "#  venv=${ENV_DIR}  pip_cache=${PIP_CACHE_DIR}"
echo "####################################################################"

# The theory benchmarks are recomputed from iso_market/node_network.py at run
# time and stored in each run's config.json, so results are only meaningful for
# the network they were played on. Archive any results from a previous network
# instead of mixing them into this paper package.
for base in "$RESULTS_BASE" "$FIGURES_BASE"; do
    if [ -d "$base" ] && [ -n "$(ls -A "$base" 2>/dev/null)" ]; then
        STAMP="$(date +%Y%m%d_%H%M%S)"
        echo "Archiving stale ${base} -> ${base}_old_${STAMP} (network may have changed)"
        mv "$base" "${base}_old_${STAMP}"
    fi
done

# Run one treatment: $1 = memory | parity, $2 = history_len, $3 = output subdir.
run_mode() {
    local mode="$1"
    local hlen="${2:-$HISTORY_LEN}"
    local subdir="${3:-$mode}"
    local out_dir="${RESULTS_BASE}/${subdir}"
    local fig_dir="${FIGURES_BASE}/${subdir}"
    local mode_arg="--ppo-parity"
    [ "$mode" = "memory" ] && mode_arg="--legacy-memory"

    echo ""
    echo "==== [$subdir] (${mode}, H=${hlen}) game -> ${out_dir} ============="
    python llm_market/run_llm_market.py \
        --backend "$BACKEND" \
        --model "$MODEL" \
        --num-sessions "$SESSIONS" \
        --num-periods "$PERIODS" \
        --warmup-competitive "$WARMUP_COMP" \
        --history-len "$hlen" \
        --history-window "$WINDOW" \
        --goal "$GOAL" \
        ${GOALS:+--goals "$GOALS"} \
        --goals-start "$GOALS_START" \
        $mode_arg \
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
        --output-dir "$out_dir"

    # Plot exactly like the PPO runs.
    python experiments/plot_results.py "$out_dir" --calvano-paper       --save "$fig_dir" || true
    python experiments/plot_results.py "$out_dir" --per-firm-profit     --save "$fig_dir" || true
    python experiments/plot_results.py "$out_dir" --variance-funnel     --save "$fig_dir" || true
    # The punishment / retaliation figure (now populated for the LLM agents).
    python experiments/plot_results.py "$out_dir" --deviation-explainer --save "$fig_dir" || true
}

TABLE_DIRS=()
case "$MODE" in
    pilot)
        # Cheap smoke test: memory treatment + deviation figure only.
        run_mode memory "$HISTORY_LEN" memory
        TABLE_DIRS=("${RESULTS_BASE}/memory")
        ;;
    all)
        # The three-arm paper package: treatment + info-matched + memoryless baseline.
        run_mode memory "$HISTORY_LEN" memory
        run_mode parity 8 parity_h8
        run_mode parity 1 parity_h1
        TABLE_DIRS=("${RESULTS_BASE}/memory" "${RESULTS_BASE}/parity_h8" "${RESULTS_BASE}/parity_h1")
        ;;
    both)
        run_mode memory
        run_mode parity
        TABLE_DIRS=("${RESULTS_BASE}/memory" "${RESULTS_BASE}/parity")
        ;;
    memory|parity)
        run_mode "$MODE"
        TABLE_DIRS=("${RESULTS_BASE}/${MODE}")
        ;;
    *)
        echo "ERROR: MODE must be pilot, all, memory, parity or both (got '$MODE')" >&2
        exit 1
        ;;
esac

# Paper-ready reference table: frozen-LLM outcomes vs the theory benchmarks.
python experiments/make_llm_paper_table.py "${TABLE_DIRS[@]}" \
    --save "$FIGURES_BASE" || true

echo ""
echo "Done."
echo "  results -> ${RESULTS_BASE}/"
echo "  figures -> ${FIGURES_BASE}/"
echo "  paper   -> ${FIGURES_BASE}/paper_table.tex (and .md)"
