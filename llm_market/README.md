# LLM-Agent Electricity Market (IBM Granite)

Each firm is driven by an open-source **IBM Granite** model instead of PPO. The agents
play the **same** `ElectricityMarketEnv` (DC-OPF clearing) and are scored with the
**same** benchmarks and Δ metric as the PPO experiments, so results are directly
comparable and plot with the existing `experiments/plot_results.py`.

## The four-stage pipeline

| Step | File | Role |
|------|------|------|
| 1. State Translator | `state_translator.py` | Raw market data (LMPs, generation, profit) → text |
| 2. Memory & Prompt | `prompt_builder.py` | Sliding-window history → chat messages |
| 3. Reasoning Engine | `granite_engine.py` | Granite (vLLM/transformers/mock) → rationale + JSON |
| 4. Action Parser | `action_parser.py` | JSON → validated MW per plant (clipped to capacity) |

`run_llm_market.py` drives the loop: every period it builds one prompt per firm,
sends **all agents' prompts in a single batched call**, parses the actions, clears the
market, updates each agent's memory, and logs a row compatible with the plotter.

## Backends

- **`mock`** — no GPU, no model. Returns valid JSON heuristically so you can test the
  whole pipeline on a laptop. (Already verified end-to-end.)
- **`vllm`** — high-throughput batched inference on the A100 (use this on Gilbreth).
  Uses structured JSON decoding when the installed vLLM supports it.
- **`transformers`** — Hugging Face fallback (slower).

## Local test (no GPU) — already works

```bash
source venv/bin/activate
python llm_market/run_llm_market.py --backend mock \
    --num-sessions 3 --num-periods 40 --output-dir results/llm_mock
```

## Run Granite on the Gilbreth A100

**You only need to do the install once.** Everything below runs on the Gilbreth login node.

1. Pull the code:
   ```bash
   cd ~/ppo-collusion
   git fetch origin && git checkout without-episodes && git pull
   ```
2. Submit the job. **The default is the 8B model in bf16 on one A100-40GB** — full
   precision, fits comfortably (~18 GB), no quantization caveats. First run creates a
   separate `~/envs/ppo-llm` venv and installs deps; vLLM auto-downloads the Apache-2.0
   Granite weights from Hugging Face — **no token needed**:
   ```bash
   sbatch -J llm-granite8 run_gilbreth_llm.sh
   ```
3. Monitor:
   ```bash
   squeue -u $USER
   tail -f slurm-llm-*.out
   ```

### Model / hardware choices (set via env vars)

| Config | Fits on | Command |
|--------|---------|---------|
| **8B, bf16 (default)** | 1× A100-40GB | `sbatch -J llm-granite8 run_gilbreth_llm.sh` |
| 3B, bf16 (fastest) | 1× A100-40GB | `MODEL=ibm-granite/granite-4.1-3b-instruct sbatch run_gilbreth_llm.sh` |
| 30B, bf16, 2 GPUs | 2× A100-40GB | `MODEL=ibm-granite/granite-4.1-30b-instruct TP=2 sbatch -G2 -J llm-granite30 run_gilbreth_llm.sh` |

The 8B model is IBM's recommended balanced choice and (per IBM) matches or beats the
older Granite 4.0 32B MoE — strong enough for the strategic reasoning here while fitting
on one GPU at full precision.

Other overridable env vars: `SESSIONS`, `PERIODS`, `WINDOW`, `TEMPERATURE`, `MAXTOK`,
`SEED`, `OUTPUT_DIR`, `FIGURES_DIR`.

## Plotting (identical to the PPO runs)

```bash
python experiments/plot_results.py results/llm_granite/h1 --calvano-paper   --save figures/llm/
python experiments/plot_results.py results/llm_granite/h1 --per-firm-profit --save figures/llm/
python experiments/plot_results.py results/llm_granite/h1 --variance-funnel --save figures/llm/
```

## What you must install / provide

- **Nothing on your Mac** beyond the existing `venv` (the `mock` backend already runs).
- **On Gilbreth**: the Slurm script auto-creates `~/envs/ppo-llm` and `pip install -r
  requirements_llm.txt` on first submit. Granite is Apache-2.0 and ungated, so **no
  Hugging Face token or paid API key is required**. If `module load cuda` uses a
  different name on your allocation, set `PYTHON_MODULE`/adjust the `module load` line.

## Notes

- Output layout (`config.json`, `sessions/session_*/session.json`, `aggregate.json`)
  mirrors the PPO runs exactly, so all existing plotting/analysis works.
- The system prompt frames each firm as a self-interested **repeated** competitor and
  does **not** instruct it to collude — any supra-competitive behavior is emergent.
- Cost: **$0 per token** — the model runs on your own GPU allocation.
