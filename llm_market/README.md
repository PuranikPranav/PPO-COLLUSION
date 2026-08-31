# LLM-Agent Electricity Market (IBM Granite)

Each firm is driven by an open-source **IBM Granite** model instead of PPO. The agents
play the **same** `ElectricityMarketEnv` (DC-OPF clearing) and are scored with the
**same** benchmarks and Δ metric as the PPO experiments, so results are directly
comparable and plot with the existing `experiments/plot_results.py`.

## No training loop — collusion is elicited *in context*

Granite is **frozen**: there is no gradient training, so there is no PPO-style learning
curve. That is also why a memoryless agent cannot collude — a frozen model fed only a
single-period snapshot has neither channel (it can't *learn* the equilibrium and can't
*reason over history*), so it plays myopically near Nash. Collusion from a frozen LLM is
elicited by giving it (a) **memory** of the joint history, (b) a repeated-game framing
that *permits but never instructs* reward/punishment, and (c) **enough rounds** for a
tacit norm to stabilise. The x-axis is **rounds of the repeated game**; the headline
number is the **stationary Δ** (last-half average) across seeds.

### Three experimental arms

| Arm | How to run | Role |
|-----|-----------|------|
| **Memory treatment** | `MODE=memory` (`--legacy-memory`) | Narrative window of own+rival quantities/price/profit + a carry-forward **strategy note** the model writes in words. The collusion treatment + direct qualitative evidence. |
| **High-H parity** | `MODE=parity HISTORY_LEN=8` | The strict 19×H observation vector with H>1, so the obs itself carries a multi-period window. Directly comparable to PPO at the same H. |
| **Memoryless baseline** | `MODE=parity HISTORY_LEN=1` | The deliberately-crippled control: same state, one period back. Expected to stay near Nash. The treatment-vs-baseline contrast is itself a result. |

**`MODE=all` (default) runs all three arms in one job** → `memory/`, `parity_h8/`,
`parity_h1/` under the results base, each with its own figures, plus the combined
paper table. Every session also writes `sessions/session_*/transcripts.jsonl` —
the model's per-period `reasoning`/`strategy` text (grep it for the smoking-gun
quotes; disable with `--no-transcripts`).

### Comparability with the PPO runs

- Metrics rows log **instantaneous per-period values** (no within-session
  averaging); the plot layer averages **across sessions** per step — identical to
  the PPO convention.
- Every session's metrics start with the **t=0 competitive anchor row** (same as
  `ppo.py`), so all figures begin at the competitive benchmark.
- The limit-strategy LMP sweep is derived from the run's own benchmarks
  (competitive→monopoly, padded), so it tracks whatever network parameters are in
  `iso_market/node_network.py`.

### Punishment / impulse-response (the retaliation figure)

After each session the driver runs `run_deviation_experiment_llm` (in `llm_dynamics.py`),
the LLM analogue of `experiments/ppo.run_deviation_experiment`: warm up to a resting
point, **force one firm to over-produce for a single period**, then let both play their
normal LLM policy and record how the rival reacts. A rival that cuts output / lets price
fall is exhibiting a trigger (reward-punishment) strategy. This populates
`deviation_experiment` so `plot_results.py --deviation-explainer` produces the punishment
figure (previously the field was always empty). Tunables: `DEV_FRAC`, `DEV_WARMUP`,
`DEV_HORIZON`, `DEV_SESSIONS`. `LIMIT_STRATEGY=1` additionally records each firm's
reaction function (parity mode only).

## The four-stage pipeline

| Step | File | Role |
|------|------|------|
| 1. State Translator | `state_translator.py` | Raw market data (LMPs, generation, profit) → text |
| 2. Memory & Prompt | `prompt_builder.py` | Sliding-window history → chat messages |
| 3. Reasoning Engine | `granite_engine.py` | Granite (vLLM/transformers/mock) → rationale + JSON |
| 4. Action Parser | `action_parser.py` | JSON → validated MW per plant (clipped to capacity) |
| 5. Dynamics & experiments | `llm_dynamics.py` | One-period play + punishment/impulse-response + limit strategy |

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

## Recommended first submit (pilot → full)

Granite is **frozen** — there is no training. Collusion (if any) emerges *in context*
over repeated rounds. A short competitive warmup helps the Δ plot start cleanly; the
real “trial phase” is just enough periods for a tacit resting point, then the
**forced-deviation** test asks whether the rival punishes a cheat.

**1. Pilot first** (memory treatment only, 3×80 rounds + deviation figure):
```bash
cd ~/ppo-collusion && git pull
MODE=pilot sbatch -J llm-pilot run_gilbreth_llm.sh
```
Outputs: `results/llm_granite8/memory/`, `figures/llm_granite8/memory/`
(deviation explainer PNGs + `sessions/*/transcripts.jsonl`).

**2. Full paper package** once the pilot looks sane:
```bash
MODE=all SESSIONS=20 PERIODS=300 sbatch -J llm-granite8 run_gilbreth_llm.sh
```

**3. Optional unilateral-collusion probe** — answers “what happens if ONE agent
colludes?” Firm 0 gets an explicitly collusive **price-leader** prompt
(`collude`: restrain output, hold the price high, wait for the rival to match);
firm 1 keeps the ordinary self-interested prompt, so anything firm 1 does in
response — following the leader up, or free-riding on the high price — is
**emergent, not instructed**:
```bash
MODE=pilot GOALS=collude,own_profit \
  RESULTS_BASE=results/llm_granite8_asym FIGURES_BASE=figures/llm_granite8_asym \
  sbatch -J llm-asym run_gilbreth_llm.sh
```
`GOALS=own_profit,joint_profit` is the softer variant (firm 1 told to maximize
industry profit, no explicit leadership framing).

**Trial phase (recommended):** add `GOALS_START=40` and both firms play the normal
selfish prompt for the first 40 rounds, then firm 0’s collusion prompt switches on.
The figures then show a within-session before/after contrast around the moment one
agent starts colluding — cleaner causal evidence than comparing across runs:
```bash
MODE=pilot GOALS=collude,own_profit GOALS_START=40 PERIODS=160 \
  RESULTS_BASE=results/llm_granite8_asym FIGURES_BASE=figures/llm_granite8_asym \
  sbatch -J llm-asym run_gilbreth_llm.sh
```
What to read off this run: firm 1’s generation/profit after period 40 (does it
restrain toward the leader → tacit collusion, or expand into the higher price →
free-riding), the LMP path, and firm 1’s `reasoning` in
`sessions/session_*/transcripts.jsonl` (does it *notice* the restraint?). The
post-hoc deviation/limit probes run with the asymmetric goals fully active.

### What to look at for collusion

| Evidence | Where |
|----------|--------|
| Stationary Δ above Nash | Calvano Δ plot / `aggregate.json` |
| Restraint in quantities / high LMP | gen + LMP figures |
| Rival floods after a forced cheat | `--deviation-explainer` figures |
| Model’s own words (“match rival”, “punish”) | `sessions/session_*/transcripts.jsonl` |

### Recommended publishable run (all three arms, ONE submission)

```bash
# memory/ + parity_h8/ + parity_h1/ + figures + combined paper table:
MODE=all SESSIONS=20 PERIODS=300 sbatch -J llm-granite8 run_gilbreth_llm.sh
```

Or submit the arms as separate jobs (e.g. to parallelize across GPUs):

```bash
MODE=memory  SESSIONS=20 PERIODS=300 WINDOW=10  sbatch -J llm-mem run_gilbreth_llm.sh
MODE=parity  HISTORY_LEN=8 SESSIONS=20 PERIODS=300 LIMIT_STRATEGY=1 \
    sbatch -J llm-h8 run_gilbreth_llm.sh
MODE=parity  HISTORY_LEN=1 SESSIONS=20 PERIODS=300 sbatch -J llm-h1 run_gilbreth_llm.sh
```
(When splitting into separate jobs, set a distinct `RESULTS_BASE`/`FIGURES_BASE`
per job or the stale-results archiving will shuffle the previous arm away.)

### Model / hardware choices (set via env vars)

| Config | Fits on | Command |
|--------|---------|---------|
| **8B, bf16 (default)** | 1× A100-40GB | `sbatch -J llm-granite8 run_gilbreth_llm.sh` |
| 2B, bf16 (fastest) | 1× A100-40GB | `MODEL=ibm-granite/granite-3.3-2b-instruct sbatch run_gilbreth_llm.sh` |

The default is `ibm-granite/granite-3.3-8b-instruct` — a public, dense Granite model
that is well-supported by vLLM 0.11 and fits on one A100-40GB at full (bf16) precision.

Overridable env vars: `MODE` (pilot|memory|parity|both|all), `HISTORY_LEN`, `WINDOW`,
`GOAL`, `GOALS` (optional per-firm, e.g. `collude,own_profit`), `GOALS_START`
(period at which `GOALS` switches on; before that both play `GOAL`), `SESSIONS`,
`PERIODS`, `WARMUP_COMP`, `TEMPERATURE`, `MAXTOK`, `SEED`, `DEV_FRAC`, `DEV_WARMUP`,
`DEV_HORIZON`, `DEV_SESSIONS`, `LIMIT_STRATEGY`, `RESULTS_BASE`, `FIGURES_BASE`.

> **Independent sessions:** each session now uses a distinct base seed threaded into the
> engine sampler (`engine.chat(..., seed=...)`), so the cross-session error bars are real.
> Previously the engine was built once with a fixed seed and every "session" sampled
> identically.

## Plotting (identical to the PPO runs)

```bash
python experiments/plot_results.py results/llm_granite8/memory --calvano-paper       --save figures/llm/
python experiments/plot_results.py results/llm_granite8/memory --per-firm-profit     --save figures/llm/
python experiments/plot_results.py results/llm_granite8/memory --variance-funnel     --save figures/llm/
python experiments/plot_results.py results/llm_granite8/memory --deviation-explainer --save figures/llm/  # punishment figure
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
