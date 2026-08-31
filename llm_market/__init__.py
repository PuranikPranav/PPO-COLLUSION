"""
LLM-agent ecosystem for the electricity-market collusion study.

Each firm is driven by an open-source IBM Granite model instead of PPO. The
four-stage pipeline mirrors the project design:

    Step 1  State Translator   raw market data  -> text          (state_translator.py)
    Step 2  Memory & Prompt     sliding window   -> chat messages (prompt_builder.py)
    Step 3  Reasoning Engine    Granite (vLLM)   -> rationale+JSON (granite_engine.py)
    Step 4  Action Parser       JSON             -> MW per plant  (action_parser.py)

The same `ElectricityMarketEnv` (DC-OPF clearing) and benchmarks used by the PPO
experiments are reused, so the LLM results are directly comparable and can be
plotted with the existing `experiments/plot_results.py`.
"""
