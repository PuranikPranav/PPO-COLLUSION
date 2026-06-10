"""Canonical repo paths for training runs and deviation-experiment figures."""
from __future__ import annotations

from pathlib import Path

# Training run: sessions/, config.json, aggregate.json, *.pt checkpoints
DEFAULT_RUN_DIR_NAME = "latest_results"

# Plots from plot_calvano_proof, impulse_response, plot_top4_cartels, etc.
DEVIATION_EXPERIMENT_DIR_NAME = "deviation_experiment"

# CLI aliases for renamed folders (local layout vs cluster mirror)
_RUN_DIR_ALIASES = {
    "h1": DEFAULT_RUN_DIR_NAME,
    "results/delta_cont/h1": DEFAULT_RUN_DIR_NAME,
    "results/delta/h1": DEFAULT_RUN_DIR_NAME,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_run_dir_key(run_dir: Path | str) -> str:
    p = Path(run_dir)
    parts = [x for x in p.parts if x not in (".", "")]
    if not parts:
        return DEFAULT_RUN_DIR_NAME
    if len(parts) >= 2:
        tail = "/".join(parts[-2:])
        if tail in _RUN_DIR_ALIASES:
            return _RUN_DIR_ALIASES[tail]
    name = parts[-1]
    if name in _RUN_DIR_ALIASES:
        return _RUN_DIR_ALIASES[name]
    return str(p)


def resolve_run_dir(run_dir: Path | str | None = None) -> Path:
    """Resolve a run directory relative to repo root; default ``latest_results``."""
    root = repo_root()
    if run_dir is None:
        return root / DEFAULT_RUN_DIR_NAME
    key = _normalize_run_dir_key(run_dir)
    if key == DEFAULT_RUN_DIR_NAME:
        return root / DEFAULT_RUN_DIR_NAME
    p = Path(run_dir)
    if not p.is_absolute():
        p = root / p
    return p


def deviation_figures_dir(run_dir: Path | str) -> Path:
    """Output directory for Calvano / static-BR impulse plots."""
    return resolve_run_dir(run_dir) / DEVIATION_EXPERIMENT_DIR_NAME


def stochastic_deviation_output_dir(run_dir: Path | str) -> Path:
    """Default output for ``stochastic_deviation.py`` rollouts."""
    resolved = resolve_run_dir(run_dir)
    if resolved.name == DEFAULT_RUN_DIR_NAME:
        return resolved / DEVIATION_EXPERIMENT_DIR_NAME / "stochastic_deviation"
    try:
        rel = resolved.relative_to(repo_root() / "results")
        return repo_root() / "figures" / rel / "stochastic_deviation"
    except ValueError:
        return resolved / "stochastic_deviation"
