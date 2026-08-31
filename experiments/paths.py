"""Canonical repo paths for training runs, figures, and archived results."""
from __future__ import annotations

from pathlib import Path

# Primary local run: sessions/, config.json, aggregate.json, deviation_experiment/
DEFAULT_RUN_DIR_NAME = "latest_results"

# Archived cross-history runs (local only; was results/delta_crosshistory)
OLD_RESULTS_ROOT_NAME = "old_results"
ARCHIVE_CROSSHISTORY_DIR_NAME = "delta_crosshistory"

# Gilbreth cluster output roots (optional local mirror under results/ after rsync)
CLUSTER_RESULTS_ROOT_NAME = "results/delta_cont"

# Subdirs under a run directory
DEVIATION_EXPERIMENT_DIR_NAME = "deviation_experiment"
TRAINING_FIGURES_DIR_NAME = "figures"

# CLI aliases → DEFAULT_RUN_DIR_NAME
_RUN_DIR_ALIASES = {
    "h1": DEFAULT_RUN_DIR_NAME,
    "results/delta_cont/h1": DEFAULT_RUN_DIR_NAME,
    "results/delta/h1": DEFAULT_RUN_DIR_NAME,
    "figures/h1": DEFAULT_RUN_DIR_NAME,  # legacy plot folder name
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


def archive_crosshistory_dir() -> Path:
    """``old_results/delta_crosshistory`` (H=1,2,3 episodic cross-history archive)."""
    return repo_root() / OLD_RESULTS_ROOT_NAME / ARCHIVE_CROSSHISTORY_DIR_NAME


def archive_crosshistory_run(h: int | str) -> Path:
    return archive_crosshistory_dir() / f"h{h}"


def training_figures_dir(run_dir: Path | str | None = None) -> Path:
    """Training convergence / Calvano paper plots for a run."""
    return resolve_run_dir(run_dir) / TRAINING_FIGURES_DIR_NAME


def deviation_figures_dir(run_dir: Path | str | None = None) -> Path:
    """Calvano / static-BR impulse plots."""
    return resolve_run_dir(run_dir) / DEVIATION_EXPERIMENT_DIR_NAME


def stochastic_deviation_output_dir(run_dir: Path | str | None = None) -> Path:
    """Default output for ``stochastic_deviation.py`` rollouts."""
    resolved = resolve_run_dir(run_dir)
    if resolved.name == DEFAULT_RUN_DIR_NAME:
        return deviation_figures_dir(resolved) / "stochastic_deviation"
    root = repo_root()
    for base in (root / OLD_RESULTS_ROOT_NAME, root / "results"):
        try:
            rel = resolved.relative_to(base)
            return base / "figures" / rel / "stochastic_deviation"
        except ValueError:
            continue
    return resolved / "stochastic_deviation"
