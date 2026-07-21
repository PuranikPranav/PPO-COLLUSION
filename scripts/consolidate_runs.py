"""Merge sessions from multiple run dirs (e.g. SLURM array tasks) into one
run dir so plot_results.py averages across ALL sessions.

Usage:
    python scripts/consolidate_runs.py <out_dir> <run_dir> [<run_dir> ...]

Example (after a SLURM array run):
    python scripts/consolidate_runs.py results/gp_twofirm/merged results/gp_twofirm/task_*
"""
import json
import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    run_dirs = [Path(p) for p in sys.argv[2:]]
    (out / "sessions").mkdir(parents=True, exist_ok=True)

    config_written = False
    i = 0
    for rd in run_dirs:
        cfg = rd / "config.json"
        if not cfg.exists():
            print(f"skip {rd} (no config.json)")
            continue
        if not config_written:
            shutil.copy(cfg, out / "config.json")
            config_written = True
        for sd in sorted((rd / "sessions").glob("session_*")):
            sj = sd / "session.json"
            if not sj.exists():
                print(f"skip {sd} (no session.json)")
                continue
            dst = out / "sessions" / f"session_{i}"
            dst.mkdir(exist_ok=True)
            for f in sd.iterdir():
                shutil.copy(f, dst / f.name)
            i += 1
    # quick summary
    deltas = []
    for sj in sorted((out / "sessions").glob("session_*/session.json")):
        s = json.loads(sj.read_text())
        tail = s["metrics"][-60:]
        deltas.append(sum(r["delta_combined"] for r in tail) / max(1, len(tail)))
    print(f"consolidated {i} sessions -> {out}")
    if deltas:
        import statistics
        print(f"tail-averaged Δ: mean {statistics.mean(deltas):+.3f}, "
              f"per-session {[round(d, 3) for d in deltas]}")


if __name__ == "__main__":
    main()
