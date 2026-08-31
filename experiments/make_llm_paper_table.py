"""Build a paper-ready results table from one or more run directories.

Works for both the LLM (Granite) runs and the PPO runs: any directory produced
by llm_market/run_llm_market.py or experiments/ppo.py that contains
config.json + sessions/session_*/session.json.

The table reports, per run, tail-window (converged-play) averages across
sessions, next to the three theory benchmarks (Competitive / Nash-Cournot /
Monopoly) taken from the run's own config.json, so every number in the table
is internally consistent with the network the run was played on.

Usage:
    python experiments/make_llm_paper_table.py RESULTS_DIR [RESULTS_DIR ...] \
        [--tail-frac 0.25] [--save OUT_DIR] [--caption "..."] [--label tab:llm]

Outputs (with --save): OUT_DIR/paper_table.tex and OUT_DIR/paper_table.md.
The Markdown table is always printed to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _run_label(config: dict) -> str:
    """Human-readable row label for a run directory."""
    agent = config.get("agent_type", "")
    model = str(config.get("model", "") or "")
    if "llm" in agent or model:
        name = model.split("/")[-1] if model else "LLM"
        if config.get("ppo_parity", False):
            h = config.get("history_len", 1)
            return f"{name} (parity, H={h})"
        w = config.get("history_window", "?")
        return f"{name} (memory, W={w})"
    return "PPO agents"


def _session_tail_stats(session: dict, tail_frac: float) -> dict | None:
    """Tail-window means of one session's logged metrics."""
    rows = session.get("metrics", [])
    if not rows:
        return None
    k = max(1, int(len(rows) * tail_frac))
    tail = rows[-k:]

    def col(name):
        return np.array([r[name] for r in tail if name in r], dtype=float)

    out = {
        "avg_lmp": float(np.mean(col("avg_lmp"))),
        "delta": float(np.mean(col("delta_combined"))),
    }
    for f in (0, 1, 2):
        g = col(f"firm_{f}_avg_gen")
        p = col(f"firm_{f}_avg_step_profit")
        out[f"gen_{f}"] = float(np.mean(g)) if g.size else float("nan")
        out[f"profit_{f}"] = float(np.mean(p)) if p.size else float("nan")
    out["total_gen"] = float(np.nansum([out[f"gen_{f}"] for f in (0, 1, 2)]))
    out["total_profit"] = float(np.nansum([out[f"profit_{f}"] for f in (0, 1, 2)]))
    return out


def summarize_run(run_dir: Path, tail_frac: float) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    sess_dirs = sorted(
        (run_dir / "sessions").glob("session_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    per_session, stationary = [], []
    for sd in sess_dirs:
        f = sd / "session.json"
        if not f.exists():
            continue
        sess = json.loads(f.read_text())
        stats = _session_tail_stats(sess, tail_frac)
        if stats is not None:
            per_session.append(stats)
            stationary.append(bool(sess.get("converged", False)))
    if not per_session:
        raise SystemExit(f"{run_dir}: no usable sessions/session_*/session.json")

    def agg(key):
        v = np.array([s[key] for s in per_session], dtype=float)
        return float(np.mean(v)), float(np.std(v))

    summary = {"label": _run_label(config), "n_sessions": len(per_session),
               "stationary_fraction": float(np.mean(stationary)),
               "benchmarks": config.get("benchmarks", {})}
    for key in ("avg_lmp", "total_gen", "profit_0", "profit_1",
                "total_profit", "delta"):
        summary[key] = agg(key)
    return summary


def _bench_rows(benchmarks: dict) -> list[dict]:
    rows = []
    for key, label, delta in (
        ("competitive", "Competitive (theory)", None),
        ("cournot_nash", "Nash--Cournot (theory)", 0.0),
        ("monopoly", "Joint monopoly (theory)", 1.0),
    ):
        b = benchmarks.get(key)
        if not b:
            continue
        rows.append({
            "label": label,
            "avg_lmp": b["avg_lmp"],
            "total_gen": b["total_gen"],
            "profit_0": float(b["profits"]["0"]),
            "profit_1": float(b["profits"]["1"]),
            "total_profit": b["total_profit"],
            "delta": delta,
        })
    return rows


def _fmt(mean, std=None, nd=1):
    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def _fmt_md(mean, std=None, nd=1):
    if mean is None:
        return "--"
    if std is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


HEADERS = ["Avg. LMP ($/MWh)", "Total gen. (MW)", "Firm 0 profit ($)",
           "Firm 1 profit ($)", "Total profit ($)", r"$\Delta_{\mathrm{comb}}$"]
KEYS = ["avg_lmp", "total_gen", "profit_0", "profit_1", "total_profit", "delta"]


def build_tables(summaries: list[dict], caption: str, label: str):
    bench = _bench_rows(summaries[0]["benchmarks"])

    md, tex = [], []
    md.append("| Agent / benchmark | " + " | ".join(
        h.replace("$\\Delta_{\\mathrm{comb}}$", "Δ_comb").replace("$", "")
        for h in HEADERS) + " |")
    md.append("|---" * (len(HEADERS) + 1) + "|")

    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{{caption}}}")
    tex.append(rf"\label{{{label}}}")
    tex.append(r"\begin{tabular}{l" + "c" * len(HEADERS) + "}")
    tex.append(r"\toprule")
    tex.append("Agent / benchmark & " + " & ".join(HEADERS) + r" \\")
    tex.append(r"\midrule")

    for r in bench:
        nd = {"delta": 2}
        cells_tex = [_fmt(r[k], nd=nd.get(k, 1)) if r[k] is not None else "--" for k in KEYS]
        cells_md = [_fmt_md(r[k], nd=nd.get(k, 1)) if r[k] is not None else "--" for k in KEYS]
        tex.append(r["label"] + " & " + " & ".join(cells_tex) + r" \\")
        md.append("| " + r["label"].replace("--", "–") + " | " + " | ".join(cells_md) + " |")

    tex.append(r"\midrule")
    for s in summaries:
        nd = {"delta": 2}
        cells_tex = [_fmt(*s[k], nd=nd.get(k, 1)) for k in KEYS]
        cells_md = [_fmt_md(*s[k], nd=nd.get(k, 1)) for k in KEYS]
        note = f" ({s['n_sessions']} sessions)"
        tex.append(s["label"] + note + " & " + " & ".join(cells_tex) + r" \\")
        md.append("| " + s["label"] + note + " | " + " | ".join(cells_md) + " |")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--tail-frac", type=float, default=0.25,
                    help="Final fraction of each session used as the converged-play window.")
    ap.add_argument("--save", type=Path, default=None,
                    help="Directory for paper_table.tex / paper_table.md.")
    ap.add_argument("--caption", type=str,
                    default="Market outcomes of frozen LLM agents (inference only) "
                            "against the theory benchmarks. Mean $\\pm$ s.d. over "
                            "sessions, averaged over the final quarter of each session.")
    ap.add_argument("--label", type=str, default="tab:llm_reference")
    args = ap.parse_args()

    summaries = [summarize_run(d, args.tail_frac) for d in args.run_dirs]
    md, tex = build_tables(summaries, args.caption, args.label)

    print(md)
    for s in summaries:
        print(f"[{s['label']}] stationary sessions: {s['stationary_fraction'] * 100:.0f}%")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        (args.save / "paper_table.md").write_text(md)
        (args.save / "paper_table.tex").write_text(tex)
        print(f"\nSaved -> {args.save}/paper_table.tex and paper_table.md")


if __name__ == "__main__":
    main()
