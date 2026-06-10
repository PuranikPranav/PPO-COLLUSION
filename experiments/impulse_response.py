"""
Calvano Fig. 4 impulse response — Firm 1 one-period quantity cheat, Firm 0 on policy.

Protocol
--------
  Warmup + t = -2, -1 : both play frozen greedy policy (long-run collusion)
  t = 0               : Firm 0 = greedy (unchanged strategy)
                        Firm 1 = manual deviation ONLY this period
  t ≥ 1               : both released to greedy policy (free interaction)

Deviation modes (--deviation-mode)
--------------------------------
  cap        : Firm 1 → plant capacity (default; largest one-period Cournot cheat)
  frac       : Firm 1 → greedy MW × (1 + --deviation-frac)
  static_br  : Firm 1 → myopic profit-max MW given Firm 0's greedy output (Calvano BR analog)

Calvano Fig. 4 plot (Cournot / LMP mapping)
-------------------------------------------
  Deviator (Firm 1)     : actual nodal avg LMP each period
  Nondeviator (Firm 0)  : long-run nodal LMP at t ≤ 0 (strategy unchanged at cheat),
                          then actual nodal LMP from t ≥ 1 (market reaction / punishment)

Usage: python experiments/impulse_response.py --run-dir latest_results
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_mpl_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ppo-collusion-matplotlib-cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import ElectricityMarketEnv, NUM_FIRMS, FIRM_PLANT_IDX, PLANTS
from experiments.paths import DEFAULT_RUN_DIR_NAME, deviation_figures_dir, resolve_run_dir
from experiments.stochastic_deviation import load_session_agents, load_or_warm_normalizers

DEVIATOR_FID = 1
NONDEVIATOR_FID = 0


def firm_avg_nodal_lmp(lmps, fid: int) -> float:
    nodes = [PLANTS[pidx]["node"] for pidx in FIRM_PLANT_IDX[fid]]
    return float(np.mean([lmps[n] for n in nodes]))


def firm_total_mw(actions, fid: int) -> float:
    return float(np.sum(actions[fid]))


def greedy_actions(obs, agents, normalizers) -> dict:
    actions = {}
    for fid, agent in agents.items():
        normalizers[fid].update(obs[fid])
        obs_norm = normalizers[fid].normalize(obs[fid])
        actions[fid] = agent.deterministic_action(obs_norm)
    return actions


def market_step(env, obs, actions):
    obs_next, rewards, done, info = env.step(actions)
    if done and info.get("error"):
        obs_next = env.reset()
    return obs_next, rewards, done, info


def profit_firm(info, actions, fid: int) -> float:
    lmps = info["lmps"]
    p = 0.0
    for j, pidx in enumerate(FIRM_PLANT_IDX[fid]):
        plant = PLANTS[pidx]
        g = float(actions[fid][j])
        p += lmps[plant["node"]] * g - plant["mc"] * g - 0.5 * plant["qc"] * g * g
    return p


def apply_deviation(actions_policy: dict, mode: str, frac: float, env) -> dict:
    """Manual Firm 1 deviation at t=0; Firm 0 keeps policy output."""
    out = {fid: actions_policy[fid].copy() for fid in range(NUM_FIRMS)}
    if mode == "cap":
        for j, pidx in enumerate(FIRM_PLANT_IDX[DEVIATOR_FID]):
            out[DEVIATOR_FID][j] = PLANTS[pidx]["cap"]
    elif mode == "frac":
        bumped = out[DEVIATOR_FID] * (1.0 + frac)
        for j, pidx in enumerate(FIRM_PLANT_IDX[DEVIATOR_FID]):
            bumped[j] = min(bumped[j], PLANTS[pidx]["cap"])
        out[DEVIATOR_FID] = bumped
    elif mode == "static_br":
        out[DEVIATOR_FID] = _static_best_response_mw(out[NONDEVIATOR_FID], env)
    else:
        raise ValueError(f"Unknown deviation mode: {mode}")
    return out


def _static_best_response_mw(f0_mw: np.ndarray, env) -> np.ndarray:
    """Grid search F1 MW to maximize profit given fixed F0 (one-period BR)."""
    caps = np.array([PLANTS[pidx]["cap"] for pidx in FIRM_PLANT_IDX[DEVIATOR_FID]])
    best_mw = caps.copy()
    best_pi = -np.inf
    grid = np.linspace(0.0, 1.0, 41)
    for frac in grid:
        trial = frac * caps
        gen_node = np.zeros(5)
        for j, pidx in enumerate(FIRM_PLANT_IDX[NONDEVIATOR_FID]):
            gen_node[PLANTS[pidx]["node"]] += float(f0_mw[j])
        for j, pidx in enumerate(FIRM_PLANT_IDX[DEVIATOR_FID]):
            gen_node[PLANTS[pidx]["node"]] += float(trial[j])
        lmps, _, _, _ = env._clear_market(gen_node)
        if lmps is None:
            continue
        acts = {NONDEVIATOR_FID: f0_mw, DEVIATOR_FID: trial}
        info = {"lmps": lmps}
        pi = profit_firm(info, acts, DEVIATOR_FID)
        if pi > best_pi:
            best_pi = pi
            best_mw = trial.copy()
    return best_mw


def run_firm1_impulse(
    env,
    agents,
    normalizers,
    *,
    deviation_mode: str = "cap",
    deviation_frac: float = 0.2,
    warmup: int = 20,
    horizon: int = 15,
):
    np.random.seed(43)
    obs = env.reset()

    for _ in range(warmup):
        actions = greedy_actions(obs, agents, normalizers)
        obs, _, _, _ = market_step(env, obs, actions)

    trace = {
        "lmp_dev_actual": [],
        "lmp_non_actual": [],
        "lmp_dev_plot": [],
        "lmp_non_plot": [],
        "lmp_sys": [],
        "gen_f0": [],
        "gen_f1": [],
    }

    def record_actual(info, actions):
        trace["lmp_dev_actual"].append(firm_avg_nodal_lmp(info["lmps"], DEVIATOR_FID))
        trace["lmp_non_actual"].append(firm_avg_nodal_lmp(info["lmps"], NONDEVIATOR_FID))
        trace["lmp_sys"].append(float(info["avg_lmp"]))
        trace["gen_f0"].append(firm_total_mw(actions, NONDEVIATOR_FID))
        trace["gen_f1"].append(firm_total_mw(actions, DEVIATOR_FID))

    # t = -2, -1
    for _ in (-2, -1):
        actions = greedy_actions(obs, agents, normalizers)
        obs, _, _, info = market_step(env, obs, actions)
        record_actual(info, actions)

    resting_lmp_dev = trace["lmp_dev_actual"][-1]
    resting_lmp_non = trace["lmp_non_actual"][-1]
    resting_gen_f0 = trace["gen_f0"][-1]

    # t = 0 — F0 policy, F1 manual deviation
    actions_policy = greedy_actions(obs, agents, normalizers)
    actions_exec = apply_deviation(actions_policy, deviation_mode, deviation_frac, env)
    obs, _, _, info = market_step(env, obs, actions_exec)
    record_actual(info, actions_exec)

    # t = 1 … horizon — both on policy
    for _ in range(1, horizon + 1):
        actions = greedy_actions(obs, agents, normalizers)
        obs, _, _, info = market_step(env, obs, actions)
        record_actual(info, actions)

    # Calvano Fig. 4 mapping (Bertrand price → Cournot nodal LMP):
    #   Deviator: actual LMP throughout
    #   Nondeviator: long-run at t≤1 (no price cut yet), actual from t≥2 (punishment lag)
    n_pts = len(trace["lmp_dev_actual"])
    for i in range(n_pts):
        rel_t = i - 2
        trace["lmp_dev_plot"].append(trace["lmp_dev_actual"][i])
        if rel_t <= 1:
            trace["lmp_non_plot"].append(resting_lmp_non)
        else:
            trace["lmp_non_plot"].append(trace["lmp_non_actual"][i])

    for key in trace:
        trace[key] = np.array(trace[key], dtype=float)

    trace["resting_lmp_dev"] = resting_lmp_dev
    trace["resting_lmp_non"] = resting_lmp_non
    trace["resting_gen_f0"] = resting_gen_f0
    return trace


def classify_firm0(trace, thresh=0.02):
    resting = trace["resting_gen_f0"]
    gen_t1 = trace["gen_f0"][3]
    rel = (gen_t1 - resting) / resting if resting > 1e-6 else 0.0
    if rel > thresh:
        return "A"
    if rel < -thresh:
        return "C"
    return "B"


def plot_fig4(traces, benchmarks, *, deviation_mode, deviation_frac, case_counts, save_path):
    n = len(traces)
    horizon = traces[0]["lmp_sys"].shape[0] - 3
    t_axis = np.arange(-2, horizon + 1)

    def ms(key):
        m = np.array([tr[key] for tr in traces])
        return m.mean(0), m.std(0)

    dev_m, dev_s = ms("lmp_dev_plot")
    non_m, non_s = ms("lmp_non_plot")
    lr_dev = np.mean([tr["resting_lmp_dev"] for tr in traces])
    lr_non = np.mean([tr["resting_lmp_non"] for tr in traces])

    bench = benchmarks or {}
    nash = bench.get("cournot_nash", {}).get("avg_lmp")
    mono = bench.get("monopoly", {}).get("avg_lmp")

    dev_label = {
        "cap": "Firm 1 → capacity",
        "frac": f"Firm 1 → +{int(deviation_frac * 100)}%",
        "static_br": "Firm 1 → static BR quantity",
    }.get(deviation_mode, deviation_mode)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        t_axis, dev_m, color="#c0392b", marker="s", markevery=2, ms=6, lw=2.5,
        label="Deviating agent (Firm 1 nodal avg LMP)",
    )
    ax.fill_between(t_axis, dev_m - dev_s, dev_m + dev_s, color="#c0392b", alpha=0.12)
    ax.plot(
        t_axis, non_m, color="#2980b9", marker="^", markevery=2, ms=6, lw=2.5, ls="--",
        label="Nondeviating agent (F0; long-run until t=1, then reacts)",
    )
    ax.fill_between(t_axis, non_m - non_s, non_m + non_s, color="#2980b9", alpha=0.10)

    ax.axhline(lr_non, color="0.45", ls="-", lw=1.2, alpha=0.75,
               label=f"Long-run LMP (${lr_non:.2f})")
    if nash is not None:
        ax.axhline(float(nash), color="grey", ls=":", lw=1.0, alpha=0.65, label="Nash avg LMP")
    if mono is not None:
        ax.axhline(float(mono), color="grey", ls="-.", lw=1.0, alpha=0.65, label="Monopoly avg LMP")
    ax.axvline(0, color="red", ls=":", lw=1.8, alpha=0.8, label="F1 deviation (t=0)")

    a, b, c = case_counts["A"], case_counts["B"], case_counts["C"]
    ax.set_title(
        f"Fig. 4 style — {dev_label}\n"
        f"F0 at t=1: retaliate {a/n*100:.0f}% | ignore {b/n*100:.0f}% | accommodate {c/n*100:.0f}%",
        fontsize=11,
    )
    ax.set_xlabel("Time (periods relative to deviation)")
    ax.set_ylabel("Average LMP ($/MWh)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)
    ax.set_xticks([x for x in (-2, -1, 0, 1, 2, 5, 10, 15) if x <= t_axis[-1]])

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Calvano Fig.4 — F1 deviation, F0 response")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"Training run directory (default: {DEFAULT_RUN_DIR_NAME}/)",
    )
    parser.add_argument(
        "--deviation-mode",
        choices=("cap", "frac", "static_br"),
        default="cap",
        help="Firm 1 cheat at t=0 (default cap = max MW)",
    )
    parser.add_argument("--deviation-frac", type=float, default=0.2,
                        help="Used when --deviation-mode=frac")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    fig_dir = deviation_figures_dir(run_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sessions_root = run_dir / "sessions"
    session_dirs = sorted(
        d for d in sessions_root.iterdir()
        if d.is_dir() and (d / "agent_0.pt").exists()
    )
    if args.max_sessions:
        session_dirs = session_dirs[: args.max_sessions]

    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    benchmarks = config.get("benchmarks", {})

    traces = []
    case_counts = {"A": 0, "B": 0, "C": 0}

    print(f"Deviation: {args.deviation_mode} | sessions: {len(session_dirs)}")

    for s_dir in session_dirs:
        env = ElectricityMarketEnv(
            history_len=int(config.get("history_len", 1)),
            episode_len=int(config.get("episode_len", 168)),
        )
        agents = load_session_agents(s_dir, env)
        normalizers = load_or_warm_normalizers(s_dir, env, agents, warmup_steps=500)
        tr = run_firm1_impulse(
            env, agents, normalizers,
            deviation_mode=args.deviation_mode,
            deviation_frac=args.deviation_frac,
            warmup=args.warmup,
            horizon=args.horizon,
        )
        traces.append(tr)
        case_counts[classify_firm0(tr)] += 1

    out = args.save or (fig_dir / "calvano_fig4_f1_deviation.png")
    plot_fig4(
        traces, benchmarks,
        deviation_mode=args.deviation_mode,
        deviation_frac=args.deviation_frac,
        case_counts=case_counts,
        save_path=out,
    )
    legacy = fig_dir / "clean_lmp_impulse.png"
    if out != legacy:
        plot_fig4(
            traces, benchmarks,
            deviation_mode=args.deviation_mode,
            deviation_frac=args.deviation_frac,
            case_counts=case_counts,
            save_path=legacy,
        )


if __name__ == "__main__":
    main()
