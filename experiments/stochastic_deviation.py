"""
Stochastic deviation experiment for trained PPO collusion agents.

Motivation (advisor meeting, 2026-05-19)
---------------------------------------
The original one-shot deviation analysis was "not very convincing." The advisor
proposed: at every period, with small probability p (~1e-3) force one of the
firms to deviate from its PPO-learned policy by multiplying its action by a
random factor X drawn from a small set (e.g. {1.5, 2, 3}). Run for many steps
and study (i) the long-run behavior with rare shocks, and (ii) the average
impulse response across many naturally occurring deviation events.

Assumptions
-----------
* Trained policies are frozen — both agents act on the Beta policy mean (greedy).
* Rival learns nothing about the deviation directly: it only observes nodal
  LMPs through the history window (imperfect monitoring).
* On a deviation step the deviator's MW per plant is set to
      X * deterministic_policy_output_this_step
  clipped to [0, plant_cap].
* The observation normalizer is not saved by train_session; we re-warm it on
  the trained policies for `--warmup-steps` before logging begins.

Outputs (per session, under --output-dir)
-----------------------------------------
* timeseries.json   — full per-step log (gen, lmp, profit, deviation flags)
* timeseries.png    — three-row plot: generation / avg LMP / profit
* event_study.png   — mean ± std response in a window around deviation events
* summary.json      — aggregated statistics (events per firm, mean impact, …)

Default output dir: ``latest_results/deviation_experiment/stochastic_deviation/``
(for runs under ``old_results/`` or ``results/``, defaults to ``<root>/figures/<subpath>/stochastic_deviation/``).

CLI
---
    python experiments/stochastic_deviation.py \
        --run-dir latest_results \
        --sessions 0,1,2 \
        --num-steps 10000 \
        --deviation-prob 1e-3 \
        --x-values 1.5,2,3 \
        --deviator random \
        --warmup-steps 500 \
        --seed 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_mpl_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ppo-collusion-matplotlib-cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iso_market.market_env import (
    ElectricityMarketEnv,
    NUM_FIRMS,
    NUM_NODES,
    FIRM_PLANT_IDX,
    PLANTS,
)
from experiments.paths import DEFAULT_RUN_DIR_NAME, resolve_run_dir, stochastic_deviation_output_dir
from experiments.ppo import PPOAgent, RunningNormalizer


# ----------------------------------------------------------------------
# Loading trained agents
# ----------------------------------------------------------------------
def load_session_agents(session_dir: Path, env: ElectricityMarketEnv,
                        hidden: int = 64, device: str = "cpu") -> dict:
    """Load agent_0.pt / agent_1.pt from a saved session directory."""
    agents = {}
    for fid in range(NUM_FIRMS):
        weight_path = session_dir / f"agent_{fid}.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing weights: {weight_path}")
        caps = np.array(
            [PLANTS[i]["cap"] for i in FIRM_PLANT_IDX[fid]], dtype=np.float32
        )
        agent = PPOAgent(
            agent_id=fid,
            obs_dim=env.obs_dim,
            act_dim=env.action_dims[fid],
            caps=caps,
            hidden=hidden,
            lr=3e-4,
            rollout_len=1,
            device=device,
        )
        state = torch.load(weight_path, map_location=device, weights_only=True)
        agent.ac.load_state_dict(state)
        agent.ac.eval()
        agents[fid] = agent
    return agents


def warmup_normalizers(env: ElectricityMarketEnv, agents: dict,
                       normalizers: dict, n_steps: int) -> dict:
    """Roll deterministic policies forward to warm Welford running stats."""
    obs = env.reset()
    for _ in range(n_steps):
        actions = {}
        for fid, agent in agents.items():
            normalizers[fid].update(obs[fid])
            obs_norm = normalizers[fid].normalize(obs[fid])
            actions[fid] = agent.deterministic_action(obs_norm)
        obs, _rewards, done, info = env.step(actions)
        if done and info.get("error"):
            obs = env.reset()
    return obs


def load_or_warm_normalizers(
    session_dir: Path,
    env: ElectricityMarketEnv,
    agents: dict,
    warmup_steps: int,
) -> dict:
    """
    Prefer normalizer_<fid>.json saved at the end of training (bit-exact
    reproduction of the trained obs-distribution). Fall back to running a
    `warmup_steps` deterministic-policy rollout when the JSON file is missing
    (e.g. for older sessions trained before the normalizer-save patch).
    """
    normalizers = {fid: RunningNormalizer(env.obs_dim) for fid in range(NUM_FIRMS)}

    all_present = all(
        (session_dir / f"normalizer_{fid}.json").exists() for fid in range(NUM_FIRMS)
    )
    if all_present:
        for fid in range(NUM_FIRMS):
            with open(session_dir / f"normalizer_{fid}.json") as f:
                normalizers[fid].load_state_dict(json.load(f))
        print(f"  loaded saved normalizers from {session_dir.name}/")
        return normalizers

    warmup_normalizers(env, agents, normalizers, n_steps=warmup_steps)
    print(
        f"  no normalizer_<fid>.json in {session_dir.name}/ — "
        f"warmed up via {warmup_steps} deterministic steps"
    )
    return normalizers


# ----------------------------------------------------------------------
# Single rollout
# ----------------------------------------------------------------------
def run_stochastic_rollout(env: ElectricityMarketEnv, agents: dict,
                           normalizers: dict, *, num_steps: int,
                           deviation_prob: float, x_sampler, deviator_mode: str,
                           rng: np.random.Generator) -> dict:
    """
    Run `num_steps` of stochastic-deviation simulation. Returns dict of per-step
    arrays. Deterministic policy at every step except when a deviation event
    fires, in which case the deviator's plants are scaled by X.

    deviator_mode:
        "random"   — when an event fires, pick deviator uniformly
        "0" / "1"  — fixed deviator (events only fire for that firm)
        "each"     — each firm independently considered each step
    """
    obs = env.reset()
    # Resume warmed-up obs distribution by stepping deterministic once
    # (caller already warmed normalizers; env.reset() above just zeroes the t).

    log = {
        "gen_firm": {fid: np.zeros(num_steps) for fid in range(NUM_FIRMS)},
        "gen_plant": {pidx: np.zeros(num_steps) for pidx in range(len(PLANTS))},
        "profit_firm": {fid: np.zeros(num_steps) for fid in range(NUM_FIRMS)},
        "lmps": np.zeros((num_steps, NUM_NODES)),
        "avg_lmp": np.zeros(num_steps),
        "is_deviation": np.zeros(num_steps, dtype=bool),
        "deviator": -np.ones(num_steps, dtype=int),
        "x_used": np.zeros(num_steps),
        "policy_action_mw_firm": {fid: [] for fid in range(NUM_FIRMS)},
    }

    for t in range(num_steps):
        actions_policy = {}
        for fid, agent in agents.items():
            normalizers[fid].update(obs[fid])
            obs_norm = normalizers[fid].normalize(obs[fid])
            actions_policy[fid] = agent.deterministic_action(obs_norm).astype(np.float64)
            log["policy_action_mw_firm"][fid].append(actions_policy[fid].copy())

        actions_exec = {fid: a.copy() for fid, a in actions_policy.items()}

        # Decide who (if anyone) deviates this step
        deviators_this_step = []
        if deviator_mode == "each":
            for fid in range(NUM_FIRMS):
                if rng.random() < deviation_prob:
                    deviators_this_step.append(fid)
        else:
            if rng.random() < deviation_prob:
                if deviator_mode == "random":
                    deviators_this_step.append(int(rng.integers(0, NUM_FIRMS)))
                elif deviator_mode in ("0", "1"):
                    deviators_this_step.append(int(deviator_mode))

        # Apply multiplicative deviation
        if deviators_this_step:
            # Multi-firm-event: log only the first deviator (rare event by design).
            primary = deviators_this_step[0]
            x = float(x_sampler(rng))
            for fid in deviators_this_step:
                caps = agents[fid].caps
                actions_exec[fid] = np.minimum(actions_exec[fid] * x, caps)
            log["is_deviation"][t] = True
            log["deviator"][t] = primary
            log["x_used"][t] = x

        obs_next, rewards, done, info = env.step(actions_exec)

        log["lmps"][t] = info["lmps"]
        log["avg_lmp"][t] = info["avg_lmp"]
        for fid in range(NUM_FIRMS):
            log["gen_firm"][fid][t] = float(np.sum(actions_exec[fid]))
            log["profit_firm"][fid][t] = float(rewards[fid])
        for pidx, mw in info["gen"].items():
            log["gen_plant"][pidx][t] = float(mw)

        obs = obs_next
        if done and info.get("error"):
            obs = env.reset()

    return log


# ----------------------------------------------------------------------
# Event-study aggregation
# ----------------------------------------------------------------------
def build_event_study(log: dict, *, window: int = 30) -> dict:
    """
    Align each deviation event at t=0 and average ±window steps around it,
    one bucket per deviator firm.
    """
    T = log["avg_lmp"].shape[0]
    buckets = {fid: {"gen_self": [], "gen_rival": [], "lmp": [],
                      "profit_self": [], "profit_rival": [], "x": []}
               for fid in range(NUM_FIRMS)}

    for t in np.where(log["is_deviation"])[0]:
        if t - window < 0 or t + window >= T:
            continue
        dev = int(log["deviator"][t])
        rival = 1 - dev
        sl = slice(t - window, t + window + 1)
        buckets[dev]["gen_self"].append(log["gen_firm"][dev][sl].copy())
        buckets[dev]["gen_rival"].append(log["gen_firm"][rival][sl].copy())
        buckets[dev]["lmp"].append(log["avg_lmp"][sl].copy())
        buckets[dev]["profit_self"].append(log["profit_firm"][dev][sl].copy())
        buckets[dev]["profit_rival"].append(log["profit_firm"][rival][sl].copy())
        buckets[dev]["x"].append(float(log["x_used"][t]))

    summarized = {}
    for fid, b in buckets.items():
        if not b["gen_self"]:
            summarized[fid] = None
            continue
        summarized[fid] = {
            "n_events": len(b["gen_self"]),
            "x_mean": float(np.mean(b["x"])),
            "x_std": float(np.std(b["x"])),
            "gen_self_mean": np.mean(b["gen_self"], axis=0),
            "gen_self_std":  np.std(b["gen_self"],  axis=0),
            "gen_rival_mean": np.mean(b["gen_rival"], axis=0),
            "gen_rival_std":  np.std(b["gen_rival"],  axis=0),
            "lmp_mean": np.mean(b["lmp"], axis=0),
            "lmp_std":  np.std(b["lmp"],  axis=0),
            "profit_self_mean": np.mean(b["profit_self"], axis=0),
            "profit_self_std":  np.std(b["profit_self"],  axis=0),
            "profit_rival_mean": np.mean(b["profit_rival"], axis=0),
            "profit_rival_std":  np.std(b["profit_rival"],  axis=0),
        }
    return summarized


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_timeseries(log: dict, *, out_path: Path, benchmarks: dict,
                    title_suffix: str = "") -> None:
    T = log["avg_lmp"].shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "Stochastic deviation simulation" + (f" — {title_suffix}" if title_suffix else ""),
        fontsize=13,
    )

    # Generation
    ax = axes[0]
    for fid in range(NUM_FIRMS):
        ax.plot(t, log["gen_firm"][fid], color=f"C{fid}", lw=0.9, label=f"Firm {fid}")
    bench = benchmarks or {}
    if bench:
        comp = bench.get("competitive", {}).get("gens", [])
        mono = bench.get("monopoly", {}).get("gens", [])
        if comp:
            f0c, f1c = comp[0] + comp[1], comp[2]
            ax.axhline(f0c, ls="--", color="C0", alpha=0.4, lw=0.8)
            ax.axhline(f1c, ls="--", color="C1", alpha=0.4, lw=0.8)
        if mono:
            f0m, f1m = mono[0] + mono[1], mono[2]
            ax.axhline(f0m, ls=":", color="C0", alpha=0.5, lw=0.9)
            ax.axhline(f1m, ls=":", color="C1", alpha=0.5, lw=0.9)
    for tt in np.where(log["is_deviation"])[0]:
        c = f"C{int(log['deviator'][tt])}"
        ax.axvline(tt, color=c, alpha=0.25, lw=0.8)
    ax.set_ylabel("Generation (MW)")
    ax.set_title("Generation — vertical markers are deviation events (colored by deviator)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    # LMP
    ax = axes[1]
    ax.plot(t, log["avg_lmp"], color="black", lw=0.9, label="Avg LMP")
    if bench:
        ax.axhline(bench["competitive"]["avg_lmp"], ls="--", color="grey",
                   alpha=0.7, lw=1.0, label=f"Competitive (${bench['competitive']['avg_lmp']:.2f})")
        ax.axhline(bench["monopoly"]["avg_lmp"], ls=":", color="black",
                   alpha=0.8, lw=1.0, label=f"Monopoly (${bench['monopoly']['avg_lmp']:.2f})")
    for tt in np.where(log["is_deviation"])[0]:
        c = f"C{int(log['deviator'][tt])}"
        ax.axvline(tt, color=c, alpha=0.25, lw=0.8)
    ax.set_ylabel("Avg LMP ($/MWh)")
    ax.set_title("Quantity-weighted average LMP")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    # Profit
    ax = axes[2]
    for fid in range(NUM_FIRMS):
        ax.plot(t, log["profit_firm"][fid], color=f"C{fid}", lw=0.9, label=f"Firm {fid} profit/step")
    if bench:
        comp_pi = bench.get("competitive", {}).get("profits", {})
        mono_pi = bench.get("monopoly", {}).get("profits", {})
        for fid in range(NUM_FIRMS):
            if str(fid) in comp_pi:
                ax.axhline(float(comp_pi[str(fid)]), ls="--", color=f"C{fid}", alpha=0.35, lw=0.8)
            if str(fid) in mono_pi:
                ax.axhline(float(mono_pi[str(fid)]), ls=":", color=f"C{fid}", alpha=0.5, lw=0.9)
    for tt in np.where(log["is_deviation"])[0]:
        c = f"C{int(log['deviator'][tt])}"
        ax.axvline(tt, color=c, alpha=0.25, lw=0.8)
    ax.set_ylabel("Profit ($/step)")
    ax.set_xlabel("Step")
    ax.set_title("Per-firm profit")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_event_study(summary: dict, *, out_path: Path, window: int,
                     deviation_prob: float, title_suffix: str = "") -> None:
    deviators = [fid for fid in summary if summary[fid] is not None]
    if not deviators:
        print("No events recorded — skipping event-study plot.")
        return

    fig, axes = plt.subplots(3, len(deviators), figsize=(7.0 * len(deviators), 9),
                              squeeze=False)
    fig.suptitle(
        f"Event study (p={deviation_prob:.0e}, window=±{window}) "
        + (title_suffix or ""),
        fontsize=13,
    )

    for col, fid in enumerate(deviators):
        s = summary[fid]
        rival = 1 - fid
        t = np.arange(-window, window + 1)

        # Generation
        ax = axes[0, col]
        ax.plot(t, s["gen_self_mean"], color=f"C{fid}", lw=1.8,
                label=f"Firm {fid} (deviator)")
        ax.fill_between(t, s["gen_self_mean"] - s["gen_self_std"],
                        s["gen_self_mean"] + s["gen_self_std"], color=f"C{fid}", alpha=0.18)
        ax.plot(t, s["gen_rival_mean"], color=f"C{rival}", lw=1.5, ls="--",
                label=f"Firm {rival} (rival)")
        ax.fill_between(t, s["gen_rival_mean"] - s["gen_rival_std"],
                        s["gen_rival_mean"] + s["gen_rival_std"], color=f"C{rival}", alpha=0.15)
        ax.axvline(0, color="red", ls="--", alpha=0.6, lw=1.0)
        ax.set_title(
            f"Firm {fid} deviates  →  generation  "
            f"(n={s['n_events']}, X̄={s['x_mean']:.2f})"
        )
        ax.set_xlabel("Steps relative to event")
        ax.set_ylabel("Generation (MW)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

        # LMP
        ax = axes[1, col]
        ax.plot(t, s["lmp_mean"], color="black", lw=1.8, label="Avg LMP")
        ax.fill_between(t, s["lmp_mean"] - s["lmp_std"],
                        s["lmp_mean"] + s["lmp_std"], color="black", alpha=0.18)
        ax.axvline(0, color="red", ls="--", alpha=0.6, lw=1.0)
        ax.set_title(f"Firm {fid} deviates  →  Avg LMP")
        ax.set_xlabel("Steps relative to event")
        ax.set_ylabel("Avg LMP ($/MWh)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

        # Profit
        ax = axes[2, col]
        ax.plot(t, s["profit_self_mean"], color=f"C{fid}", lw=1.8,
                label=f"Firm {fid} profit (deviator)")
        ax.fill_between(t, s["profit_self_mean"] - s["profit_self_std"],
                        s["profit_self_mean"] + s["profit_self_std"],
                        color=f"C{fid}", alpha=0.18)
        ax.plot(t, s["profit_rival_mean"], color=f"C{rival}", lw=1.5, ls="--",
                label=f"Firm {rival} profit (rival)")
        ax.fill_between(t, s["profit_rival_mean"] - s["profit_rival_std"],
                        s["profit_rival_mean"] + s["profit_rival_std"],
                        color=f"C{rival}", alpha=0.15)
        ax.axvline(0, color="red", ls="--", alpha=0.6, lw=1.0)
        ax.set_title(f"Firm {fid} deviates  →  profit")
        ax.set_xlabel("Steps relative to event")
        ax.set_ylabel("Profit ($/step)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ----------------------------------------------------------------------
# Sampling X
# ----------------------------------------------------------------------
def build_x_sampler(args):
    if args.x_distribution == "discrete":
        values = np.array(
            [float(v) for v in args.x_values.split(",") if v.strip()],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError("--x-values is empty")
        def sample(rng: np.random.Generator) -> float:
            return float(values[rng.integers(0, values.size)])
        return sample, f"discrete{{{','.join(str(v) for v in values.tolist())}}}"

    lo, hi = [float(v) for v in args.x_range.split(",")]
    if lo >= hi:
        raise ValueError("--x-range must be 'lo,hi' with lo < hi")
    def sample(rng: np.random.Generator) -> float:
        return float(rng.uniform(lo, hi))
    return sample, f"uniform[{lo},{hi}]"


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------
def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Stochastic deviation analysis on trained PPO agents."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=f"Training run dir (default: {DEFAULT_RUN_DIR_NAME}/; aliases: h1, results/delta_cont/h1, figures/h1)",
    )
    p.add_argument("--sessions", type=str, default="0",
                   help='Comma-separated session ids, or "all". Default: "0"')
    p.add_argument("--num-steps", type=int, default=10_000,
                   help="Steps per rollout after normalizer warmup (default 10000)")
    p.add_argument("--deviation-prob", type=float, default=1e-3,
                   help="Per-step probability of a deviation event (default 1e-3)")
    p.add_argument("--x-distribution", type=str, default="discrete",
                   choices=("discrete", "uniform"),
                   help="discrete: --x-values; uniform: --x-range")
    p.add_argument("--x-values", type=str, default="1.5,2,3",
                   help="Comma-separated multipliers when --x-distribution discrete")
    p.add_argument("--x-range", type=str, default="1.5,3.0",
                   help="lo,hi when --x-distribution uniform")
    p.add_argument("--deviator", type=str, default="random",
                   choices=("random", "0", "1", "each"),
                   help="random: pick uniformly per event; 0/1: fixed; "
                   "each: each firm independently with prob p")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="Steps to warm up the obs normalizer before logging")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="Must match training --hidden-dim (default 64)")
    p.add_argument("--event-window", type=int, default=30,
                   help="±window for event-study (default 30 steps)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_RUN_DIR_NAME}/deviation_experiment/stochastic_deviation/ "
        "or <old_results|results>/figures/<subpath>/stochastic_deviation/",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    run_dir = resolve_run_dir(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir not found: {run_dir}")

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    benchmarks = config.get("benchmarks", {})
    history_len = int(config.get("history_len", 1))
    episode_len = int(config.get("episode_len", 168))

    sessions_root = run_dir / "sessions"
    if not sessions_root.is_dir():
        raise FileNotFoundError(f"No sessions/ folder in {run_dir}")

    if args.sessions.strip().lower() == "all":
        session_dirs = sorted(
            d for d in sessions_root.iterdir() if d.is_dir() and (d / "agent_0.pt").exists()
        )
    else:
        ids = [s.strip() for s in args.sessions.split(",") if s.strip()]
        session_dirs = [sessions_root / f"session_{sid}" for sid in ids]
        session_dirs = [d for d in session_dirs if d.is_dir()]
    if not session_dirs:
        raise FileNotFoundError(
            "No matching session directories with agent weights — check --sessions"
        )

    out_dir = args.output_dir or stochastic_deviation_output_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_sampler, x_label = build_x_sampler(args)
    base_seed = args.seed

    print(
        "\n=== Stochastic deviation experiment ===\n"
        f"  run_dir         : {run_dir}\n"
        f"  history_len     : {history_len}\n"
        f"  episode_len     : {episode_len}\n"
        f"  sessions        : {[d.name for d in session_dirs]}\n"
        f"  num_steps       : {args.num_steps}\n"
        f"  warmup_steps    : {args.warmup_steps}\n"
        f"  deviation_prob  : {args.deviation_prob}\n"
        f"  deviator_mode   : {args.deviator}\n"
        f"  x_distribution  : {x_label}\n"
        f"  event_window    : ±{args.event_window}\n"
        f"  device          : {device}\n"
        f"  output_dir      : {out_dir}\n"
        + "=" * 40
    )

    summary_all_sessions = []

    for sess_idx, session_dir in enumerate(session_dirs):
        sess_name = session_dir.name
        sess_seed = base_seed + sess_idx
        torch.manual_seed(sess_seed)
        np.random.seed(sess_seed)
        rng = np.random.default_rng(sess_seed)

        env = ElectricityMarketEnv(history_len=history_len, episode_len=episode_len)
        agents = load_session_agents(session_dir, env, hidden=args.hidden_dim, device=device)
        normalizers = load_or_warm_normalizers(
            session_dir, env, agents, warmup_steps=args.warmup_steps
        )

        log = run_stochastic_rollout(
            env=env, agents=agents, normalizers=normalizers,
            num_steps=args.num_steps,
            deviation_prob=args.deviation_prob,
            x_sampler=x_sampler,
            deviator_mode=args.deviator,
            rng=rng,
        )
        summary = build_event_study(log, window=args.event_window)

        sess_out = out_dir / sess_name
        sess_out.mkdir(parents=True, exist_ok=True)

        # Save full time series
        ts_payload = {
            "config": {
                "run_dir": str(run_dir),
                "session": sess_name,
                "seed": sess_seed,
                "history_len": history_len,
                "num_steps": args.num_steps,
                "warmup_steps": args.warmup_steps,
                "deviation_prob": args.deviation_prob,
                "deviator_mode": args.deviator,
                "x_distribution": x_label,
                "event_window": args.event_window,
            },
            "log": _to_jsonable(log),
        }
        (sess_out / "timeseries.json").write_text(json.dumps(ts_payload))

        # Save summary
        sess_summary = {
            "session": sess_name,
            "n_events_total": int(log["is_deviation"].sum()),
            "n_events_by_deviator": {
                str(fid): int(np.sum(log["deviator"] == fid)) for fid in range(NUM_FIRMS)
            },
            "mean_lmp_overall": float(np.mean(log["avg_lmp"])),
            "mean_lmp_nondev": float(np.mean(log["avg_lmp"][~log["is_deviation"]]))
                if (~log["is_deviation"]).any() else None,
            "mean_lmp_dev_step": float(np.mean(log["avg_lmp"][log["is_deviation"]]))
                if log["is_deviation"].any() else None,
            "mean_profit_by_firm_nondev": {
                str(fid): (float(np.mean(log["profit_firm"][fid][~log["is_deviation"]]))
                           if (~log["is_deviation"]).any() else None)
                for fid in range(NUM_FIRMS)
            },
            "event_study": _to_jsonable({
                str(fid): (None if v is None else {k: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                                   for k, vv in v.items()})
                for fid, v in summary.items()
            }),
        }
        (sess_out / "summary.json").write_text(json.dumps(sess_summary, indent=2))
        summary_all_sessions.append(sess_summary)

        if not args.no_plots:
            plot_timeseries(
                log,
                out_path=sess_out / "timeseries.png",
                benchmarks=benchmarks,
                title_suffix=f"{run_dir.name}/{sess_name}  (p={args.deviation_prob:.0e})",
            )
            plot_event_study(
                summary,
                out_path=sess_out / "event_study.png",
                window=args.event_window,
                deviation_prob=args.deviation_prob,
                title_suffix=f"{run_dir.name}/{sess_name}",
            )

        print(f"[{sess_name}] events={sess_summary['n_events_total']}  "
              f"by_deviator={sess_summary['n_events_by_deviator']}  "
              f"meanLMP={sess_summary['mean_lmp_overall']:.2f}")

    # Cross-session aggregate summary
    (out_dir / "summary_all_sessions.json").write_text(
        json.dumps(summary_all_sessions, indent=2)
    )
    print(f"\nAll done. Aggregate summary → {out_dir / 'summary_all_sessions.json'}")


if __name__ == "__main__":
    main()
