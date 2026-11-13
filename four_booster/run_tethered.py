"""Train two tethered landers (lunar_tethered.py), each with its own instance of a model's Agent.

    python run.py --suite tethered [--model mlp] [--episodes N]   # outputs/tethered_<model>/
    python run_tethered.py [--episodes N] [--seed S]               # outputs/tethered_mlp_seed<S>/

Each lander's Agent sees only its own observation row and learns from its own rewards; they update together
every model.BATCH_EPISODES episodes. Outputs: lander<k>_ckpt_{init,final}.pt, log.json (per-episode returns,
length, outcome), returns.png, final.gif (one episode with the final policies), and episodes.json (every
RECORD_EVERY-th episode, frame by frame) for ../view_episodes.py.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lunar_tethered import N, TetheredLanders
from models import mlp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # shared episode_recorder.py
from episode_recorder import EpisodeRecorder  # noqa: E402
EPISODES = 3000
MAX_STEPS = 1000
RECORD_EVERY = 100
RECORDER_OUTCOME = {"out_of_bounds": "out of bounds", "timeout": "timed out", "landed_on_targets": "landed on targets"}


def run_episode(env, agents, seed=None, frames=None, recorder=None, episode=None):
    obs, _ = env.reset(seed=seed)
    if recorder:
        recorder.start(env, episode)
    log_probs, rewards = [[] for _ in agents], [[] for _ in agents]
    info = {}
    for t in range(MAX_STEPS):
        actions = []
        for k, agent in enumerate(agents):
            a, lp = agent.act(obs[k])
            actions.append(a)
            log_probs[k].append(lp)
        obs, r, terminated, _, info = env.step(np.stack(actions))
        for k in range(len(agents)):
            rewards[k].append(float(r[k]))
        if recorder:
            recorder.step(env, r)
        if frames is not None and t % 2 == 0:
            frames.append(Image.fromarray(env.render()))
        if terminated:
            break
    outcome = info.get("outcome", "timeout")
    if outcome == "landed" and all(info["on_target"]):
        outcome = "landed_on_targets"
    if recorder:
        each = [sum(rw) for rw in rewards]
        recorder.end(env, float(np.mean(each)), terminated, not terminated,
                     outcome=RECORDER_OUTCOME.get(outcome, outcome), returns_each=each)
    return [(torch.stack(lp), rw) for lp, rw in zip(log_probs, rewards)], outcome


def plot(log, path, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k in range(N):
        r = np.array([e["returns"][k] for e in log])
        m = np.convolve(r, np.ones(20) / 20, mode="valid") if len(r) >= 20 else r
        ax1.plot(np.arange(len(m)) + len(r) - len(m), m, label=f"lander {k} (20-episode mean)")
    ax1.set_ylabel("return"); ax1.set_title(title); ax1.legend(loc="lower right")
    w = 100
    for name in ("crashed", "out_of_bounds", "timeout", "landed", "landed_on_targets"):
        x = np.array([e["outcome"] == name for e in log], dtype=float)
        if len(x) >= w and x.any():
            ax2.plot(np.arange(w - 1, len(x)), np.convolve(x, np.ones(w) / w, mode="valid"), label=name)
    ax2.set_xlabel("episode"); ax2.set_ylabel(f"fraction, last {w}")
    if ax2.lines:
        ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def train(model, episodes, out, title, seed=None):
    """model: a models/ module with Agent and BATCH_EPISODES. seed=None leaves torch and the env unseeded."""
    if seed is not None:
        torch.manual_seed(seed)
    env = TetheredLanders(render_mode="rgb_array")
    env.reset(seed=seed)  # later resets continue the env's RNG
    obs_dim, act_dim = env.observation_space.shape[1], env.action_space.shape[1]
    agents = [model.Agent(obs_dim, act_dim) for _ in range(N)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for k, agent in enumerate(agents):
        agent.save(out / f"lander{k}_ckpt_init.pt")

    recorder = EpisodeRecorder(out / "episodes.json", env, run=title, every=RECORD_EVERY, last_episode=episodes - 1)
    log = []
    for ep in range(0, episodes, model.BATCH_EPISODES):
        batches = [[] for _ in agents]
        for e in range(ep, min(ep + model.BATCH_EPISODES, episodes)):
            episode, outcome = run_episode(env, agents, recorder=recorder, episode=e)
            for k, (lp, rw) in enumerate(episode):
                batches[k].append((lp, rw))
            log.append({"returns": [sum(rw) for _, rw in episode], "len": len(episode[0][1]), "outcome": outcome})
        for agent, batch in zip(agents, batches):
            agent.update(batch)
        recent = log[-100:]
        print(f"episode {len(log):5d}  avg return (last 100): "
              + "  ".join(f"{np.mean([e['returns'][k] for e in recent]):8.1f}" for k in range(N))
              + f"   landed: {np.mean([e['outcome'].startswith('landed') for e in recent]):.2f}", flush=True)

    for k, agent in enumerate(agents):
        agent.save(out / f"lander{k}_ckpt_final.pt")
    (out / "log.json").write_text(json.dumps(log))
    plot(log, out / "returns.png", title)
    frames = []
    with torch.no_grad():
        run_episode(env, agents, frames=frames)
    frames[0].save(out / "final.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = HERE / "outputs" / f"tethered_mlp_seed{args.seed}"
    train(mlp, args.episodes, out, f"tethered landers · mlp · seed {args.seed}", seed=args.seed)
    print(f"saved {out.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
