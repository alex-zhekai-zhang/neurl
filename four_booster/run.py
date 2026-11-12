"""Train a model on this folder's task suite: 4 engines (left/right angled boosters + left/right side engines).

    python run.py                                  # every task, mlp
    python run.py --suite booster                  # one task
    python run.py --suite booster 2landing --model mlp --episodes 1000

Tasks
    booster    lunar_booster.py: the left angled booster is weakened for part of training (BOOSTER_* below)
    2landing   lunar_2landing.py: two pads, +100 and -100; which pad is good can switch (PAD_* below)

Outputs go to outputs/<task>_<model>/: checkpoints, returns.png, and episodes.json for ../view_episodes.py.
"""
import argparse
import importlib
import random
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # shared episode_recorder.py
from episode_recorder import EpisodeRecorder  # noqa: E402

import lunar_2landing  # noqa: E402
import lunar_booster  # noqa: E402

SUITE_NAME = "4 boosters"
EPISODES = 9000
RECORD_EVERY = 100  # every RECORD_EVERY-th episode is saved for the viewer

# ---- booster task: weakened engine ----
BOOSTER_ENGINE = "left_main_scale"    # env attribute scaling the left angled booster's power
BOOSTER_SCALE = 0.0                   # power while weakened
BOOSTER_WEAK = (1 / 3, 2 / 3)         # weakened between these fractions of training

# ---- 2landing task: which pad is good ----
PAD_START = "left"          # good pad at the start of training
PAD_SWITCH = "at"           # "never", "at" (swap at each PAD_SWITCH_AT fraction of training),
                            # "every" (swap every PAD_SWITCH_EVERY episodes), "random" (coin flip per episode)
PAD_SWITCH_AT = (1 / 3, 2 / 3)
PAD_SWITCH_EVERY = 1500
PAD_SEED = 0                # for "random"


def booster_schedule(episodes):
    lo, hi = (int(f * episodes) for f in BOOSTER_WEAK)

    def schedule(env, episode):
        scale = BOOSTER_SCALE if lo <= episode < hi else 1.0
        if getattr(env, BOOSTER_ENGINE) != scale:
            setattr(env, BOOSTER_ENGINE, scale)
            return f"{BOOSTER_ENGINE} -> {scale}"
        return None
    return schedule


def pad_schedule(episodes):
    other = {"left": "right", "right": "left"}
    rng = random.Random(PAD_SEED)

    def good_pad(episode):
        if PAD_SWITCH == "never":
            return PAD_START
        if PAD_SWITCH == "random":
            return rng.choice(("left", "right"))
        if PAD_SWITCH == "at":
            swaps = sum(episode >= int(f * episodes) for f in PAD_SWITCH_AT)
        elif PAD_SWITCH == "every":
            swaps = episode // PAD_SWITCH_EVERY
        else:
            raise ValueError(f"unknown PAD_SWITCH {PAD_SWITCH!r}")
        return PAD_START if swaps % 2 == 0 else other[PAD_START]

    def schedule(env, episode):
        pad = good_pad(episode)
        if env.good_pad != pad:
            env.good_pad = pad
            return None if PAD_SWITCH == "random" else f"good pad -> {pad}"
        return None
    return schedule


TASKS = {
    "booster": (lunar_booster.LunarLander, booster_schedule),
    "2landing": (lunar_2landing.LunarLander, pad_schedule),
}
MODELS = sorted(p.stem for p in (HERE / "models").glob("*.py") if p.stem != "__init__")


def plot_returns(returns, changes, path, title):
    r = np.asarray(returns)
    mean = np.convolve(r, np.ones(20) / 20, mode="valid") if len(r) >= 20 else r
    plt.figure(figsize=(9, 4))
    plt.plot(r, color="#3987e5", alpha=0.25, linewidth=1)
    plt.plot(np.arange(len(mean)) + len(r) - len(mean), mean, color="#3987e5", linewidth=2, label="20-episode mean")
    for e in changes[:50]:
        plt.axvline(e, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("episode"); plt.ylabel("return"); plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(path); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="+", default=["all"], choices=["all", *TASKS], help="tasks to run (default: all)")
    ap.add_argument("--model", default="mlp", choices=MODELS)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()

    tasks = list(TASKS) if "all" in args.suite else list(dict.fromkeys(args.suite))
    model = importlib.import_module(f"models.{args.model}")
    for task in tasks:
        make_env, make_schedule = TASKS[task]
        print(f"=== {SUITE_NAME} · {task} · {args.model} ({args.episodes} episodes)")
        # gym.make adds a 1000-step TimeLimit; constructing the class directly does not
        env = gym.wrappers.TimeLimit(make_env(continuous=True), max_episode_steps=1000)
        out = HERE / "outputs" / f"{task}_{args.model}"
        recorder = EpisodeRecorder(out / "episodes.json", env, run=f"{SUITE_NAME} · {task} · {args.model}",
                                   every=RECORD_EVERY, last_episode=args.episodes - 1)
        changes, inner = [], make_schedule(args.episodes)

        def schedule(env_u, episode, inner=inner, changes=changes):
            msg = inner(env_u, episode)
            if msg:
                changes.append(episode)
            return msg

        returns = model.train(env, args.episodes, schedule, recorder, out)
        plot_returns(returns, changes, out / "returns.png", f"{SUITE_NAME} · {task} · {args.model}")
        env.close()
        print(f"saved {out.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
