"""Metrics from results/. Run: python report.py

IQM = mean of the middle 50% of seeds; [lo, hi] = 95% percentile bootstrap over seeds.
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
WINDOW = 50          # episodes
SOLVED = 200.0       # criterion: mean online return over WINDOW consecutive episodes
BINS = 8             # learning-set curve: training split into this many equal step bins


def load(experiment):
    runs = {}
    for f in sorted((RESULTS / experiment).glob("*.json")):
        r = json.loads(f.read_text())
        r["log"] = {k: np.array(v) for k, v in r["log"].items()}
        runs.setdefault(r["agent"], []).append(r)
    return runs


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float(x[n // 4: n - n // 4].mean())


def ci(x, stat=iqm, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    boots = [stat(x[rng.integers(len(x), size=len(x))]) for _ in range(reps)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(x, digits=2):
    lo, hi = ci(x)
    return f"{iqm(x):.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def to_criterion(log, phase):
    """(steps, episodes, reached) from the start of a phase until the criterion is first met."""
    m = log["phase"] == phase
    ret, step = log["ret"][m], log["step"][m]
    start = log["step"][~m & (log["step"] < step[0])].max() if (log["step"] < step[0]).any() else 0
    roll = np.convolve(ret, np.ones(WINDOW) / WINDOW, mode="valid")
    hit = np.flatnonzero(roll >= SOLVED)
    if len(hit) == 0:
        return int(step[-1] - start), len(ret), False
    k = hit[0] + WINDOW - 1
    return int(step[k] - start), int(k + 1), True


def certify():
    print("\n## certify: PPO from scratch, deterministic eval over 100 episodes")
    print("task | seeds | eval return IQM [95% CI] | seeds with eval >= 200 | landed fraction (mean)")
    for task, runs in load("certify").items():
        ret = [r["evals"]["final"]["return"] for r in runs]
        landed = np.mean([r["evals"]["final"]["outcomes"].get("landed", 0.0) for r in runs])
        print(f"{task} | {len(runs)} | {fmt(ret, 1)} | {sum(x >= SOLVED for x in ret)}/{len(runs)} | {landed:.2f}")


def savings():
    print(f"\n## savings: A0 -> B1 -> A1 -> B2; criterion = {WINDOW}-episode mean online return >= {SOLVED:.0f}")
    print("agent | seeds | reached B1 | reached B2 | episodes to criterion B1 | B2 | savings index 1 - B2/B1 (episodes) | seeds with index > 0")
    runs = load("savings")
    crit = {a: {p: [to_criterion(r["log"], p) for r in rs] for p in ("B1", "B2")} for a, rs in runs.items()}
    for agent, rs in runs.items():
        b1, b2 = crit[agent]["B1"], crit[agent]["B2"]
        idx = [1 - y[1] / x[1] for x, y in zip(b1, b2)]
        print(f"{agent} | {len(rs)} | {sum(x[2] for x in b1)}/{len(rs)} | {sum(x[2] for x in b2)}/{len(rs)} | "
              f"{fmt([x[1] for x in b1], 0)} | {fmt([x[1] for x in b2], 0)} | {fmt(idx)} | {sum(i > 0 for i in idx)}/{len(rs)}")
    print("\nagent | env steps (thousands) to criterion B1 | B2 | savings index 1 - B2/B1 (steps)")
    for agent in runs:
        b1, b2 = ([x[0] / 1e3 for x in crit[agent][p]] for p in ("B1", "B2"))
        print(f"{agent} | {fmt(b1, 0)} | {fmt(b2, 0)} | {fmt([1 - y / x for x, y in zip(b1, b2)])}")
    print("(runs that never reach the criterion count as the full phase length)")
    print("\nagent | deterministic eval return on B at: end A0 (before B1) | end B1 | end A1 (before B2) | end B2")
    for agent, rs in runs.items():
        print(agent, *[fmt([r["evals"][p]["B"]["return"] for r in rs], 0) for p in ("A0", "B1", "A1", "B2")], sep=" | ")
    print("\nagent | deterministic eval return on A at: end A0 | end B1 | end A1")
    for agent, rs in runs.items():
        print(agent, *[fmt([r["evals"][p]["A"]["return"] for r in rs], 0) for p in ("A0", "B1", "A1")], sep=" | ")


def learning_set():
    print("\n## learning set: two pads, good pad flips every 8 episodes per env")
    print("trial-2 accuracy = P(landed on good pad on the 2nd episode after a reversal); chance for a fixed pad = 0.5")
    runs = load("learning_set")
    total = max(r["log"]["step"].max() for rs in runs.values() for r in rs)
    edges = np.linspace(0, total, BINS + 1)

    def acc(log, trial, lo, hi):
        m = (log["trial"] == trial) & (log["rev"] > 0) & (log["step"] > lo) & (log["step"] <= hi)
        return float((log["outcome"][m] == "landed_good").mean()) if m.any() else np.nan

    print(f"\nagent | seeds | trial-2 accuracy IQM by training bin ({total / BINS / 1e6:.1f}M steps each)")
    for agent, rs in runs.items():
        print(agent, len(rs), *[f"{iqm([acc(r['log'], 2, a, b) for r in rs]):.2f}" for a, b in zip(edges, edges[1:])], sep=" | ")
    print("\nagent | last bin: trial-1 accuracy | trial-2 accuracy | trial-8 accuracy | good-pad landings, all trials | seeds with trial-2 > 0.5")
    for agent, rs in runs.items():
        a, b = edges[-2], edges[-1]
        t2 = [acc(r["log"], 2, a, b) for r in rs]
        every = [float((r["log"]["outcome"][r["log"]["step"] > a] == "landed_good").mean()) for r in rs]
        print(agent, fmt([acc(r["log"], 1, a, b) for r in rs]), fmt(t2), fmt([acc(r["log"], 8, a, b) for r in rs]),
              fmt(every), f"{sum(x > 0.5 for x in t2)}/{len(rs)}", sep=" | ")
    print("\nagent | last bin outcome fractions (mean over seeds): landed_good, landed_bad, landed_off, timeout, crashed, out_of_bounds")
    for agent, rs in runs.items():
        fr = [[float((r["log"]["outcome"][r["log"]["step"] > edges[-2]] == o).mean()) for r in rs]
              for o in ("landed_good", "landed_bad", "landed_off", "timeout", "crashed", "out_of_bounds")]
        print(agent, *[f"{np.mean(x):.2f}" for x in fr], sep=" | ")


if __name__ == "__main__":
    for name, fn in (("certify", certify), ("savings", savings), ("learning_set", learning_set)):
        if (RESULTS / name).exists():
            fn()
