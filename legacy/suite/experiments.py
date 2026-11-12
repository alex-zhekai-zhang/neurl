"""Certificates and the two signature tests.

    python experiments.py certify       # PPO from scratch on A, B, B_dead
    python experiments.py savings       # A0 -> B1 -> A1 -> B2
    python experiments.py learning_set  # two pads, good pad flips every REVERSAL_EVERY episodes
    options: --seeds 0 1 2 ...  --agents ...  --workers 12  --ls-steps N

Writes results/<experiment>/<agent>_seed<k>.json (per-episode log + deterministic evals). Then: python report.py
"""
import argparse
import copy
import json
import time
from multiprocessing import Pool
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
N_ENVS = 16

# single pad, angled layout; B weakens the left booster
TASKS = {"A": 1.0, "B": 0.7, "B_dead": 0.0}
CERTIFY_STEPS = 2_000_000
SAVINGS_PHASES = [("A0", "A", 2_000_000), ("B1", "B", 1_500_000), ("A1", "A", 1_500_000), ("B2", "B", 1_500_000)]
# ppo_reset: networks re-initialised at every switch; ppo_cued: engine scales in obs;
# ppo_snapshot: told the task id, restores its networks from the end of the last exposure to that task
SAVINGS_AGENTS = ("ppo", "ppo_reset", "ppo_cued", "ppo_snapshot")

TWO_PADS = {"pads": "two", "rest_steps": 50, "pad_shaping": "good"}
REVERSAL_EVERY = 8
LEARNING_SET_STEPS = 3_000_000
# memory: the agent's last pad landing in obs; feedback: last episode's good pad in obs; cued: current good pad in obs
LEARNING_SET_AGENTS = ("ppo", "ppo_memory", "ppo_feedback", "ppo_cued")

AGENTS = {"certify": tuple(TASKS), "savings": SAVINGS_AGENTS, "learning_set": LEARNING_SET_AGENTS}


def run(job):
    import torch
    torch.set_num_threads(1)
    from legacy.suite.lander import LunarLander
    from legacy.suite.ppo import Agent, VecEnv, evaluate

    experiment, name, seed, ls_steps = job
    t0 = time.time()

    cued = experiment == "savings" and name == "ppo_cued"

    def single(task):
        return lambda: LunarLander(engine_scale={"left_booster": TASKS[task]}, observe_context=cued)

    def evals(agent):
        out = {}
        for task in ("A", "B"):
            ret, outcomes = evaluate(agent, single(task), episodes=50)
            out[task] = {"return": ret, "outcomes": outcomes}
        return out

    result = {"experiment": experiment, "agent": name, "seed": seed, "evals": {}}
    if experiment == "certify":
        vec = VecEnv(single(name), N_ENVS, seed)
        agent = Agent(vec.obs_dim, vec.act_dim, seed)
        agent.learn(vec, CERTIFY_STEPS, name)
        ret, outcomes = evaluate(agent, single(name), episodes=100)
        result["evals"]["final"] = {"return": ret, "outcomes": outcomes}
    elif experiment == "savings":
        vec = VecEnv(single("A"), N_ENVS, seed)
        agent = Agent(vec.obs_dim, vec.act_dim, seed)
        snapshots = {}
        for phase, task, steps in SAVINGS_PHASES:
            vec.apply(lambda e: e.engine_scale.update(left_booster=TASKS[task]))
            if name == "ppo_reset" and phase != "A0":
                agent.init_networks()
            if name == "ppo_snapshot" and task in snapshots:
                nets, agent.obs_rms, agent.ret_rms = copy.deepcopy(snapshots[task])
                for module, state in zip((agent.actor, agent.critic, agent.opt), nets[:3]):
                    module.load_state_dict(state)
                agent.log_std.data.copy_(nets[3])
            agent.learn(vec, steps, phase)
            snapshots[task] = copy.deepcopy((
                (agent.actor.state_dict(), agent.critic.state_dict(), agent.opt.state_dict(), agent.log_std.data),
                agent.obs_rms, agent.ret_rms))
            result["evals"][phase] = evals(agent)  # at the end of the phase, on both tasks
    elif experiment == "learning_set":
        kw = dict(TWO_PADS, observe_context=name == "ppo_cued")
        vec = VecEnv(lambda: LunarLander(**kw), N_ENVS, seed, memory={"ppo_memory": "outcome", "ppo_feedback": "feedback"}.get(name),
                     reversal_every=REVERSAL_EVERY)
        agent = Agent(vec.obs_dim, vec.act_dim, seed)
        agent.learn(vec, ls_steps, "LS")

    result["hp"] = agent.hp
    result["log"] = {k: [e.get(k) for e in agent.log] for k in agent.log[0]}
    out = RESULTS / experiment / f"{name}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    return f"{experiment} {name} seed {seed}: {len(agent.log)} episodes, {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(AGENTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--agents", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ls-steps", type=int, default=LEARNING_SET_STEPS)
    args = ap.parse_args()
    jobs = [(args.experiment, a, s, args.ls_steps) for a in (args.agents or AGENTS[args.experiment]) for s in args.seeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)
