"""reinforce_lunar.py plus one change: at the switch, exploration is reset to its untrained level.

The reset multiplies the last layer's weight and bias by c in (0, 1], i.e. logits -> c * logits.
Every state's action ranking is unchanged (the greedy policy is the same); only confidence drops.
c is found by bisection so the mean policy entropy on the last pre-switch batch's states equals
the mean entropy the untrained policy had on its first batch ("default" exploration).

Everything else (network, LR schedule incl. the spike, batch size, loss) is copied from
reinforce_lunar.py. --no-reset runs the same code without the reset, so with the same --seed
the two runs are identical up to the switch.

pip install "gymnasium[box2d]" torch
    python reinforce_lunar_reset.py [--seed 0]
    python reinforce_lunar_reset.py --no-reset [--seed 0]
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-reset", action="store_true", help="skip the exploration reset (baseline)")
args = ap.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
tag = f"{'noreset' if args.no_reset else 'reset'}_seed{args.seed}"

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints" / "lunar_reset"
FIG_DIR = ROOT / "figures" / "lunar_reset"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- identical to reinforce_lunar.py --------------------------------------------------------
GAMMA = 0.99

LR_default = 1e-3
LR_peak = 1e-3
LR = LR_default
EPISODES = 6000
SWITCH = 3000
BATCH_EPISODES = 10
ANNEAL_EPISODES = 30
def lr_scheduler(ep):
    if ep < SWITCH or ep >= ANNEAL_EPISODES + SWITCH :
        return LR_default
    if SWITCH <= ep < ANNEAL_EPISODES + SWITCH:
            return LR_peak - (LR_peak - LR_default) * ((ep - SWITCH) / ANNEAL_EPISODES)

class RemapActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mapping = list(range(env.action_space.n))

    def action(self, a):
        return self.mapping[a]
env = RemapActions(gym.make("LunarLander-v3"))
env.reset(seed=args.seed)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = nn.Sequential(
    nn.Linear(obs_dim, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, n_actions)
)
opt = torch.optim.Adam(policy.parameters(), lr=LR)
torch.save(policy.state_dict(), CKPT_DIR / f"{tag}_init.pt")  # untrained weights


def run_episode():
    obs, _ = env.reset()
    log_probs, rewards, entropies, states = [], [], [], []
    done = False
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32)
        logits = policy(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        states.append(x)
        entropies.append(dist.entropy().item())
        obs, r, terminated, truncated, _ = env.step(action.item())
        log_probs.append(dist.log_prob(action))
        rewards.append(r)
        done = terminated or truncated
    return torch.stack(log_probs), rewards, entropies, states


def discounted_returns(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    return torch.tensor(out[::-1], dtype=torch.float32)
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def mean_entropy(states, scale=1.0):
    """Mean policy entropy on `states` if the logits were multiplied by `scale`."""
    return torch.distributions.Categorical(logits=scale * policy(states)).entropy().mean().item()


@torch.no_grad()
def reset_exploration(states, target):
    """Scale the last layer so the mean entropy on `states` equals `target`. Returns the scale."""
    if mean_entropy(states) >= target:
        return 1.0
    lo, hi = 0.0, 1.0  # entropy at scale 0 is ln(n_actions) >= target > entropy at scale 1
    for _ in range(50):
        mid = (lo + hi) / 2
        if mean_entropy(states, mid) > target:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    policy[-1].weight.mul_(c)
    policy[-1].bias.mul_(c)
    return c


recent, entropy_hist = [], []
default_entropy, last_states = None, None
for ep in range(0, EPISODES, BATCH_EPISODES):
    if ep >= SWITCH and env.mapping != [0, 3, 2, 1]:
        print("switching controls")
        torch.save(policy.state_dict(), CKPT_DIR / f"{tag}_switch.pt")  # weights right before the swap
        env.mapping = [0, 3, 2, 1]  #switch thrusters
        if not args.no_reset:
            before = mean_entropy(last_states)
            c = reset_exploration(last_states, default_entropy)
            print(f"exploration reset: entropy {before:.3f} -> {mean_entropy(last_states):.3f} nats "
                  f"(untrained default {default_entropy:.3f}), logits scaled by {c:.4f}")
    all_logp, all_ret, batch_ent, batch_states = [], [], [], []
    for _ in range(BATCH_EPISODES):
        logp, rewards, ents, states = run_episode()
        all_logp.append(logp)
        all_ret.append(discounted_returns(rewards))
        recent = (recent + [sum(rewards)])
        batch_ent += ents
        batch_states += states
    last_states = torch.stack(batch_states)
    if default_entropy is None:
        default_entropy = float(np.mean(batch_ent))  # untrained policy, first batch
    entropy_hist.append(float(np.mean(batch_ent)))
    LR = lr_scheduler(ep)
    for g in opt.param_groups:
        g["lr"] = LR
    logp = torch.cat(all_logp)
    ret = torch.cat(all_ret)
    adv = (ret - ret.mean()) / (ret.std() + 1e-8)  # normalized returns as baseline
    loss = -(logp * adv).sum() / BATCH_EPISODES

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (ep // BATCH_EPISODES) % 10 == 0:
        print(f"episode {ep + BATCH_EPISODES:5d}  avg return (last 100): {np.mean(recent[-100:]):8.1f}   "
              f"lr: {opt.param_groups[0]['lr']}   entropy: {entropy_hist[-1]:.3f}", flush=True)

recent_smoothed = [sum([k/10 for k in recent[n - 10: n]]) for n in range(10, len(recent))]
torch.save(policy.state_dict(), CKPT_DIR / f"{tag}_final.pt")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
a1.plot(recent_smoothed); a1.axvline(SWITCH, color="k", ls="--", lw=1); a1.set_ylabel("return (10-ep avg)")
a2.plot(np.arange(len(entropy_hist)) * BATCH_EPISODES, entropy_hist)
a2.axhline(default_entropy, color="gray", ls=":", lw=1); a2.axvline(SWITCH, color="k", ls="--", lw=1)
a2.set_ylabel("policy entropy (nats)"); a2.set_xlabel("episode")
fig.suptitle(f"reinforce_lunar {'with exploration reset' if not args.no_reset else '(no reset)'} ({tag})")
fig.tight_layout(); fig.savefig(FIG_DIR / f"{tag}.png")
env.close()
