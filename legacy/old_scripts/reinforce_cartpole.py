"""REINFORCE with a simple MLP policy on CartPole.

Partway through training the two actions (push left / push right) are swapped
to see how the agent re-adapts.

pip install gymnasium torch
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figures"
CKPT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)


GAMMA = 0.99

LR_default = 1e-3
LR_peak = 1e-1
LR = LR_default
EPISODES = 1000
SWITCH = 500
BATCH_EPISODES = 10
ANNEAL_EPISODES = 10  
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


# CartPole-v1 actions: 0 = push cart left, 1 = push cart right
NORMAL = [0, 1]
SWAPPED = [1, 0]  # left <-> right

env = RemapActions(gym.make("CartPole-v1"))
assert env.action_space.n == len(SWAPPED)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = nn.Sequential(
    nn.Linear(obs_dim, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, n_actions),
)
opt = torch.optim.Adam(policy.parameters(), lr=LR)
torch.save(policy.state_dict(), CKPT_DIR / "cartpole_ckpt_init.pt")  # untrained weights


def run_episode():
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        logits = policy(torch.as_tensor(obs, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        obs, r, terminated, truncated, _ = env.step(action.item())
        log_probs.append(dist.log_prob(action))
        rewards.append(r)
        done = terminated or truncated
    return torch.stack(log_probs), rewards


def discounted_returns(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    return torch.tensor(out[::-1], dtype=torch.float32)


recent = []
for ep in range(0, EPISODES, BATCH_EPISODES):
    if ep >= SWITCH and env.mapping != SWAPPED:
        print("switching controls")
        torch.save(policy.state_dict(), CKPT_DIR / "cartpole_ckpt_switch.pt")  # weights right before the swap
        env.mapping = SWAPPED  # swap left/right
    all_logp, all_ret = [], []
    for _ in range(BATCH_EPISODES):
        logp, rewards = run_episode()
        all_logp.append(logp)
        all_ret.append(discounted_returns(rewards))
        recent = (recent + [sum(rewards)])
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

    if (ep // BATCH_EPISODES) % 1 == 0:
        print(f"episode {ep + BATCH_EPISODES:5d}  avg return (last 100): {np.mean(recent[-100:]):8.1f}   lr: {opt.param_groups[0]['lr']}")

recent_smoothed = [sum([k/10 for k in recent[n - 10: n]]) for n in range(10, len(recent))]
torch.save(policy.state_dict(), CKPT_DIR / "reinforce_cartpole.pt")
plt.plot(recent_smoothed); plt.savefig(FIG_DIR / "cartpole_figure.png")
env.close()
