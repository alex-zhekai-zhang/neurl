"""REINFORCE with an MLP Gaussian policy (continuous actions).

The network outputs the mean of each action number; a learned, state-independent log_std sets the spread.
Actions are sampled from Normal(mean, std) and clipped to [-1, 1] by the env.
Agent holds one policy; train() runs it on a single-lander env. run_tethered.py uses one Agent per lander.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.99
LR = 1e-3
BATCH_EPISODES = 10
HIDDEN = (256, 128, 128)


def discounted_returns(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    return torch.tensor(out[::-1], dtype=torch.float32)


class Agent:
    """One policy network, its log_std and its optimiser."""

    def __init__(self, obs_dim, act_dim):
        layers, d = [], obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.policy = nn.Sequential(*layers, nn.Linear(d, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # std starts at 1
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + [self.log_std], lr=LR)

    def act(self, obs):
        """Sampled action (numpy) and its log-probability (tensor, for update)."""
        mean = self.policy(torch.as_tensor(obs, dtype=torch.float32))
        dist = torch.distributions.Normal(mean, self.log_std.exp())
        action = dist.sample()
        return action.numpy(), dist.log_prob(action).sum()  # independent numbers: joint log-prob is the sum

    def update(self, episodes):
        """One gradient step from a batch; episodes = [(stacked log-probs, rewards), ...]."""
        logp = torch.cat([lp for lp, _ in episodes])
        ret = torch.cat([discounted_returns(r) for _, r in episodes])
        adv = (ret - ret.mean()) / (ret.std() + 1e-8)  # normalized returns as baseline
        loss = -(logp * adv).sum() / len(episodes)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "log_std": self.log_std.detach()}, path)


def train(env, episodes, schedule, recorder, out_dir):
    """See models/__init__.py for the interface."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0])

    def run_episode(episode):
        obs, _ = env.reset()
        recorder.start(env, episode)
        log_probs, rewards = [], []
        done = False
        while not done:
            action, logp = agent.act(obs)
            log_probs.append(logp)
            obs, r, terminated, truncated, _ = env.step(action)  # env clips to [-1, 1]
            recorder.step(env, r)
            rewards.append(r)
            done = terminated or truncated
        recorder.end(env, sum(rewards), terminated, truncated)
        return torch.stack(log_probs), rewards

    agent.save(out_dir / "ckpt_init.pt")
    returns = []
    for ep in range(0, episodes, BATCH_EPISODES):
        batch = []
        for e in range(ep, min(ep + BATCH_EPISODES, episodes)):
            change = schedule(env.unwrapped, e)
            if change:
                agent.save(out_dir / f"ckpt_ep{e}.pt")  # weights right before the env changes
                print(f"episode {e}: {change}")
            logp, rewards = run_episode(e)
            batch.append((logp, rewards))
            returns.append(sum(rewards))
        agent.update(batch)

        print(f"episode {len(returns):5d}  avg return (last 100): {np.mean(returns[-100:]):8.1f}"
              f"   std: {agent.log_std.exp().detach().numpy().round(3)}")

    agent.save(out_dir / "ckpt_final.pt")
    return returns
