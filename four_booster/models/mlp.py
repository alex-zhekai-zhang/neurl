"""REINFORCE with an MLP Gaussian policy (continuous actions).

The network outputs the mean of each action number; a learned, state-independent log_std sets the spread.
Actions are sampled from Normal(mean, std) and clipped to [-1, 1] by the env.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GAMMA = 0.99
LR = 1e-3
BATCH_EPISODES = 10
HIDDEN = (256, 128, 128)


def train(env, episodes, schedule, recorder, out_dir):
    """See models/__init__.py for the interface."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    layers, d = [], obs_dim
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    policy = nn.Sequential(*layers, nn.Linear(d, act_dim))
    log_std = nn.Parameter(torch.zeros(act_dim))  # std starts at 1
    opt = torch.optim.Adam(list(policy.parameters()) + [log_std], lr=LR)

    def save(name):
        torch.save({"policy": policy.state_dict(), "log_std": log_std.detach()}, out_dir / name)

    def run_episode(episode):
        obs, _ = env.reset()
        recorder.start(env, episode)
        log_probs, rewards = [], []
        done = False
        while not done:
            mean = policy(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Normal(mean, log_std.exp())
            action = dist.sample()
            log_probs.append(dist.log_prob(action).sum())  # independent numbers: joint log-prob is the sum
            obs, r, terminated, truncated, _ = env.step(action.numpy())  # env clips to [-1, 1]
            recorder.step(env, r)
            rewards.append(r)
            done = terminated or truncated
        recorder.end(env, sum(rewards), terminated, truncated)
        return torch.stack(log_probs), rewards

    def discounted_returns(rewards):
        G, out = 0.0, []
        for r in reversed(rewards):
            G = r + GAMMA * G
            out.append(G)
        return torch.tensor(out[::-1], dtype=torch.float32)

    save("ckpt_init.pt")
    returns = []
    for ep in range(0, episodes, BATCH_EPISODES):
        all_logp, all_ret = [], []
        for e in range(ep, min(ep + BATCH_EPISODES, episodes)):
            change = schedule(env.unwrapped, e)
            if change:
                save(f"ckpt_ep{e}.pt")  # weights right before the env changes
                print(f"episode {e}: {change}")
            logp, rewards = run_episode(e)
            all_logp.append(logp)
            all_ret.append(discounted_returns(rewards))
            returns.append(sum(rewards))
        logp = torch.cat(all_logp)
        ret = torch.cat(all_ret)
        adv = (ret - ret.mean()) / (ret.std() + 1e-8)  # normalized returns as baseline
        loss = -(logp * adv).sum() / len(all_logp)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"episode {len(returns):5d}  avg return (last 100): {np.mean(returns[-100:]):8.1f}"
              f"   std: {log_std.exp().detach().numpy().round(3)}")

    save("ckpt_final.pt")
    return returns
