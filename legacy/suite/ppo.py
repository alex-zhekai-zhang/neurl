"""PPO with a Gaussian policy, an in-process vector env, and a per-episode log."""
import numpy as np
import torch
import torch.nn as nn

HP = dict(n_envs=16, n_steps=512, epochs=4, minibatch=128, lr=3e-4, gamma=0.999, lam=0.98, clip=0.2,
          ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, hidden=64, max_episode_steps=1000)


class RunningStat:
    def __init__(self, shape):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):
        m, v, n = x.mean(0), x.var(0), x.shape[0]
        delta, tot = m - self.mean, self.count + n
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.mean = self.mean + delta * n / tot
        self.count = tot


class VecEnv:
    """n envs stepped in-process with auto-reset and a time limit.

    memory appends 3 numbers about past episodes to the observation:
      "outcome"   the agent's last pad landing: [on left, on right, +1 if that pad was good / -1 if bad]
      "feedback"  which pad was good in the previous episode: [left, right, 0]
    reversal_every=L flips each env's good pad after every L of its episodes (random initial pad);
    episodes are then logged with their reversal count and trial number within the block (from 1).
    """

    def __init__(self, make_env, n, seed, max_steps=HP["max_episode_steps"], memory=None, reversal_every=None):
        self.envs = [make_env() for _ in range(n)]
        self.seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(n)]
        self.max_steps, self.memory, self.seeded = max_steps, memory, False
        self.mem = np.zeros((n, 3), dtype=np.float32)
        self.t, self.ret = np.zeros(n, dtype=int), np.zeros(n)
        self.obs_dim = self.envs[0].observation_space.shape[0] + (3 if memory else 0)
        self.act_dim = self.envs[0].action_space.shape[0]
        self.reversal_every = reversal_every
        self.block_ep, self.n_rev = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        if reversal_every:
            rng = np.random.default_rng(seed)
            for e in self.envs:
                e.good_pad = ("left", "right")[rng.integers(2)]

    def _obs(self, i, o):
        return np.concatenate([o, self.mem[i]]) if self.memory else o

    def apply(self, fn):
        for e in self.envs:
            fn(e)

    def reset(self):
        obs = [self._obs(i, e.reset(seed=None if self.seeded else self.seeds[i])[0]) for i, e in enumerate(self.envs)]
        self.seeded = True
        self.t[:], self.ret[:] = 0, 0.0
        return np.stack(obs)

    def step(self, actions):
        """Returns obs, reward, terminated, truncated, {env: final obs of a truncated episode}, finished episodes."""
        n = len(self.envs)
        obs, rew = [None] * n, np.zeros(n)
        term, trunc = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        final, episodes = {}, []
        for i, e in enumerate(self.envs):
            o, r, te, _, info = e.step(actions[i])
            self.t[i] += 1
            self.ret[i] += r
            rew[i], term[i] = r, te
            trunc[i] = not te and self.t[i] >= self.max_steps
            if te or trunc[i]:
                episodes.append({"ret": float(self.ret[i]), "len": int(self.t[i]),
                                 "outcome": info.get("outcome", "timeout"), "env": i})
                was_good = getattr(e, "good_pad", None)  # before any reversal below
                if self.reversal_every:
                    episodes[-1].update(rev=int(self.n_rev[i]), trial=int(self.block_ep[i]) + 1)
                    self.block_ep[i] += 1
                    if self.block_ep[i] == self.reversal_every:
                        e.good_pad = "right" if e.good_pad == "left" else "left"
                        self.block_ep[i] = 0
                        self.n_rev[i] += 1
                if trunc[i]:
                    final[i] = self._obs(i, o)
                if self.memory == "feedback":
                    self.mem[i] = [was_good == "left", was_good == "right", 0.0]
                if self.memory == "outcome" and info.get("pad"):
                    good = info["outcome"] == "landed_good"
                    self.mem[i] = [info["pad"] == "left", info["pad"] == "right", 1.0 if good else -1.0]
                o = e.reset()[0]
                self.t[i], self.ret[i] = 0, 0.0
            obs[i] = self._obs(i, o)
        return np.stack(obs), rew, term, trunc, final, episodes


def mlp(inp, out, hidden, out_gain):
    layers, d = [], inp
    for _ in range(2):
        lin = nn.Linear(d, hidden)
        nn.init.orthogonal_(lin.weight, 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = hidden
    lin = nn.Linear(d, out)
    nn.init.orthogonal_(lin.weight, out_gain)
    nn.init.zeros_(lin.bias)
    return nn.Sequential(*layers, lin)


class Agent:
    def __init__(self, obs_dim, act_dim, seed, hp=None):
        self.hp = {**HP, **(hp or {})}
        self.obs_dim, self.act_dim, self.seed = obs_dim, act_dim, seed
        self.steps, self.log = 0, []  # log: one dict per finished training episode
        self.obs_rms, self.ret_rms = RunningStat(obs_dim), RunningStat(())
        self.gen = torch.Generator().manual_seed(seed)
        self.init_networks()

    def init_networks(self):
        """(Re)initialise weights and optimiser; keeps normalisation statistics and the log."""
        torch.manual_seed(self.seed + 7919 * self.steps)
        self.actor = mlp(self.obs_dim, self.act_dim, self.hp["hidden"], 0.01)
        self.critic = mlp(self.obs_dim, 1, self.hp["hidden"], 1.0)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))
        self.params = [*self.actor.parameters(), *self.critic.parameters(), self.log_std]
        self.opt = torch.optim.Adam(self.params, lr=self.hp["lr"], eps=1e-5)

    def norm(self, obs):
        x = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return torch.as_tensor(np.clip(x, -10, 10), dtype=torch.float32)

    def act(self, obs):
        """Deterministic action (policy mean) for one observation."""
        with torch.no_grad():
            return self.actor(self.norm(obs)).numpy()

    def learn(self, vec, steps, phase):
        hp = self.hp
        n, T, gamma = len(vec.envs), hp["n_steps"], hp["gamma"]
        obs = vec.reset()
        disc = np.zeros(n)
        for _ in range(max(1, steps // (n * T))):
            b_obs = torch.zeros(T, n, self.obs_dim)
            b_act = torch.zeros(T, n, self.act_dim)
            b_logp, b_val, b_rew, b_done = (torch.zeros(T, n) for _ in range(4))
            for t in range(T):
                self.obs_rms.update(obs)
                x = self.norm(obs)
                with torch.no_grad():
                    dist = torch.distributions.Normal(self.actor(x), self.log_std.exp())
                    a = dist.mean + dist.stddev * torch.randn(n, self.act_dim, generator=self.gen)
                    b_logp[t] = dist.log_prob(a).sum(-1)
                    b_val[t] = self.critic(x).squeeze(-1)
                b_obs[t], b_act[t] = x, a
                obs, rew, term, trunc, final, episodes = vec.step(a.numpy())
                disc = disc * gamma + rew
                self.ret_rms.update(disc)
                r = np.clip(rew / np.sqrt(self.ret_rms.var + 1e-8), -10, 10)
                for i, o in final.items():  # time limit is not a terminal state: bootstrap
                    with torch.no_grad():
                        r[i] += gamma * self.critic(self.norm(o)).item()
                done = term | trunc
                disc[done] = 0.0
                b_rew[t] = torch.as_tensor(r, dtype=torch.float32)
                b_done[t] = torch.as_tensor(done, dtype=torch.float32)
                for ep in episodes:
                    self.log.append({**ep, "step": self.steps + (t + 1) * n, "phase": phase})
            self.steps += n * T

            with torch.no_grad():
                next_val = self.critic(self.norm(obs)).squeeze(-1)
            adv, last = torch.zeros(T, n), torch.zeros(n)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else b_val[t + 1]
                live = 1.0 - b_done[t]
                last = b_rew[t] + gamma * nv * live - b_val[t] + gamma * hp["lam"] * live * last
                adv[t] = last
            ret = adv + b_val

            f_obs, f_act = b_obs.reshape(-1, self.obs_dim), b_act.reshape(-1, self.act_dim)
            f_logp, f_adv, f_ret = b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
            for _ in range(hp["epochs"]):
                for idx in torch.randperm(n * T, generator=self.gen).split(hp["minibatch"]):
                    dist = torch.distributions.Normal(self.actor(f_obs[idx]), self.log_std.exp())
                    ratio = (dist.log_prob(f_act[idx]).sum(-1) - f_logp[idx]).exp()
                    a_ = f_adv[idx]
                    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                    pg = -torch.min(ratio * a_, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * a_).mean()
                    vf = 0.5 * (self.critic(f_obs[idx]).squeeze(-1) - f_ret[idx]).pow(2).mean()
                    loss = pg + hp["vf_coef"] * vf - hp["ent_coef"] * dist.entropy().sum(-1).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.params, hp["max_grad_norm"])
                    self.opt.step()


def evaluate(agent, make_env, episodes=100, seed=10_000, max_steps=HP["max_episode_steps"]):
    """Deterministic policy on fixed seeds. Returns (mean return, {outcome: fraction})."""
    env, rets, outcomes = make_env(), [], {}
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        total, info, term = 0.0, {}, False
        for _ in range(max_steps):
            obs, r, term, _, info = env.step(agent.act(obs))
            total += r
            if term:
                break
        rets.append(total)
        o = info.get("outcome", "timeout") if term else "timeout"
        outcomes[o] = outcomes.get(o, 0) + 1 / episodes
    return float(np.mean(rets)), outcomes
