"""REINFORCE on LunarLander where a control swap is learned by one small layer.

    obs -> trunk (MLP) -> 4 logits -> action-connect (4x4 linear, no bias) -> action logits

Phase 1, normal controls:  action-connect frozen at the identity, trunk trains.
Phase 2, swapped controls:
    --phase2 connect (default): trunk frozen, only the 16 action-connect weights train.
    --phase2 all:               trunk and action-connect both train (baseline).

The swap is undone exactly by setting action-connect to a permutation matrix
(checked bit-for-bit at the switch), so phase 2 tests whether REINFORCE finds it.

--reset-entropy: at the switch, scale the trunk's output layer so the policy's mean entropy
matches the untrained network's on the same states (action rankings unchanged).

pip install "gymnasium[box2d]" torch matplotlib
    python remap_lunar.py [--phase2 connect|all] [--reset-entropy] [--switch 3000] [--post-episodes 5000]
                          [--lr 1e-3] [--connect-lr 1e-2] [--trunk-lr 1e-3] [--seed 0]
"""
import argparse
import copy
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints" / "remap_lunar"
FIG_DIR = ROOT / "figures" / "remap_lunar"

GAMMA = 0.99
BATCH_EPISODES = 10

# LunarLander-v3 actions: 0 = noop, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine
NORMAL = [0, 1, 2, 3]
SWAPPED = [0, 3, 2, 1]  # left and right engines swapped


class RemapActions(gym.ActionWrapper):
    """Agent action a is executed in the env as mapping[a]."""

    def __init__(self, env, mapping):
        super().__init__(env)
        assert sorted(mapping) == list(range(env.action_space.n))
        self.mapping = list(mapping)

    def action(self, a):
        return self.mapping[a]


def make_env(mapping):
    return RemapActions(gym.make("LunarLander-v3"), mapping)


class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, n_actions),
        )
        self.connect = nn.Linear(n_actions, n_actions, bias=False)  # action-connect layer
        self.set_connect(NORMAL)

    def set_connect(self, mapping):
        """Set action-connect to the permutation that undoes `mapping`.

        The trunk's logit z[e] scores env action e. In an env that sends agent
        action a to env action mapping[a], agent action a needs logit z[mapping[a]],
        so row a of the matrix is the one-hot vector e_{mapping[a]}.
        """
        with torch.no_grad():
            self.connect.weight.copy_(permutation(mapping))

    def forward(self, obs):
        return self.connect(self.trunk(obs))


def permutation(mapping):
    return torch.eye(len(mapping))[mapping]


def discounted_returns(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    return torch.tensor(out[::-1], dtype=torch.float32)


def run_episode(env, policy, seed=None, greedy=False):
    obs, _ = env.reset(seed=seed)
    log_probs, rewards = [], []
    done = False
    while not done:
        logits = policy(torch.as_tensor(obs, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if greedy else dist.sample()
        obs, r, terminated, truncated, _ = env.step(action.item())
        log_probs.append(dist.log_prob(action))
        rewards.append(r)
        done = terminated or truncated
    return torch.stack(log_probs), rewards


def train_phase(policy, env, param_groups, episodes, label, returns, track=None):
    """REINFORCE with a fresh Adam over `param_groups` ([{"params": ..., "lr": ...}, ...])."""
    opt = torch.optim.Adam(param_groups)
    t0 = time.time()
    for ep in range(0, episodes, BATCH_EPISODES):
        all_logp, all_ret = [], []
        for _ in range(BATCH_EPISODES):
            logp, rewards = run_episode(env, policy)
            all_logp.append(logp)
            all_ret.append(discounted_returns(rewards))
            returns.append(sum(rewards))
        logp, ret = torch.cat(all_logp), torch.cat(all_ret)
        adv = (ret - ret.mean()) / (ret.std() + 1e-8)  # normalized returns as baseline
        loss = -(logp * adv).sum() / BATCH_EPISODES
        opt.zero_grad()
        loss.backward()
        opt.step()
        extra = track() if track else ""
        if (ep + BATCH_EPISODES) % 100 == 0:
            print(f"[{label}] episode {ep + BATCH_EPISODES:5d}  avg return (last 100): "
                  f"{np.mean(returns[-100:]):8.1f}{extra}  ({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def evaluate(policy, env_mapping, seeds, greedy):
    """Mean/std return with the policy's current weights. Leaves the training RNG untouched."""
    rng = torch.get_rng_state()
    torch.manual_seed(0)  # same sampling noise for every evaluation
    env = make_env(env_mapping)
    returns = np.array([sum(run_episode(env, policy, seed=s, greedy=greedy)[1]) for s in seeds])
    env.close()
    torch.set_rng_state(rng)
    return returns


@torch.no_grad()
def check_permutation_exact(policy, seeds):
    """With action-connect = swap permutation, the swapped env must reproduce the normal env exactly."""
    obs = torch.as_tensor(np.array([make_env(NORMAL).observation_space.sample() for _ in range(2000)]),
                          dtype=torch.float32)
    policy.set_connect(NORMAL)
    z_normal = policy(obs)
    base = evaluate(policy, NORMAL, seeds, greedy=True)
    broken = evaluate(policy, SWAPPED, seeds, greedy=True)
    policy.set_connect(SWAPPED)
    z_swapped = policy(obs)
    fixed = evaluate(policy, SWAPPED, seeds, greedy=True)
    policy.set_connect(NORMAL)

    env_logits = torch.empty_like(z_swapped)
    env_logits[:, SWAPPED] = z_swapped  # agent action a is env action SWAPPED[a]
    print(f"  logits exactly re-indexed: {torch.equal(env_logits, z_normal)}")
    print(f"  greedy: normal env {base.mean():7.1f} | swapped env, identity {broken.mean():7.1f} | "
          f"swapped env, permutation {fixed.mean():7.1f}  (episode-for-episode equal: {np.array_equal(base, fixed)})")


@torch.no_grad()
def collect_states(policy, env_mapping, seeds):
    """States visited by the sampled policy on fixed seeds. Leaves the training RNG untouched."""
    rng = torch.get_rng_state()
    torch.manual_seed(0)
    env = make_env(env_mapping)
    states = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32)
            states.append(x)
            a = torch.distributions.Categorical(logits=policy(x)).sample()
            obs, _, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated
    env.close()
    torch.set_rng_state(rng)
    return torch.stack(states)


@torch.no_grad()
def mean_entropy(policy, states, scale=1.0):
    """Mean policy entropy on `states` if the logits were multiplied by `scale`."""
    return torch.distributions.Categorical(logits=scale * policy(states)).entropy().mean().item()


@torch.no_grad()
def reset_entropy(policy, states, target):
    """Scale the trunk's output layer by c in (0, 1] so the mean entropy on `states` equals `target`.

    Logits are linear in that layer's weight and bias (action-connect is linear with no bias),
    so this multiplies every logit by c: action rankings are unchanged, confidence drops.
    Returns c.
    """
    if mean_entropy(policy, states) >= target:
        return 1.0
    lo, hi = 0.0, 1.0  # entropy at scale 0 is ln(n_actions) >= target > entropy at scale 1
    for _ in range(50):
        mid = (lo + hi) / 2
        if mean_entropy(policy, states, mid) > target:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    out = policy.trunk[-1]
    out.weight.mul_(c)
    out.bias.mul_(c)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=int, default=3000, help="episodes of phase 1 (normal controls)")
    ap.add_argument("--post-episodes", type=int, default=5000, help="episodes of phase 2 (swapped controls)")
    ap.add_argument("--lr", type=float, default=1e-3, help="trunk learning rate, phase 1")
    ap.add_argument("--connect-lr", type=float, default=1e-2, help="action-connect learning rate, phase 2")
    ap.add_argument("--phase2", choices=["connect", "all"], default="connect",
                    help="phase 2 trains only action-connect, or the whole network")
    ap.add_argument("--trunk-lr", type=float, default=1e-3, help="trunk learning rate, phase 2 (--phase2 all only)")
    ap.add_argument("--reset-entropy", action="store_true",
                    help="at the switch, scale the logits so policy entropy matches the untrained network's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = f"seed{args.seed}" if args.phase2 == "connect" else f"seed{args.seed}_all_trunklr{args.trunk_lr:g}"
    if args.reset_entropy:
        tag += "_entreset"
    seeds = range(10_000, 10_000 + args.eval_episodes)
    target = permutation(SWAPPED)

    probe = make_env(NORMAL)
    policy = Policy(probe.observation_space.shape[0], probe.action_space.n)
    untrained = copy.deepcopy(policy)  # initial weights, for --reset-entropy (deepcopy uses no RNG)
    probe.close()
    returns = []

    # ---- phase 1: normal controls, trunk learns, action-connect frozen at identity
    policy.trunk.requires_grad_(True)
    policy.connect.requires_grad_(False)
    env = make_env(NORMAL)
    env.reset(seed=args.seed)
    train_phase(policy, env, [{"params": policy.trunk.parameters(), "lr": args.lr}], args.switch, "phase 1", returns)
    env.close()
    assert torch.equal(policy.connect.weight, torch.eye(policy.n_actions)), "action-connect changed in phase 1"
    torch.save(policy.state_dict(), CKPT_DIR / f"{tag}_switch.pt")

    print("\nswitching controls")
    check_permutation_exact(policy, seeds)
    base_sampled = evaluate(policy, NORMAL, seeds, greedy=False)
    base_greedy = evaluate(policy, NORMAL, seeds, greedy=True)
    if args.reset_entropy:
        # states the trained policy visits under the old controls; target = untrained network on those states
        states = collect_states(policy, NORMAL, seeds)
        default = mean_entropy(untrained, states)
        before = mean_entropy(policy, states)
        c = reset_entropy(policy, states, default)
        print(f"  entropy reset: {before:.3f} -> {mean_entropy(policy, states):.3f} nats "
              f"(untrained network on the same states: {default:.3f}), logits scaled by {c:.4f}")
    print()

    # ---- phase 2: swapped controls, action-connect learns from the identity (+ trunk if --phase2 all)
    train_trunk = args.phase2 == "all"
    policy.trunk.requires_grad_(train_trunk)
    policy.connect.requires_grad_(True)
    groups = [{"params": policy.connect.parameters(), "lr": args.connect_lr}]
    if train_trunk:
        groups.append({"params": policy.trunk.parameters(), "lr": args.trunk_lr})
    trunk_at_switch = {k: v.clone() for k, v in policy.trunk.state_dict().items()}
    dist_hist = []

    def trunk_drift():
        """Per-layer ||W - W_switch|| / ||W_switch|| for the trunk's weight matrices."""
        sd = policy.trunk.state_dict()
        return [((sd[k] - v).norm() / v.norm()).item() for k, v in trunk_at_switch.items() if k.endswith("weight")]

    def track():
        W = policy.connect.weight.detach()
        dist_hist.append((W - target).norm().item())
        rows_ok = (W.argmax(1) == target.argmax(1)).sum().item()
        drift = "  trunk drift " + " ".join(f"{d:.2f}" for d in trunk_drift()) if train_trunk else ""
        return f"  |W - P| = {dist_hist[-1]:.2f}  rows argmax-correct: {rows_ok}/4{drift}"

    env = make_env(SWAPPED)
    env.reset(seed=args.seed + 1)
    train_phase(policy, env, groups, args.post_episodes, "phase 2", returns, track)
    env.close()
    if not train_trunk:
        for k, v in policy.trunk.state_dict().items():
            assert torch.equal(v, trunk_at_switch[k]), f"trunk param {k} changed in phase 2"
    torch.save(policy.state_dict(), CKPT_DIR / f"{tag}_final.pt")

    np.set_printoptions(precision=2, suppress=True)
    if train_trunk:
        print(f"\ntrunk drift per layer (||W - W_switch|| / ||W_switch||): "
              + " ".join(f"{d:.2f}" for d in trunk_drift()))
    else:
        print(f"\ntrunk unchanged in phase 2: True")
    print(f"learned action-connect W:\n{policy.connect.weight.detach().numpy()}\ntarget permutation P:\n{target.numpy()}")
    for greedy, base in ((True, base_greedy), (False, base_sampled)):
        learned = evaluate(policy, SWAPPED, seeds, greedy)
        print(f"{'greedy ' if greedy else 'sampled'} eval: before switch (normal env) {base.mean():7.1f} ± {base.std():5.1f}"
              f"   after learning (swapped env) {learned.mean():7.1f} ± {learned.std():5.1f}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6))
    smooth = np.convolve(returns, np.ones(20) / 20, mode="valid")
    a1.plot(returns, alpha=0.3)
    a1.plot(np.arange(len(smooth)) + 19, smooth)
    a1.axvline(args.switch, color="k", ls="--", lw=1)
    a1.set_ylabel("return"); a1.set_xlabel("episode (dashed: controls swapped)")
    a2.plot(args.switch + np.arange(1, len(dist_hist) + 1) * BATCH_EPISODES, dist_hist)
    a2.set_ylabel("|W - P|  (Frobenius)"); a2.set_xlabel("episode")
    phase2_desc = "whole network learns" if train_trunk else "only action-connect learns"
    fig.suptitle(f"Phase 1: trunk learns | Phase 2: {phase2_desc} ({tag})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}.png")
    print(f"saved checkpoints/remap_lunar/{tag}_{{switch,final}}.pt and figures/remap_lunar/{tag}.png")


if __name__ == "__main__":
    main()
