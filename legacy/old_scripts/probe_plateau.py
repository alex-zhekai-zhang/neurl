"""Probe the landscape around remap_lunar.py checkpoints: plateau or local optimum?

For each checkpoint, on swapped controls, measures:
  - policy entropy on visited states (max ln 4 = 1.39 nats; ~0 means a near-deterministic policy)
  - REINFORCE gradient norm, split into trunk / action-connect
  - return at theta + sigma * d for random directions d, perturbing either all weights or only
    action-connect. Each tensor of d is scaled to the norm of the corresponding weight tensor
    (Li et al. 2018), so sigma = 0.1 moves every tensor by 10% of its size.

Plateau:         gradient ~0, return flat for small sigma, some larger-sigma directions improve.
Local optimum:   every direction makes the return worse.

    python probe_plateau.py seed1_switch.pt seed1_all_trunklr0.01_final.pt ...   (names in checkpoints/remap_lunar/)
"""
import argparse

import numpy as np
import torch

from legacy.old_scripts.remap_lunar import CKPT_DIR, SWAPPED, Policy, discounted_returns, make_env

SIGMAS = [0.01, 0.03, 0.1, 0.3]


def rollouts(policy, seeds, grad=False, sample_seed=0):
    """Sampled episodes on fixed env seeds and a fixed sampling seed (common random numbers)."""
    rng = torch.get_rng_state()
    torch.manual_seed(sample_seed)
    env = make_env(SWAPPED)
    rets, ents, logps, discs = [], [], [], []
    with torch.set_grad_enabled(grad):
        for s in seeds:
            obs, _ = env.reset(seed=s)
            done, R, L = False, [], []
            while not done:
                logits = policy(torch.as_tensor(obs, dtype=torch.float32))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                ents.append(dist.entropy().item())
                L.append(dist.log_prob(a))
                obs, r, terminated, truncated, _ = env.step(a.item())
                R.append(r)
                done = terminated or truncated
            rets.append(sum(R))
            logps.append(torch.stack(L))
            discs.append(discounted_returns(R))
    env.close()
    torch.set_rng_state(rng)
    return np.array(rets), float(np.mean(ents)), logps, discs


def grad_norms(policy, seeds):
    rets, ent, logps, discs = rollouts(policy, seeds, grad=True)
    ret = torch.cat(discs)
    adv = (ret - ret.mean()) / (ret.std() + 1e-8)
    loss = -(torch.cat(logps) * adv).sum() / len(seeds)
    policy.zero_grad()
    loss.backward()
    g_trunk = torch.sqrt(sum(p.grad.norm() ** 2 for p in policy.trunk.parameters())).item()
    g_conn = policy.connect.weight.grad.norm().item()
    return rets, ent, g_trunk, g_conn


def random_direction(policy, which, gen):
    d = {}
    for name, p in policy.named_parameters():
        if which == "connect" and not name.startswith("connect"):
            d[name] = torch.zeros_like(p)
            continue
        r = torch.randn(p.shape, generator=gen)
        d[name] = r / r.norm() * p.detach().norm()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--directions", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    seeds = range(20_000, 20_000 + args.episodes)

    probe = make_env(SWAPPED)
    obs_dim, n_actions = probe.observation_space.shape[0], probe.action_space.n
    probe.close()

    for ckpt in args.ckpts:
        policy = Policy(obs_dim, n_actions)
        policy.load_state_dict(torch.load(CKPT_DIR / ckpt))
        base_rets, ent, g_trunk, g_conn = grad_norms(policy, seeds)
        base = base_rets.mean()
        lines = [f"== {ckpt}",
                 f"  base return {base:8.1f}   entropy {ent:.3f} nats (max 1.386)   "
                 f"grad norm: trunk {g_trunk:.2e}  connect {g_conn:.2e}"]
        theta = {n: p.detach().clone() for n, p in policy.named_parameters()}
        for which in ("all", "connect"):
            gen = torch.Generator().manual_seed(0)
            dirs = [random_direction(policy, which, gen) for _ in range(args.directions)]
            for sigma in SIGMAS:
                deltas, ents = [], []
                for d in dirs:
                    with torch.no_grad():
                        for n, p in policy.named_parameters():
                            p.copy_(theta[n] + sigma * d[n])
                    r, e, _, _ = rollouts(policy, seeds)
                    deltas.append(r.mean() - base)
                    ents.append(e)
                deltas = np.array(deltas)
                lines.append(f"  perturb {which:7s} sigma={sigma:<5}  return change: mean {deltas.mean():+8.1f}  "
                             f"best {deltas.max():+8.1f}  worst {deltas.min():+8.1f}  "
                             f"improved >25: {(deltas > 25).sum()}/{len(deltas)}   entropy {np.mean(ents):.3f}")
            with torch.no_grad():
                for n, p in policy.named_parameters():
                    p.copy_(theta[n])
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
