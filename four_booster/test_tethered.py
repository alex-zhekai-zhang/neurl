"""Checks for lunar_tethered.py. Run: python test_tethered.py"""
import numpy as np

import lunar_booster
from lunar_tethered import FPS, TETHER_LENGTH, TetheredLanders


def random_rollout(env, seed, steps=1000):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [obs]
    for _ in range(steps):
        obs, r, term, _, info = env.step(rng.uniform(-1, 1, (2, 3)))
        out.append(obs)
        if term:
            break
    return out, r, info


def test_shapes_and_determinism():
    env = TetheredLanders()
    a, _, _ = random_rollout(env, 0)
    b, _, _ = random_rollout(env, 0)
    assert a[0].shape == (2, 15) and len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_rope_never_exceeds_length():
    env = TetheredLanders()
    for seed in range(20):
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            _, _, term, _, _ = env.step(rng.uniform(-1, 1, (2, 3)))
            assert (env.landers[0].position - env.landers[1].position).length <= TETHER_LENGTH + 0.02
            if term:
                break


def test_tether_force_is_equal_and_opposite_when_taut():
    env = TetheredLanders()
    env.reset(seed=0)
    env.landers[0].linearVelocity, env.landers[1].linearVelocity = (-6.0, 0.0), (6.0, 0.0)
    forces = []
    for _ in range(10):
        obs, *_ = env.step(np.zeros((2, 3)))
        forces.append(obs[:, 13:])
    f = max(forces, key=lambda x: np.abs(x).sum())
    assert f[0, 0] > 0.1 and f[1, 0] < -0.1  # left lander pulled right, right lander pulled left
    assert np.allclose(f[0], -f[1], atol=1e-5)


def test_slack_rope_leaves_thrust_unchanged():
    """With the rope slack, full boosters give the same lift as on a single lunar_booster lander."""
    def lift(env, act, body):
        env.reset(seed=0)
        v0 = body(env).linearVelocity.y
        for _ in range(10):
            env.step(act)
        return (body(env).linearVelocity.y - v0) * FPS / 10

    single = lift(lunar_booster.LunarLander(continuous=True), np.array([0.0, 1.0, 1.0]), lambda e: e.lander)
    pair = lift(TetheredLanders(), np.tile([0.0, 1.0, 1.0], (2, 1)), lambda e: e.landers[0])
    assert abs(single - pair) < 0.3, (single, pair)


def test_shared_terminal_rewards():
    env = TetheredLanders()
    env.reset(seed=0)
    env.crashed[1] = True
    _, r, term, _, info = env.step(np.zeros((2, 3)))
    assert term and info["outcome"] == "crashed" and list(r) == [-100, -100]

    env.reset(seed=0)
    for body in [*env.landers, *env.legs[0], *env.legs[1]]:
        body.awake = False
    _, r, term, _, info = env.step(np.zeros((2, 3)))
    assert term and info["outcome"] == "landed" and list(r) == [100, 100] and len(info["on_target"]) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
