"""Env checks. Run: python suite/test_lander.py"""
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from legacy.suite.lander import LunarLander  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rollout(env, seed, actions, max_steps=400):
    """Returns (obs, reward, terminated) per step for a fixed action sequence."""
    out = [(env.reset(seed=seed)[0], 0.0, False)]
    for a in actions[:max_steps]:
        obs, r, term, _, _ = env.step(a)
        out.append((obs, r, term))
        if term:
            break
    return out


def same(a, b, exact=True):
    if len(a) != len(b):
        return False
    for (o1, r1, t1), (o2, r2, t2) in zip(a, b):
        if t1 != t2 or not np.array_equal(o1, o2):
            return False
        if (r1 != r2) if exact else (abs(r1 - r2) > 1e-9):
            return False
    return True


def actions(seed, dim, n=400):
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, dim)).astype(np.float32)


def test_stock_matches_gymnasium():
    ref = gym.make("LunarLander-v3", continuous=True).unwrapped
    env = LunarLander(layout="stock")
    for seed in range(20):
        assert same(rollout(ref, seed, actions(seed, 2)), rollout(env, seed, actions(seed, 2))), seed


def test_angled_matches_fork():
    ref = load(ROOT.parent / "four_booster" / "lunar_booster.py").LunarLander()
    env = LunarLander(layout="angled")
    for seed in range(20):
        assert same(rollout(ref, seed, actions(seed, 3)), rollout(env, seed, actions(seed, 3))), seed
    ref.left_main_scale = 0.4
    env.engine_scale["left_booster"] = 0.4
    for seed in range(5):
        assert same(rollout(ref, seed, actions(seed, 3)), rollout(env, seed, actions(seed, 3))), seed


def test_two_pads_matches_fork():
    ref = load(ROOT.parent / "four_booster" / "lunar_2landing.py").LunarLander()
    env = LunarLander(layout="angled", pads="two", good_pad="right")
    ref.good_pad = "right"
    for seed in range(20):
        assert same(rollout(ref, seed, actions(seed, 3)), rollout(env, seed, actions(seed, 3)), exact=False), seed


def test_deterministic():
    env = LunarLander()
    assert same(rollout(env, 3, actions(3, 3)), rollout(env, 3, actions(3, 3)))
    assert not same(rollout(env, 3, actions(3, 3)), rollout(env, 4, actions(3, 3)))


def test_scale_zero_is_engine_off():
    acts = actions(0, 3)
    off = acts.copy()
    off[:, 1] = -1.0  # left booster commanded off
    dead = LunarLander(engine_scale={"left_booster": 0.0})
    a, b = rollout(dead, 0, acts), rollout(LunarLander(), 0, off)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))  # same motion; fuel cost also zero
    assert same(a, b)


def test_fault_fires_at_step():
    acts = actions(1, 3)
    plain = rollout(LunarLander(), 1, acts)
    fault = rollout(LunarLander(fault=(30, "right_booster", 0.0)), 1, acts)
    assert same(plain[:31], fault[:31])
    assert not same(plain[:60], fault[:60])
    env = LunarLander(fault=(30, "right_booster", 0.0))
    assert same(rollout(env, 1, acts), rollout(env, 1, acts))  # fault resets each episode


def test_action_perm_and_sign():
    acts = actions(2, 3)
    mirrored = acts[:, [0, 2, 1]] * np.array([-1, 1, 1], dtype=np.float32)
    env = LunarLander(action_perm=(0, 2, 1), action_sign=(-1, 1, 1))
    assert same(rollout(env, 2, mirrored), rollout(LunarLander(), 2, acts))


def test_context_and_outcome():
    env = LunarLander(pads="two", good_pad="right", observe_context=True, engine_scale={"left_side": 0.5})
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (13,)
    assert list(obs[8:]) == [1.0, 1.0, 0.5, 1.0, 1.0]
    term, info = False, {}
    while not term:
        obs, _, term, _, info = env.step(np.zeros(3))
    assert info["outcome"] in ("crashed", "out_of_bounds", "landed_good", "landed_bad", "landed_off")
    assert LunarLander(layout="stock").action_space.shape == (2,)


def test_pad_shaping_good():
    acts = np.tile(np.array([[0.0, -1.0, 1.0]], dtype=np.float32), (40, 1))
    total, moved_right = {}, None
    for good in ("left", "right"):
        traj = rollout(LunarLander(pads="two", good_pad=good, pad_shaping="good"), 0, acts)
        total[good] = sum(r for _, r, _ in traj)
        moved_right = traj[-1][0][0] > traj[0][0][0]
    assert (total["right"] > total["left"]) == moved_right  # same motion, rewarded only toward the good pad


def test_rest_steps():
    def descend(env):
        """Guided upright descent with the side engine left on, so Box2D never puts the lander to sleep."""
        env.reset(seed=0)
        contact = []
        for _ in range(1000):
            if not (contact and contact[-1]):
                env.lander.linearVelocity = (0.0, max(env.lander.linearVelocity.y, -1.5))
                env.lander.angularVelocity = 0.0
                env.lander.angle = 0.0
            obs, r, term, _, info = env.step(np.array([0.6 if contact and contact[-1] else 0.0, -1.0, -1.0]))
            contact.append(bool(obs[6] and obs[7]))
            if term:
                return contact, r, info
        return contact, None, {}

    contact, r, info = descend(LunarLander(rest_steps=5))
    assert info.get("outcome") == "landed" and r == 100
    assert contact[-5:] == [True] * 5 and not contact[-6]
    longer, _, _ = descend(LunarLander())
    assert len(longer) > len(contact)  # stock rule keeps going


def test_vecenv_reversal_and_memory():
    from legacy.suite.ppo import VecEnv
    kw = {"pads": "two", "rest_steps": 50, "pad_shaping": "good"}
    for memory in ("feedback", "outcome"):
        vec = VecEnv(lambda: LunarLander(**kw), 2, seed=0, memory=memory, reversal_every=2)
        obs = vec.reset()
        assert obs.shape == (2, 11) and not obs[:, 8:].any()
        good = [e.good_pad for e in vec.envs]
        done, stale = [0, 0], 0
        while min(done) < 6:
            obs, _, _, _, _, episodes = vec.step(np.zeros((2, 3)))
            for ep in episodes:
                i = ep["env"]
                assert (ep["rev"], ep["trial"]) == (done[i] // 2, done[i] % 2 + 1)
                done[i] += 1
                if memory == "feedback":  # the pad that was good in the episode just finished
                    assert list(obs[i, 8:]) == [good[i] == "left", good[i] == "right", 0.0]
                    stale += good[i] != vec.envs[i].good_pad
                else:  # free fall never rests on a pad, so nothing to remember
                    assert not obs[i, 8:].any()
                assert (vec.envs[i].good_pad != good[i]) == (done[i] % 2 == 0)  # flips every 2 episodes
                good[i] = vec.envs[i].good_pad
        assert memory == "outcome" or stale >= 4  # after a reversal the memory points at the old pad


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
