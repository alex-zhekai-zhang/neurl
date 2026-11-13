"""Records LunarLander training episodes as compact JSON for view_episodes.py.

Every `every`-th episode (and the last one) is stored frame by frame: lander and leg poses, the engines
fired that step (env.flames), and the step reward. Every episode's return and every change to the env's
task parameters (*_scale attributes, good_pad) are logged too, so the viewer can draw the learning curve
and the switch periods.
Envs with several landers (env.landers, e.g. four_booster/lunar_tethered.py) are stored with per-lander lists
L, G, F and r in each frame, plus the targets; pass outcome and returns_each to end().
"""
import json
import os
import sys
from pathlib import Path

# names shown in the viewer for the env's task parameters
PARAM_LABELS = {
    "left_engine_scale": "left side engine",
    "left_main_scale": "left angled booster",
    "good_pad": "good pad",
}


def _task_params(u):
    return {k: (v if isinstance(v, str) else float(v)) for k, v in vars(u).items()
            if k.endswith("_scale") or k == "good_pad"}


def _r(x):
    return round(float(x), 3)


class EpisodeRecorder:
    def __init__(self, path, env, run, every=100, last_episode=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every, self.last_episode = every, last_episode
        self.cur, self.params = None, {}
        mod = sys.modules[type(env.unwrapped).__module__]  # the env file's geometry constants
        self.data = {
            "run": run,
            "env": mod.__name__,
            "every": every,
            "geometry": {
                "scale": mod.SCALE,
                "viewport": [mod.VIEWPORT_W, mod.VIEWPORT_H],
                "fps": mod.FPS,
                "lander_poly": mod.LANDER_POLY,
                "leg_half": [mod.LEG_W, mod.LEG_H],
                "tether_length": getattr(mod, "TETHER_LENGTH", None),
            },
            "param_labels": PARAM_LABELS,
            "returns": [],
            "param_log": [],  # [episode, attribute, value] whenever a task parameter changes
            "episodes": [],
        }

    def start(self, env, episode):
        """Call right after env.reset()."""
        u = env.unwrapped
        params = _task_params(u)
        for k, v in params.items():
            if self.params.get(k) != v:
                self.data["param_log"].append([episode, k, v])
        self.params = params

        if episode % self.every == 0 or episode == self.last_episode:
            terrain = [p[0] for p in u.sky_polys] + [u.sky_polys[-1][1]]
            if hasattr(u, "landers"):  # tethered: one wide pad
                pads = [[_r(u.pad[0]), _r(u.pad[1]), _r(u.helipad_y), "pad", 1]]
            elif hasattr(u, "pads"):  # [x1, x2, y, label, is_good]
                pads = [[_r(x1), _r(x2), _r(u.helipad_y), name, int(name == u.good_pad)]
                        for name, (x1, x2) in u.pads.items()]
            else:
                pads = [[_r(u.helipad_x1), _r(u.helipad_x2), _r(u.helipad_y), "pad", 1]]
            self.cur = {
                "episode": episode,
                "params": params,
                "terrain": [[_r(x), _r(y)] for x, y in terrain],
                "pads": pads,
                "two_pads": hasattr(u, "pads"),
                "frames": [],
            }
            if hasattr(u, "landers"):
                self.cur["targets"] = [_r(x) for x in u.targets]
            self._frame(u, [0.0] * len(u.landers) if hasattr(u, "landers") else 0.0)

    def step(self, env, reward):
        """Call after every env.step()."""
        if self.cur is not None:
            self._frame(env.unwrapped, reward)

    def end(self, env, ret, terminated, truncated, outcome=None, returns_each=None):
        """Call once the episode is over. outcome and returns_each are for multi-lander envs."""
        self.data["returns"].append(_r(ret))
        if self.cur is None:
            return
        u = env.unwrapped
        # mirrors the env's termination checks: body touched the ground, flew off-screen, or came to rest
        if outcome is not None:
            pass
        elif u.game_over:
            outcome = "crashed"
        elif terminated and not u.lander.awake:
            outcome = "landed"
            if hasattr(u, "pads"):
                outcome = ("landed off pad" if u.landed_pad is None else
                           "landed on good pad" if u.landed_pad == u.good_pad else "landed on bad pad")
        elif terminated:
            outcome = "out of bounds"
        else:
            outcome = "timed out"
        self.cur["return"] = _r(ret)
        if returns_each is not None:
            self.cur["returns_each"] = [_r(x) for x in returns_each]
        self.cur["outcome"] = outcome
        self.data["episodes"].append(self.cur)
        self.cur = None
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, separators=(",", ":")))
        os.replace(tmp, self.path)  # atomic, so the viewer never reads a half-written file

    def _frame(self, u, reward):
        if hasattr(u, "landers"):
            self.cur["frames"].append({
                "L": [[_r(l.position.x), _r(l.position.y), _r(l.angle)] for l in u.landers],
                "G": [[[_r(g.position.x), _r(g.position.y), _r(g.angle), int(g.ground_contact)] for g in legs]
                      for legs in u.legs],
                "F": [[[_r(v) for v in f] for f in fl] for fl in getattr(u, "flames", [[] for _ in u.landers])],
                "r": [_r(x) for x in reward],
            })
            return
        l = u.lander
        self.cur["frames"].append({
            "l": [_r(l.position.x), _r(l.position.y), _r(l.angle)],
            "g": [[_r(g.position.x), _r(g.position.y), _r(g.angle), int(g.ground_contact)] for g in u.legs],
            "f": [[_r(v) for v in f] for f in getattr(u, "flames", [])],
            "r": _r(reward),
        })
