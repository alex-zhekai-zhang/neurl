"""Render one LunarLander episode per checkpoint into a single GIF.

Run reinforce_lunar.py first to produce the checkpoints in checkpoints/.
"""
import gymnasium as gym
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figures"
CKPT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)


SEED = 0         # same terrain and starting push for every segment
FRAME_SKIP = 2   # env runs at 50 fps; keep every 2nd frame -> 25 fps GIF in real time
HOLD_FRAMES = 15 # freeze on the last frame of each episode for ~0.6 s
NORMAL, SWAPPED = [0, 1, 2, 3], [0, 3, 2, 1]

SEGMENTS = [
    ("Untrained agent", CKPT_DIR / "lunar_ckpt_init.pt", NORMAL),
    ("Trained (before switch)", CKPT_DIR / "lunar_ckpt_switch.pt", NORMAL),
    ("Same agent, controls swapped", CKPT_DIR / "lunar_ckpt_switch.pt", SWAPPED),
    ("Final agent (swapped controls)", CKPT_DIR / "reinforce_lunar.pt", SWAPPED),
]

env = gym.make("LunarLander-v3", render_mode="rgb_array")
policy = nn.Sequential(  # must match the network in reinforce_lunar.py
    nn.Linear(env.observation_space.shape[0], 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, env.action_space.n),
)
font = ImageFont.load_default(size=20)


def label(frame, title, ret):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), title, fill="white", font=font)
    draw.text((10, 36), f"return: {ret:7.1f}", fill="white", font=font)
    return img


frames = []
torch.manual_seed(SEED)
for title, ckpt, mapping in SEGMENTS:
    policy.load_state_dict(torch.load(ckpt))
    obs, _ = env.reset(seed=SEED)
    ret, t, done = 0.0, 0, False
    while not done:
        if t % FRAME_SKIP == 0:
            frames.append(label(env.render(), title, ret))
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
        action = torch.distributions.Categorical(logits=logits).sample().item()
        obs, r, terminated, truncated, _ = env.step(mapping[action])
        ret += r
        t += 1
        done = terminated or truncated
    frames += [label(env.render(), title, ret)] * HOLD_FRAMES
    print(f"{title:32s} return {ret:7.1f}  ({t} steps)")

frames[0].save(FIG_DIR / "lunar.gif", save_all=True, append_images=frames[1:],
               duration=1000 * FRAME_SKIP // 50, loop=0)
env.close()
print(f"saved figures/lunar.gif ({len(frames)} frames)")
