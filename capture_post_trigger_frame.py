"""One-off exploration: replay the same demo, step past its end with no-ops, and
save screenshots at key points (right before the x-jump, right after, and further
along) as PNGs to visually inspect what's actually on screen - numbers alone
(x/y/lives/score) don't tell us if this is a cutscene, Act 2 start, or a boss arena.
"""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
actions = d["actions"]

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
obs, info = env.reset()
for i in range(actions.shape[0]):
    obs, r, terminated, truncated, info = env.step(actions[i].astype(np.int8))

noop = np.zeros(len(env.buttons), dtype=np.int8)
checkpoints = [0, 500, 590, 595, 600, 650, 800, 1199]
last_i = 0
for target in checkpoints:
    for i in range(last_i, target):
        obs, r, terminated, truncated, info = env.step(noop)
    last_i = target
    Image.fromarray(obs).save(f"/tmp/frame_step{target}.png")
    print(f"saved frame at step {target}: x={info.get('x')} y={info.get('y')}")

env.close()
