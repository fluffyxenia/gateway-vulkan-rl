import sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
demo_actions = d["actions"]
boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
buttons = env.buttons
right_idx = buttons.index("RIGHT")
b_idx = buttons.index("B")
noop = np.zeros(len(buttons), dtype=np.int8)

rng = np.random.default_rng(2000)
tracker = CheckpointTracker(boxes, extend_past_end=True)
obs, info = env.reset()
tracker.reset()
t_total = 0
for i in range(demo_actions.shape[0]):
    obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))
    t_total += 1
for i in range(650):
    obs, r, terminated, truncated, info = env.step(noop)
    t_total += 1

action = np.zeros(len(buttons), dtype=np.int8)
last_x, stuck_count, jump_hold = None, 0, 0
for t in range(6500):
    x_now = last_x if last_x is not None else 0
    in_loop_zone = 3000 <= x_now <= 3450
    action[:] = 0
    action[right_idx] = 1
    if not in_loop_zone:
        if jump_hold > 0:
            action[b_idx] = 1
            jump_hold -= 1
        elif stuck_count > 4 or rng.random() < 0.1:
            jump_hold = int(rng.integers(3, 12))
    obs, r, terminated, truncated, info = env.step(action)
    x = info.get("x", 0)
    if last_x is not None:
        stuck_count = stuck_count + 1 if x <= last_x else 0
    last_x = x
    if x >= 3880:
        Image.fromarray(obs).save(f"/tmp/wall_frame_{t}.png")
        print(f"step {t}: x={x} y={info.get('y')} saved screenshot")
    if x >= 3893 and stuck_count > 100:
        print("confirmed stuck")
        break
env.close()
