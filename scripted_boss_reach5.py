import sys
import numpy as np
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
up_idx = buttons.index("UP")
noop = np.zeros(len(buttons), dtype=np.int8)

for seed in range(8):
    rng = np.random.default_rng(seed)
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    obs, info = env.reset()
    tracker.reset()
    t_total = 0
    for i in range(demo_actions.shape[0]):
        obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        rr, cp_term, event = tracker.step(x, y, score, t_total)
        t_total += 1
    for i in range(650):
        obs, r, terminated, truncated, info = env.step(noop)
        t_total += 1

    action = np.zeros(len(buttons), dtype=np.int8)
    last_x = None
    stuck_count = 0
    boss_reached = False
    max_x = 0
    jump_hold = 0
    for t in range(5500):
        action[:] = 0
        action[right_idx] = 1
        if jump_hold > 0:
            action[b_idx] = 1
            jump_hold -= 1
        elif stuck_count > 6 or rng.random() < 0.08:
            jump_hold = int(rng.integers(3, 12))  # variable jump hold duration
        obs, r, terminated, truncated, info = env.step(action)
        x = info.get("x", 0)
        score = info.get("score", 0)
        if last_x is not None:
            stuck_count = stuck_count + 1 if x <= last_x else 0
            max_x = max(max_x, x)
        last_x = x
        rr, cp_term, event = tracker.step(x, info.get("y", 0), score, t_total)
        t_total += 1
        if event is not None and event.startswith("boss_trigger"):
            boss_reached = True
            print(f"seed {seed}: BOSS TRIGGER REACHED at step {t}, total={t_total}")
            break
        if cp_term or terminated or truncated:
            break
    print(f"seed {seed}: max_x={max_x} final_x={info.get('x')} boss_reached={boss_reached} event={event}")

env.close()
