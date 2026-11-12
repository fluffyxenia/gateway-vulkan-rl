import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
demo_actions = d["actions"]

boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")
tracker = CheckpointTracker(boxes, extend_past_end=True)

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
buttons = env.buttons
right_idx = buttons.index("RIGHT")
b_idx = buttons.index("B")
noop = np.zeros(len(buttons), dtype=np.int8)

for trial in range(3):
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
    print(f"trial {trial}: after replay+wait, x={info.get('x')}")

    action = np.zeros(len(buttons), dtype=np.int8)
    last_x = None
    stuck_count = 0
    jump_timer = 0
    boss_reached = False
    max_x = 0
    for t in range(6000):
        action[:] = 0
        action[right_idx] = 1
        jump_timer += 1
        if stuck_count > 8 or jump_timer % 15 == 0:
            action[b_idx] = 1
            jump_timer = 0
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
            print(f"  BOSS TRIGGER REACHED at scripted step {t}, total_steps={t_total}")
            break
        if cp_term or terminated or truncated:
            print(f"  episode ended at scripted step {t}, event={event}, x={x}, max_x={max_x}")
            break
    print(f"trial {trial}: FINAL x={info.get('x')} max_x={max_x} boss_reached={boss_reached}")

env.close()
