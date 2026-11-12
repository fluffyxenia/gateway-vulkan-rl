"""Fixed version: replay real demo -> WAIT (no-op) through the cutscene until x
genuinely resets to the new small-value space and starts increasing again (not a
fixed step count guess) -> THEN switch to scripted heuristic."""
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

for trial in range(2):
    obs, info = env.reset()
    tracker.reset()
    t_total = 0
    for i in range(demo_actions.shape[0]):
        obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        rr, cp_term, event = tracker.step(x, y, score, t_total)
        t_total += 1

    print(f"trial {trial}: after demo replay, x={info.get('x')}")

    # Wait through the cutscene with no-op, watching for the x-reset + resumed movement
    reset_detected = False
    for i in range(3000):
        obs, r, terminated, truncated, info = env.step(noop)
        x = info.get("x", 0)
        t_total += 1
        if i % 500 == 0:
            print(f"  wait step {i}: x={x} y={info.get('y')}")
        if x < 1000 and not reset_detected:
            reset_detected = True
            print(f"  x-reset detected at wait step {i}: x={x}")
    print(f"trial {trial}: after 3000-step wait, x={info.get('x')} reset_detected={reset_detected}")

    if reset_detected:
        last_x = None
        stuck_count = 0
        jump_timer = 0
        boss_reached = False
        action = np.zeros(len(buttons), dtype=np.int8)
        for t in range(5000):
            action[:] = 0
            action[right_idx] = 1
            jump_timer += 1
            if stuck_count > 10 or jump_timer % 20 == 0:
                action[b_idx] = 1
                jump_timer = 0
            obs, r, terminated, truncated, info = env.step(action)
            x = info.get("x", 0)
            score = info.get("score", 0)
            if last_x is not None:
                stuck_count = stuck_count + 1 if x <= last_x else 0
            last_x = x
            rr, cp_term, event = tracker.step(x, info.get("y", 0), score, t_total)
            t_total += 1
            if event is not None and event.startswith("boss_trigger"):
                boss_reached = True
                print(f"  BOSS TRIGGER REACHED at scripted step {t}")
                break
            if cp_term or terminated or truncated:
                print(f"  episode ended during scripted phase at step {t}, event={event}, x={x}")
                break
        print(f"trial {trial}: FINAL x={info.get('x')} boss_reached={boss_reached}")

env.close()
