import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, N_BUTTONS, buttons_to_multibinary
from boss_hit_counter import BossHitDetector, structural_offset

LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/train_ep271_outcome-boss_defeated.npy"
actions = np.load(LOG)
print(f"actions length: {len(actions)}")
mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
tracker.reset(pre_visited={tracker.end_idx})
obs, info = env.reset()

detector = BossHitDetector()
started = False
trigger_frame = None
ram_history = []
decrement_frames = []

for t in range(len(actions)):
    mb = buttons_to_multibinary(actions[t].astype(np.int8))
    obs, _r, term, trunc, info = env.step(mb)
    x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
    r, cp_term, event = tracker.step(x, y, score, t)
    ram = env.unwrapped.get_ram()
    if event and "boss_trigger" in event and info.get("act") == 1 and not started:
        detector.start(ram)
        started = True
        trigger_frame = t
        continue
    if started and detector.confirmed_addr is None:
        confirmed = detector.step(ram)
        rel = t - trigger_frame
        ram_history.append((rel, ram.copy()))
        if confirmed is not None:
            print(f"CONFIRMED addr=0x{confirmed:x} so={structural_offset(confirmed)} at t+{rel}, total frames captured={len(ram_history)}")
            break

env.close()

targets_bright = {0x0888, 0x0AAA, 0x0EEE}
targets_dim = {0x0644, 0x0240, 0x0020}
candidates = {}
for rel, ram in ram_history:
    for addr in range(0xC000, 0xFFFE):
        val = (int(ram[addr]) << 8) | int(ram[addr+1])
        if val in targets_bright or val in targets_dim:
            candidates.setdefault(addr, set()).add(val)

found_any = False
for addr, vals in candidates.items():
    if (vals & targets_bright) and (vals & targets_dim):
        found_any = True
        print(f"addr=0x{addr:x}: saw both dim and bright: {[hex(v) for v in vals]}")
if not found_any:
    print("No address found matching the full dim+bright flash signature in this window.")
