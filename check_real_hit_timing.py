"""Verify Foxo's ~29-31 frame minimum-gap estimate against REAL observed decrement
timing from a genuinely long, already-confirmed-legitimate 2b boss fight (episode
149, len=1494, not in the earlier disqualified list)."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, N_BUTTONS, START_MODES
from boss_hit_counter import structural_offset

LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/train_ep149_outcome-boss_defeated.npy"
actions = np.load(LOG)
print(f"actions shape: {actions.shape}")

mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
tracker.reset(pre_visited={tracker.end_idx})
obs, info = env.reset()

# Manually run the SAME warp-to-6 + monotonic-decrease scan as BossHitDetector,
# but log EVERY decrement event (not just confirm-at-0) for every candidate, to
# see real inter-decrement frame gaps for the eventually-confirmed address.
prev_ram = None
candidates = {}  # addr -> (last_val, [frames_of_decrement])
disqualified = set()
trigger_frame = None
confirmed_addr = None
WARP_VALUE = 6
WARP_WINDOW = 10

for t in range(len(actions)):
    from sonic_rl_train import buttons_to_multibinary
    mb = buttons_to_multibinary(actions[t].astype(np.int8))
    obs, _r, term, trunc, info = env.step(mb)
    x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
    r, cp_term, event = tracker.step(x, y, score, t)
    ram = env.unwrapped.get_ram()
    if event and "boss_trigger" in event and info.get("act") == 1:
        trigger_frame = t
        prev_ram = ram.copy()
        print(f"boss_trigger at frame {t}")
        continue
    if trigger_frame is not None and confirmed_addr is None:
        if t - trigger_frame <= WARP_WINDOW:
            warped = np.nonzero((ram == WARP_VALUE) & (prev_ram != WARP_VALUE))[0]
            for addr in warped:
                addr = int(addr)
                if addr not in candidates and addr not in disqualified:
                    candidates[addr] = (WARP_VALUE, [])
        for addr in list(candidates.keys()):
            val, decrements = candidates[addr]
            new_val = int(ram[addr])
            if new_val > val:
                del candidates[addr]
                disqualified.add(addr)
                continue
            if new_val < val:
                decrements.append(t)
                candidates[addr] = (new_val, decrements)
            else:
                candidates[addr] = (val, decrements)
            if new_val == 0:
                confirmed_addr = addr
                gaps = [decrements[i] - decrements[i-1] for i in range(1, len(decrements))]
                print(f"CONFIRMED addr=0x{addr:x} structural_offset={structural_offset(addr)} "
                      f"decrement frames={decrements} gaps_between_decrements={gaps}")
                break
        prev_ram = ram.copy()
    if confirmed_addr is not None:
        break

env.close()
if confirmed_addr is None:
    print("No confirmation reached in this replay (unexpected - was legitimate per earlier verification).")
