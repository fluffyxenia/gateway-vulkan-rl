"""Check WHY a couple of newly-disqualified episodes get rejected, to confirm
the new step-size check is working as intended and not over-rejecting due to
a bug."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, N_BUTTONS, buttons_to_multibinary
from boss_hit_counter import structural_offset

for ep_prefix in ["baseline_ep001", "train_ep110", "train_ep262"]:
    import glob
    matches = glob.glob(f"/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/{ep_prefix}_outcome-boss_defeated.npy")
    if not matches:
        continue
    actions = np.load(matches[0])
    mode_cfg = START_MODES["boss"]
    env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
    boxes = parse_coords_file(COORDS_FILE)
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    tracker.reset(pre_visited={tracker.end_idx})
    obs, info = env.reset()
    started = False
    trigger_frame = None
    candidate_values = {}  # addr -> list of (frame, val)
    for t in range(len(actions)):
        mb = buttons_to_multibinary(actions[t].astype(np.int8))
        obs, _r, term, trunc, info = env.step(mb)
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        r, cp_term, event = tracker.step(x, y, score, t)
        ram = env.unwrapped.get_ram()
        if event and "boss_trigger" in event and info.get("act") == 1 and not started:
            started = True
            trigger_frame = t
            prev_ram = ram.copy()
            continue
        if started and (t - trigger_frame) <= 10:
            warped = np.nonzero((ram == 6) & (prev_ram != 6))[0]
            for addr in warped:
                addr = int(addr)
                candidate_values.setdefault(addr, []).append((t - trigger_frame, 6))
            prev_ram = ram.copy()
        if started and candidate_values and (t - trigger_frame) <= 60:
            for addr in list(candidate_values.keys()):
                val = int(ram[addr])
                last_val = candidate_values[addr][-1][1]
                if val != last_val:
                    candidate_values[addr].append((t - trigger_frame, val))
        if started and (t - trigger_frame) > 60:
            break
    env.close()
    print(f"=== {ep_prefix} ===")
    for addr, seq in candidate_values.items():
        if seq[-1][1] == 0 or len(seq) > 2:
            print(f"  addr=0x{addr:x} so={structural_offset(addr)} sequence={seq}")
