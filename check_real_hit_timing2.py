"""Verify Foxo's ~29-31 frame minimum-gap estimate against REAL observed decrement
timing, using the ACTUAL (singleton-fixed) BossHitDetector class directly, with a
side-channel history tracker to log every decrement frame for whichever candidate
it ultimately confirms."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, N_BUTTONS, START_MODES, buttons_to_multibinary
from boss_hit_counter import BossHitDetector, structural_offset

LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/train_ep149_outcome-boss_defeated.npy"
actions = np.load(LOG)

mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
tracker.reset(pre_visited={tracker.end_idx})
obs, info = env.reset()

detector = BossHitDetector()
started = False
history = {}  # addr -> list of (frame, val)

for t in range(len(actions)):
    mb = buttons_to_multibinary(actions[t].astype(np.int8))
    obs, _r, term, trunc, info = env.step(mb)
    x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
    r, cp_term, event = tracker.step(x, y, score, t)
    ram = env.unwrapped.get_ram()
    if event and "boss_trigger" in event and info.get("act") == 1 and not started:
        print(f"boss_trigger at frame {t}")
        detector.start(ram)
        started = True
        continue
    if started and detector.confirmed_addr is None:
        # snapshot live (non-disqualified) candidates' values before stepping,
        # to build a full per-candidate value history for whichever one wins
        for addr, val in detector.candidates.items():
            history.setdefault(addr, []).append((t, val))
        confirmed = detector.step(ram)
        if confirmed is not None:
            print(f"CONFIRMED addr=0x{confirmed:x} structural_offset={structural_offset(confirmed)}")
            hist = history.get(confirmed, [])
            # dedupe consecutive same-value entries, keep only frames where value changed
            decrements = []
            last_val = None
            for frame, val in hist:
                if val != last_val:
                    decrements.append(frame)
                    last_val = val
            gaps = [decrements[i] - decrements[i-1] for i in range(1, len(decrements))]
            print(f"  decrement frames (approx, first-seen-at-new-value): {decrements}")
            print(f"  gaps between decrements: {gaps}")
            break

env.close()
