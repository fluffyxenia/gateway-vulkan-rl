"""Scan the FULL 64KB RAM array for the exact word-value signature of the
AIZ1-miniboss hit-flash ($0644/$0888 at one slot, $0240/$0AAA at another,
$0020/$0EEE at a third) during a genuinely-confirmed real hit, rather than
trust manual byte-counting through 40+ lines of asm declarations."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, N_BUTTONS, buttons_to_multibinary
from boss_hit_counter import BossHitDetector, structural_offset

LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/train_ep271_outcome-boss_defeated.npy"
actions = np.load(LOG)
mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
tracker.reset(pre_visited={tracker.end_idx})
obs, info = env.reset()

detector = BossHitDetector()
started = False
trigger_frame = None
ram_history = []  # (rel_frame, full_ram_copy)

for t in range(2000):  # cap for safety
    mb = buttons_to_multibinary(actions[t].astype(np.int8)) if t < len(actions) else np.zeros(12, dtype=bool)
    obs, _r, term, trunc, info = env.step(mb)
    x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
    r, cp_term, event = tracker.step(x, y, score, t)
    ram = env.unwrapped.get_ram()
    if event and "boss_trigger" in event and info.get("act") == 1 and not started:
        detector.start(ram)
        started = True
        trigger_frame = t
    if started:
        rel = t - trigger_frame
        if 0 <= rel <= 40:
            ram_history.append((rel, ram.copy()))
        confirmed = detector.step(ram) if detector.confirmed_addr is None else detector.confirmed_addr
        if confirmed is not None and rel > 40:
            break
        if rel > 45:
            break

env.close()

# Now scan: for each 16-bit-aligned address, check if its value sequence across
# these ~40 frames matches EITHER expected target word ($888,$AAA,$EEE) appearing
# at all (appearing+disappearing = flash), cross-referenced with the dim values
# ($644,$240,$20) at the SAME address in other frames.
targets_bright = {0x0888, 0x0AAA, 0x0EEE}
targets_dim = {0x0644, 0x0240, 0x0020}
candidates = {}
for rel, ram in ram_history:
    for addr in range(0xC000, 0xFFFE, 1):  # scan a broad high-RAM range, byte-aligned (words can start on odd bytes too in principle, but typically even)
        val = (int(ram[addr]) << 8) | int(ram[addr+1])
        if val in targets_bright or val in targets_dim:
            candidates.setdefault(addr, set()).add(val)

# report addresses that show BOTH a bright AND a dim value across the window (real flash)
for addr, vals in candidates.items():
    if (vals & targets_bright) and (vals & targets_dim):
        print(f"addr=0x{addr:x}: saw both dim and bright values: {[hex(v) for v in vals]}")
