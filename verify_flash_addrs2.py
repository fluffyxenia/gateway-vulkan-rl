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
FLASH_ADDRS = [0xFC2E, 0xFC34, 0xFC36, 0xFC3C]
trigger_frame = None

def word_at(ram, addr):
    return (int(ram[addr]) << 8) | int(ram[addr+1])

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
        if 0 <= rel <= 40:
            vals = [hex(word_at(ram, a)) for a in FLASH_ADDRS]
            print(f"t+{rel}: flash_words={vals}")
        if confirmed is not None:
            print(f"CONFIRMED addr=0x{confirmed:x} so={structural_offset(confirmed)} at t+{rel}")
            break
    if started and (t - trigger_frame) > 50:
        break

env.close()
