"""For 2b's 10 FINAL (triple-fix-verified) legitimate boss defeats, check WHEN
(relative to boss_trigger) each winning candidate was first discovered (its
warp-to-6 moment) - directly tests whether WARP_WINDOW_FRAMES=10 is actually
cutting off real candidates in practice, and informs whether Foxo's
disqualification-expiry idea would even help without also widening this window.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, N_BUTTONS, buttons_to_multibinary
from boss_hit_counter import BossHitDetector, structural_offset

verified_names = ["train_ep005", "train_ep110", "train_ep262", "train_ep271",
                   "train_ep283", "train_ep287", "baseline_ep001", "baseline_ep002",
                   "baseline_ep003", "baseline_ep007"]

mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)

for ep_prefix in verified_names:
    path = f"/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/{ep_prefix}_outcome-boss_defeated.npy"
    actions = np.load(path)
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    tracker.reset(pre_visited={tracker.end_idx})
    obs, info = env.reset()
    detector = BossHitDetector()
    started = False
    trigger_frame = None
    discovered_frame = {}
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
            pre_candidates = set(detector.candidates.keys())
            confirmed = detector.step(ram)
            new_candidates = set(detector.candidates.keys()) - pre_candidates
            for addr in new_candidates:
                if addr not in discovered_frame:
                    discovered_frame[addr] = t - trigger_frame
            if confirmed is not None:
                disc = discovered_frame.get(confirmed, "unknown(pre-existing?)")
                print(f"{ep_prefix}: confirmed addr=0x{confirmed:x} so={structural_offset(confirmed)} "
                      f"discovered at frame+{disc} relative to trigger, confirmed at frame+{t-trigger_frame}")
                break
env.close()
