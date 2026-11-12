import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, buttons_to_multibinary
from boss_hit_counter import BossHitDetector, structural_offset

episodes = ["train_ep174", "train_ep138", "train_ep157", "train_ep120", "train_ep033"]
mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)

for ep_prefix in episodes:
    import glob
    matches = glob.glob(f"/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/{ep_prefix}_outcome-boss_defeated.npy")
    if not matches:
        continue
    actions = np.load(matches[0])
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    tracker.reset(pre_visited={tracker.end_idx})
    obs, info = env.reset()
    detector = BossHitDetector()
    started = False
    history = {}
    for t in range(len(actions)):
        mb = buttons_to_multibinary(actions[t].astype(np.int8))
        obs, _r, term, trunc, info = env.step(mb)
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        r, cp_term, event = tracker.step(x, y, score, t)
        ram = env.unwrapped.get_ram()
        if event and "boss_trigger" in event and info.get("act") == 1 and not started:
            detector.start(ram)
            started = True
            continue
        if started and detector.confirmed_addr is None:
            for addr, val in detector.candidates.items():
                history.setdefault(addr, []).append((t, val))
            confirmed = detector.step(ram)
            if confirmed is not None:
                hist = history.get(confirmed, [])
                decrements, last_val = [], None
                for frame, val in hist:
                    if val != last_val:
                        decrements.append(frame)
                        last_val = val
                gaps = [decrements[i] - decrements[i-1] for i in range(1, len(decrements))]
                print(f"{ep_prefix}: addr=0x{confirmed:x} so={structural_offset(confirmed)} decrements={decrements} gaps={gaps}")
                break
env.close()
