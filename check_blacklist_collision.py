"""Check whether any of 2b's final confirmed addresses were ALSO, at some earlier
point in the SAME episode's boss_trigger window, already disqualified (e.g. by
singleton/increase/frame-gap) before later becoming a legitimate fresh candidate -
this would be the within-window version of Foxo's object-pooling concern,
observable even without needing a long post-window delay."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, N_BUTTONS, buttons_to_multibinary
from boss_hit_counter import structural_offset

verified_names = ["train_ep005", "train_ep110", "train_ep271", "train_ep283",
                   "train_ep287", "baseline_ep001", "baseline_ep002",
                   "baseline_ep003", "baseline_ep007"]

mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
WARP_VALUE = 6
WARP_WINDOW = 10

for ep_prefix in verified_names:
    path = f"/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/{ep_prefix}_outcome-boss_defeated.npy"
    actions = np.load(path)
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    tracker.reset(pre_visited={tracker.end_idx})
    obs, info = env.reset()
    started = False
    trigger_frame = None
    prev_ram = None
    ever_seen_addrs = {}  # addr -> list of (frame, event) for ANY address that ever touched 6 in the window
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
        if started and (t - trigger_frame) <= WARP_WINDOW + 5:  # a bit past window for context
            warped = np.nonzero((ram == WARP_VALUE) & (prev_ram != WARP_VALUE))[0]
            for addr in warped:
                addr = int(addr)
                ever_seen_addrs.setdefault(addr, []).append(t - trigger_frame)
            prev_ram = ram.copy()
        if started and t - trigger_frame > WARP_WINDOW + 20:
            break
    # report any address that warped to 6 MORE THAN ONCE within this window - that's
    # a slot getting reused/reallocated during the critical window itself
    multi = {a: frames for a, frames in ever_seen_addrs.items() if len(frames) > 1}
    print(f"{ep_prefix}: {len(ever_seen_addrs)} distinct addrs warped-to-6 in window; "
          f"{len(multi)} warped MORE THAN ONCE (slot reuse within window): {multi}")
env.close()
