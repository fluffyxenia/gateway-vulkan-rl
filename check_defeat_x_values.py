"""Check (c): what x value do the two CONFIRMED (timer-freeze-verified) demos
actually show at the moment of defeat - is 4500 really the right threshold,
verified against known-good data, not just inferred?"""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME
from timer_defeat_check import check_timer_freeze_defeat, TIMER_BYTE_ADDR

for i in [0, 3]:
    path = f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz"
    d = np.load(path, allow_pickle=True)
    actions_full = d["actions"]
    actions = actions_full[:, :12]
    env = retro.make(game=GAME, render_mode=None)
    obs, info0 = env.reset()
    scores, timer_bytes, xs, acts = [], [], [], []
    for t in range(len(actions)):
        mb = actions[t].astype(np.int8)
        obs, _r, term, trunc, info = env.step(mb)
        scores.append(info.get("score", 0))
        xs.append(info.get("x", 0))
        acts.append(info.get("act"))
        ram = env.unwrapped.get_ram()
        timer_bytes.append(int(ram[TIMER_BYTE_ADDR]))
        if term or trunc:
            print(f"demo {i}: ENV TERMINATED/TRUNCATED at frame {t} (of {len(actions)} recorded actions)")
            break
    env.close()
    hits = check_timer_freeze_defeat(scores, timer_bytes)
    for h in hits:
        print(f"demo {i}: defeat at frame {h}, x={xs[h]}, act={acts[h]}, total_frames_in_recording={len(actions)}, frames_after_defeat={len(actions)-h}")
