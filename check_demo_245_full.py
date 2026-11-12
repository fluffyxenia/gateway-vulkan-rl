"""Check (a) whether demos 2/4/5 terminate/truncate, and (b) whether there's
ANY second large x discontinuity after the main Act1->Act2 warp that might
represent an unaccounted-for second coordinate epoch."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME

for i in [2, 4, 5]:
    path = f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz"
    d = np.load(path, allow_pickle=True)
    actions_full = d["actions"]
    actions = actions_full[:, :12]
    env = retro.make(game=GAME, render_mode=None)
    obs, info0 = env.reset()
    xs, acts, prev_x = [], [], None
    discontinuities = []
    terminated_at = None
    for t in range(len(actions)):
        mb = actions[t].astype(np.int8)
        obs, _r, term, trunc, info = env.step(mb)
        x = info.get("x", 0)
        act = info.get("act")
        if prev_x is not None and act == 1 and abs(x - prev_x) > 200:
            discontinuities.append((t, prev_x, x))
        xs.append(x)
        acts.append(act)
        prev_x = x
        if term or trunc:
            terminated_at = t
            break
    env.close()
    print(f"demo {i}: total_recorded_actions={len(actions)}, replay_reached_frame={len(xs)}, "
          f"terminated_at={terminated_at}, final_x={xs[-1]}, final_act={acts[-1]}")
    print(f"  discontinuities (|dx|>200 within act==1) after main warp: {discontinuities}")
