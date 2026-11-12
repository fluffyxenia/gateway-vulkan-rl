"""For the 4 WholeAct demos without a timer-freeze hit, check directly whether
they reached the boss-trigger region (act==1 and x>=4500ish) versus never
getting there - distinguishing 'reached but didn't win' from 'never reached'."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME

for i in [1, 2, 4, 5]:
    path = f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz"
    d = np.load(path, allow_pickle=True)
    actions_full = d["actions"]
    actions = actions_full[:, :12]
    env = retro.make(game=GAME, render_mode=None)
    obs, info0 = env.reset()
    max_x_in_act1 = -1
    reached_boss_x = False
    act1_frame_count = 0
    last_act = 0
    for t in range(len(actions)):
        mb = actions[t].astype(np.int8)
        obs, _r, term, trunc, info = env.step(mb)
        act = info.get("act")
        x = info.get("x", 0)
        if act == 1:
            act1_frame_count += 1
            if x > max_x_in_act1:
                max_x_in_act1 = x
            if x >= 4500:
                reached_boss_x = True
        last_act = act
        if term or trunc:
            break
    env.close()
    print(f"demo {i}: {len(actions)} total frames, act1_frames={act1_frame_count}, "
          f"max_x_in_act1={max_x_in_act1}, reached_boss_trigger_x(>=4500)={reached_boss_x}, "
          f"final_act={last_act}")
