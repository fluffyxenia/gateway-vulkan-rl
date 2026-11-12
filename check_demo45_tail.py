import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME

for i in [4, 5]:
    path = f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz"
    d = np.load(path, allow_pickle=True)
    actions_full = d["actions"]
    actions = actions_full[:, :12]
    env = retro.make(game=GAME, render_mode=None)
    obs, info0 = env.reset()
    tail = []
    for t in range(len(actions)):
        mb = actions[t].astype(np.int8)
        obs, _r, term, trunc, info = env.step(mb)
        if t >= len(actions) - 40:
            tail.append((t, info.get('x'), info.get('y'), info.get('rings'), info.get('lives'), info.get('score')))
    env.close()
    print(f"=== demo {i} last 40 frames ===")
    for row in tail:
        print(row)
