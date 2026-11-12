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
    scores = []
    for t in range(len(actions)):
        mb = actions[t].astype(np.int8)
        obs, _r, term, trunc, info = env.step(mb)
        scores.append(info.get("score", 0))
        if term or trunc:
            break
    env.close()
    # print last 30 frames' score, and any jump anywhere
    print(f"=== demo {i}: {len(scores)} frames ===")
    prev = scores[0]
    for t in range(1, len(scores)):
        if scores[t] != prev:
            print(f"  score change at frame {t}: {prev} -> {scores[t]} ({len(scores)-t} frames remain)")
        prev = scores[t]
