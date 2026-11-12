import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME

path = "/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000002.npz"
d = np.load(path, allow_pickle=True)
actions_full = d["actions"]
actions = actions_full[:, :12]
env = retro.make(game=GAME, render_mode=None)
obs, info0 = env.reset()
for t in range(len(actions)):
    mb = actions[t].astype(np.int8)
    obs, _r, term, trunc, info = env.step(mb)
    if 6110 <= t <= 6383:
        print(f"t={t}: x={info.get('x')} y={info.get('y')} act={info.get('act')} zone={info.get('zone')} "
              f"lives={info.get('lives')} score={info.get('score')} rings={info.get('rings')}")
env.close()
