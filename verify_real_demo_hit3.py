import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME

d = np.load("/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000000.npz", allow_pickle=True)
actions_full = d["actions"]
actions = actions_full[:, :12]

env = retro.make(game=GAME, render_mode=None)
obs, info0 = env.reset()

candidates = [0xdda6, 0xfcf9, 0xfcfd]
history = {a: [] for a in candidates}
for t in range(4600):
    mb = actions[t].astype(np.int8)
    obs, _r, term, trunc, info = env.step(mb)
    if 4500 <= t <= 4599:
        ram = env.unwrapped.get_ram()
        for a in candidates:
            val = (int(ram[a]) << 8) | int(ram[a+1])
            history[a].append((t, hex(val)))
env.close()
for a in candidates:
    # print only value CHANGES to keep it readable
    seq = history[a]
    changes = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i][1] != changes[-1][1]:
            changes.append(seq[i])
    print(f"addr=0x{a:x} value changes: {changes}")
