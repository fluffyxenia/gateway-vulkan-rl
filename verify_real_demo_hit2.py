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

ram_window = {}
for t in range(4600):
    mb = actions[t].astype(np.int8)
    obs, _r, term, trunc, info = env.step(mb)
    if 4500 <= t <= 4599:
        ram_window[t] = env.unwrapped.get_ram().copy()
env.close()
print(f"captured {len(ram_window)} frames (4500-4599)")

targets_bright = {0x0888, 0x0AAA, 0x0EEE}
targets_dim = {0x0644, 0x0240, 0x0020}
candidates = {}
for t, ram in ram_window.items():
    for addr in range(0, 0xFFFE):
        val = (int(ram[addr]) << 8) | int(ram[addr+1])
        if val in targets_bright or val in targets_dim:
            candidates.setdefault(addr, set()).add(val)
found = False
for addr, vals in sorted(candidates.items()):
    if (vals & targets_bright) and (vals & targets_dim):
        found = True
        print(f"FLASH addr=0x{addr:x}: both dim+bright seen: {[hex(v) for v in vals]}")
if not found:
    print("Still no address with BOTH dim and bright seen across 4500-4599.")
    # report ANY address that touched $EEE specifically (the clearest "white" value, least likely to collide by chance)
    eee_addrs = [a for a, v in candidates.items() if 0x0EEE in v]
    print(f"Addresses that touched exactly 0x0EEE at any point: {[hex(a) for a in sorted(eee_addrs)]}")
