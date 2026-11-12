"""Use the genuine human-played full-act demo (score jumped exactly 30->130
between frames 4547/4548, matching the decomp source's +100 boss-defeat bonus)
as ground truth to find the REAL hit-counter address and REAL flash signature,
bypassing all of tonight's uncertain RL-rollout "confirmations" which have
repeatedly turned out to be false positives."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME

d = np.load("/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000000.npz", allow_pickle=True)
actions_full = d["actions"]  # (T, 24) - 2-player, slice to player 0
actions = actions_full[:, :12]

env = retro.make(game=GAME, render_mode=None)
obs, info0 = env.reset()

ram_window = {}  # frame -> ram copy, for frames 4530-4560
for t in range(4560):
    mb = actions[t].astype(np.int8)
    obs, _r, term, trunc, info = env.step(mb)
    if 4530 <= t <= 4559:
        ram_window[t] = env.unwrapped.get_ram().copy()
    if t == 4559:
        break
env.close()

print(f"captured {len(ram_window)} frames")

# (a) find any byte that DECREASED to exactly 0 right at the score-jump moment (4547->4548)
ram_4546 = ram_window[4546]
ram_4548 = ram_window[4548]
ram_4549 = ram_window[4549]
decreased_to_zero = []
for addr in range(0x10000):
    if ram_4548[addr] == 0 and ram_4546[addr] != 0 and ram_4546[addr] < 20:
        decreased_to_zero.append((addr, int(ram_4546[addr])))
print(f"Bytes that were small(<20) nonzero at frame 4546 and exactly 0 at frame 4548: {decreased_to_zero[:30]}")

# (b) scan for the flash dim/bright signature across the window
targets_bright = {0x0888, 0x0AAA, 0x0EEE}
targets_dim = {0x0644, 0x0240, 0x0020}
candidates = {}
for t, ram in ram_window.items():
    for addr in range(0xC000, 0xFFFE):
        val = (int(ram[addr]) << 8) | int(ram[addr+1])
        if val in targets_bright or val in targets_dim:
            candidates.setdefault(addr, set()).add(val)
found = False
for addr, vals in candidates.items():
    if (vals & targets_bright) and (vals & targets_dim):
        found = True
        print(f"FLASH addr=0x{addr:x}: saw both dim and bright: {[hex(v) for v in vals]}")
if not found:
    print("No flash signature address found in this window either.")
