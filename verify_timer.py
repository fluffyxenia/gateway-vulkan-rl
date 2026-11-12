"""Verify the calibrated Timer address (0xfe22, derived by counting bytes from
the already-confirmed `score` address at 0xfe26 backward through Ring_count+
Timer's declarations in sonic3k.constants.asm - NOT long-distance hand-counting
like the palette attempt) against real human demo data, before trusting it."""
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

TIMER_MIN, TIMER_SEC, TIMER_FRAME = 0xfe23, 0xfe24, 0xfe25
RING_COUNT = 0xfe20

for t in range(4560):
    mb = actions[t].astype(np.int8)
    obs, _r, term, trunc, info = env.step(mb)
    if t in (0, 1, 60, 120, 600, 1200, 4540, 4545, 4546, 4547, 4548, 4549, 4550, 4555, 4559):
        ram = env.unwrapped.get_ram()
        print(f"t={t}: timer={ram[TIMER_MIN]}:{ram[TIMER_SEC]:02d}:{ram[TIMER_FRAME]:02d} "
              f"ram_ring_count={(int(ram[RING_COUNT])<<8)|int(ram[RING_COUNT+1])} "
              f"info_rings={info.get('rings')} info_score={info.get('score')}")
env.close()
