"""Check #1: is Tails (Player_2) actually active in 2b/2c's savestates? Check
Player_2's object slot (0xb04a, confirmed via the Touch_Enemy research earlier
tonight: "($FFFFB000) for main character, ($FFFFB04A) for sidekick") directly -
if it's moving/updating over a few no-op steps, Tails is genuinely active."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import START_MODES

P1_BASE = 0xb000
P2_BASE = 0xb04a
X_POS_OFF, Y_POS_OFF, STATUS_OFF, ROUTINE_OFF = 0x10, 0x14, 0x2A, 0x05

def dump_char(ram, base, label):
    x = (int(ram[base+X_POS_OFF]) << 8) | int(ram[base+X_POS_OFF+1])
    y = (int(ram[base+Y_POS_OFF]) << 8) | int(ram[base+Y_POS_OFF+1])
    status = int(ram[base+STATUS_OFF])
    routine = int(ram[base+ROUTINE_OFF])
    print(f"    {label}: x={x} y={y} status=0x{status:02x} routine={routine}")

for mode in ["boss", "mid"]:
    cfg = START_MODES[mode]
    print(f"=== {mode} ({cfg['state']}) ===")
    env = retro.make(game=cfg["game"], state=cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
    obs, info0 = env.reset()
    for t in range(5):
        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=bool))
        ram = env.unwrapped.get_ram()
        print(f"  step {t}:")
        dump_char(ram, P1_BASE, "Player_1 (Sonic)")
        dump_char(ram, P2_BASE, "Player_2 (Tails?)")
    env.close()
