import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import START_MODES

cfg = START_MODES["boss"]
env = retro.make(game=cfg["game"], state=cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
obs, info0 = env.reset()
obs, r, term, trunc, info = env.step(np.zeros(12, dtype=bool))
ram = env.unwrapped.get_ram()
target = info.get('x')
print(f"info['x'] = {target}, scanning for matching 16-bit big-endian value in RAM...")
matches = []
for addr in range(0, 0xFFFE):
    val = (int(ram[addr]) << 8) | int(ram[addr+1])
    if val == target:
        matches.append(addr)
print(f"found {len(matches)} matching addresses: {[hex(a) for a in matches[:30]]}")
env.close()
