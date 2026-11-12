import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import START_MODES

cfg = START_MODES["boss"]
env = retro.make(game=cfg["game"], state=cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
obs, info0 = env.reset()
for t in range(3):
    obs, r, term, trunc, info = env.step(np.zeros(12, dtype=bool))
    ram = env.unwrapped.get_ram()
    direct_x = (int(ram[0xb010]) << 8) | int(ram[0xb011])
    print(f"step {t}: info['x']={info.get('x')}  direct_read(0xb010)={direct_x}")
env.close()
