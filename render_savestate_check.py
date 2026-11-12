import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import START_MODES
from PIL import Image

for mode in ["boss", "mid"]:
    cfg = START_MODES[mode]
    env = retro.make(game=cfg["game"], state=cfg["state"], inttype=retro.data.Integrations.ALL, render_mode="rgb_array")
    obs, info0 = env.reset()
    for t in range(60):
        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=bool))
    frame = env.render()
    Image.fromarray(frame).save(f"/home/foxo/gateway-vulkan-rl/check_{mode}_frame.png")
    env.close()
    print(f"saved check_{mode}_frame.png")
