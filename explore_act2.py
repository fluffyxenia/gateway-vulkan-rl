"""One-off exploration: load AngelIslandZone.Act2 directly via its built-in state,
check spawn x/y, then step forward holding RIGHT (simple heuristic, not a real
policy) to see how much traversal separates Act2's start from any cutscene/boss
gate - informs whether boss training (point 2) is tractable without full Act2 demo
data, or needs real human recordings first."""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", state="AngelIslandZone.Act2", render_mode=None)
obs, info = env.reset()
print(f"Act2 info right after reset: {info}")

buttons = env.buttons
right_idx = buttons.index("RIGHT")
action = np.zeros(len(buttons), dtype=np.int8)
action[right_idx] = 1

xs = []
for i in range(2400):
    obs, r, terminated, truncated, info = env.step(action)
    x, y = info.get("x"), info.get("y")
    xs.append((i, x, y, info.get("lives"), info.get("score")))
    if i % 200 == 0:
        print(f"step {i}: x={x} y={y} lives={info.get('lives')}")
        Image.fromarray(obs).save(f"/tmp/act2_step{i}.png")
    if terminated or truncated:
        print(f"env ended at step {i}: terminated={terminated} truncated={truncated}")
        break

env.close()
print("\nfinal state:", xs[-1])
