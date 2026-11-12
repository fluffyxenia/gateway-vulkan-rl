"""Test whether pressing START or A periodically advances the frozen cutscene -
pure no-op waiting for 15000 steps (250s) did NOT work, so something else must be
needed."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
demo_actions = d["actions"]

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
buttons = env.buttons
print("buttons:", buttons)
start_idx = buttons.index("START")
a_idx = buttons.index("A")
b_idx = buttons.index("B")

obs, info = env.reset()
for i in range(demo_actions.shape[0]):
    obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))
print(f"after demo replay: x={info.get('x')}")

action = np.zeros(len(buttons), dtype=np.int8)
for i in range(3000):
    action[:] = 0
    # press start/A/B periodically, each on its own phase to distinguish effects
    if i % 90 < 3:
        action[start_idx] = 1
    elif i % 90 < 6:
        action[a_idx] = 1
    elif i % 90 < 9:
        action[b_idx] = 1
    obs, r, terminated, truncated, info = env.step(action)
    x = info.get("x", 0)
    if i % 300 == 0:
        print(f"step {i}: x={x} y={info.get('y')} lives={info.get('lives')}")
    if x < 1000 and x != 32:
        print(f"MOVEMENT at step {i}: x={x}")
        break

print(f"final: x={info.get('x')}")
env.close()
