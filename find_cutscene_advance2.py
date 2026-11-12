import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
demo_actions = d["actions"]

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
buttons = env.buttons
right_idx = buttons.index("RIGHT")
noop = np.zeros(len(buttons), dtype=np.int8)

obs, info = env.reset()
for i in range(demo_actions.shape[0]):
    obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))

# wait through the x-reset transition (no-op, ~600 steps based on prior tests)
for i in range(650):
    obs, r, terminated, truncated, info = env.step(noop)
print(f"after transition wait: x={info.get('x')} y={info.get('y')}")

action = np.zeros(len(buttons), dtype=np.int8)
action[right_idx] = 1
for i in range(3000):
    obs, r, terminated, truncated, info = env.step(action)
    x = info.get("x", 0)
    if i % 300 == 0:
        print(f"hold-right step {i}: x={x} y={info.get('y')}")
    if x > 40:
        print(f"MOVEMENT at step {i}: x={x}")
        break
print(f"final: x={info.get('x')}")
env.close()
