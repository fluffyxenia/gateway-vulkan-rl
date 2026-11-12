"""One-off exploration (not standing infra): replays a real demo's exact action
sequence (deterministic), then continues stepping with neutral (no-op) input well
past the recording's end, logging x/y/lives/score/done-flags every few frames, to
understand what actually happens after the current end-trigger (x>=12210) - cutscene,
miniboss arena, zone transition, etc. - before designing boss-training/cutscene-
handling work around it.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
actions = d["actions"]
print(f"Demo has {actions.shape[0]} recorded frames")

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
obs, info = env.reset()

for i in range(actions.shape[0]):
    obs, r, terminated, truncated, info = env.step(actions[i].astype(np.int8))
    if terminated or truncated:
        print(f"env ended early at replay step {i}: terminated={terminated} truncated={truncated}")
        break

print(f"After replaying full demo: x={info.get('x')} y={info.get('y')} lives={info.get('lives')} score={info.get('score')}")

noop = np.zeros(len(env.buttons), dtype=np.int8)
EXTRA_STEPS = 1200
for i in range(EXTRA_STEPS):
    obs, r, terminated, truncated, info = env.step(noop)
    if i % 30 == 0 or terminated or truncated:
        print(f"extra step {i}: x={info.get('x')} y={info.get('y')} lives={info.get('lives')} "
              f"score={info.get('score')} terminated={terminated} truncated={truncated}")
    if terminated or truncated:
        print("env signaled done - stopping")
        break

env.close()
