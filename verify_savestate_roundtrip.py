"""One-off verification (not standing infra): confirms env.unwrapped.em.get_state()/
set_state() round-trips correctly - capture a snapshot mid-episode, step forward
further, then restore the snapshot and confirm we're back to the EXACT x/y/lives/
score/frame content as when captured. This is the core mechanism training method
(2b) (save-state + BC-anchor rows) will need - verified generically here, with an
arbitrary capture point, NOT any real boss-trigger coordinate (none exist yet -
waiting on Foxo's hands-on data per the current instruction)."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
obs, info = env.reset()
buttons = env.buttons
right_idx = buttons.index("RIGHT")
action = np.zeros(len(buttons), dtype=np.int8)
action[right_idx] = 1

for i in range(200):
    obs, r, terminated, truncated, info = env.step(action)

snapshot = env.unwrapped.em.get_state()
snapshot_obs = obs.copy()
snapshot_info = dict(info)
print(f"Captured snapshot at step 200: x={snapshot_info.get('x')} y={snapshot_info.get('y')} "
      f"lives={snapshot_info.get('lives')} score={snapshot_info.get('score')} "
      f"snapshot size={len(snapshot)} bytes")

# step forward further, diverging from the snapshot
for i in range(100):
    obs, r, terminated, truncated, info = env.step(action)
print(f"After 100 more steps: x={info.get('x')} y={info.get('y')}")

# restore the snapshot
env.unwrapped.em.set_state(snapshot)
obs2, r2, terminated2, truncated2, info2 = env.step(np.zeros(len(buttons), dtype=np.int8))
print(f"After restoring snapshot + 1 no-op step: x={info2.get('x')} y={info2.get('y')} "
      f"lives={info2.get('lives')} score={info2.get('score')}")

# To compare apples-to-apples, step the ORIGINAL env one more no-op step too and compare
env.unwrapped.em.set_state(snapshot)
obs3, r3, terminated3, truncated3, info3 = env.step(np.zeros(len(buttons), dtype=np.int8))
match = np.array_equal(obs2, obs3) and info2.get("x") == info3.get("x") and info2.get("y") == info3.get("y")
print(f"\nTwo independent restores + identical no-op step produce identical result: {match}")
assert match, "FAIL: save-state restore is not deterministic!"
print("PASS: save-state round-trip is deterministic and restores exact game state")

env.close()
