"""Find out how long the Act1->Act2 cutscene actually takes to finish via pure
no-op waiting, with a much longer budget, and capture a save-state the moment real
interactive gameplay resumes (x starts increasing from the reset position) - this
save-state can be reused for all subsequent boss-reach attempts AND doubles as the
method-2c save-state (right after the first cutscene)."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro

DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
demo_actions = d["actions"]

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
buttons = env.buttons
noop = np.zeros(len(buttons), dtype=np.int8)

obs, info = env.reset()
for i in range(demo_actions.shape[0]):
    obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))
print(f"after demo replay: x={info.get('x')}")

last_x = None
stable_count = 0
for i in range(15000):
    obs, r, terminated, truncated, info = env.step(noop)
    x = info.get("x", 0)
    if i % 1000 == 0:
        print(f"wait step {i}: x={x} y={info.get('y')} lives={info.get('lives')}")
    if last_x is not None and x > last_x and x < 1000:
        # real forward movement resumed, in the new post-cutscene space
        print(f"MOVEMENT RESUMED at wait step {i}: x went {last_x} -> {x}")
        snapshot = env.unwrapped.em.get_state()
        with open("/home/foxo/gateway-vulkan-rl/aiz1_post_cutscene_savestate.state", "wb") as f:
            f.write(snapshot)
        print(f"Captured save-state ({len(snapshot)} bytes) -> aiz1_post_cutscene_savestate.state")
        break
    last_x = x
else:
    print("Movement never resumed within 15000 wait steps - something else is going on")

env.close()
