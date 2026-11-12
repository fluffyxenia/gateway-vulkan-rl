"""
Generate a synthetic "fake human demo" .npz in EXACTLY the format
stable_retro.scripts.playback_movie.py --npy-actions --info-dict produces (actions bool
array + info object array, both saved into one .npz via np.savez_compressed) - for
testing the ingest_demo.py / bc_pretrain.py pipeline before any real human recordings
exist. Uses a real (short) rollout through the actual env so "actions" and "info" are
genuinely causally consistent with each other (info is captured by literally stepping
the env with these exact actions), which lets the ingestion pipeline's determinism
cross-check be meaningfully tested (it should PASS, since this really is deterministic
self-consistent data) - only the cutscene-trigger sanity check is expected to fail,
since a short random rollout has no realistic chance of reaching the real end point.
"""
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from sonic_rl_train import GAME, BUTTONS

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/foxo/gateway-vulkan-rl/fake_demo.npz"

env = retro.make(game=GAME, render_mode=None)
obs, info = env.reset()
rng = np.random.default_rng(55)

actions = np.zeros((N_STEPS, len(BUTTONS)), dtype=bool)
info_steps = []
for t in range(N_STEPS):
    # bias toward RIGHT to get some real motion, like a human mashing forward
    mb = (rng.random(len(BUTTONS)) < 0.15).astype(bool)
    mb[BUTTONS.index("RIGHT")] = rng.random() < 0.8
    mb[BUTTONS.index("MODE")] = False
    mb[BUTTONS.index("START")] = False
    actions[t] = mb
    obs, reward, terminated, truncated, info = env.step(mb.astype(np.int8))
    info_steps.append(info)
    if terminated or truncated:
        actions = actions[:t + 1]
        break

env.close()
np.savez_compressed(OUT, actions=actions, info=np.array(info_steps, dtype=object))
print(f"wrote {OUT}: actions shape={actions.shape}, {len(info_steps)} info entries")
print(f"final x,y: {info_steps[-1].get('x')}, {info_steps[-1].get('y')}")
