"""One-off diagnostic (not standing infra): splits real per-episode wall-clock time into
rollout (env.step + INFER round-trips) vs TRAIN call, to see where time actually goes
with the new persistent-server design vs the old per-episode-subprocess design's
~17-18s/episode baseline."""
import sys
import time
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import sonic_rl_train as m
import stable_retro as retro
import numpy as np
from checkpoints import parse_coords_file, CheckpointTracker

boxes = parse_coords_file(m.COORDS_FILE)
tracker = CheckpointTracker(boxes)
env = retro.make(game=m.GAME, render_mode=None)
rng = np.random.default_rng(555)
server = m.PolicyServerClient(m.SERVER_BIN, m.WEIGHTS_FILE, m.IW, m.IH, m.N_BUTTONS)

for i in range(4):
    t0 = time.time()
    ep = m.run_episode(env, tracker, server=server, rng=rng, max_steps=3000)
    t1 = time.time()
    losses = m.train_on_episode(ep, server)
    t2 = time.time()
    print(f"ep {i+1}: rollout={t1-t0:.2f}s train={t2-t1:.2f}s total={t2-t0:.2f}s "
          f"len={ep['length']} reward={ep['total_reward']:+.1f} loss={losses.mean():+.4f}")

server.quit()
env.close()
