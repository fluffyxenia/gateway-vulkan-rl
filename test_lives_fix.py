"""Quick isolated test: confirm run_episode() now actually ends an episode the moment
info['lives'] decreases, with correct accounting (reward includes DEATH_PENALTY, outcome
tagged 'died'). No training, no weight changes - just exercises the fixed function with
a random policy over enough steps that at least one death is likely (falling in a pit /
touching an enemy without defeating it are both common early with a random policy)."""
import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import numpy as np
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import run_episode, GAME, COORDS_FILE, DEATH_PENALTY

boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes)
env = retro.make(game=GAME, render_mode=None)
rng = np.random.default_rng(42)

n_died = 0
for ep_i in range(15):
    ep = run_episode(env, tracker, policy=None, rng=rng, max_steps=7200)
    marker = ""
    if ep["outcome"] == "died":
        n_died += 1
        marker = f"  <-- DIED (last step reward={ep['rewards'][-1]:+.1f}, expect it to include DEATH_PENALTY={DEATH_PENALTY})"
    print(f"ep {ep_i+1}: outcome={ep['outcome']:12s} length={ep['length']:4d} "
          f"total_reward={ep['total_reward']:+.1f}{marker}")

env.close()
print(f"\n{n_died}/10 episodes ended via death-detection in this quick test "
      f"(random policy, 3000-step cap, mostly just sanity-checking the mechanism fires "
      f"and isn't silently broken - not a real baseline run).")
