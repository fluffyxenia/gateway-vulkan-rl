"""Step 29 verification: replay method 2a's episode 218 (the one confirmed
boss_defeated at structural_offset=47, under the OLD pre-singleton-check detector)
using the EXACT same recorded action sequence, through run_episode()'s real logic,
so the NEW (singleton-check-fixed) BossHitDetector sees the identical RAM trajectory
live and we get a direct, verified answer: does this specific defeat survive the fix,
or does it get disqualified as a crowded/generic-field false positive?
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import GAME, COORDS_FILE, run_episode, N_BUTTONS, LEARNABLE_BUTTONS, BUTTONS

ACTION_LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_223601/train_ep218_outcome-boss_defeated.npy"
recorded_actions = np.load(ACTION_LOG)  # (T, len(LEARNABLE_BUTTONS)) int8, multi-hot over LEARNABLE_BUTTONS only
print(f"Loaded recorded action sequence: {recorded_actions.shape}")


class ReplayPolicy:
    """Drop-in replacement for PolicyNet in run_episode() - ignores the observation
    entirely and just plays back the exact recorded learnable-button vector for
    each step in order, so the resulting RAM trajectory is byte-identical to the
    original episode (same deterministic env, same actions => same trajectory)."""
    def __init__(self, actions):
        self.actions = actions
        self.t = 0

    def sample_buttons(self, gray, rng):
        if self.t >= len(self.actions):
            # shouldn't happen - original episode ended by self.t steps
            vec = np.zeros(N_BUTTONS, dtype=np.int8)
        else:
            vec = self.actions[self.t].astype(np.int8)
        self.t += 1
        return vec


boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
env = retro.make(game=GAME, render_mode=None)
rng = np.random.default_rng(123)

policy = ReplayPolicy(recorded_actions)
ep = run_episode(env, tracker, policy=policy, rng=rng, max_steps=len(recorded_actions) + 10)
env.close()

print(f"\nReplay result: outcome={ep['outcome']} length={ep['length']} "
      f"total_reward={ep['total_reward']:.2f} n_checkpoints={ep['n_checkpoints']}")
print("Original episode 218: outcome=boss_defeated length=6551")
if ep["outcome"] == "boss_defeated":
    print("=> CONFIRMED: this defeat SURVIVES the singleton-check fix - it was a real, "
          "legitimate boss defeat, not a crowded-offset false positive.")
else:
    print("=> DISQUALIFIED: this defeat does NOT survive the singleton-check fix - the "
          "original structural_offset=47 detection was almost certainly a false "
          "positive (a crowded/generic shared field, same category as the ros_bit "
          "offset=60 cluster found in method 2b's data), not a real boss defeat.")
