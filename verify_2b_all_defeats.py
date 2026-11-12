"""Step 29 full re-verification: replay EVERY one of method 2b's 109 original
boss_defeated episodes (both train and baseline) through the real run_episode()
logic with the NOW-FIXED (singleton-check) BossHitDetector, using the actual
decomp-ROM + Act1Boss savestate env setup 2b really used. Tally survived vs
disqualified counts - this replaces the earlier crude "offset==42 only" aggregate
estimate with a real, direct, per-episode verified answer.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, run_episode, N_BUTTONS, START_MODES

LOG_DIR = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031"
mode_cfg = START_MODES["boss"]


class ReplayPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.t = 0

    def sample_buttons(self, gray, rng):
        if self.t >= len(self.actions):
            vec = np.zeros(N_BUTTONS, dtype=np.int8)
        else:
            vec = self.actions[self.t].astype(np.int8)
        self.t += 1
        return vec


files = sorted(glob.glob(os.path.join(LOG_DIR, "*outcome-boss_defeated.npy")))
print(f"Found {len(files)} boss_defeated action logs to re-verify.")

boxes = parse_coords_file(COORDS_FILE)
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"],
                  inttype=retro.data.Integrations.ALL, render_mode=None)
rng = np.random.default_rng(123)

survived, disqualified = [], []
for i, path in enumerate(files):
    actions = np.load(path)
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    policy = ReplayPolicy(actions)
    ep = run_episode(env, tracker, policy=policy, rng=rng, max_steps=len(actions) + 10,
                      pre_visited={tracker.end_idx})
    name = os.path.basename(path)
    if ep["outcome"] == "boss_defeated":
        survived.append(name)
    else:
        disqualified.append(name)
    if (i + 1) % 20 == 0:
        print(f"  ...{i+1}/{len(files)} checked so far ({len(survived)} survived, {len(disqualified)} disqualified)")

env.close()
print(f"\n=== FINAL: {len(survived)}/{len(files)} survive the fixed detector, "
      f"{len(disqualified)}/{len(files)} disqualified as false positives ===")
print("Disqualified episodes:", disqualified[:20], "..." if len(disqualified) > 20 else "")
