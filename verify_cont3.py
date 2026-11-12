"""Verify the 3 boss_defeated episodes from the 2a continuation run (launched on
pre-act-gate-fix code) by replaying their exact recorded actions through the
NOW-FIXED run_episode() - same methodology as episode 218's verification."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import GAME, COORDS_FILE, run_episode, N_BUTTONS

LOG_DIR = "/home/foxo/gateway-vulkan-rl/action_logs/20261110_032240"
episodes = ["train_ep081_outcome-boss_defeated.npy",
            "train_ep129_outcome-boss_defeated.npy",
            "train_ep291_outcome-boss_defeated.npy"]

class ReplayPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.t = 0
    def sample_buttons(self, gray, rng):
        vec = self.actions[self.t].astype(np.int8) if self.t < len(self.actions) else np.zeros(N_BUTTONS, dtype=np.int8)
        self.t += 1
        return vec

boxes = parse_coords_file(COORDS_FILE)
env = retro.make(game=GAME, render_mode=None)
rng = np.random.default_rng(123)

for name in episodes:
    actions = np.load(f"{LOG_DIR}/{name}")
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    policy = ReplayPolicy(actions)
    ep = run_episode(env, tracker, policy=policy, rng=rng, max_steps=len(actions)+10)
    print(f"{name}: replay outcome={ep['outcome']} length={ep['length']} reward={ep['total_reward']:.2f} n_checkpoints={ep['n_checkpoints']}")

env.close()
