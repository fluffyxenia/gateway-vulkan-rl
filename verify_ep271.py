import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, run_episode, N_BUTTONS, START_MODES

LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/train_ep271_outcome-boss_defeated.npy"
actions = np.load(LOG)
mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
rng = np.random.default_rng(123)

class ReplayPolicy:
    def __init__(self, a):
        self.a = a
        self.t = 0
    def sample_buttons(self, gray, rng):
        v = self.a[self.t].astype(np.int8) if self.t < len(self.a) else np.zeros(N_BUTTONS, dtype=np.int8)
        self.t += 1
        return v

policy = ReplayPolicy(actions)
ep = run_episode(env, tracker, policy=policy, rng=rng, max_steps=len(actions)+10, pre_visited={tracker.end_idx})
print(f"ep271 replay outcome={ep['outcome']} length={ep['length']} reward={ep['total_reward']:.2f}")
env.close()
