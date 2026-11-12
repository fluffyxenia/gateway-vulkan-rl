"""One-off diagnostic (not standing infra): direct side-by-side of rollout cost - old
numpy CPU PolicyNet.forward path vs new server.infer() IPC path - on the SAME machine,
SAME env, SAME number of steps, run back to back, to settle whether swapping rollout
inference onto the persistent GPU server was actually a net win or a net regression."""
import sys
import time
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import numpy as np
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from policy_net import PolicyNet, IW, IH
import sonic_rl_train as m

N_STEPS = 3000
WEIGHTS = m.WEIGHTS_FILE

boxes = parse_coords_file(m.COORDS_FILE)
tracker = CheckpointTracker(boxes)
env = retro.make(game=m.GAME, render_mode=None)
rng = np.random.default_rng(42)

print("=== OLD path: numpy CPU PolicyNet.forward + sample_buttons ===")
policy = PolicyNet(m.N_BUTTONS)
policy.load(WEIGHTS)
obs, info = env.reset()
tracker.reset()
t0 = time.time()
for t in range(N_STEPS):
    gray = m.downsample_gray(obs, IW, IH)
    button_vec = policy.sample_buttons(gray, rng)
    mb = m.buttons_to_multibinary(button_vec)
    obs, _, terminated, truncated, info = env.step(mb)
    if terminated or truncated:
        obs, info = env.reset()
t1 = time.time()
print(f"OLD numpy-CPU rollout: {N_STEPS} steps in {t1-t0:.2f}s ({(t1-t0)/N_STEPS*1000:.3f}ms/step, {N_STEPS/(t1-t0):.1f} it/s)")

print("\n=== NEW path: server.infer() over IPC pipe ===")
server = m.PolicyServerClient(m.SERVER_BIN, WEIGHTS, IW, IH, m.N_BUTTONS)
obs, info = env.reset()
tracker.reset()
t0 = time.time()
for t in range(N_STEPS):
    gray = m.downsample_gray(obs, IW, IH)
    probs = server.infer(gray)
    button_vec = (rng.random(m.N_BUTTONS) < probs).astype(np.int8)
    mb = m.buttons_to_multibinary(button_vec)
    obs, _, terminated, truncated, info = env.step(mb)
    if terminated or truncated:
        obs, info = env.reset()
t1 = time.time()
print(f"NEW server-IPC rollout: {N_STEPS} steps in {t1-t0:.2f}s ({(t1-t0)/N_STEPS*1000:.3f}ms/step, {N_STEPS/(t1-t0):.1f} it/s)")
server.quit()

print("\n=== Isolating env.step() alone (no policy at all, fixed no-op action) ===")
obs, info = env.reset()
noop = np.zeros(len(m.BUTTONS), dtype=np.int8)
t0 = time.time()
for t in range(N_STEPS):
    obs, _, terminated, truncated, info = env.step(noop)
    if terminated or truncated:
        obs, info = env.reset()
t1 = time.time()
print(f"env.step()-only: {N_STEPS} steps in {t1-t0:.2f}s ({(t1-t0)/N_STEPS*1000:.3f}ms/step)")

env.close()
