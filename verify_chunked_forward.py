"""One-off verification (not standing infra): confirms the Step 16/17 chunking fix to
PolicyNet.forward() is (1) numerically identical to the old single-shot behavior, and
(2) actually keeps RSS flat/small on the real 10984-frame health-check batch - the
exact scenario that caused the real incident - before trusting it.
"""
import sys
import resource
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from policy_net import PolicyNet, FORWARD_CHUNK_SIZE

net = PolicyNet(10)
net.load("/home/foxo/gateway-vulkan-rl/sonic_policy_weights_bc_pretrained.bin")

print(f"FORWARD_CHUNK_SIZE = {FORWARD_CHUNK_SIZE}")

# --- Correctness: chunked forward on N=500 (> chunk size, forces multiple chunks)
# must exactly match individual per-row forward calls (N=1 each) ---
rng = np.random.default_rng(7)
frames = rng.random((500, 56, 80)).astype(np.float32)
batched = net.forward(frames)
individual = np.concatenate([net.forward(frames[i:i+1]) for i in range(500)], axis=0)
max_diff = np.max(np.abs(batched - individual))
print(f"Correctness: max|diff| between chunked-batch and per-row forward = {max_diff:.3e}")
assert max_diff < 1e-5, "FAIL: chunked forward diverges from per-row forward!"
print("PASS: chunked forward is numerically identical to per-row forward")

# --- Memory safety: full 10984-frame health-check-sized batch, watch RSS ---
print(f"\nRSS before full-batch forward: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB")
big_frames = rng.random((10984, 56, 80)).astype(np.float32)
print(f"RSS after allocating input array: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB")
probs = net.forward(big_frames)
print(f"RSS after forward on all 10984 frames: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB")
print(f"output shape: {probs.shape}")
