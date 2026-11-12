"""Redo of verify_anchor_loss.py using REAL demo frames (not random noise) to avoid
extreme probability saturation that makes a clipped-numpy-BCE comparison meaningless."""
import numpy as np
import struct
import subprocess
import sys
import shutil
import time
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from policy_net import PolicyNet
from bc_pretrain import build_combined_batch

IW, IH, N_BUTTONS = 80, 56, 10
SERVER = "/home/foxo/gateway-vulkan-rl/sonic_policy_server"
WEIGHTS = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights.bin"
TEST_W = "/tmp/anchor_loss_test_weights2.bin"
shutil.copyfile(WEIGHTS, TEST_W)

demos = [
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-BottomRoute-000000.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000000.npz",
]
all_frames, all_actions, _ = build_combined_batch(demos)
rng = np.random.default_rng(321)
T = 400
idx = rng.choice(all_frames.shape[0], T, replace=False)
frames = all_frames[idx]
mask = all_actions[idx]
coef = np.concatenate([
    rng.standard_normal(T // 2).astype(np.float32),
    np.full(T - T // 2, 1.0, dtype=np.float32),
])

net = PolicyNet(N_BUTTONS)
net.load(TEST_W)
probs = net.forward(frames)
print("min prob:", probs.min(), "max prob:", probs.max())
eps = 1e-7
p = np.clip(probs, eps, 1 - eps)
bce_action = -(mask * np.log(p) + (1 - mask) * np.log(1 - p)).sum(axis=1)
expected_loss = float(np.mean(bce_action * coef))
print(f"numpy-computed expected first-minibatch loss: {expected_loss:.6f}")

proc = subprocess.Popen([SERVER, TEST_W, str(IW), str(IH), str(N_BUTTONS)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(0.5)
NBATCH = 400
NEPOCH = 1
req = (b"T" + struct.pack("<qqq", T, NBATCH, NEPOCH) + struct.pack("<ff", 0.003, 0.1)
       + frames.astype(np.float32).tobytes() + mask.astype(np.float32).tobytes() + coef.tobytes())
proc.stdin.write(req)
proc.stdin.flush()
status = struct.unpack("<i", proc.stdout.read(4))[0]
server_loss = struct.unpack(f"<{NEPOCH}f", proc.stdout.read(4 * NEPOCH))[0]
proc.stdin.write(b"Q")
proc.stdin.flush()
proc.wait(timeout=10)

print(f"server-reported loss: {server_loss:.6f}  status={status}")
diff = abs(expected_loss - server_loss)
print(f"abs diff: {diff:.3e}")
print("PASS" if diff < 1e-2 else "FAIL")
