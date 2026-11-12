"""One-off verification (not standing infra): confirms the Step 16 loss formula
(entropy removed entirely, BC-anchor rows mixed into the same batch via the
per-sample coefficient) is correctly implemented in sonic_policy_server.cpp BEFORE
trusting it in real training - compares the server's first-reported loss (computed
at the CURRENT, pre-update weights, before that minibatch's optimizer step) against
an independent numpy computation of mean(bce_action * coefficient) over the exact
same synthetic batch and weights.
"""
import numpy as np
import struct
import subprocess
import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from policy_net import PolicyNet

IW, IH, N_BUTTONS = 80, 56, 10
NE_IN = IW * IH
SERVER = "/home/foxo/gateway-vulkan-rl/sonic_policy_server"
WEIGHTS = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights.bin"  # clean BC checkpoint
TEST_W = "/tmp/anchor_loss_test_weights.bin"
import shutil
shutil.copyfile(WEIGHTS, TEST_W)

rng = np.random.default_rng(321)
T = 400
frames = rng.random((T, IH, IW), dtype=np.float32)
mask = (rng.random((T, N_BUTTONS)) < 0.3).astype(np.float32)
# half "episode-like" rows with real-ish advantage values, half "anchor-like" rows
# with the fixed ANCHOR_WEIGHT=1.0 coefficient - exactly mirrors how
# train_on_episode() concatenates the two groups.
coef = np.concatenate([
    rng.standard_normal(T // 2).astype(np.float32),       # episode rows: real advantages
    np.full(T - T // 2, 1.0, dtype=np.float32),            # anchor rows: fixed weight
])

net = PolicyNet(N_BUTTONS)
net.load(TEST_W)
probs = net.forward(frames.reshape(T, IH, IW))
eps = 1e-7
p = np.clip(probs, eps, 1 - eps)
bce_action = -(mask * np.log(p) + (1 - mask) * np.log(1 - p)).sum(axis=1)
expected_loss = float(np.mean(bce_action * coef))
print(f"numpy-computed expected first-minibatch loss: {expected_loss:.6f}")

proc = subprocess.Popen([SERVER, TEST_W, str(IW), str(IH), str(N_BUTTONS)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
import time
time.sleep(0.5)
NBATCH = 400  # one minibatch covering the whole batch, nepoch=1 -> first reported loss
             # is computed at the ORIGINAL (pre-update) weights, directly comparable
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
err = proc.stderr.read().decode(errors="replace")

print(f"server-reported loss (epoch 1, no shuffling since 1 minibatch covers everything): {server_loss:.6f}")
print(f"status={status} (0=ok)")
diff = abs(expected_loss - server_loss)
print(f"abs diff: {diff:.3e}")
print("PASS" if diff < 1e-3 else "FAIL", "- within tolerance" if diff < 1e-3 else "- TOO LARGE, investigate")
print("\n--- server stderr ---")
print(err[-1500:])
