"""One-off equivalence check (not standing infra): confirms sonic_policy_server's TRAIN
and INFER opcodes produce the same numbers as the old per-process design
(sonic_policy_train subprocess + policy_net.py/forward_check.cpp forward pass) on
identical inputs, before sonic_rl_train.py is switched over to talk to the server.
Run with: python3 -I verify_server_equivalence.py
"""
import numpy as np
import struct
import subprocess
import sys
import os

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")

IW, IH, N_BUTTONS = 80, 56, 10
NE_IN = IW * IH
TRAINER = "/home/foxo/gateway-vulkan-rl/sonic_policy_train"
SERVER = "/home/foxo/gateway-vulkan-rl/sonic_policy_server"
BASE_WEIGHTS = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights.bin"  # current live (episode 327) weights - real, not synthetic
OLD_W = "/tmp/eq_test_weights_old.bin"
NEW_W = "/tmp/eq_test_weights_new.bin"
BATCH_FILE = "/tmp/eq_test_batch.bin"

rng = np.random.default_rng(777)
T = 600
NBATCH = 200
NEPOCH = 3
LR = 0.003
ENTROPY_COEF = 0.02
WEIGHT_DECAY = 0.01

frames = rng.random((T, IH, IW), dtype=np.float32)
mask = (rng.random((T, N_BUTTONS)) < 0.3).astype(np.float32)
advantage = rng.standard_normal(T).astype(np.float32)

with open(BATCH_FILE, "wb") as f:
    f.write(frames.astype(np.float32).tobytes())
    f.write(mask.astype(np.float32).tobytes())
    f.write(advantage.astype(np.float32).tobytes())

# identical starting weights for both runs
import shutil
shutil.copyfile(BASE_WEIGHTS, OLD_W)
shutil.copyfile(BASE_WEIGHTS, NEW_W)

print("=== TRAIN equivalence: old subprocess-per-call trainer vs new persistent server ===")

# ---- OLD: subprocess-per-call trainer (exactly as sonic_rl_train.py invokes it today) ----
result = subprocess.run(
    [TRAINER, OLD_W, BATCH_FILE, str(IW), str(IH), str(T), str(NBATCH), str(N_BUTTONS),
     str(NEPOCH), str(LR), str(ENTROPY_COEF), str(WEIGHT_DECAY)],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("OLD TRAINER FAILED:", result.stderr)
    sys.exit(1)
old_epoch_losses = [float(line.split("=")[-1]) for line in result.stdout.splitlines() if "average pg_loss" in line]
print("old trainer per-epoch avg losses:", old_epoch_losses)

# ---- NEW: persistent server, one TRAIN request ----
proc = subprocess.Popen([SERVER, NEW_W, str(IW), str(IH), str(N_BUTTONS)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
import time
time.sleep(0.5)  # let it load weights before we send anything

req = b"T" + struct.pack("<qqq", T, NBATCH, NEPOCH) + struct.pack("<fff", LR, ENTROPY_COEF, WEIGHT_DECAY)
req += frames.astype(np.float32).tobytes()
req += mask.astype(np.float32).tobytes()
req += advantage.astype(np.float32).tobytes()
proc.stdin.write(req)
proc.stdin.flush()

resp_status = proc.stdout.read(4)
status = struct.unpack("<i", resp_status)[0]
resp_losses = proc.stdout.read(4 * NEPOCH)
new_epoch_losses = list(struct.unpack(f"<{NEPOCH}f", resp_losses))
print("new server per-epoch avg losses:", new_epoch_losses, "status(0=ok,1=nonfinite seen)=", status)

proc.stdin.write(b"Q")
proc.stdin.flush()
proc.wait(timeout=10)
server_err = proc.stderr.read().decode(errors="replace")

max_loss_diff = max(abs(a - b) for a, b in zip(old_epoch_losses, new_epoch_losses))
print(f"max |loss diff| across {NEPOCH} epochs: {max_loss_diff:.3e}")

old_final = np.fromfile(OLD_W, dtype=np.float32)
new_final = np.fromfile(NEW_W, dtype=np.float32)
weight_max_diff = np.max(np.abs(old_final - new_final))
print(f"final weights: old size={old_final.size} new size={new_final.size} max|diff|={weight_max_diff:.3e}")
print(f"final weights byte-identical: {np.array_equal(old_final, new_final)}")

print("\n=== INFER equivalence: forward_check (CPU) vs server INFER (Vulkan) ===")
test_frame = rng.random((IH, IW), dtype=np.float32)
frame_file = "/tmp/eq_test_frame.bin"
test_frame.astype(np.float32).tofile(frame_file)

fc = subprocess.run(["/home/foxo/gateway-vulkan-rl/forward_check", BASE_WEIGHTS, frame_file,
                      str(IW), str(IH), str(N_BUTTONS)], capture_output=True, text=True)
print("forward_check stdout:", fc.stdout.strip())
cpu_probs = np.array([float(x) for x in fc.stdout.strip().split()[3:]], dtype=np.float32)

proc2 = subprocess.Popen([SERVER, BASE_WEIGHTS, str(IW), str(IH), str(N_BUTTONS)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(0.5)
proc2.stdin.write(b"I" + test_frame.astype(np.float32).tobytes())
proc2.stdin.flush()
resp = proc2.stdout.read(4 * N_BUTTONS)
server_probs = np.array(struct.unpack(f"<{N_BUTTONS}f", resp), dtype=np.float32)
proc2.stdin.write(b"Q")
proc2.stdin.flush()
proc2.wait(timeout=10)

print("cpu (forward_check) probs:   ", cpu_probs)
print("server (Vulkan INFER) probs:", server_probs)
print(f"max|diff|: {np.max(np.abs(cpu_probs - server_probs)):.3e}  (expect ~1e-4 bar, CPU vs Vulkan backend, same bar used throughout this project)")

print("\n--- server stderr log (TRAIN call) ---")
print(server_err[-2000:])
