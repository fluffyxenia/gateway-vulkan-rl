"""One-off throughput benchmark (not standing infra): measures INFER call latency/it-s
on the persistent sonic_policy_server, separating first-call (one-time Vulkan/shader
warm-up) from steady-state, same methodology as sonic_policy_train.cpp's own timing
summary."""
import numpy as np
import struct
import subprocess
import sys
import time

IW, IH, N_BUTTONS = 80, 56, 10
NE_IN = IW * IH
SERVER = "/home/foxo/gateway-vulkan-rl/sonic_policy_server"
WEIGHTS = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights.bin"
N_CALLS = 3000

rng = np.random.default_rng(99)
frames = rng.random((N_CALLS, IH, IW), dtype=np.float32)

proc = subprocess.Popen([SERVER, WEIGHTS, str(IW), str(IH), str(N_BUTTONS)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(0.5)

first_t = None
steady_total = 0.0
steady_n = 0
for i in range(N_CALLS):
    t0 = time.perf_counter()
    proc.stdin.write(b"I" + frames[i].astype(np.float32).tobytes())
    proc.stdin.flush()
    resp = proc.stdout.read(4 * N_BUTTONS)
    t1 = time.perf_counter()
    dt = t1 - t0
    if i == 0:
        first_t = dt
    else:
        steady_total += dt
        steady_n += 1

proc.stdin.write(b"Q")
proc.stdin.flush()
proc.wait(timeout=10)

print(f"first INFER call (incl. one-time warm-up): {first_t*1000:.2f}ms ({1.0/first_t:.1f} it/s)")
print(f"steady-state over remaining {steady_n} calls: {steady_total/steady_n*1000:.3f}ms/call avg "
      f"({steady_n/steady_total:.1f} it/s)")
