"""One-off measurement (not standing infra): samples the real sonic_policy_server
process's RSS/peak-RSS from /proc/<pid>/status during a real-sized TRAIN call
(T=3300, minibatch_size=1650 - matching the actual combined episode+anchor batch
size used in real training), to answer Foxo's question with a measured number
instead of an estimate."""
import numpy as np
import struct
import subprocess
import sys
import threading
import time

IW, IH, N_BUTTONS = 80, 56, 10
NE_IN = IW * IH
SERVER = "/home/foxo/gateway-vulkan-rl/sonic_policy_server"
WEIGHTS = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights.bin"
TEST_W = "/tmp/rss_measure_weights.bin"
import shutil
shutil.copyfile(WEIGHTS, TEST_W)

T = 3300
NBATCH = 1650
NEPOCH = 1
rng = np.random.default_rng(99)
frames = rng.random((T, IH, IW)).astype(np.float32)
mask = (rng.random((T, N_BUTTONS)) < 0.3).astype(np.float32)
coef = rng.standard_normal(T).astype(np.float32)

proc = subprocess.Popen([SERVER, TEST_W, str(IW), str(IH), str(N_BUTTONS)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(0.5)
server_pid = proc.pid

samples = []
stop_flag = threading.Event()

def sampler():
    while not stop_flag.is_set():
        try:
            with open(f"/proc/{server_pid}/status") as f:
                vmrss = vmhwm = None
                for line in f:
                    if line.startswith("VmRSS:"):
                        vmrss = int(line.split()[1])
                    elif line.startswith("VmHWM:"):
                        vmhwm = int(line.split()[1])
                samples.append((time.time(), vmrss, vmhwm))
        except FileNotFoundError:
            break
        time.sleep(0.05)

t = threading.Thread(target=sampler)
t.start()

req = (b"T" + struct.pack("<qqq", T, NBATCH, NEPOCH) + struct.pack("<ff", 0.003, 0.1)
       + frames.tobytes() + mask.tobytes() + coef.tobytes())
t0 = time.time()
proc.stdin.write(req)
proc.stdin.flush()
status = struct.unpack("<i", proc.stdout.read(4))[0]
losses = struct.unpack(f"<{NEPOCH}f", proc.stdout.read(4 * NEPOCH))
t1 = time.time()

time.sleep(0.3)
stop_flag.set()
t.join()

proc.stdin.write(b"Q")
proc.stdin.flush()
proc.wait(timeout=10)

print(f"TRAIN call wall time: {t1-t0:.2f}s, status={status}, loss={losses}")
peak_rss = max(s[1] for s in samples if s[1] is not None)
peak_hwm = max(s[2] for s in samples if s[2] is not None)
print(f"Peak VmRSS during call: {peak_rss/1024:.1f} MB")
print(f"Peak VmHWM (all-time high-water mark) during call: {peak_hwm/1024:.1f} MB")
print(f"Number of samples: {len(samples)}")
print("\nFull timeline (seconds since start, VmRSS MB):")
for ts, rss, hwm in samples:
    if rss is not None:
        print(f"  t={ts-t0:+.2f}s  VmRSS={rss/1024:.1f} MB")
