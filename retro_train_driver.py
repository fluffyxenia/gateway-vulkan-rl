"""
Step 4 pipeline-proof driver: collects a batch of real frames from stable-retro's
Sonic 3 & Knuckles environment (random policy - no RL algorithm here, that's explicitly
out of scope per the task), builds a placeholder per-frame target (delta in rings+score
since the previous frame - a stand-in, NOT the final reward design the user will specify
later), downsamples frames to grayscale, writes a raw binary batch file, and invokes the
ggml Vulkan+CPU-fallback trainer binary on it. Repeats for a few outer iterations to show
the loop actually running end-to-end and consuming GPU, not just a single call.
"""
import struct
import subprocess
import sys
import time

import numpy as np
import stable_retro as retro

GAME = "SonicAndKnuckles3-Genesis-v0"
IW, IH = 80, 56     # downsampled grayscale frame size fed to the conv net
NBATCH = 32
NEPOCH = 20
N_OUTER_ITERS = 5
TRAINER = "/home/foxo/gateway-vulkan-rl/retro_convnet_train"
BATCH_FILE = "/home/foxo/gateway-vulkan-rl/retro_batch.bin"


def downsample_gray(frame: np.ndarray, iw: int, ih: int) -> np.ndarray:
    """frame: (H,W,3) uint8 -> (ih,iw) float32 in [0,1], simple strided subsample."""
    h, w, _ = frame.shape
    ys = (np.linspace(0, h - 1, ih)).astype(np.int64)
    xs = (np.linspace(0, w - 1, iw)).astype(np.int64)
    sub = frame[ys][:, xs]  # (ih, iw, 3)
    gray = sub.astype(np.float32).mean(axis=2) / 255.0
    return gray


def main():
    env = retro.make(game=GAME, render_mode=None)
    obs, info = env.reset()
    prev_score = info.get("score", 0) if info else 0
    prev_rings = info.get("rings", 0) if info else 0

    for outer in range(N_OUTER_ITERS):
        frames = np.zeros((NBATCH, IH, IW), dtype=np.float32)
        targets = np.zeros((NBATCH,), dtype=np.float32)
        t0 = time.time()
        for i in range(NBATCH):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score = info.get("score", prev_score)
            rings = info.get("rings", prev_rings)
            placeholder_target = float((score - prev_score) + (rings - prev_rings))
            prev_score, prev_rings = score, rings
            frames[i] = downsample_gray(obs, IW, IH)
            targets[i] = placeholder_target
            if terminated or truncated:
                obs, info = env.reset()
                prev_score = info.get("score", 0) if info else 0
                prev_rings = info.get("rings", 0) if info else 0
        collect_dt = time.time() - t0

        with open(BATCH_FILE, "wb") as f:
            f.write(frames.astype(np.float32).tobytes())
            f.write(targets.astype(np.float32).tobytes())

        print(f"=== outer iter {outer+1}/{N_OUTER_ITERS}: collected {NBATCH} real frames "
              f"from stable-retro in {collect_dt:.3f}s, target range "
              f"[{targets.min():.1f},{targets.max():.1f}] ===")
        sys.stdout.flush()

        t0 = time.time()
        result = subprocess.run(
            [TRAINER, BATCH_FILE, str(IW), str(IH), str(NBATCH), str(NEPOCH)],
            capture_output=True, text=True,
        )
        train_dt = time.time() - t0
        print(result.stdout)
        if result.returncode != 0:
            print("TRAINER STDERR:", result.stderr, file=sys.stderr)
            print(f"Trainer FAILED with exit code {result.returncode}")
            sys.exit(1)
        print(f"(trainer subprocess took {train_dt:.3f}s)")

    env.close()
    print("\nAll outer iterations completed - retro env -> ggml Vulkan training pipeline PROVEN end to end.")


if __name__ == "__main__":
    main()
