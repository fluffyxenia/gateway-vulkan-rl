"""
Forensic trace: find the exact step of the 7367px x-displacement spotted in episode 8
of the original diagnostic batch (same seed 777, same rl_run4 weights, same rollout
order as sonic_rl_diagnose.py / sonic_rl_diagnose_deltas.py). Logs FULL per-step
(x, y, frame_idx) for the whole episode (not just windowed around enemy_zone_entered
events, since the jump isn't necessarily near one), finds the single step with the
largest |dx| (expected ~7367), and prints raw x,y for ~10 steps before and after.
"""
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker
from policy_net import PolicyNet, IW, IH
from sonic_rl_train import GAME, COORDS_FILE, WEIGHTS_FILE, N_ACTIONS, action_to_multibinary, downsample_gray

TARGET_EPISODE = 8  # 1-indexed, matching the original diagnostic batch


def main():
    boxes = parse_coords_file(COORDS_FILE)
    tracker = CheckpointTracker(boxes)
    policy = PolicyNet(N_ACTIONS)
    loaded = policy.load(WEIGHTS_FILE)
    print(f"Loaded trained weights: {loaded}")
    if not loaded:
        print("WARNING: no weights file - aborting.")
        return

    env = retro.make(game=GAME, render_mode=None)
    rng = np.random.default_rng(777)  # SAME seed as the original diagnostic run

    for ep_i in range(1, TARGET_EPISODE + 1):
        obs, info = env.reset()
        tracker.reset()
        xs, ys, fidx, lives_list, score_list = [], [], [], [], []
        for t in range(7200):
            gray = downsample_gray(obs, IW, IH)
            a_idx = policy.sample_action(gray, rng)
            mb = action_to_multibinary(a_idx)
            obs, _env_reward, terminated, truncated, info = env.step(mb)
            x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
            lives = info.get("lives")
            r, cp_terminated, event = tracker.step(x, y, score, t)
            if ep_i == TARGET_EPISODE:
                xs.append(x)
                ys.append(y)
                fidx.append(t)
                lives_list.append(lives)
                score_list.append(score)
            if cp_terminated or terminated or truncated:
                break
        print(f"episode {ep_i}: length={t+1}" + (" (TARGET - analyzing)" if ep_i == TARGET_EPISODE else ""))

    env.close()

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_mag = np.sqrt(dx**2 + dy**2)

    # Scan the whole episode for every point lives actually decreased, for context.
    lives_arr = lives_list
    print("\nAll lives-decrease events in this episode (full 7200-step scan):")
    found_any = False
    for i in range(1, len(lives_arr)):
        if lives_arr[i] is not None and lives_arr[i-1] is not None and lives_arr[i] < lives_arr[i-1]:
            found_any = True
            print(f"  t={fidx[i-1]} -> t={fidx[i]}: lives {lives_arr[i-1]} -> {lives_arr[i]}  "
                  f"(x,y before={xs[i-1]:.0f},{ys[i-1]:.0f}  x,y after={xs[i]:.0f},{ys[i]:.0f})")
    if not found_any:
        print("  (none found - lives never decreased during this recorded episode)")

    jump_i = int(np.argmax(step_mag))  # index into dx/dy; corresponds to transition from step jump_i to jump_i+1

    # how far back does the pre-jump freeze at (7280,1133) actually extend?
    freeze_start = jump_i
    while freeze_start > 0 and xs[freeze_start-1] == xs[jump_i] and ys[freeze_start-1] == ys[jump_i]:
        freeze_start -= 1
    print(f"Pre-jump freeze at (x={xs[jump_i]:.0f},y={ys[jump_i]:.0f}) extends from "
          f"t={fidx[freeze_start]} through t={fidx[jump_i]} ({jump_i - freeze_start + 1} identical steps).")

    print(f"\nLargest single-step displacement: {step_mag[jump_i]:.1f} px, "
          f"at transition from t={fidx[jump_i]} -> t={fidx[jump_i+1]}")
    print(f"  dx={dx[jump_i]:+.1f}  dy={dy[jump_i]:+.1f}")

    # find how long (x,y) stays at (0,0) and where it resolves to
    zero_end = jump_i + 1
    while zero_end < len(xs) and xs[zero_end] == 0.0 and ys[zero_end] == 0.0:
        zero_end += 1
    print(f"\n(x,y)=(0,0) persists from t={fidx[jump_i+1]} through t={fidx[zero_end-1]} "
          f"({zero_end - (jump_i+1)} steps), then resolves to:")
    if zero_end < len(xs):
        print(f"  t={fidx[zero_end]}: x={xs[zero_end]:.0f} y={ys[zero_end]:.0f} lives={lives_list[zero_end]} score={score_list[zero_end]}")
        for extra in range(zero_end, min(zero_end + 8, len(xs))):
            print(f"  t={fidx[extra]}: x={xs[extra]:.0f} y={ys[extra]:.0f} lives={lives_list[extra]} score={score_list[extra]}")
    else:
        print("  (stayed at (0,0) for the rest of this recorded episode)")

    lo = max(0, jump_i - 10)
    hi = min(len(xs), jump_i + 12)
    print(f"\nRaw (x,y,lives,score) for t={fidx[lo]} through t={fidx[hi-1]} "
          f"(jump is between t={fidx[jump_i]} and t={fidx[jump_i+1]}):")
    for i in range(lo, hi):
        marker = ""
        if i == jump_i:
            marker = "   <-- step BEFORE the jump"
        elif i == jump_i + 1:
            marker = "   <-- step AFTER the jump (this is where x/y land)"
        print(f"  t={fidx[i]:5d}: x={xs[i]:8.1f}  y={ys[i]:8.1f}  lives={lives_list[i]}  score={score_list[i]}{marker}")

    # also print the dx,dy sequence explicitly for the same window for easy eyeballing
    print(f"\nPer-step deltas (dx, dy) leading up to and through the jump:")
    for i in range(lo, min(hi, len(dx))):
        marker = "   <== THE JUMP" if i == jump_i else ""
        print(f"  t={fidx[i]:5d}->t={fidx[i+1]:5d}: dx={dx[i]:+8.1f}  dy={dy[i]:+8.1f}{marker}")


if __name__ == "__main__":
    main()
