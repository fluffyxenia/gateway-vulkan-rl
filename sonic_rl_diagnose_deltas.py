"""
Read-only diagnostic (no training, no weight changes): reproduces the exact same 8
diagnostic episodes from sonic_rl_diagnose.py (same trained weights, same rng seed 777,
same rollout order) but this time records the RAW per-step (x, y) position every single
step - not just checkpoint events - so we can look at actual frame-to-frame displacement
during the "first enemy" stall cases and answer: is the agent truly idling (near-zero
displacement) or oscillating/wandering (nonzero but directionless, consistent with
knockback-driven movement)?

Usage: python3 -I sonic_rl_diagnose_deltas.py
"""
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker
from policy_net import PolicyNet, IW, IH
from sonic_rl_train import GAME, COORDS_FILE, WEIGHTS_FILE, N_ACTIONS, action_to_multibinary, downsample_gray

# Episodes to analyze in detail (1-indexed, matching the original diagnostic run):
#   5 - the full-7200-step stall (4x enemy_zone_entered, never resolved, never escaped)
#   3 - re-entered the enemy zone twice, eventually did move on and reach further boxes
#   8 - re-entered the enemy zone twice, had already passed box #4 via a different path
TARGET_EPISODES = {3, 5, 8}
ENEMY_BOX_IDX = 3  # confirmed index of "first enemy" from the earlier diagnostic run


def run_episode_with_deltas(env, tracker, policy, rng, max_steps, record):
    obs, info = env.reset()
    tracker.reset()
    xs, ys = [], []
    enemy_zone_entries = []  # step indices where enemy_zone_entered fired
    for t in range(max_steps):
        gray = downsample_gray(obs, IW, IH)
        a_idx = policy.sample_action(gray, rng)
        mb = action_to_multibinary(a_idx)
        obs, _env_reward, terminated, truncated, info = env.step(mb)
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        r, cp_terminated, event = tracker.step(x, y, score, t)
        if record:
            xs.append(x)
            ys.append(y)
            if event is not None and "enemy_zone_entered" in event:
                enemy_zone_entries.append(t)
        if cp_terminated or terminated or truncated:
            break
    return xs, ys, enemy_zone_entries, t + 1


def analyze(xs, ys, enemy_zone_entries, ep_num, length):
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_mag = np.sqrt(dx**2 + dy**2)

    n_idle = int(np.sum((dx == 0) & (dy == 0)))
    n_total = len(dx)
    net_disp = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    path_len = np.sum(step_mag)

    print(f"\n=== Episode {ep_num}: raw movement analysis (length={length} steps) ===")
    print(f"  idle steps (dx=dy=0): {n_idle}/{n_total} ({100*n_idle/n_total:.1f}%)")
    print(f"  mean |step| = {step_mag.mean():.2f} px/frame, max |step| = {step_mag.max():.1f} px/frame")
    print(f"  net displacement (start->end, straight-line) = {net_disp:.1f} px")
    print(f"  total path length (sum of |step|) = {path_len:.1f} px")
    print(f"  ratio net_disp/path_len = {net_disp/path_len if path_len > 0 else float('nan'):.4f} "
          f"(near 0 = wandered/oscillated without net progress; near 1 = moved in a straight line)")

    print(f"  enemy_zone_entered fired at steps: {enemy_zone_entries}")
    for entry_t in enemy_zone_entries:
        lo = max(0, entry_t - 10)
        hi = min(len(xs), entry_t + 70)
        print(f"  --- window around entry at step {entry_t} (steps {lo}-{hi-1}) ---")
        line = []
        for t in range(lo, hi):
            tag = " <ENTER" if t == entry_t else ""
            line.append(f"t={t}:(x={int(xs[t])},y={int(ys[t])}){tag}")
        # print compactly, 6 per line
        for i in range(0, len(line), 6):
            print("    " + "  ".join(line[i:i+6]))
        window_dx = np.diff(xs[lo:hi])
        window_dy = np.diff(ys[lo:hi])
        window_mag = np.sqrt(window_dx**2 + window_dy**2)
        window_net = np.sqrt((xs[hi-1]-xs[lo])**2 + (ys[hi-1]-ys[lo])**2)
        window_path = window_mag.sum()
        n_window_idle = int(np.sum((window_dx == 0) & (window_dy == 0)))
        print(f"    window stats: idle={n_window_idle}/{len(window_dx)}, "
              f"mean|step|={window_mag.mean():.2f}, net_disp={window_net:.1f}, path_len={window_path:.1f}, "
              f"ratio={window_net/window_path if window_path > 0 else float('nan'):.3f}")


def main():
    boxes = parse_coords_file(COORDS_FILE)
    tracker = CheckpointTracker(boxes)
    policy = PolicyNet(N_ACTIONS)
    loaded = policy.load(WEIGHTS_FILE)
    print(f"Loaded trained weights: {loaded}")
    if not loaded:
        print("WARNING: no weights file - aborting, this needs the trained rl_run4 weights.")
        return

    env = retro.make(game=GAME, render_mode=None)
    rng = np.random.default_rng(777)  # SAME seed as the original diagnostic run

    for ep_i in range(1, 9):  # reproduce all 8 in order so the rng stream matches exactly
        record = ep_i in TARGET_EPISODES
        xs, ys, enemy_zone_entries, length = run_episode_with_deltas(
            env, tracker, policy, rng, max_steps=7200, record=record)
        if record:
            analyze(xs, ys, enemy_zone_entries, ep_i, length)
        else:
            print(f"(episode {ep_i}: not a target, skipped detailed recording, length={length})")

    env.close()


if __name__ == "__main__":
    main()
