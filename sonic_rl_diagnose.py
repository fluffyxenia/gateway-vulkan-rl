"""
Read-only diagnostic: no training, no weight changes. Loads a trained weights file
(v2 multi-label architecture - independent per-button sigmoid outputs, see
sonic_rl_train.py) and rolls out a handful of episodes exactly like a training rollout
(same policy.sample_buttons per-button Bernoulli sampling, same env/reward logic) - but
instead of only counting how many checkpoints were hit, records the ordered
sequence of SPECIFIC checkpoint boxes (index + comment text) each episode actually
touched, so the path taken (top route vs bottom route vs mixed) can be read off directly.

Usage: python3 -I sonic_rl_diagnose.py [n_episodes] [max_steps]
"""
import os
import re
import sys
import time

import numpy as np
import stable_retro as retro

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker
from policy_net import PolicyNet, IW, IH
from sonic_rl_train import (
    GAME, COORDS_FILE, WEIGHTS_FILE, N_BUTTONS, buttons_to_multibinary, downsample_gray,
    ACTION_LOG_ROOT, save_action_log, write_run_metadata,
)

EVENT_BOX_RE = re.compile(r"box=(\d+)")


def box_label(idx, boxes):
    b = boxes[idx]
    tag = ""
    c = b.comment.lower()
    if "top route" in c or "top loop" in c or "top section" in c:
        tag = " [TOP]"
    elif "bottom route" in c or "bottom loop" in c or "bottom section" in c:
        tag = " [BOTTOM]"
    return f"#{idx} {b.kind}: \"{b.comment.strip()}\"{tag}"


def run_diagnostic_episode(env, tracker, boxes, policy, rng, max_steps, end_idx):
    obs, info = env.reset()
    tracker.reset()
    path = []  # list of (step_idx, box_idx, label)
    actions = []
    for t in range(max_steps):
        gray = downsample_gray(obs, IW, IH)
        button_vec = policy.sample_buttons(gray, rng)
        actions.append(button_vec)
        mb = buttons_to_multibinary(button_vec)
        obs, _env_reward, terminated, truncated, info = env.step(mb)
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        r, cp_terminated, event = tracker.step(x, y, score, t)
        if event is not None:
            m = EVENT_BOX_RE.search(event)
            idx = int(m.group(1)) if m else (end_idx if event == "level_end_trigger" else None)
            if idx is not None:
                path.append((t, idx, event))
        if cp_terminated or terminated or truncated:
            break
    return path, t + 1, np.array(actions, dtype=np.int8)


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 7200

    boxes = parse_coords_file(COORDS_FILE)
    tracker = CheckpointTracker(boxes)
    policy = PolicyNet(N_BUTTONS)
    loaded = policy.load(WEIGHTS_FILE)
    print(f"Loaded trained weights: {loaded} (from {WEIGHTS_FILE})")
    if not loaded:
        print("WARNING: no weights file found - this would just be the zero-init policy, not the trained one.")
        return

    env = retro.make(game=GAME, render_mode=None)
    rng = np.random.default_rng(777)  # different seed than training, for a fresh look

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(ACTION_LOG_ROOT, f"diagnose_{run_id}")
    write_run_metadata(log_dir, {"n_episodes": n_episodes, "max_steps": max_steps, "kind": "diagnose"})
    print(f"Action-sequence logs for this diagnostic run -> {log_dir}/")

    all_paths = []
    for ep_i in range(n_episodes):
        path, length, actions = run_diagnostic_episode(env, tracker, boxes, policy, rng, max_steps, tracker.end_idx)
        all_paths.append(path)
        furthest_for_name = path[-1][1] if path else -1
        save_action_log(actions, log_dir, f"diag_ep{ep_i+1:03d}_furthest-{furthest_for_name}")
        print(f"\n=== Episode {ep_i+1}/{n_episodes} (length={length} steps) ===")
        if not path:
            print("  (no checkpoints reached at all)")
        for t, idx, event in path:
            print(f"  step {t:5d}: {box_label(idx, boxes)}  [{event}]")
        furthest_idx = path[-1][1] if path else -1
        print(f"  -> furthest box index reached: {furthest_idx} "
              f"({box_label(furthest_idx, boxes) if furthest_idx >= 0 else 'none'})")

    env.close()

    # ---- aggregate route analysis ----
    print("\n=== ROUTE SUMMARY ===")
    top_hits = {"top route", "top loop", "top section"}
    bottom_hits = {"bottom route", "bottom loop", "bottom section"}
    for ep_i, path in enumerate(all_paths):
        comments = [boxes[idx].comment.lower() for _, idx, _ in path]
        took_top = any(any(k in c for k in top_hits) for c in comments)
        took_bottom = any(any(k in c for k in bottom_hits) for c in comments)
        if took_top and took_bottom:
            route = "MIXED (both top and bottom boxes touched)"
        elif took_top:
            route = "TOP route"
        elif took_bottom:
            route = "BOTTOM route"
        else:
            route = "neither (didn't reach the branching section)"
        max_idx = path[-1][1] if path else -1
        print(f"  ep {ep_i+1}: furthest=#{max_idx:2d}  n_boxes_hit={len(path):2d}  route={route}")

    bottom_first_idx = next(i for i, b in enumerate(boxes) if "bottom route first section start" in b.comment.lower())
    reached_bottom_first = sum(1 for path in all_paths if any(idx == bottom_first_idx for _, idx, _ in path))
    print(f"\n'bottom route first section start' is box #{bottom_first_idx}. "
          f"Reached by {reached_bottom_first}/{n_episodes} diagnostic episodes.")


if __name__ == "__main__":
    main()
