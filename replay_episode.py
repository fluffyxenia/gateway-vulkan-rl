"""
Replay a previously-logged episode's exact action sequence, with real rendering - meant
to be run on a non-headless machine (Gateway itself has no GL/display; this is the
"standing infrastructure" requested so any episode, especially one flagged as
anomalous, can be visually inspected after the fact without needing to watch headless
training runs live).

Requires: `pip install stable-retro` (same version used during the original run) and
the game's ROM already imported via `python3 -m retro.import <rom_dir>` on this machine
too - stable-retro's game data isn't tied to Gateway, it's a per-machine install.

The action log (.npy, (T, N_BUTTONS) int8 multi-hot button vectors - v2 action space:
independent per-button outputs, not a categorical index over a fixed menu) plus the
run's run_metadata.json (full button order + which of those are learnable + game id) are
everything needed - no policy weights, no RNG state, nothing Gateway-specific. The
emulator is deterministic given a fixed start state (env.reset() with the default/
initial state) and a fixed input sequence, so replaying the same actions in the same
order exactly reproduces the original episode.

Usage:
  python3 replay_episode.py <path/to/some_episode.npy> [--rgb-array-every N]

  Default: render_mode="human" (opens a window, needs a real display).
  --rgb-array-every N: instead of opening a window (e.g. no display available even on
      the "non-headless" machine, or running over a remote desktop without GL forwarding),
      save every Nth frame as a PNG into ./replay_frames/<episode-name>/ instead.
"""
import argparse
import json
import os
import sys

import numpy as np
import stable_retro as retro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action_log", help="path to a .npy action-index file saved by sonic_rl_train.py / sonic_rl_diagnose.py")
    ap.add_argument("--rgb-array-every", type=int, default=0,
                     help="if >0, don't open a window - save every Nth frame as a PNG instead")
    ap.add_argument("--state", type=str, default=None,
                     help="Step 31 addition: save-state name to start from (e.g. "
                          "'AngelIslandZone.Act1Boss' for method 2b episodes, "
                          "'AngelIslandZone.Act1Midpoint' for 2c) - required for any "
                          "episode that wasn't a plain default-start (method 2a)  "
                          "rollout, since run_metadata.json's own 'game' field doesn't "
                          "capture which start_mode was actually used. Omit for 2a "
                          "episodes (default env.reset() behavior, unchanged).")
    ap.add_argument("--inttype-all", action="store_true",
                     help="pass inttype=retro.data.Integrations.ALL to retro.make() - "
                          "required alongside --state for the decomp-ROM-based "
                          "save-states (method 2b/2c), which live under the ALL "
                          "integrations search path, not the default stable one.")
    ap.add_argument("--game", type=str, default=None,
                     help="override the game id instead of trusting run_metadata.json's "
                          "own 'game' field - that field is NOT start_mode-aware and is "
                          "confirmed wrong for 2b/2c episodes (it always records the real-"
                          "ROM game id even when the episode actually ran against "
                          "'SonicKnuckles3Decomp-Genesis'). Pass the real game id "
                          "explicitly for any non-default start_mode episode.")
    args = ap.parse_args()

    log_dir = os.path.dirname(os.path.abspath(args.action_log))
    meta_path = os.path.join(log_dir, "run_metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: no run_metadata.json found next to {args.action_log} "
              f"(expected at {meta_path}) - can't reconstruct the action->button mapping without it.")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    actions = np.load(args.action_log)  # (T, N_BUTTONS) 0/1, or legacy (T,) int indices - see fallback below
    buttons = meta["buttons"]
    game = args.game if args.game else meta["game"]

    if "learnable_buttons" in meta:
        # v2: independent per-button multi-hot vectors
        learnable_buttons = meta["learnable_buttons"]
        learnable_idx_in_full = [buttons.index(b) for b in learnable_buttons]

        def action_to_multibinary(button_vec):
            v = np.zeros(len(buttons), dtype=np.int8)
            v[learnable_idx_in_full] = button_vec
            return v
    elif "discrete_actions" in meta:
        # v1 legacy format (categorical index over a fixed button-bundle menu) - kept so
        # old action logs from before the v2 redesign can still be replayed if needed.
        discrete_actions = meta["discrete_actions"]

        def action_to_multibinary(a_idx):
            v = np.zeros(len(buttons), dtype=np.int8)
            for name in discrete_actions[int(a_idx)]:
                v[buttons.index(name)] = 1
            return v
    else:
        print("ERROR: run_metadata.json has neither 'learnable_buttons' (v2) nor "
              "'discrete_actions' (v1 legacy) - don't know how to interpret this action log.")
        sys.exit(1)

    print(f"Replaying {args.action_log}")
    print(f"  game={game}  n_steps={len(actions)}  buttons={buttons}")

    render_mode = None if args.rgb_array_every > 0 else "human"
    make_kwargs = {"game": game, "render_mode": "rgb_array" if args.rgb_array_every > 0 else "human"}
    if args.state:
        make_kwargs["state"] = args.state
    if args.inttype_all:
        make_kwargs["inttype"] = retro.data.Integrations.ALL
    env = retro.make(**make_kwargs)
    obs, info = env.reset()

    frame_out_dir = None
    if args.rgb_array_every > 0:
        ep_name = os.path.splitext(os.path.basename(args.action_log))[0]
        frame_out_dir = os.path.join("replay_frames", ep_name)
        os.makedirs(frame_out_dir, exist_ok=True)
        print(f"  saving every {args.rgb_array_every}th frame to {frame_out_dir}/")
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: --rgb-array-every needs Pillow installed (`pip install pillow`).")
            sys.exit(1)

    for t, a in enumerate(actions):
        mb = action_to_multibinary(a)
        obs, reward, terminated, truncated, info = env.step(mb)
        if args.rgb_array_every > 0 and t % args.rgb_array_every == 0:
            frame = env.render()
            Image.fromarray(frame).save(os.path.join(frame_out_dir, f"frame_{t:05d}.png"))
        else:
            env.render()
        if terminated or truncated:
            print(f"  env signaled terminated/truncated at step {t} "
                  f"(action log has {len(actions)} steps total - this may be expected "
                  f"if the original episode ended via the env's own done condition)")
            break

    print(f"Replay finished. Final info: {info}")
    env.close()


if __name__ == "__main__":
    main()
