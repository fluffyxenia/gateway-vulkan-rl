"""
Behavior-cloning pretraining: supervised per-button binary cross-entropy against one or
more recorded human demo playthroughs (see ingest_demo.py for how those get turned into
frame/action pairs), producing a weights file to use as the INITIAL weights for the
REINFORCE loop instead of random init - same idea as pretrain->SFT->RL for LLMs.

No new C++ trainer needed: sonic_policy_train.cpp's loss is already
    -mean(logprob(action)*advantage) - entropy_coef*mean(entropy)
which, with advantage fixed to a constant 1.0 for every frame and entropy_coef=0,
reduces exactly to standard multi-label behavior-cloning BCE:
    -mean(logprob(action))  =  -mean_i( sum_k[a_k*log(p_k)+(1-a_k)*log(1-p_k)] )
So this script just builds a batch file the normal way and reuses the existing
`sonic_policy_train` binary with a constant-1 advantage array and entropy_coef=0,
instead of writing a second trainer.

Usage: python3 bc_pretrain.py <demo1.npz> [demo2.npz ...] [--out WEIGHTS] [--nepoch N] [--lr LR]

Multiple demos are concatenated frame-wise into one combined batch (BC treats every
frame as an independent (frame, action) supervised pair - no episode-boundary/return
computation needed the way REINFORCE's discounted-return-to-go requires, so
concatenating across separate playthroughs is perfectly valid, and desirable for
variety).

IMPORTANT, deliberate design choice given this project's own recent incident (the
rl_run4 weights were destroyed by an unrelated test run that didn't back them up
first): this script does NOT write to sonic_rl_train.py's live WEIGHTS_FILE by default.
It writes to a separate --out path (default sonic_policy_weights_bc_pretrained.bin).
Copying that over WEIGHTS_FILE to actually start an RL run from it is a deliberate,
separate, visible step - not an automatic side effect of running this script.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from ingest_demo import ingest
from policy_net import IW, IH
from sonic_rl_train import N_BUTTONS, LEARNABLE_BUTTONS, SAFE_MINIBATCH

TRAINER = "/home/foxo/gateway-vulkan-rl/sonic_policy_train"
BATCH_FILE = "/home/foxo/gateway-vulkan-rl/bc_pretrain_batch.bin"
DEFAULT_OUT = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights_bc_pretrained.bin"


def build_combined_batch(demo_paths, box_override=None, exclude_paths=None):
    """Step 27: demos under demos_fullact/ are Foxo's 6 new full-AIZ1-act recordings
    (cutscene through the boss approach) - auto-detected by directory name rather
    than requiring a separate CLI flag per path, since that's exactly what
    distinguishes them from the original 5 (which stop at the old cutscene
    trigger). For those, full_act=True is passed to ingest() so the old
    truncate-at-cutscene-trigger logic is skipped (it would otherwise discard
    exactly the new second-half coverage these demos exist to add) - see
    ingest_demo.py's ingest() docstring. exclude_paths: paths to skip entirely
    (e.g. a demo found NOT to reach the boss trigger under the merged
    coords.list, confirmed by inspection rather than discarded reflexively)."""
    exclude_paths = set(exclude_paths or ())
    all_frames, all_actions = [], []
    per_demo_stats = []
    for path in demo_paths:
        if path in exclude_paths:
            print(f"  [build_combined_batch] SKIPPING {path} (explicitly excluded)")
            continue
        full_act = "demos_fullact/" in path or "demos_fullact\\" in path
        result = ingest(path, box_override=box_override, full_act=full_act)
        all_frames.append(result["frames"])
        all_actions.append(result["actions"])
        per_demo_stats.append((path, result["n_frames"], result["sanity_passed"]))

    frames = np.concatenate(all_frames, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    assert frames.shape[0] == actions.shape[0]
    assert actions.shape[1] == N_BUTTONS, f"expected {N_BUTTONS} button columns, got {actions.shape[1]}"
    return frames, actions, per_demo_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demos", nargs="+", help="one or more .npz demo files from playback_movie.py --npy-actions --info-dict")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output weights file (NOT the live RL weights file by default - see module docstring)")
    ap.add_argument("--nepoch", type=int, default=50, help="BC is supervised - reusing the same batch for many epochs is normal/expected, unlike RL's 1-epoch-per-on-policy-batch")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.1,
                     help="AdamW weight decay - added after two real BC runs diverged to "
                          "NaN around epoch ~20 at this LR with wd implicitly 0.0 (ggml's "
                          "own default, never explicitly set before). wd=0.01 delayed the "
                          "divergence (to ~epoch 38) but didn't prevent it - effective "
                          "per-step decay at lr=0.01 was only lr*wd=0.0001 (0.01%), too "
                          "weak; 0.1 gives lr*wd=0.001 (0.1%/step), ~30% cumulative shrink "
                          "pressure over 300 steps. See AGENTS.md 'Step 12/13'.")
    ap.add_argument("--minibatch-size", type=int, default=None,
                     help="samples per gradient step/ggml graph - the v3 two-conv architecture "
                          "hits a real Vulkan memory ceiling somewhere between 3000 and 4049 "
                          f"samples in one graph (see AGENTS.md 'Step 12'). Default: computed "
                          f"automatically to divide the combined dataset into the fewest "
                          f"equal-ish chunks each <= {SAFE_MINIBATCH}, so real demo frames aren't "
                          f"needlessly dropped to the trainer's per-minibatch remainder (unlike "
                          f"blindly using {SAFE_MINIBATCH} regardless of total size, which could "
                          f"drop a meaningful fraction of a real, hard-to-replace dataset).")
    ap.add_argument("--box-override", type=float, nargs=4, default=None,
                     metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                     help="use this box instead of coords.list's end-trigger box for the "
                          "sanity-check/truncation step - diagnostic use only, does NOT "
                          "modify coords.list (see ingest_demo.py docstring)")
    ap.add_argument("--exclude", nargs="*", default=None,
                     help="demo path(s) to skip entirely despite being listed in `demos` - "
                          "e.g. a full_act demo confirmed by direct inspection to never "
                          "reach the boss_trigger box (stuck/low-quality run), rather than "
                          "silently dropped or silently kept")
    args = ap.parse_args()

    print(f"=== Building combined BC batch from {len(args.demos)} demo file(s) ===")
    box_override = tuple(args.box_override) if args.box_override else None
    frames, actions, per_demo_stats = build_combined_batch(args.demos, box_override=box_override, exclude_paths=args.exclude)
    T = frames.shape[0]
    print(f"\nCombined batch: {T} total frames from {len(args.demos)} demo(s)")
    for path, n, passed in per_demo_stats:
        print(f"  {path}: {n} frames, sanity_check_passed={passed}")
    if not all(s[2] for s in per_demo_stats):
        print("  WARNING: at least one demo failed/warned on the cutscene-trigger sanity "
              "check (see detail above) - still including it in the combined batch, "
              "but worth reviewing before trusting the pretrained result.")

    advantage = np.ones((T,), dtype=np.float32)  # constant -> reduces REINFORCE loss to plain BCE

    with open(BATCH_FILE, "wb") as f:
        f.write(frames.tobytes())
        f.write(actions.tobytes())
        f.write(advantage.tobytes())

    if os.path.exists(args.out):
        os.remove(args.out)  # start pretraining from scratch (random init), not continue some stale file

    if args.minibatch_size is not None:
        minibatch_size = args.minibatch_size
    else:
        n_chunks = max(1, -(-T // SAFE_MINIBATCH))  # ceil division
        minibatch_size = T // n_chunks               # FLOOR here (not ceil) - see the
        # detailed comment in sonic_rl_train.py's train_on_episode(): ceil here would
        # make minibatch_size*n_chunks >= T, which combined with the trainer's own floor
        # -division remainder could drop a full extra chunk's worth of real demo frames.
    print(f"\n=== Running BC pretraining: {T} frames, minibatch_size={minibatch_size}, "
          f"{args.nepoch} epochs, lr={args.lr}, entropy_coef=0 (pure BCE), weight_decay={args.weight_decay} ===")
    result = subprocess.run(
        [TRAINER, args.out, BATCH_FILE, str(IW), str(IH), str(T), str(minibatch_size),
         str(N_BUTTONS), str(args.nepoch), str(args.lr), "0.0", str(args.weight_decay)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("TRAINER STDERR:", result.stderr, file=sys.stderr)
        raise RuntimeError(f"trainer failed with exit code {result.returncode}")

    print(f"BC-pretrained weights written to {args.out}")
    print(f"LEARNABLE_BUTTONS order (for reference): {LEARNABLE_BUTTONS}")
    print("\nTo actually use this as the RL starting point: copy/rename this file over "
          "sonic_rl_train.py's WEIGHTS_FILE (sonic_policy_weights.bin) BEFORE launching "
          "a training run - this script deliberately does not do that automatically.")


if __name__ == "__main__":
    main()
