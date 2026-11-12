"""
Ingest a human demo recording (produced via stable_retro's own tools - no custom
recording tool needed, per the scope reduction):

    1. python -m stable_retro.examples.interactive --game SonicAndKnuckles3-Genesis-v0 --record .
       -> produces a .bk2 movie file
    2. python -m stable_retro.scripts.playback_movie <file>.bk2 --npy-actions --info-dict --no-video
       -> produces <file>.npz with TWO keys: "actions" ((T, 12) bool, full Genesis button
          order, confirmed to match BUTTONS in sonic_rl_train.py exactly - no reordering
          needed) and "info" ((T,) object array of per-frame info dicts: x, y, lives,
          score, etc). NOTE: despite --info-dict implying a separate .json, reading
          stable_retro's playback_movie.py source directly (not just assuming) shows
          that when --npy-actions is also given, "info" is embedded in the SAME .npz as
          a second key - there is no separate .json file produced by this exact
          invocation. Confirmed empirically with a synthetic round-trip test before
          writing this, not just read off the source.

This module turns one or more such .npz files into the batch format
sonic_policy_train.cpp already expects (frames + multi-hot button mask + advantage),
for behavior-cloning pretraining (see bc_pretrain.py).

CAUSAL FRAME/ACTION PAIRING (the thing explicitly flagged as needing to be right):
stable-retro's own recording already operates at per-frame granularity - movie.get_key()
is queried fresh every single frame, so a held-down button shows up as the same 1 in
every frame it's held, not a one-off "button pressed" event. This matches our own
rollout's convention exactly: run_episode() computes `gray = downsample(obs)` BEFORE
calling `policy.sample_buttons(gray, ...)` and env.step() - i.e. frame_t is "what the
agent/human saw right before deciding action_t", and action_t is applied to PRODUCE
frame_{t+1}, not derived from it. This script reconstructs frames the identical way: for
t=0 use the observation from env.reset(); for t>0 use the observation produced by
env.step(action_{t-1}) - never pairing action_t with the frame IT caused.
"""
import sys

import numpy as np
import stable_retro as retro

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker  # noqa: F401 (re-exported for bc_pretrain.py)
from policy_net import IW, IH
from sonic_rl_train import GAME, COORDS_FILE, BUTTONS, multibinary_to_buttons, downsample_gray


def load_demo_npz(path: str):
    """Returns (actions_full: (T,12) bool, info: (T,) array-of-dicts or None).

    Step 27 addition: Foxo's 6 new full-AIZ1-act .bk2 recordings (demos_fullact/)
    include a second controller/player slot (Tails) in the .bk2 movie, which
    playback_movie.py's own key-building loop (`for p in range(movie.players): for i
    in range(emulator.num_buttons): keys.append(...)`) lays out player-major - i.e.
    a 2-player recording produces (T, 24) actions with columns [0:12]=player 0
    (confirmed to be Sonic - the one with real, varied input) and [12:24]=player 1
    (Tails). Checked directly on the first converted demo before assuming this:
    player-0's columns had 3855/4914 nonzero frames with a sane button-press
    distribution (RIGHT=2811, matching a right-heavy platformer run); player-1's
    columns were ALL ZERO across all 4914 frames - Tails was never actually
    controlled in this recording. Given that, we slice to player 0's 12 columns and
    verify (not assume) the discarded half really is empty before doing so, on
    every file this is applied to - if a future demo has a non-trivial Tails track,
    this will raise loudly instead of silently dropping real input.
    """
    d = np.load(path, allow_pickle=True)
    if "actions" not in d:
        raise ValueError(f"{path}: no 'actions' key found - is this really a playback_movie.py --npy-actions output?")
    actions_full = d["actions"]
    info = d["info"] if "info" in d else None

    n_cols = actions_full.shape[1]
    n_buttons = len(BUTTONS)
    if n_cols == n_buttons:
        pass  # normal single-player recording, nothing to do
    elif n_cols % n_buttons == 0 and n_cols > n_buttons:
        n_players = n_cols // n_buttons
        p0 = actions_full[:, :n_buttons]
        rest = actions_full[:, n_buttons:]
        rest_nonzero_frames = int(np.any(rest, axis=1).sum())
        if rest_nonzero_frames > 0:
            raise ValueError(
                f"{path}: {n_players}-player recording ({n_cols} action columns) but "
                f"the non-player-0 columns have {rest_nonzero_frames} nonzero frames - "
                f"this is NOT a silent/idle extra controller like the ones checked so "
                f"far (Tails, all-zero). Refusing to silently drop real input - "
                f"inspect this demo manually before ingesting it."
            )
        print(f"  [load_demo_npz] {path}: {n_players}-player recording detected "
              f"({n_cols} columns) - non-player-0 columns confirmed all-zero "
              f"({rest_nonzero_frames} nonzero frames across {len(rest)} steps), "
              f"slicing to player 0's {n_buttons} columns.")
        actions_full = p0
    else:
        raise ValueError(
            f"{path}: actions has {n_cols} columns, not a multiple of "
            f"len(BUTTONS)={n_buttons} - can't interpret this as any number of "
            f"players' worth of standard Genesis button input."
        )

    return actions_full, info


def replay_and_extract(actions_full: np.ndarray, recorded_info=None, verbose=True):
    """Replays actions_full (T, len(BUTTONS)) through a FRESH env (standard default
    reset - i.e. this assumes the recording started from the normal Act 1 start state,
    not some other save state; this is checked, not assumed - see below), extracting
    causally-correct (frame_t, action_t) pairs. If recorded_info is given, cross-checks
    the replay's own x/y/score/lives against it at every step as a determinism/fidelity
    check (if this project's repeated "the emulator is deterministic given a fixed
    start state and fixed inputs" finding holds, and the recording really did start from
    the default state, every single step should match exactly).

    Returns dict with "frames" (T,IH,IW) float32, "actions" (T,len(BUTTONS)) int8,
    "info" (T,) list of dicts (freshly captured by this replay, for the sanity check),
    and "mismatch_count" (how many steps, if any, disagreed with recorded_info - 0 means
    either no recorded_info was given, or a perfect match, i.e. the recording did start
    from the assumed default state).
    """
    env = retro.make(game=GAME, render_mode=None)
    obs, info0 = env.reset()

    T = len(actions_full)
    frames = np.zeros((T, IH, IW), dtype=np.float32)
    replay_info = []
    mismatch_count = 0

    if recorded_info is not None and len(recorded_info) > 0:
        first_recorded = recorded_info[0]
        # info0 from env.reset() is always {} by this library's own convention
        # (confirmed earlier this project) - so the real first-step comparison has to be
        # against recorded_info[0], which corresponds to the state AFTER the recording's
        # first action was applied, matching replay_info[0] below (both are "after action
        # 0" snapshots) - not env.reset()'s own (always-empty) return.
        pass  # comparison happens in the loop below, once replay_info[0] exists

    for t in range(T):
        gray = downsample_gray(obs, IW, IH)
        frames[t] = gray
        mb = actions_full[t].astype(np.int8)
        obs, _reward, terminated, truncated, info = env.step(mb)
        replay_info.append(info)
        if recorded_info is not None and t < len(recorded_info):
            rec = recorded_info[t]
            keys_to_check = ("x", "y", "score", "lives")
            if any(info.get(k) != rec.get(k) for k in keys_to_check):
                mismatch_count += 1
                if verbose and mismatch_count <= 3:
                    print(f"  [ingest warning] step {t}: replay info {({k: info.get(k) for k in keys_to_check})} "
                          f"!= recorded info {({k: rec.get(k) for k in keys_to_check})}")
        if terminated or truncated:
            if verbose:
                print(f"  env signaled terminated/truncated at step {t}/{T} during replay")
            frames = frames[:t + 1]
            actions_full = actions_full[:t + 1]
            break

    env.close()

    if recorded_info is not None:
        n_checked = min(T, len(recorded_info))
        if mismatch_count == 0:
            print(f"  [ingest] Determinism check PASSED: all {n_checked} replayed steps "
                  f"exactly match the recording's own saved info (x/y/score/lives) - "
                  f"confirms the recording started from the standard default reset state.")
        else:
            print(f"  [ingest] Determinism check: {mismatch_count}/{n_checked} steps MISMATCHED "
                  f"recorded info - the recording may not have started from the standard "
                  f"default reset state (e.g. a non-default save state was used), or "
                  f"something else is off. Frames extracted by THIS replay are what will "
                  f"be used for BC training regardless, but if this count is high, the "
                  f"frame/x,y pairing may not reflect the human's actual playthrough "
                  f"faithfully - worth a closer look before trusting the result.")

    return {
        "frames": frames,
        "actions": actions_full.astype(np.int8),
        "info": replay_info,
    }


def check_ends_at_cutscene_trigger(info_list, lookback=30, box_override=None):
    """Sanity check: do the LAST few frames of this recording actually fall inside the
    known cutscene-trigger box (the "end" box from coords.list/CheckpointTracker)? This
    is a free, cheap way to confirm a human recording really did stop at the right
    point, using info we already get for free from the recording, rather than just
    trusting it. Returns (passed: bool, first_entry_step: int or None, detail: str).

    box_override: optional (x_min,x_max,y_min,y_max) to use INSTEAD of the box read
    from coords.list - for diagnostic use only when there's hard evidence the
    coords.list definition itself is off (see the 5-real-demo analysis: all 5
    independent human playthroughs crossed x=[11210,11213] at y in [987,1000], not the
    coords.list-defined y=[700,900] - this is NOT a per-demo quirk, it's the same ~90
    -290px discrepancy on every single one, strongly pointing at the box definition
    itself rather than anything wrong with the demos). This does NOT modify coords.list
    - it's a way to proceed with truncation using a corrected window while being
    explicit in the output that this isn't the canonical, as-recorded definition."""
    boxes = parse_coords_file(COORDS_FILE)
    tracker = CheckpointTracker(boxes)

    def describe(b):
        """Human-readable position description for either box shape coords.list
        supports (see checkpoints.py) - a plain bounding box, or a one-sided threshold
        (the "end" trigger is now the latter - see AGENTS.md Step 10)."""
        if b.condition == "threshold":
            return f"x>={b.threshold_x}"
        return f"x=[{b.x_min},{b.x_max}] y=[{b.y_min},{b.y_max}]"

    if box_override is not None:
        class _Box:
            def __init__(self, x_min, x_max, y_min, y_max):
                self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
                self.condition = "box"
            def contains(self, x, y):
                return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
        end_box = _Box(*box_override)
        print(f"  [sanity check] NOTE: using OVERRIDE box {describe(end_box)} "
              f"instead of coords.list's {describe(boxes[tracker.end_idx])} - "
              f"coords.list itself has NOT been modified.")
    else:
        end_box = boxes[tracker.end_idx]

    first_entry = None
    for t, info in enumerate(info_list):
        x, y = info.get("x"), info.get("y")
        if x is None or y is None:
            continue
        if end_box.contains(x, y):
            first_entry = t
            break

    n = len(info_list)
    if first_entry is None:
        return False, None, (
            f"NEVER entered the cutscene-trigger box (#{tracker.end_idx}, "
            f"{describe(end_box)}) "
            f"anywhere in this {n}-step recording - this demo does NOT reach the end "
            f"point as expected."
        )

    steps_after = n - 1 - first_entry
    detail = (
        f"Entered the cutscene-trigger box (#{tracker.end_idx}) at step {first_entry}/{n} "
        f"({steps_after} steps remain in the recording afterward)."
    )
    passed = steps_after <= lookback
    if not passed:
        detail += (
            f" That's MORE than {lookback} steps before the recording ends - either the "
            f"recording kept going well past the trigger (e.g. into the cutscene/next "
            f"area), or the trigger was entered and left again earlier than intended. "
            f"Worth a look, not necessarily wrong."
        )
    return passed, first_entry, detail


def ingest(npz_path: str, truncate_at_trigger: bool = True, box_override=None, full_act: bool = False):
    """Full pipeline for one demo file: load, replay+extract, sanity-check, optionally
    truncate to the cutscene-trigger point. Returns a dict ready to feed into BC
    training: {"frames": (T,IH,IW) f32, "actions": (T,N_BUTTONS) i8 in LEARNABLE_BUTTONS
    order, "n_frames": T, "sanity_passed": bool, "sanity_detail": str}.
    box_override: see check_ends_at_cutscene_trigger - NOT the default, only pass this
    when there's a documented reason the coords.list box itself is wrong.

    full_act (Step 27, for Foxo's 6 new demos_fullact/*.bk2 recordings that continue
    PAST the old cutscene trigger into the post-cutscene continuation and the boss
    approach): when True, the old truncate-at-cutscene-trigger logic is skipped
    entirely - truncating there would throw away exactly the new second-half
    coverage these demos exist to provide. check_ends_at_cutscene_trigger is still
    run and reported (as a sanity check that the demo DOES pass through the old
    trigger on the way through, which it always should), but its result no longer
    drives truncation. Instead, the full replayed recording is run back through
    CheckpointTracker(extend_past_end=True) over the merged coords.list to report
    which checkpoints (including the boss_trigger box) were actually reached -
    giving a real, measured coverage readout instead of just assuming the demo
    reaches as far as advertised.
    """
    print(f"Ingesting {npz_path}" + (" [full_act mode]" if full_act else ""))
    actions_full, recorded_info = load_demo_npz(npz_path)
    print(f"  raw actions shape (post player-0 slice if applicable): {actions_full.shape} "
          f"(expect (T, {len(BUTTONS)}))")
    if actions_full.shape[1] != len(BUTTONS):
        raise ValueError(f"{npz_path}: actions has {actions_full.shape[1]} columns, "
                          f"expected {len(BUTTONS)} (BUTTONS={BUTTONS}) - button order mismatch?")

    replayed = replay_and_extract(actions_full, recorded_info)

    passed, first_entry, detail = check_ends_at_cutscene_trigger(replayed["info"], box_override=box_override)
    print(f"  [sanity check: old cutscene trigger] {'PASSED' if passed else 'FAILED/WARNING (expected for full_act demos)' if full_act else 'FAILED/WARNING'}: {detail}")

    frames = replayed["frames"]
    actions_bt = multibinary_to_buttons(replayed["actions"])  # (T, N_BUTTONS)

    coverage_detail = None
    if full_act:
        boxes = parse_coords_file(COORDS_FILE)
        tracker = CheckpointTracker(boxes, extend_past_end=True)
        events = []
        for t, info in enumerate(replayed["info"]):
            x, y = info.get("x"), info.get("y")
            score = info.get("score")
            if x is None or y is None:
                continue
            _reward, terminated, event = tracker.step(x, y, score, t)
            if event:
                events.append((t, event))
            if terminated:
                break
        boss_reached = any("boss_trigger" in e for _, e in events)
        coverage_detail = (
            f"extend_past_end coverage over {len(replayed['info'])} frames: "
            f"{len(events)} checkpoint events fired, boss_trigger reached: {boss_reached}. "
            f"Events: {events[-6:] if len(events) > 6 else events}"
        )
        print(f"  [full_act coverage check] {coverage_detail}")
        if not boss_reached:
            print(f"  [full_act coverage check] WARNING: this demo never reached the "
                  f"boss_trigger box under the merged coords.list - this demo's actual "
                  f"content may not cover as far as expected, or the merged coords.list "
                  f"still needs adjustment. NOT silently assumed fine.")
    elif truncate_at_trigger and first_entry is not None:
        frames = frames[:first_entry + 1]
        actions_bt = actions_bt[:first_entry + 1]
        print(f"  truncated to {len(frames)} frames (up to and including the cutscene-trigger entry)")

    return {
        "frames": frames,
        "actions": actions_bt,
        "n_frames": len(frames),
        "sanity_passed": passed,
        "sanity_detail": detail,
        "coverage_detail": coverage_detail,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 ingest_demo.py [--full-act] <demo1.npz> [demo2.npz ...]")
        sys.exit(1)
    args = sys.argv[1:]
    full_act = "--full-act" in args
    if full_act:
        args.remove("--full-act")
    for path in args:
        result = ingest(path, full_act=full_act)
        print(f"  -> {result['n_frames']} usable frames, sanity_passed={result['sanity_passed']}\n")
