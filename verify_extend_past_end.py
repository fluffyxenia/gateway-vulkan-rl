"""One-off verification (not standing infra): confirms CheckpointTracker's new
extend_past_end flag behaves correctly on both settings before trusting it:
1. extend_past_end=False (default): reaching the "end" box terminates, exactly as
   every existing BC-pretraining call site (ingest_demo.py etc.) depends on.
2. extend_past_end=True: reaching the "end" box does NOT terminate - episode
   continues to evaluate the new continuation boxes, including the boss_trigger.
"""
import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker

boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")

# --- Test 1: default behavior unchanged ---
tracker = CheckpointTracker(boxes)  # extend_past_end=False by default
tracker.reset()
r, terminated, event = tracker.step(12210, 500, score=0, frame_idx=2500)
print(f"Test 1 (extend_past_end=False, reach end box): reward={r} terminated={terminated} event={event}")
assert terminated, "FAIL: default behavior should still terminate at the end box"
assert event == "level_end_trigger"
print("PASS: default behavior unchanged (still terminates)")

# --- Test 2: extend_past_end=True continues past the end box, AND same-step
# cascading through chained prerequisites is blocked (Step 22 fix) ---
tracker2 = CheckpointTracker(boxes, extend_past_end=True)
tracker2.reset()
r2, terminated2, event2 = tracker2.step(12210, 500, score=0, frame_idx=2500)
print(f"\nTest 2a (extend_past_end=True, reach end box, SAME step as everything downstream "
      f"of it numerically): reward={r2} terminated={terminated2} event={event2}")
assert not terminated2, "FAIL: extend_past_end=True should NOT terminate at the end box"
assert r2 == CheckpointTracker.END_REWARD, (
    f"FAIL: expected ONLY END_REWARD this step ({CheckpointTracker.END_REWARD}), got {r2} - "
    f"means box 37 ('Survived cutscene') and/or the boss trigger fired in the SAME "
    f"step as the end-trigger, which is exactly the same-step-cascade bug the "
    f"visited-snapshot fix is supposed to prevent.")
print("PASS: end box grants reward, does not terminate, and does NOT cascade into "
      "downstream prerequisite-chained boxes within this same step")

# a LATER, separate step reaching x=472 in the (simulated) new post-cutscene space -
# box 37's prerequisite (box 36 visited) is now satisfied, from a PRIOR step
r2b, term2b, ev2b = tracker2.step(472, 500, score=0, frame_idx=2600)
print(f"\nTest 2b (later step, x=472, post-cutscene 'Survived cutscene' threshold): "
      f"reward={r2b} terminated={term2b} event={ev2b}")
assert r2b == CheckpointTracker.CHECKPOINT_REWARD
assert not term2b

# continue stepping through the new continuation to reach the boss trigger, on yet
# another later step
r3, terminated3, event3 = tracker2.step(4465, 500, score=0, frame_idx=3000)
print(f"\nTest 2c (continue to boss trigger x=4465): reward={r3} terminated={terminated3} event={event3}")
assert not terminated3, "FAIL: boss_trigger should not terminate the episode itself"
assert event3 is not None and "boss_trigger" in event3
assert r3 == CheckpointTracker.CHECKPOINT_REWARD
print("PASS: boss trigger reached, rewarded, episode continues (no auto-termination)")

# --- Test 3: kill_checkpoint (special stage room) terminates immediately, no time-gate ---
tracker3 = CheckpointTracker(boxes)
tracker3.reset()
r4, terminated4, event4 = tracker3.step(700, 1000, score=0, frame_idx=5)  # inside [452,983]x[825,1249], very early frame_idx
print(f"\nTest 3 (kill_checkpoint / special stage room, frame_idx=5): reward={r4} terminated={terminated4} event={event4}")
assert terminated4, "FAIL: kill_checkpoint should terminate immediately, no time-gate"
assert r4 == CheckpointTracker.KILL_CHECKPOINT_PENALTY
assert event4 is not None and "kill_checkpoint" in event4
print("PASS: kill_checkpoint terminates immediately regardless of frame_idx (no time-gate, unlike start_zone_kill)")

print("\n=== ALL EXTEND_PAST_END / BOSS_TRIGGER / KILL_CHECKPOINT BEHAVIORS VERIFIED ===")
