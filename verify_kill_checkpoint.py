"""One-off verification (not standing infra): confirms the new start-zone kill
checkpoint behaves correctly before trusting it in real training:
1. Does NOT trigger at the real measured spawn position (x=5002, y=1051).
2. Does NOT trigger on a normal, fast pass-through en route to checkpoint 1
   (frame_idx below the time gate) - this is the bug a real smoke test caught: a
   plain position check fires on every episode that makes ANY forward progress,
   since reaching checkpoint 1 requires passing through this box first.
3. DOES trigger (terminate + penalty) when the player is STILL in the box after the
   time gate has passed - genuine non-progression.
4. Becomes permanently inert for the rest of the episode once the first real
   checkpoint has been reached (no retroactive kill on a later backtrack).
"""
import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker

boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")
tracker = CheckpointTracker(boxes)

kill_idx = [i for i, b in enumerate(boxes) if b.kind == "start_zone_kill"]
assert len(kill_idx) == 1, f"expected exactly 1 start_zone_kill box, found {len(kill_idx)}"
kill_idx = kill_idx[0]
print(f"start_zone_kill box is #{kill_idx}: {boxes[kill_idx]}")

# --- Test 1: real spawn position must NOT trigger it ---
tracker.reset()
SPAWN_X, SPAWN_Y = 5002, 1051
r, terminated, event = tracker.step(SPAWN_X, SPAWN_Y, score=0, frame_idx=0)
print(f"\nTest 1 (spawn x={SPAWN_X},y={SPAWN_Y}): reward={r} terminated={terminated} event={event}")
assert not terminated, "FAIL: spawn position triggered episode termination!"
assert event is None, f"FAIL: spawn position triggered an event: {event}"
print("PASS: spawn position does not trigger anything")

KILL_X, KILL_Y = 5025, 1051  # comfortably inside [5010,5045]

# --- Test 2: normal fast pass-through (frame_idx below the time gate) must NOT fire ---
# Real human demos cross spawn->checkpoint1 in 45-74 steps (measured directly from all
# 5 demos) - frame_idx=100 represents a normal, if slightly slow, policy still well
# under the 300-frame gate.
tracker.reset()
r, terminated, event = tracker.step(KILL_X, KILL_Y, score=0, frame_idx=100)
print(f"\nTest 2 (kill-zone x={KILL_X},y={KILL_Y}, frame_idx=100, normal pass-through): "
      f"reward={r} terminated={terminated} event={event}")
assert not terminated, "FAIL: kill box fired on a normal fast pass-through!"
assert event is None, f"FAIL: kill box produced an event on a normal pass-through: {event}"
print("PASS: normal pass-through before the time gate does not trigger anything")

# --- Test 3: still in the box AFTER the time gate DOES terminate with penalty ---
tracker.reset()
r, terminated, event = tracker.step(KILL_X, KILL_Y, score=0, frame_idx=350)
print(f"\nTest 3 (kill-zone x={KILL_X},y={KILL_Y}, frame_idx=350, past the time gate): "
      f"reward={r} terminated={terminated} event={event}")
assert terminated, "FAIL: kill box did not terminate after the time gate"
assert r < 0, f"FAIL: expected a negative penalty, got {r}"
assert event is not None and "start_zone_kill" in event
print(f"PASS: kill box terminates with penalty={r} (milder than HAZARD_PENALTY=-2.0)")

# --- Test 4: once checkpoint 1 is reached, the kill box becomes permanently inert,
# even past the time gate ---
tracker.reset()
r1, term1, ev1 = tracker.step(5055, 1051, score=0, frame_idx=50)  # reach checkpoint 1 first
print(f"\nTest 4 setup: reached checkpoint 1 -> reward={r1} terminated={term1} event={ev1}")
assert not term1 and ev1 is not None and "checkpoint" in ev1
r2, term2, ev2 = tracker.step(KILL_X, KILL_Y, score=0, frame_idx=400)  # backtrack, past gate
print(f"Test 4 (backtrack into kill-zone AFTER checkpoint 1, past time gate): "
      f"reward={r2} terminated={term2} event={ev2}")
assert not term2, "FAIL: kill box retroactively killed a run after checkpoint 1 was reached"
assert ev2 is None, f"FAIL: kill box still fired an event after being disabled: {ev2}"
print("PASS: kill box is permanently inert for the rest of the episode after checkpoint 1")

print("\n=== ALL REQUIRED BEHAVIORS VERIFIED ===")
