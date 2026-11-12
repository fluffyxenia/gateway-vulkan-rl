"""Direct unit test for the new stagnation penalty on repeated enemy-box entries.
Simulates re-entering box #3 ("first enemy") five times in a row, each time letting the
30-60 frame window expire with no score increase (so it never resolves to a kill), and
checks: entries 1-2 give reward 0.0, entries 3+ give exactly STAGNATION_PENALTY (-0.5)."""
import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker

boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")
tracker = CheckpointTracker(boxes)
tracker.reset()

ENEMY_IDX = 3
ex, ey = (boxes[ENEMY_IDX].x_min + boxes[ENEMY_IDX].x_max) // 2, (boxes[ENEMY_IDX].y_min + boxes[ENEMY_IDX].y_max) // 2
outside_x = boxes[ENEMY_IDX].x_min - 50

frame = 0
score = 0
all_ok = True
for attempt in range(1, 6):
    # step outside the box first (so enemy_pending resets cleanly / box not already entered)
    r, term, event = tracker.step(outside_x, ey, score, frame); frame += 1
    # enter the box
    r, term, event = tracker.step(ex, ey, score, frame); frame += 1
    expected = 0.0 if attempt <= 2 else CheckpointTracker.STAGNATION_PENALTY
    ok = abs(r - expected) < 1e-9
    all_ok &= ok
    print(f"{'OK  ' if ok else 'FAIL'} attempt {attempt}: reward={r:+.2f} (expected {expected:+.2f})  event={event}")
    # let the window expire without a score increase (65 frames of no progress, still outside)
    for _ in range(65):
        r2, term, event2 = tracker.step(outside_x, ey, score, frame); frame += 1
        if event2 is not None:
            print(f"     (unexpected event during expiry wait: {event2})")
            all_ok = False

print("\nALL PASSED" if all_ok else "\nSOME FAILED")
print(f"\nFinal enemy_attempts dict: {tracker.enemy_attempts}")
assert ENEMY_IDX not in tracker.visited, "enemy box should never get marked visited in this test (never defeated)"
print("Confirmed: box never marked visited (never actually defeated in this test) - penalty applies independent of outcome.")
