"""Confirm the stagnation penalty and a later genuine kill are independent: enter the
enemy box 3 times (first two expire unresolved, free; 3rd gets the -0.5 stagnation
penalty on entry), then on this 3rd attempt let the score actually increase within the
window - should still get the full +2.0 ENEMY_REWARD when it resolves, on top of (not
instead of) the -0.5 already charged on entry."""
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

# Two free unresolved attempts
for attempt in range(2):
    tracker.step(outside_x, ey, score, frame); frame += 1
    tracker.step(ex, ey, score, frame); frame += 1
    for _ in range(65):
        tracker.step(outside_x, ey, score, frame); frame += 1

# Third attempt: should get -0.5 on entry, then a kill 40 frames later (score goes up)
tracker.step(outside_x, ey, score, frame); frame += 1
r_entry, _, event_entry = tracker.step(ex, ey, score, frame); frame += 1
print(f"3rd entry: reward={r_entry:+.2f} event={event_entry}")
assert abs(r_entry - CheckpointTracker.STAGNATION_PENALTY) < 1e-9

for i in range(39):
    tracker.step(outside_x, ey, score, frame); frame += 1
score += 10  # score goes up within the 30-60 frame window
r_kill, term, event_kill = tracker.step(outside_x, ey, score, frame); frame += 1
print(f"kill step: reward={r_kill:+.2f} event={event_kill}")
assert abs(r_kill - CheckpointTracker.ENEMY_REWARD) < 1e-9, f"expected +2.0 ENEMY_REWARD, got {r_kill}"
assert ENEMY_IDX in tracker.visited
print("\nPASSED: stagnation penalty (-0.5 on entry) and the later +2.0 kill reward are independent/additive, "
      "and the box correctly ends up marked visited once actually defeated.")
