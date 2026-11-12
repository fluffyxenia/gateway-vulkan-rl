"""
Full-pipeline verification of the new threshold+prerequisite condition type against all
5 real demo recordings - not just the END box's position-only check (ingest_demo.py's
check_ends_at_cutscene_trigger), but the COMPLETE CheckpointTracker state machine,
stepped through every recorded frame exactly as a real training rollout would, to
confirm:
  1. camera-lock-1 (threshold x>=11707, requires box #33 "DO NOT LET GO" already
     visited) fires correctly, and - critically - NEVER fires before its prerequisite
     has actually been visited (which would indicate the prerequisite gating is broken).
  2. the end-trigger (threshold x>=11215, no prerequisite) fires correctly via the full
     tracker, consistent with ingest_demo.py's independent position-only check.
  3. box #33 ("DO NOT LET GO...") itself gets visited at some point before camera-lock-1
     fires, for every demo (sanity: the prerequisite is actually satisfiable on these
     real playthroughs, not just theoretically referenced).
"""
import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker
from ingest_demo import load_demo_npz
import glob

boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")

# Step 22 fix: these were hardcoded numeric indices, which already silently went
# stale once before (the Step 15 start_zone_kill box insertion shifted every later
# index by +1, undetected until this exact script was re-run for the Step 22 merge
# verification). Looking these up by their known, unique comment text instead makes
# this robust to any future box insertion earlier in the file, not just the current
# one.
def _find_unique(predicate, label):
    hits = [i for i, b in enumerate(boxes) if predicate(b)]
    assert len(hits) == 1, f"expected exactly one box matching {label!r}, found {len(hits)}: {hits}"
    return hits[0]

PREREQ_IDX = _find_unique(lambda b: b.comment.startswith("**JUST HOLD RIGHT"), "JUST HOLD RIGHT prefix")
CAMLOCK_IDX = _find_unique(lambda b: b.comment == "camera lock 1 trigger", "camera lock 1 trigger")
END_IDX = _find_unique(lambda b: b.comment == "interactive-cutscene trigger", "interactive-cutscene trigger")

all_ok = True
for f in sorted(glob.glob("/home/foxo/gateway-vulkan-rl/demos/*.npz")):
    actions, info = load_demo_npz(f)
    tracker = CheckpointTracker(boxes)
    tracker.reset()

    prereq_step = None
    camlock_step = None
    end_step = None
    camlock_fired_before_prereq = False

    for t, inf in enumerate(info):
        x, y, score = inf.get("x"), inf.get("y"), inf.get("score", 0)
        r, terminated, event = tracker.step(x, y, score, t)
        if event == f"checkpoint(box={PREREQ_IDX})" and prereq_step is None:
            prereq_step = t
        if event == f"checkpoint(box={CAMLOCK_IDX})" and camlock_step is None:
            camlock_step = t
            if prereq_step is None:
                camlock_fired_before_prereq = True
        if event == "level_end_trigger" and end_step is None:
            end_step = t
        if terminated:
            break

    name = f.split("/")[-1]
    print(f"=== {name} ===")
    print(f"  prerequisite (#{PREREQ_IDX}) visited at step: {prereq_step}")
    print(f"  camera-lock-1 (#{CAMLOCK_IDX}) fired at step: {camlock_step}")
    print(f"  end-trigger fired at step: {end_step}")

    ok = True
    if prereq_step is None:
        print("  FAIL: prerequisite never visited on this demo - can't test gating meaningfully")
        ok = False
    if camlock_step is None:
        print("  FAIL: camera-lock-1 never fired")
        ok = False
    if camlock_fired_before_prereq:
        print("  FAIL: camera-lock-1 fired BEFORE its prerequisite was visited - gating is broken!")
        ok = False
    if prereq_step is not None and camlock_step is not None and camlock_step < prereq_step:
        print("  FAIL: camera-lock-1 step is before prerequisite step (inconsistent)")
        ok = False
    if end_step is None:
        print("  FAIL: end-trigger never fired")
        ok = False
    if camlock_step is not None and end_step is not None and camlock_step > end_step:
        print("  NOTE: camera-lock-1 fired after the end-trigger (unexpected x-ordering, worth a look)")
    if ok:
        print("  OK: prerequisite -> camera-lock-1 -> end-trigger all fired in the correct order")
    print()
    all_ok &= ok

print("ALL DEMOS PASSED full-tracker verification" if all_ok else "SOME DEMOS FAILED - see above")
