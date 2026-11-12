"""Apply the CORRECT methodology (use recorded info directly, not a fresh
replay) to demos 4 and 5 too, ahead of Foxo's own eyes-on check."""
import numpy as np

for i in [4, 5]:
    path = f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz"
    d = np.load(path, allow_pickle=True)
    info = d["info"]
    xs = [j.get("x") for j in info]
    scores = [j.get("score") for j in info]
    rings = [j.get("rings") for j in info]
    acts = [j.get("act") for j in info]
    print(f"=== demo {i}: {len(info)} recorded frames ===")
    for idx in range(1, len(info)):
        if xs[idx] is not None and xs[idx-1] is not None and abs(xs[idx] - xs[idx-1]) > 300:
            print(f"  x discontinuity at frame {idx}: {xs[idx-1]} -> {xs[idx]} (act {acts[idx-1]}->{acts[idx]})")
    max_x = max((x for x, a in zip(xs, acts) if a == 1 and x is not None), default=None)
    print(f"  max x in act==1: {max_x}")
    prev_score = scores[0]
    found_jump = False
    for idx in range(1, len(scores)):
        if scores[idx] is not None and prev_score is not None and scores[idx] - prev_score == 100:
            print(f"  RECORDED +100 score jump at frame {idx}: {prev_score} -> {scores[idx]}, x={xs[idx]}")
            found_jump = True
        prev_score = scores[idx]
    if not found_jump:
        print(f"  no +100 jump found in recorded info. final 10 frames: {[(xs[j], scores[j], rings[j], acts[j]) for j in range(max(0,len(info)-10), len(info))]}")
