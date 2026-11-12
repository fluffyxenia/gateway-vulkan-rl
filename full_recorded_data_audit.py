"""Re-derive ALL 6 WholeAct demos' key milestones from RECORDED INFO directly
(no replay) - zone/act transitions, max x per act, any +100 score jumps
(defeats), lives changes - comprehensive, trustworthy ground-truth audit."""
import numpy as np

for i in range(6):
    path = f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz"
    d = np.load(path, allow_pickle=True)
    info = d["info"]
    n = len(info)
    xs = [j.get("x") for j in info]
    zones = [j.get("zone") for j in info]
    acts = [j.get("act") for j in info]
    scores = [j.get("score") for j in info]
    lives = [j.get("lives") for j in info]
    rings = [j.get("rings") for j in info]

    print(f"=== demo {i}: {n} recorded frames ===")
    print(f"  first frame: x={xs[0]} zone={zones[0]} act={acts[0]} score={scores[0]} lives={lives[0]}")
    print(f"  last frame:  x={xs[-1]} zone={zones[-1]} act={acts[-1]} score={scores[-1]} lives={lives[-1]}")

    # zone transitions
    prev_zone = zones[0]
    for idx in range(1, n):
        if zones[idx] != prev_zone:
            print(f"  ZONE change at frame {idx}: {prev_zone} -> {zones[idx]} (x {xs[idx-1]}->{xs[idx]}, act {acts[idx-1]}->{acts[idx]})")
            prev_zone = zones[idx]

    # act transitions (within same zone context tracking skipped for brevity, just report all)
    prev_act = acts[0]
    for idx in range(1, n):
        if acts[idx] != prev_act:
            print(f"  ACT change at frame {idx}: {prev_act} -> {acts[idx]} (zone={zones[idx]}, x {xs[idx-1]}->{xs[idx]})")
            prev_act = acts[idx]

    # lives changes (deaths)
    prev_lives = lives[0]
    for idx in range(1, n):
        if lives[idx] != prev_lives and lives[idx] is not None and prev_lives is not None:
            print(f"  LIVES change at frame {idx}: {prev_lives} -> {lives[idx]}")
            prev_lives = lives[idx]

    # +100 score jumps (defeats)
    prev_score = scores[0]
    for idx in range(1, n):
        if scores[idx] is not None and prev_score is not None and scores[idx] - prev_score == 100:
            print(f"  +100 SCORE JUMP (defeat) at frame {idx}: {prev_score} -> {scores[idx]}, x={xs[idx]}, zone={zones[idx]}, act={acts[idx]}")
        prev_score = scores[idx]

    # max x per (zone,act) combo
    seen_combos = {}
    for idx in range(n):
        key = (zones[idx], acts[idx])
        if xs[idx] is not None:
            seen_combos[key] = max(seen_combos.get(key, -1), xs[idx])
    print(f"  max x per (zone,act): {seen_combos}")
    print()
