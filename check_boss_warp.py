import numpy as np
d = np.load("/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000000.npz", allow_pickle=True)
info = d["info"]
# look at the full trajectory from the act transition (step 2531) to the end,
# watching for any x discontinuity (jump or reset) especially near x=4465 (boss trigger)
xs = [i.get("x") for i in info]
for idx in range(2531, len(info), 20):
    ii = info[idx]
    print(f"step {idx}: x={ii.get('x')} y={ii.get('y')} lives={ii.get('lives')} "
          f"rings={ii.get('rings')} score={ii.get('score')}")

print("\n--- checking for any large x jump/discontinuity anywhere after step 2531 ---")
for idx in range(2532, len(info)):
    dx = xs[idx] - xs[idx-1] if xs[idx] is not None and xs[idx-1] is not None else None
    if dx is not None and abs(dx) > 100:
        print(f"step {idx}: x jumped {xs[idx-1]} -> {xs[idx]} (delta={dx})")
