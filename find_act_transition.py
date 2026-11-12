import numpy as np
d = np.load("/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000000.npz", allow_pickle=True)
info = d["info"]
acts = [i.get("act") for i in info]
prev = acts[0]
for idx, a in enumerate(acts):
    if a != prev:
        print(f"act transition at step {idx}: {prev} -> {a}")
        for j in range(max(0, idx - 5), min(len(info), idx + 10)):
            ii = info[j]
            print(f"  step {j}: x={ii.get('x')} y={ii.get('y')} zone={ii.get('zone')} "
                  f"act={ii.get('act')} screen_x={ii.get('screen_x')} lives={ii.get('lives')}")
        prev = a
print("\ntotal frames:", len(info))
print("final info:", info[-1])
