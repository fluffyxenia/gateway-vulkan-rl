"""Check whether the GENUINE human demo (demo 0, confirmed real win) ever
shows a similar y-climb/fall-through after settling at x=4616 post-defeat -
this tells us whether the ground at this exact spot is normally solid."""
import numpy as np
d = np.load("/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000000.npz", allow_pickle=True)
info = d["info"]
ys = [i.get("y") for i in info]
xs = [i.get("x") for i in info]
print(f"total frames: {len(info)}")
print(f"y range from frame 4548 (defeat) to end: min={min(ys[4548:])} max={max(ys[4548:])}")
for t in range(4548, len(info), 30):
    print(f"t={t}: x={xs[t]} y={ys[t]}")
