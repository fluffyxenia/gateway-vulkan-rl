"""For each demo, find every step where x fell in [11210,11213] (the known trigger
box's x-range) regardless of y, and report the actual y values there - to see how far
off from the box's y=[700,900] window the player actually was when crossing that x
range. Also report the overall x,y trajectory's min/max y to understand the route
shape. Uses ingest_demo's own replay (real info, not the recording's raw info) for
consistency, though since the determinism check passed for all 5, using the recording's
own saved info directly is equally valid and faster - use that here."""
import sys
import glob
import numpy as np

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from ingest_demo import load_demo_npz
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE

boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes)
end_box = boxes[tracker.end_idx]
print(f"end box: x=[{end_box.x_min},{end_box.x_max}] y=[{end_box.y_min},{end_box.y_max}]\n")

for f in sorted(glob.glob("/home/foxo/gateway-vulkan-rl/demos/*.npz")):
    actions, info = load_demo_npz(f)
    xs = np.array([i.get("x") for i in info])
    ys = np.array([i.get("y") for i in info])
    print(f"=== {f.split('/')[-1]} ===")
    print(f"  n_steps={len(xs)}  x range=[{xs.min()},{xs.max()}]  y range=[{ys.min()},{ys.max()}]")
    in_x_range = (xs >= end_box.x_min) & (xs <= end_box.x_max)
    idxs = np.where(in_x_range)[0]
    if len(idxs) == 0:
        print(f"  x NEVER entered [{end_box.x_min},{end_box.x_max}] at all (monotonic jump over it between samples, or never reached)")
    else:
        print(f"  x was in [{end_box.x_min},{end_box.x_max}] at steps {idxs[0]}..{idxs[-1]} ({len(idxs)} steps)")
        print(f"  y values during that x window: {ys[idxs].tolist()}")
        print(f"  (box wants y in [{end_box.y_min},{end_box.y_max}])")
    print()
