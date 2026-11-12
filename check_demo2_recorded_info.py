"""Use the ORIGINAL RECORDED info from demo 2's .npz directly (captured during
the .bk2->npz conversion, i.e. what actually happened when the real recording
was played back) - NOT a fresh replay-of-actions through my own env.step()
loop, which has an ALREADY-DOCUMENTED (Step 27) divergence problem for this
exact demo (68% determinism mismatch starting ~step 1582, compounding
chaotically). This is likely why my earlier RAM-trace investigation found a
bogus (0,0) coordinate reset that never happened in the real recording."""
import numpy as np

d = np.load("/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-000002.npz", allow_pickle=True)
info = d["info"]
print(f"total recorded frames: {len(info)}")

xs = [i.get("x") for i in info]
scores = [i.get("score") for i in info]
rings = [i.get("rings") for i in info]
acts = [i.get("act") for i in info]

# check for any x,y degenerate reset to near-0 anywhere in the RECORDED info
for idx in range(len(info)):
    if idx > 0 and xs[idx] is not None and xs[idx-1] is not None and abs(xs[idx] - xs[idx-1]) > 300:
        print(f"recorded-info x discontinuity at frame {idx}: {xs[idx-1]} -> {xs[idx]} (act {acts[idx-1]}->{acts[idx]})")

print(f"\nmax x reached (recorded info, act==1 only): {max((x for x,a in zip(xs,acts) if a==1 and x is not None), default=None)}")
print(f"final 10 frames (recorded): {[(xs[i], scores[i], rings[i], acts[i]) for i in range(max(0,len(info)-10), len(info))]}")

# score trajectory - any +100 jump?
prev_score = scores[0]
for idx in range(1, len(scores)):
    if scores[idx] is not None and prev_score is not None and scores[idx] - prev_score == 100:
        print(f"RECORDED +100 score jump at frame {idx}: {prev_score} -> {scores[idx]}, x={xs[idx]}")
    prev_score = scores[idx]
