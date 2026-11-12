import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import COORDS_FILE, START_MODES, buttons_to_multibinary

LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/train_ep174_outcome-boss_defeated.npy"
actions = np.load(LOG)
mode_cfg = START_MODES["boss"]
env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
tracker.reset(pre_visited={tracker.end_idx})
obs, info = env.reset()
addr = 0xfeb3
for t in range(len(actions)):
    mb = buttons_to_multibinary(actions[t].astype(np.int8))
    obs, _r, term, trunc, info = env.step(mb)
    x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
    r, cp_term, event = tracker.step(x, y, score, t)
    if 1270 <= t <= 1290:
        ram = env.unwrapped.get_ram()
        print(f"t={t} x={x} y={y} score={score} rings={info.get('rings')} lives={info.get('lives')} byte@addr={ram[addr]}")
env.close()
