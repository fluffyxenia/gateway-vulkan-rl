"""Apply the timer+score defeat check to all 6 WholeAct human demos - Foxo
confirmed every one of them contains a real boss hit-and-defeat sequence
before the act ends, giving multiple independent ground-truth instances."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME
from timer_defeat_check import check_timer_freeze_defeat, TIMER_BYTE_ADDR

DEMOS = [f"/home/foxo/gateway-vulkan-rl/demos_fullact/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-WholeAct-{i:06d}.npz" for i in range(6)]

for path in DEMOS:
    d = np.load(path, allow_pickle=True)
    actions_full = d["actions"]
    actions = actions_full[:, :12]
    env = retro.make(game=GAME, render_mode=None)
    obs, info0 = env.reset()
    scores, timer_bytes = [], []
    for t in range(len(actions)):
        mb = actions[t].astype(np.int8)
        obs, _r, term, trunc, info = env.step(mb)
        ram = env.unwrapped.get_ram()
        scores.append(info.get("score", 0))
        timer_bytes.append(int(ram[TIMER_BYTE_ADDR]))
        if term or trunc:
            break
    env.close()
    hits = check_timer_freeze_defeat(scores, timer_bytes)
    name = path.split("/")[-1]
    print(f"{name}: {len(scores)} frames, timer-freeze-defeat hits at frames: {hits}")
