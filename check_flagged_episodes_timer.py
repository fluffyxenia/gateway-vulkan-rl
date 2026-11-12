"""Apply the orthogonal timer+score defeat check to every episode previously
FLAGGED as boss_defeated by the hit-counter detector (2a: 4 episodes across
both runs; 2b: 109 episodes) - an independent cross-validation of the
already-established 0/0/0 result, using game-state data instead of any
per-object RAM hit-counter guess."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import GAME, START_MODES, N_BUTTONS, buttons_to_multibinary
from timer_defeat_check import check_timer_freeze_defeat, TIMER_BYTE_ADDR

# 2a's 4 flagged episodes (real ROM, default mode)
a_logs = [
    ("/home/foxo/gateway-vulkan-rl/action_logs/20261109_223601/train_ep218_outcome-boss_defeated.npy", "default"),
    ("/home/foxo/gateway-vulkan-rl/action_logs/20261110_032240/train_ep081_outcome-boss_defeated.npy", "default"),
    ("/home/foxo/gateway-vulkan-rl/action_logs/20261110_032240/train_ep129_outcome-boss_defeated.npy", "default"),
    ("/home/foxo/gateway-vulkan-rl/action_logs/20261110_032240/train_ep291_outcome-boss_defeated.npy", "default"),
]

import glob
b_logs = [(p, "boss") for p in sorted(glob.glob("/home/foxo/gateway-vulkan-rl/action_logs/20261109_215031/*outcome-boss_defeated.npy"))]

all_logs = a_logs + b_logs
print(f"Checking {len(all_logs)} previously-flagged episodes with the orthogonal timer+score signal...")

any_hits = []
for path, mode in all_logs:
    actions = np.load(path)
    mode_cfg = START_MODES[mode]
    if mode == "default":
        env = retro.make(game=mode_cfg["game"], render_mode=None)
    else:
        env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"], inttype=retro.data.Integrations.ALL, render_mode=None)
    obs, info0 = env.reset()
    scores, timer_bytes = [], []
    for t in range(len(actions)):
        mb = buttons_to_multibinary(actions[t].astype(np.int8))
        obs, _r, term, trunc, info = env.step(mb)
        ram = env.unwrapped.get_ram()
        scores.append(info.get("score", 0))
        timer_bytes.append(int(ram[TIMER_BYTE_ADDR]))
    env.close()
    hits = check_timer_freeze_defeat(scores, timer_bytes)
    name = path.split("/")[-1]
    if hits:
        any_hits.append((name, hits))
        print(f"  {name}: TIMER-FREEZE HIT at frames {hits} !!")

print(f"\n=== {len(any_hits)}/{len(all_logs)} previously-flagged episodes independently confirmed by timer+score ===")
