import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from sonic_rl_train import buttons_to_multibinary, N_BUTTONS

actions = np.load("/home/foxo/gateway-vulkan-rl/action_logs/20261111_212735/train_ep292_outcome-died.npy")
env = retro.make(game="SonicKnuckles3Decomp-Genesis", state="AngelIslandZone.Act1Boss",
                  inttype=retro.data.Integrations.ALL, render_mode=None)
obs, info0 = env.reset()
print(f"actions shape: {actions.shape}")
for t in range(len(actions)):
    mb = buttons_to_multibinary(actions[t].astype(np.int8))
    obs, _r, term, trunc, info = env.step(mb)
    if t % 20 == 0 or t >= len(actions) - 10 or term or trunc:
        print(f"t={t}: x={info.get('x')} y={info.get('y')} lives={info.get('lives')} rings={info.get('rings')} score={info.get('score')}")
    if term or trunc:
        print(f"  -> env signaled terminated/truncated at t={t}")
        break
env.close()
print(f"total steps in action log: {len(actions)}, real-time at 60fps: {len(actions)/60:.2f} seconds")
