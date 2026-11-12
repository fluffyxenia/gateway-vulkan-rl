import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from policy_net import PolicyNet
from sonic_rl_train import N_BUTTONS, LEARNABLE_BUTTONS
from bc_pretrain import build_combined_batch

demos = [
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-BottomRoute-000000.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-BottomRoute-000001.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000000.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopToBottomRoute-000000.npz",
]
frames, actions, _ = build_combined_batch(demos, box_override=(11200, 11230, 950, 1050))

weights_path = sys.argv[1] if len(sys.argv) > 1 else "/home/foxo/gateway-vulkan-rl/sonic_policy_weights_bc_pretrained.bin"
print(f"Checking weights file: {weights_path}")
net = PolicyNet(N_BUTTONS)
net.load(weights_path)

probs = net.forward(frames)
pred = (probs > 0.5).astype(np.int8)
acc_per_button = (pred == actions).mean(axis=0)
freq_per_button = actions.mean(axis=0)

print(f"{'button':6s} {'true_freq':>9s} {'mean_pred_prob':>14s} {'accuracy':>9s}")
for b, f, p, a in zip(LEARNABLE_BUTTONS, freq_per_button, probs.mean(axis=0), acc_per_button):
    print(f"{b:6s} {f:9.3f} {p:14.3f} {a:9.3f}")
print("\noverall per-button accuracy:", acc_per_button.mean())

# Baseline: what accuracy would "always predict the majority class per button" get?
# (if the real model isn't beating this, it hasn't learned anything beyond base rates)
majority_baseline = np.maximum(freq_per_button, 1 - freq_per_button).mean()
print("majority-class baseline accuracy (always guess the more common class):", majority_baseline)

# Check for real visual conditioning: do predictions vary meaningfully across different
# frames, or are they flat (= only learned marginal base rates, didn't use the image)?
std_per_button = probs.std(axis=0)
print("\nstd of predicted probability across all frames, per button (near 0 = flat/ignoring image):")
for b, s in zip(LEARNABLE_BUTTONS, std_per_button):
    print(f"  {b:6s}: {s:.3f}")

# Spot check: a few individual frames, compare predicted probs to the actual recorded action
print("\nspot-check individual frames (predicted prob vs actual action):")
rng = np.random.default_rng(0)
for idx in rng.choice(len(frames), 5, replace=False):
    p = probs[idx]
    a = actions[idx]
    print(f"  frame {idx}:")
    for b, pb, ab in zip(LEARNABLE_BUTTONS, p, a):
        marker = " <-- actual=1" if ab == 1 else ""
        print(f"    {b:6s} pred_prob={pb:.3f}{marker}")
