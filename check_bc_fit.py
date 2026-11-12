import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from policy_net import PolicyNet
from sonic_rl_train import N_BUTTONS, LEARNABLE_BUTTONS
from ingest_demo import ingest

result = ingest("/home/foxo/gateway-vulkan-rl/fake_demo.npz")
frames = result["frames"]
true_actions = result["actions"]

net = PolicyNet(N_BUTTONS)
net.load("/home/foxo/gateway-vulkan-rl/sonic_policy_weights_bc_pretrained.bin")
probs = net.forward(frames)
pred = (probs > 0.5).astype(np.int8)
acc_per_button = (pred == true_actions).mean(axis=0)
freq_per_button = true_actions.mean(axis=0)
print(f"{'button':6s} {'true_freq':>9s} {'mean_pred_prob':>14s} {'accuracy':>9s}")
for b, f, p, a in zip(LEARNABLE_BUTTONS, freq_per_button, probs.mean(axis=0), acc_per_button):
    print(f"{b:6s} {f:9.3f} {p:14.3f} {a:9.3f}")
print("\noverall per-button accuracy:", acc_per_button.mean())
