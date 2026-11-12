#!/bin/bash
# Step 28: three-way boss-training method comparison pilot (2a/2b/2c), all starting
# from the identical fullcov BC-pretrained checkpoint, run sequentially (not
# concurrently - all three would otherwise collide on the single shared
# sonic_policy_weights.bin / sonic_policy_server process). Each method's final
# weights are saved to its own named file immediately after that method's run
# finishes, and the shared live WEIGHTS_FILE is reset back to the SAME starting
# checkpoint before the next method begins, so no method's run is contaminated by
# a previous one's training, and the live file ends in a known, documented state
# (reset back to the clean fullcov checkpoint) rather than whichever method
# happened to run last.
set -e
cd /home/foxo/gateway-vulkan-rl
source venv/bin/activate
export PYTHONPATH=/home/foxo/gateway-vulkan-rl

BASE_CKPT=sonic_policy_weights_bc_pretrained_fullcov.bin
N_TRAIN=20
N_BASELINE=3

echo "=== [pilot] method 2a (full-level-replay, real ROM, default start) ==="
cp "$BASE_CKPT" sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 7200 0 1 default > method2a_pilot.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2a_pilot_final.bin
echo "=== [pilot] method 2a done ==="

echo "=== [pilot] method 2b (save-state before boss, decomp ROM) ==="
cp "$BASE_CKPT" sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 1500 0 1 boss > method2b_pilot.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2b_pilot_final.bin
echo "=== [pilot] method 2b done ==="

echo "=== [pilot] method 2c (save-state mid-level, decomp ROM) ==="
cp "$BASE_CKPT" sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 4000 0 1 mid > method2c_pilot.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2c_pilot_final.bin
echo "=== [pilot] method 2c done ==="

# Leave the shared live WEIGHTS_FILE in a known, clean, documented state (the same
# starting checkpoint all three methods began from) rather than whichever method
# happened to run last.
cp "$BASE_CKPT" sonic_policy_weights.bin
echo "=== [pilot] ALL THREE METHODS COMPLETE - WEIGHTS_FILE reset to clean fullcov checkpoint ==="
