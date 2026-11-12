#!/bin/bash
# Step 29: at-scale (300-episode) three-way boss-training comparison, per Foxo's
# explicit direction via the coordinator - the pilot (20 episodes) was too small
# to distinguish trained-policy skill from random luck, especially for method 2b.
# Matches the original historical 2a run's scale (n_train=300, n_baseline=10,
# max_steps=7200) for apples-to-apples comparison; same per-method max_steps as
# the pilot otherwise. Sequenced 2c, 2b, 2a (most-promising-and-fastest
# information first, then the one needing the most data to resolve ambiguity,
# then the slowest/most-volume-hungry one) - run sequentially, not concurrently,
# for the same shared-GPU/shared-WEIGHTS_FILE reasons as the pilot.
set -e
cd /home/foxo/gateway-vulkan-rl
source venv/bin/activate
export PYTHONPATH=/home/foxo/gateway-vulkan-rl

BASE_CKPT=sonic_policy_weights_bc_pretrained_fullcov.bin
N_TRAIN=300
N_BASELINE=10

echo "=== [scale300] method 2c (save-state mid-level, decomp ROM) ==="
cp "$BASE_CKPT" sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 4000 0 1 mid > method2c_scale300.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2c_scale300_final.bin
echo "=== [scale300] method 2c done ==="

echo "=== [scale300] method 2b (save-state before boss, decomp ROM) ==="
cp "$BASE_CKPT" sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 1500 0 1 boss > method2b_scale300.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2b_scale300_final.bin
echo "=== [scale300] method 2b done ==="

echo "=== [scale300] method 2a (full-level-replay, real ROM, default start) ==="
cp "$BASE_CKPT" sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 7200 0 1 default > method2a_scale300.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2a_scale300_final.bin
echo "=== [scale300] method 2a done ==="

cp "$BASE_CKPT" sonic_policy_weights.bin
echo "=== [scale300] ALL THREE METHODS COMPLETE - WEIGHTS_FILE reset to clean fullcov checkpoint ==="
