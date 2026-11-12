#!/bin/bash
# Step 30: resume real training per Foxo's explicit priority shift - stop
# spending cycles on detector forensics, get actual training running again.
# Continues each method from its OWN most-advanced checkpoint (not restarting
# from the shared BC checkpoint) for another 300 episodes each. Sequential,
# not concurrent, for the same shared-GPU/shared-WEIGHTS_FILE safety reasons
# established earlier tonight. Checkpoint-progress trend is the metric to
# watch (not boss-defeat rate) until it plateaus - the four-fix detector stack
# is trustworthy ground truth for defeat-rate reporting if/when that's needed.
set -e
cd /home/foxo/gateway-vulkan-rl
source venv/bin/activate
export PYTHONPATH=/home/foxo/gateway-vulkan-rl

N_TRAIN=300
N_BASELINE=10

echo "=== [continue2] method 2a (continuing from 600ep checkpoint -> 900ep) ==="
cp sonic_policy_weights_method2a_600ep_final.bin sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 7200 0 1 default > method2a_900ep.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2a_900ep_final.bin
echo "=== [continue2] method 2a done ==="

echo "=== [continue2] method 2b (continuing from 300ep checkpoint -> 600ep) ==="
cp sonic_policy_weights_method2b_scale300_final.bin sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 1500 0 1 boss > method2b_600ep.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2b_600ep_final.bin
echo "=== [continue2] method 2b done ==="

echo "=== [continue2] method 2c (continuing from 300ep checkpoint -> 600ep) ==="
cp sonic_policy_weights_method2c_scale300_final.bin sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 4000 0 1 mid > method2c_600ep.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2c_600ep_final.bin
echo "=== [continue2] method 2c done ==="

# Leave the live WEIGHTS_FILE on 2a's result (the method with the clearest,
# most consistent improvement trend) as the most useful default state for
# whatever touches it next, rather than whichever method ran last.
cp sonic_policy_weights_method2a_900ep_final.bin sonic_policy_weights.bin
echo "=== [continue2] ALL THREE METHODS COMPLETE (2a->900ep, 2b->600ep, 2c->600ep) - WEIGHTS_FILE left on 2a's latest ==="
