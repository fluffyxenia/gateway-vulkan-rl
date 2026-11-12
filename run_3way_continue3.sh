#!/bin/bash
# Step 31: third continuation round, per Foxo's explicit instruction to use
# judgment on scope rather than wait for approval. Continues each method from
# its own latest checkpoint for another 300 episodes. Noted: 2b showed a real
# behavioral collapse (repeated identical death pattern) in the prior round -
# continuing anyway rather than intervening, since RL exploration may break
# out of it naturally and no training-correctness bug was found, just an
# RL-dynamics observation worth watching, not fixing blindly.
set -e
cd /home/foxo/gateway-vulkan-rl
source venv/bin/activate
export PYTHONPATH=/home/foxo/gateway-vulkan-rl

N_TRAIN=300
N_BASELINE=10

echo "=== [continue3] method 2a (900ep -> 1200ep) ==="
cp sonic_policy_weights_method2a_900ep_final.bin sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 7200 0 1 default > method2a_1200ep.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2a_1200ep_final.bin
echo "=== [continue3] method 2a done ==="

echo "=== [continue3] method 2b (600ep -> 900ep) ==="
cp sonic_policy_weights_method2b_600ep_final.bin sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 1500 0 1 boss > method2b_900ep.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2b_900ep_final.bin
echo "=== [continue3] method 2b done ==="

echo "=== [continue3] method 2c (600ep -> 900ep) ==="
cp sonic_policy_weights_method2c_600ep_final.bin sonic_policy_weights.bin
python3 -I sonic_rl_train.py $N_TRAIN $N_BASELINE 4000 0 1 mid > method2c_900ep.log 2>&1
cp sonic_policy_weights.bin sonic_policy_weights_method2c_900ep_final.bin
echo "=== [continue3] method 2c done ==="

cp sonic_policy_weights_method2a_1200ep_final.bin sonic_policy_weights.bin
echo "=== [continue3] ALL THREE METHODS COMPLETE (2a->1200ep, 2b->900ep, 2c->900ep) - WEIGHTS_FILE left on 2a's latest ==="
