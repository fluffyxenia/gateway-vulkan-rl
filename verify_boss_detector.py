"""One-off verification (not standing infra): confirms BossHitDetector correctly
identifies a genuine monotonic-countdown-to-0 address while disqualifying a
warp-to-6-then-increases address (animation timer), using synthetic RAM arrays -
before trusting it against real gameplay."""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from boss_hit_counter import BossHitDetector, structural_offset

det = BossHitDetector()
ram = np.zeros(65536, dtype=np.uint8)
det.start(ram)  # nothing is 6 yet

# frame 1: two addresses warp to 6 (real hit counter candidate + a decoy)
ram = ram.copy()
ram[5000] = 6   # real hit counter
ram[6000] = 6   # decoy - will increase later (animation timer)
result = det.step(ram)
print(f"frame 1: result={result}, candidates={det.candidates}")
assert result is None

# frames 2-4: real counter decreases 6->5->4, decoy increases to 7 (disqualified)
for val, decoy_val in [(5, 7), (4, 2), (3, 5)]:
    ram = ram.copy()
    ram[5000] = val
    ram[6000] = decoy_val
    result = det.step(ram)
print(f"after frames 2-4: candidates={det.candidates}, disqualified={det.disqualified}")
assert 6000 in det.disqualified, "FAIL: decoy should have been disqualified (it increased)"
assert 5000 in det.candidates, "FAIL: real counter should still be a candidate"

# frames 5-7: real counter continues down to 0
for val in [2, 1, 0]:
    ram = ram.copy()
    ram[5000] = val
    result = det.step(ram)
print(f"final result: {result}")
assert result == 5000, f"FAIL: expected confirmed address 5000, got {result}"
print(f"structural_offset(5000) = {structural_offset(5000)}")
print("PASS: BossHitDetector correctly confirmed the real counter and disqualified the decoy")
