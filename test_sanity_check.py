import sys
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from ingest_demo import check_ends_at_cutscene_trigger

info = [{"x": 5000+i, "y": 1050} for i in range(20)] + [{"x": 11211, "y": 800} for _ in range(5)]
passed, first_entry, detail = check_ends_at_cutscene_trigger(info, lookback=30)
print("passed:", passed, "first_entry:", first_entry)
print("detail:", detail)
assert passed and first_entry == 20

info2 = [{"x": 11211, "y": 800}] + [{"x": 5000, "y": 1050} for _ in range(100)]
passed2, first_entry2, detail2 = check_ends_at_cutscene_trigger(info2, lookback=30)
print("\npassed2:", passed2, "first_entry2:", first_entry2)
print("detail2:", detail2)
assert first_entry2 == 0 and passed2 == False
print("\nBoth positive-case tests PASSED")
