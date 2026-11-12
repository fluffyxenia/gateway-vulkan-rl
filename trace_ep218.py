"""Diagnostic-only, read-only replay of method 2a's episode 218 - does NOT touch
WEIGHTS_FILE or spawn sonic_policy_server, safe to run alongside a live training
run. Reproduces run_episode()'s exact stepping logic manually so every
checkpoint-tracker EVENT (not just the final outcome) can be printed, to
determine whether the "boss_trigger" event itself fired legitimately (real x
in the real post-cutscene coordinate space, real prerequisite chain satisfied)
or fired incorrectly - independent verification per Foxo's explicit ask not to
just take the coordinator's framing on faith.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from sonic_rl_train import GAME, COORDS_FILE, N_BUTTONS
from boss_hit_counter import BossHitDetector, check_boss_hit_counter_sanity

ACTION_LOG = "/home/foxo/gateway-vulkan-rl/action_logs/20261109_223601/train_ep218_outcome-boss_defeated.npy"
recorded_actions = np.load(ACTION_LOG)
print(f"Loaded recorded action sequence: {recorded_actions.shape}")

boxes = parse_coords_file(COORDS_FILE)
tracker = CheckpointTracker(boxes, extend_past_end=True)
env = retro.make(game=GAME, render_mode=None)
obs, info = env.reset()

boss_detector = BossHitDetector()
boss_fight_started = False
prev_lives = info.get("lives")

for t in range(len(recorded_actions)):
    # reconstruct the full 12-button vector from the learnable-button recording
    from sonic_rl_train import buttons_to_multibinary
    mb = buttons_to_multibinary(recorded_actions[t].astype(np.int8))
    obs, _r, terminated, truncated, info = env.step(mb)
    x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
    lives = info.get("lives")
    r, cp_terminated, event = tracker.step(x, y, score, t)
    if event:
        print(f"step {t}: EVENT={event!r} x={x} y={y} score={score} lives={lives} zone={info.get('zone')} act={info.get('act')}")
    if event and "boss_trigger" in event:
        ram = env.unwrapped.get_ram()
        sane = check_boss_hit_counter_sanity(ram)
        print(f"  -> boss_trigger fired at step {t}! sanity_check={sane}, starting BossHitDetector")
        boss_detector.start(ram)
        boss_fight_started = True
    elif boss_fight_started and boss_detector.confirmed_addr is None:
        ram = env.unwrapped.get_ram()
        confirmed = boss_detector.step(ram)
        if confirmed is not None:
            print(f"  -> BossHitDetector CONFIRMED at step {t}: addr=0x{confirmed:x}")
    if lives is not None and prev_lives is not None and lives < prev_lives:
        print(f"step {t}: DIED (lives {prev_lives}->{lives}) x={x} y={y}")
    prev_lives = lives
    if terminated or truncated:
        print(f"step {t}: env terminated/truncated")
        break

env.close()
print(f"\nFinal tracker state: n_reached={tracker.n_reached} visited={sorted(tracker.visited)}")
print(f"Box 36 (end/cutscene-trigger) visited: {36 in tracker.visited}")
print(f"Box 37 (Survived cutscene) visited: {37 in tracker.visited}")
print(f"Box 71 (boss_trigger) visited: {71 in tracker.visited}")
