"""Bulk statistical data-gathering (one-off, not standing infra) on the AIZ1
boss-hit-counter mechanism, per the coordinator's request:
1. Warp-to-6 detection + mod-0x4A structural-offset check across many episodes.
2. Respawn-on-boss-restart viability: does a death near the boss lead to an
   in-episode retry, or does the episode just end?
3. (Production side handled separately in sonic_rl_train.py: boss-defeated is now
   a clean episode-end condition there.)

Uses a scripted heuristic (demo replay through the validated Act1 route, a wait
through the Act1->Act2 transition cutscene, then "hold right + jump, suppressing
jumps through the loop zone" for the new continuation) rather than the current
trained RL policy - checked first and confirmed the trained policy (as of this
check, 196/300 episodes into the method 2a run) has a max of 26/67 checkpoints and
has NEVER reached the boss trigger, so it would yield near-zero usable episodes for
this analysis. The scripted heuristic reaches much further (median ~3800-3900 out
of the 4465 needed) - reported honestly below, including whenever it does NOT reach
the actual trigger, since that itself is part of the real result.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
import stable_retro as retro
from checkpoints import parse_coords_file, CheckpointTracker
from boss_hit_counter import BossHitDetector, structural_offset, BOSS_HIT_COUNTER_ADDR

N_EPISODES = 30
DEMO = "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz"
d = np.load(DEMO, allow_pickle=True)
demo_actions = d["actions"]
boxes = parse_coords_file("/home/foxo/gateway-vulkan-rl/coords.list")

env = retro.make(game="SonicAndKnuckles3-Genesis-v0", render_mode=None)
buttons = env.buttons
right_idx = buttons.index("RIGHT")
b_idx = buttons.index("B")
noop = np.zeros(len(buttons), dtype=np.int8)

results = []  # one dict per episode

for ep in range(N_EPISODES):
    rng = np.random.default_rng(1000 + ep)
    tracker = CheckpointTracker(boxes, extend_past_end=True)
    detector = BossHitDetector()
    obs, info = env.reset()
    tracker.reset()
    t_total = 0
    for i in range(demo_actions.shape[0]):
        obs, r, terminated, truncated, info = env.step(demo_actions[i].astype(np.int8))
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        rr, cp_term, event = tracker.step(x, y, score, t_total)
        t_total += 1
    for i in range(650):
        obs, r, terminated, truncated, info = env.step(noop)
        t_total += 1

    action = np.zeros(len(buttons), dtype=np.int8)
    last_x, stuck_count, max_x, jump_hold = None, 0, 0, 0
    boss_reached = False
    boss_fight_started = False
    prev_lives = info.get("lives")
    death_events = []  # (step, continued_after: bool)
    warped_candidates_seen = set()
    confirmed_addr = None
    outcome = "timeout"

    for t in range(6000):
        x_now = last_x if last_x is not None else 0
        in_loop_zone = 3000 <= x_now <= 3450
        action[:] = 0
        action[right_idx] = 1
        if not in_loop_zone:
            if jump_hold > 0:
                action[b_idx] = 1
                jump_hold -= 1
            elif stuck_count > 6 or rng.random() < 0.08:
                jump_hold = int(rng.integers(3, 12))
        obs, r, terminated, truncated, info = env.step(action)
        x = info.get("x", 0)
        score = info.get("score", 0)
        lives = info.get("lives")
        if last_x is not None:
            stuck_count = stuck_count + 1 if x <= last_x else 0
            max_x = max(max_x, x)
        last_x = x
        rr, cp_term, event = tracker.step(x, info.get("y", 0), score, t_total)
        t_total += 1

        died = prev_lives is not None and lives is not None and lives < prev_lives
        if died and boss_fight_started:
            death_events.append([t, None])  # fill in "continued" after a few more frames
        prev_lives = lives

        if event is not None and event.startswith("boss_trigger"):
            boss_reached = True
            boss_fight_started = True
            ram = env.unwrapped.get_ram()
            detector.start(ram)
            print(f"ep {ep}: boss trigger reached at step {t}")
        elif boss_fight_started:
            ram = env.unwrapped.get_ram()
            addr = detector.step(ram)
            warped_candidates_seen |= set(detector.candidates.keys()) | detector.disqualified
            if addr is not None and confirmed_addr is None:
                confirmed_addr = addr
                outcome = "boss_defeated"
                print(f"ep {ep}: CONFIRMED hit-counter address 0x{addr:04x} "
                      f"(structural_offset={structural_offset(addr)})")

        # resolve pending death-continuation checks (did the episode keep going
        # for a few more frames after a death near the boss, i.e. a real in-episode
        # respawn, vs. terminated/truncated firing)
        for de in death_events:
            if de[1] is None and t - de[0] >= 3:
                de[1] = not (terminated or truncated)

        if confirmed_addr is not None:
            break
        if cp_term or terminated or truncated:
            outcome = "died" if died else ("cp_terminated" if cp_term else "env_done")
            break

    for de in death_events:
        if de[1] is None:
            de[1] = False  # episode ended before we could confirm a continuation

    results.append({
        "episode": ep,
        "max_x": max_x,
        "final_x": info.get("x"),
        "boss_reached": boss_reached,
        "confirmed_addr": confirmed_addr,
        "deaths_near_boss": len(death_events),
        "deaths_continued": sum(1 for de in death_events if de[1]),
        "outcome": outcome,
    })
    print(f"ep {ep}: max_x={max_x} boss_reached={boss_reached} "
          f"confirmed_addr={confirmed_addr} deaths_near_boss={len(death_events)} "
          f"outcome={outcome}")

env.close()

print("\n=== AGGREGATE RESULTS ===")
n_reached = sum(1 for r in results if r["boss_reached"])
n_confirmed = sum(1 for r in results if r["confirmed_addr"] is not None)
print(f"Episodes reaching boss trigger: {n_reached}/{N_EPISODES}")
print(f"Episodes with a CONFIRMED hit-counter address: {n_confirmed}/{N_EPISODES}")
addrs = [r["confirmed_addr"] for r in results if r["confirmed_addr"] is not None]
print(f"Confirmed addresses: {[hex(a) for a in addrs]}")
offsets = [structural_offset(a) for a in addrs]
print(f"Structural offsets (addr-0x29) mod 0x4A: {offsets}")
print(f"All offsets identical: {len(set(offsets)) <= 1 if offsets else 'N/A (no confirmed addresses)'}")
total_deaths_near_boss = sum(r["deaths_near_boss"] for r in results)
total_deaths_continued = sum(r["deaths_continued"] for r in results)
print(f"Deaths near boss: {total_deaths_near_boss}, of which continued in-episode: {total_deaths_continued}")
print(f"Max x reached across all episodes: {max(r['max_x'] for r in results)} (boss trigger at 4465)")
