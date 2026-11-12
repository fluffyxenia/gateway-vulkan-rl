"""
Step 4/5 real-RL driver: trains the tiny conv policy net (sonic_policy_train.cpp, run
via Vulkan+CPU-fallback ggml backend_sched) against stable-retro's
SonicAndKnuckles3-Genesis-v0, Angel Island Zone Act 1, using the user's hand-traced
checkpoint coordinate list (coords.list) for reward shaping - replacing the earlier
Delta-score/rings placeholder entirely.

Algorithm: vanilla REINFORCE (policy gradient) with a per-episode standardized
return-to-go advantage. Not PPO/A2C/anything fancier - this is intentionally the
simplest thing that can show a learning signal, consistent with how far this project has
scoped RL-algorithm work so far.

Usage: python3 -I sonic_rl_train.py [n_train_episodes] [n_baseline_episodes] [max_steps] [fresh_init] [extend_past_end] [start_mode]

start_mode (default/boss/mid, default "default"): Step 28, selects the three-way
boss-training comparison method. "default" = method 2a (full-level-replay, real
ROM, normal Act1 start, extend_past_end as given above). "boss" = method 2b
(decomp ROM, AngelIslandZone.Act1Boss save-state, extend_past_end forced True).
"mid" = method 2c (decomp ROM, AngelIslandZone.Act1Midpoint save-state,
extend_past_end forced True). See START_MODES above and CheckpointTracker.reset()'s
pre_visited parameter for why 2b/2c need to pre-seed the cutscene-trigger box as
already-visited (they start past it in-game, and it would otherwise never
naturally re-fire in that episode's x range, permanently locking the boss trigger).

fresh_init (0/1, default 0): if 1, deletes any existing WEIGHTS_FILE before training so
the run starts from the trainer's own random init, matching this script's original
unconditional-delete behavior. Default 0 PRESERVES an existing WEIGHTS_FILE (e.g. a
BC-pretrained checkpoint deliberately copied into place beforehand) and warm-starts from
it instead. This default was changed deliberately (see AGENTS.md "Step 13") - the old
unconditional `os.remove(WEIGHTS_FILE)` at the top of main() silently discarded any
BC-pretrained starting point every single time this script ran, which would have thrown
away a confirmed-clean, independently fit-checked 60-epoch BC-pretrain result the first
time this script was invoked after copying it into place. Pass fresh_init=1 explicitly
to get the old from-scratch behavior back.

extend_past_end (0/1, default 0): Step 22, method 2a (full-level-replay-to-boss). If 1,
the original end-trigger (x>=12210) becomes a one-time milestone reward instead of an
episode terminator, so the episode continues through the merged AIZ1-continuation boxes
toward the boss trigger (x>=4465) instead of stopping at the original Act1 finish. Needs
a correspondingly larger max_steps - see AGENTS.md "Step 22" for the real per-episode
timing this implies. Default 0 preserves the exact original Act1-only behavior.
"""
import json
import os
import struct
import sys
import time
import subprocess

import numpy as np
import stable_retro as retro

sys.path.insert(0, "/home/foxo/gateway-vulkan-rl")
from checkpoints import parse_coords_file, CheckpointTracker
from policy_net import PolicyNet, IW, IH
from boss_hit_counter import check_boss_hit_counter_sanity, BossHitDetector, structural_offset
# NOTE: `from bc_pretrain import build_combined_batch` is deliberately NOT imported
# here at module level - bc_pretrain imports ingest_demo, which itself imports GAME/
# COORDS_FILE/etc. FROM THIS FILE, creating a real circular import
# (sonic_rl_train -> bc_pretrain -> ingest_demo -> sonic_rl_train) that crashes with
# "cannot import name 'build_combined_batch' from partially initialized module" if
# done at the top. Imported instead inside main(), right where it's used - by the
# time main() runs, this deferred import only matters when this file is executed
# directly as __main__ (ingest_demo's nested re-import of "sonic_rl_train" as a
# plain module never reaches main() at all, since that import's __name__ is
# "sonic_rl_train" not "__main__", so the cycle never actually closes).

# Fixed, policy-independent reference batch for the standing health check (Step 15) -
# the SAME 5 real human demos used for BC-pretraining, loaded with the real (not
# diagnostic-override) coords.list end-trigger threshold.
HEALTH_CHECK_DEMOS = [
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-BottomRoute-000000.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-BottomRoute-000001.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000000.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopRoute-000001.npz",
    "/home/foxo/gateway-vulkan-rl/demos/SonicAndKnuckles3-Genesis-v0-AngelIslandZone.Act1-TopToBottomRoute-000000.npz",
]


def vmrss_mb() -> float:
    """Current resident set size of THIS process, in MiB, read straight from the
    kernel via /proc/self/status - used to get a real, externally-verifiable memory
    timeline (not an estimate) proving the frame-retention fix actually keeps memory
    flat across the run, instead of growing with episode count as it did before."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def sys_mem_available_mb() -> float:
    """Step 17: system-wide MemAvailable (MiB), read straight from /proc/meminfo -
    added after a real thrashing incident where THIS process's own VmRSS stayed
    flat/healthy the whole time while something else (a now-fixed einsum-driven
    memory spike in policy_net.py, triggered by the periodic health check) took the
    whole machine down. VmRSS alone was NOT sufficient to catch that in the log - this
    is the system-level complement, printed every episode now (not every 10) per the
    explicit ask after that incident, so a repeat of ANY large, sudden allocation
    (this one or a different one) shows up immediately rather than being inferred
    after the fact from a stalled log."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return -1.0

GAME = "SonicAndKnuckles3-Genesis-v0"
COORDS_FILE = "/home/foxo/gateway-vulkan-rl/coords.list"
# Step 28: three-way boss-training comparison (2a/2b/2c). 2a (full-level-replay) uses
# the real ROM/default start exactly as before. 2b/2c start from a save-state already
# past the Act1->Act2 cutscene, captured against the DECOMP ROM (confirmed
# RAM-compatible with the real ROM per the earlier lockstep test, but the save files
# themselves are ROM-build-specific) - these need a different `game`/`state`/`inttype`
# for retro.make(), selected via main()'s start_mode argument, not GAME/COORDS_FILE
# above (coords.list itself is unchanged - same x/y checkpoint definitions apply to
# both ROMs given the confirmed RAM compatibility).
DECOMP_GAME = "SonicKnuckles3Decomp-Genesis"
START_MODES = {
    "default": {"game": GAME, "state": None},                               # 2a
    "boss":    {"game": DECOMP_GAME, "state": "AngelIslandZone.Act1Boss"},   # 2b
    "mid":     {"game": DECOMP_GAME, "state": "AngelIslandZone.Act1Midpoint"},  # 2c
}
WEIGHTS_FILE = "/home/foxo/gateway-vulkan-rl/sonic_policy_weights.bin"
# Step 22: opportunistic save-state capture for boss-training method 2b - the FIRST
# time EVER (across however many runs of this script, not per-episode) that a real
# episode reaches the boss trigger, the emulator state at that moment is persisted
# here. Checked via simple file-existence (idempotent across repeated script
# invocations, not just within one run) rather than re-captured every time, since
# method 2b specifically wants ONE canonical, reusable save-state, not a different
# one each time.
BOSS_SAVESTATE_PATH = "/home/foxo/gateway-vulkan-rl/boss_pretrigger_savestate.state"
# Step 24/25: boss-defeated is now a clean episode-end condition, same tier as the
# existing end-trigger/hazard/death terminations - per explicit instruction, "going
# forward" not just for the one bulk-analysis run. Multi-act/Act2 continuation is
# explicitly deferred - defeating the boss just ends the episode cleanly, it does
# NOT try to continue into Act 2. Reward magnitude: no specific value was given, so
# this picks END_REWARD's own value (5.0) - "same tier" as the other terminal
# conditions, not a new arbitrary scale - flagged here as a judgment call, open to
# adjustment if the coordinator wants a different value once real boss-defeat data
# exists to calibrate against.
BOSS_DEFEATED_REWARD = 5.0
# Step 14: replaced the old per-episode sonic_policy_train subprocess spawn (one fresh
# process + full Vulkan device/shader re-init per episode) with ONE persistent server
# process that keeps the Vulkan device + weight tensors resident for the whole run,
# used for the TRAIN (REINFORCE gradient update) call every episode. Directly measured
# win: ~17s -> ~7s per TRAIN call (time_split_test.py), confirmed numerically
# equivalent to the old subprocess trainer first (verify_server_equivalence.py:
# max|loss diff|=4.2e-7, byte-identical final weights).
#
# Rollout action-selection DELIBERATELY STAYS on the old numpy CPU path
# (PolicyNet.forward/sample_buttons), NOT the server's INFER opcode, despite INFER
# being built, GPU-backed, and confirmed correct (max|prob diff|=2.2e-4 vs CPU, within
# this project's ~1e-4 backend-tolerance bar) and fast in isolation (0.827ms/call
# steady-state, bench_infer.py). Measured head-to-head on the SAME machine
# (rollout_compare_test.py, 3000 real steps, env.step()+policy each time): old
# numpy-CPU rollout = 2.817ms/step; new server-IPC rollout = 4.679ms/step - i.e.
# routing this tiny (288,658-param) model's rollout inference through a cross-process
# IPC round trip is ~66% SLOWER in real interleaved use than keeping it in-process,
# even though the isolated GPU compute itself is fast - the fixed per-call IPC/
# scheduling overhead dominates for a model this small and a call pattern this
# frequent (one call per env step). This matches - and now empirically confirms -
# the reasoning in the earlier "Persistent Vulkan inference server... DEFERRED" note:
# swapping a fast in-process call for cross-process IPC was never guaranteed to be a
# net win just because the far side is GPU-backed. See AGENTS.md "Step 14" for the
# full comparison. sonic_policy_server's INFER opcode is kept (verified, available,
# documented) for any future revisit - e.g. a much larger architecture where per-call
# compute time would actually dominate IPC overhead - just not wired into the live
# rollout path today.
SERVER_BIN = "/home/foxo/gateway-vulkan-rl/sonic_policy_server"
GAMMA = 0.95
LR = 0.003
NEPOCH_PER_UPDATE = 1
# Step 15: lowered from 0.02 after a real, reproducible entropy-collapse failure was
# found (not by this agent - the user watched the live agent on sc4 and it looked
# wrong). A controlled A/B on sc4 (600 real env steps, same code/frames, BC-only
# checkpoint vs the episode-327+ RL checkpoint) showed: BC-only had RIGHT at p=0.993,
# every other button correctly near 0, clean +1249 x-progress, std 0.11-0.33 across
# the buttons that should vary on real frames (matches the original BC fit-quality
# check in "Step 13"). The RL checkpoint had RIGHT NEVER exceeding p=0.5 (mean 0.47),
# five unrelated buttons stuck "on" ~100% of the time regardless of frame content,
# EVERY button's probability std across real, visually different frames in the
# 0.002-0.017 range (near-zero - the policy stopped responding to the image at all),
# and x-progress collapsed from +1249 to +317 wandering near spawn. `w_d2` (final
# layer) weight magnitude was essentially unchanged between the two checkpoints
# (mean|v| 0.0926 -> 0.0922), ruling out the earlier BC weight-decay fix as the cause.
#
# Mechanism: `entropy_coef` multiplies a dense, EVERY-SINGLE-STEP bonus that is
# maximized exactly when every button sits at p=0.5 regardless of the frame - i.e. the
# loss function literally rewards ignoring the image. The real reward signal
# (CHECKPOINT_REWARD=1.0) is sparse - most steps in a 3000-step episode get ~0 raw
# reward, so after standardizing returns to zero-mean/unit-std, the policy-gradient
# term's advantage is itself near-zero on most steps, meaning the entropy bonus's
# constant, advantage-independent pull is the ONLY consistently-nonzero gradient
# signal on the majority of steps. Accumulated over 300+ episodes, it won the
# tug-of-war and dragged a previously-sharp, correct BC-pretrained policy back toward
# input-independent noise.
#
# Why 0.02 wasn't caught before being used this long: it was chosen to fix the
# OPPOSITE failure mode - real run #1 collapsed to near-DETERMINISTIC (zero-entropy)
# behavior within ~24 episodes at max_steps=600 with no entropy bonus at all (see
# "First real run collapsed" above). 0.02 was validated by (a) an exact-match
# synthetic sanity check confirming the loss formula was wired correctly, and (b) one
# real 80-episode run at max_steps=600 (run #2) that explicitly did NOT collapse and
# showed a real, if modest, learning signal. That validation only ever exercised ONE
# direction of the failure (collapse-to-deterministic) and only at the small scale it
# was tested at. Rough cumulative-pressure accounting: run #2 had 1 gradient
# minibatch/episode (600 steps / SAFE_MINIBATCH=2000 -> 1 chunk) * 80 episodes = 80
# total entropy-weighted gradient steps with no collapse-to-uniform observed. The
# collapsed run used max_steps=3000 -> 2 minibatches/episode, and had reached episode
# 327+ (>= 654 cumulative entropy-weighted gradient steps, over 8x more than the
# validated-safe count) before anyone checked policy sharpness again. Nobody added a
# check for collapse-to-UNIFORM (the opposite direction) when scaling up episode
# length/count 5x and 6x respectively from the validated regime - that's the actual
# gap, not a one-off mistake in picking 0.02 itself at the time.
#
# FIRST attempt was 0.003 (~6.7x lower, inside the 4-10x-lower range suggested when
# this bug was reported) - picked above the naive linear extrapolation of the
# 80-step-safe budget (0.02 * 80/654 =~ 0.0024) to keep a safety margin against the
# ORIGINAL failure mode. The standing health check below caught a THIRD failure mode
# at that value, within the very first 40-episode smoke test it was meant to validate
# against: episodes 1-22 looked genuinely good (episode-20 health check: RIGHT
# mean=0.994, std=0.067 - correctly confident AND still context-sensitive), then at
# episode 23 the policy suddenly locked onto a sharp, WRONG, fully-deterministic
# combo (LEFT+DOWN+C held constant, confirmed via the episode-40 health check: LEFT
# mean=0.990, DOWN mean=0.913, C mean=1.000/std=0.0, RIGHT mean=0.021) and flatlined
# at reward=0.0 for the remaining 17 episodes straight with no recovery - a classic
# "confidently stuck in a bad local optimum, too little residual exploration pressure
# left to escape it" REINFORCE pathology, NOT the same "ignore the image entirely"
# collapse signature as before (that one had every button hovering near p=0.5; this
# one has several buttons pinned at 0 or 1 with near-zero variance, i.e. genuinely
# sharp, just wrong and un-escapable). This means 0.003 over-corrected: it fixed the
# "policy is too flat" problem but removed enough exploration pressure to expose a
# different, faster-onset problem (stuck by episode ~23, not degraded gradually by
# episode ~327).
#
# SECOND attempt was 0.01 (2x lower than the original 0.02, not 4-10x as first
# suggested) - needed real testing to find, not just reasoning: the originally-
# discussed 0.002-0.005 range was too aggressive once actually run. No NaN, no
# permanent flatline this time, BUT a third real failure was found anyway by
# upgrading the verification method: the self-play-frame health check looked clean at
# both episode 20 and 40, but re-checking against the FIXED real demo frames (the
# same ones used for BC-pretraining, independent of whatever this policy's own
# rollout happened to do) showed overall accuracy had fallen to 0.502 - WORSE than
# just always guessing the majority class (0.9644). B/DOWN/C were predicted at ~1.0
# on every frame regardless of the true label. Two different entropy_coef values
# each produced a different, real, independently-confirmed degradation within 25-40
# episodes - not a "find the right constant" problem.
#
# Step 16: REMOVED ENTIRELY, not just weakened further. Foxo's reasoning (via the
# coordinator): "Mashing buttons and holding them isn't a skill; pressing them in the
# right sequence to beat a level is." An entropy bonus structurally rewards the net
# for being UNPREDICTABLE - that was never actually a proxy for "explore useful
# behavior," it's a proxy for "be noisy," and BOTH real failures above (a sharp wrong
# deterministic lock, AND "hold everything" noise) are consistent with that mismatch
# rather than just unlucky constant values. `sonic_policy_train.cpp`/
# `sonic_policy_server.cpp` no longer compute or accept an entropy term at all - the
# TRAIN wire protocol dropped the `entropy_coef` field entirely (see
# `sonic_policy_server.cpp` "Step 16").
#
# Replaced by a BC-anchor loss instead: each TRAIN call now ALSO mixes in a small,
# fixed-weight BCE loss against a random subsample of the same 5 real demo frames
# already loaded for the health check below (see ANCHOR_WEIGHT/ANCHOR_BATCH_SIZE and
# train_on_episode()). This anchors the policy toward known-good, DEMONSTRATED
# behavior rather than toward generic randomness - a far more targeted way to say
# "don't forget what BC already taught you" than entropy ever was.
#
# Real, non-obvious lesson for whoever revisits this regardless of which mechanism is
# in use: validate any such "default" regularizer bidirectionally and at the ACTUAL
# scale it'll run at, not just once at a smaller validated scale, and not just against
# the specific failure mode it was introduced to fix - that exact asymmetry is what
# let the original entropy_coef=0.02 silently drift for 300+ episodes.
ANCHOR_BATCH_SIZE = 300  # frames per TRAIN call, re-sampled fresh each call from the
# full real-demo pool (health_frames, loaded once in main() - see its own comment).
# ~10% of a full 3000-step episode's own frame count - meaningful but not dominant by
# row count; re-sampling each call (rather than reusing one fixed subset) means the
# full demo pool gets covered in expectation over many episodes instead of overfitting
# to one slice of it.
ANCHOR_WEIGHT = 1.0  # per-sample coefficient for anchor rows, in the SAME units/scale
# as the real REINFORCE advantage (both multiply the same bce_action term - see
# sonic_policy_server.cpp). Real advantages are standardized to ~unit std and clipped
# to +/-3 (ADVANTAGE_CLIP below), so 1.0 sits at "a mildly-better-than-average episode"
# scale - deliberately not large enough to swamp a genuinely strong or weak episode's
# real signal, which can clip up to 3x this value. Validated empirically (not just by
# this reasoning) via a printed per-episode diagnostic comparing the anchor batch's
# mean BCE contribution against the episode batch's mean |advantage|-weighted BCE
# contribution - see train_on_episode().
# Step 15 (second real bug found in the SAME smoke test that validated the
# entropy_coef fix above): raised from 0.01 after the 40-episode smoke test went NaN
# at episode 12/13 - episode 12 scored reward=+16.0 (16/34 checkpoints, the best
# result in the whole run) and the very next TRAIN call on that batch produced
# non-finite loss, with every weight tensor fully NaN-contaminated from that point on
# (confirmed via the [nan-diag] dump already wired into sonic_policy_server.cpp).
# This is the EXACT SAME mechanism as the BC-pretraining NaN saga ("Step 12/13"):
# unbounded weight/logit growth eventually overflows a float32 logit to inf, and
# `inf * 0` (the button-mask multiply, for any button whose target is 0 that step) is
# NaN by IEEE754 definition - wd=0.01 was already shown insufficient to prevent this
# in BC-pretraining (only delayed it, epoch 20->38), and wd=0.1 was needed there for a
# fully clean run. RL's WEIGHT_DECAY was never revisited after that BC-pretraining
# fix - it was left at the original, never-actually-validated-for-RL 0.01.
# Lowering entropy_coef (the fix directly above) made this MUCH worse, not
# unrelated: the old entropy_coef=0.02 was simultaneously (a) causing the entropy-
# collapse correctness bug AND (b) incidentally providing numerical stability as a
# side effect, by constantly pulling every probability back toward 0.5/bounding logit
# magnitude. Removing most of that pull to fix (a) removed the protective side effect
# in (b) too, exposing the same underlying weight-decay-insufficiency this project
# already found and fixed once in BC-pretraining, just never carried over to the RL
# path. Raised to 0.1 - the SAME value validated clean for 60 full epochs with zero
# NaN in BC-pretraining, not a new guess. (AdamW's per-step decay is `lr * wd`, but
# Adam's own gradient normalization means the "wd" value itself, not lr*wd, is what
# mainly governs whether decay keeps pace with weight growth across many steps
# regardless of lr - so directly reusing the validated wd=0.1 is better-justified than
# rescaling it by RL's lower lr=0.003 vs BC's lr=0.01.) Re-verify after this change
# the same way every prior weight-decay change in this project was verified: run it
# for real and check the log for NaN, don't just trust the reasoning.
WEIGHT_DECAY = 0.1
ADVANTAGE_CLIP = 3.0
DEATH_PENALTY = -2.0  # same scale as CheckpointTracker.HAZARD_PENALTY
# Step 23: ring/score reward shaping. Checked first that nothing similar already
# existed - `score` was only ever used internally by CheckpointTracker's enemy-kill
# detection (did score increase within a window after entering an enemy box), never
# as its own continuous reward term; `rings` was never read anywhere. This is a new
# addition, not a rename/duplicate of something existing.
#
# Both are DELTA-based (reward the CHANGE this step), not based on the absolute
# current value - an absolute-value ring/score bonus would pay out every single step
# just for continuing to hold whatever was already collected, which isn't "ring
# collection" as a skill, it's "don't lose rings", a completely different (and
# already-covered, see below) signal.
RING_REWARD_RATE = 0.01    # per ring gained this step
SCORE_REWARD_RATE = 0.001  # per 100 points gained this step
# Damage/ring-loss clawback: `ring_delta = rings - prev_rings` is NEGATIVE when rings
# are lost (not gained), so `RING_REWARD_RATE * ring_delta` is already the correctly-
# signed clawback with the SAME formula/rate used for the reward itself - no separate
# constant needed. Verified the math directly before trusting it: collecting N rings
# one at a time earns `RING_REWARD_RATE * N` total; losing all N rings in one later,
# non-fatal hit then applies `RING_REWARD_RATE * (-N)` - nets to exactly 0 (confirmed
# via a standalone calculation, float noise ~2.8e-17 only) - the agent cannot profit
# from a collect-then-lose cycle, exactly as asked.
#
# Gated on `not died` deliberately: a DEATH also resets rings to 0 (the game's own
# respawn behavior), but that's an ALREADY-PENALIZED consequence via DEATH_PENALTY -
# clawing back ring-reward AGAIN on top of DEATH_PENALTY for the exact same incident
# would double-penalize it. The clawback is specifically for a real mid-episode "hit
# and survive" (rings scatter, no life lost), which is the literal mechanic Foxo
# described ("subtracts the accumulated ring-bonus ... for rings lost on that hit").
# Score never decreases in this game (no symmetric clawback needed/possible there).
# Step 18: explicit speed optimization, requested after the 500-episode BC-anchor run
# produced 6 real clears (2216-2957 steps, i.e. ~36.9-49.3s at 60fps) with no
# explicit time pressure - the only existing speed signal was implicit, via gamma
# discounting of the END_REWARD when computing discounted_returns() backward from
# episode end. That signal turns out to be far weaker in practice than it looks: the
# backward recursion G_t = r_t + gamma*G_{t+1} does discount an end-trigger reward's
# credit to EARLIER actions by gamma^(T-t), but across these episode lengths (2200-
# 3000 steps) that discount factor underflows to effectively 0 within a few hundred
# steps of the reward regardless of T - so a clear at step 2216 vs step 2957 gets
# almost identically-shaped backward credit, not a meaningfully different one. Three
# candidate mechanisms were considered:
#   (a) a small constant per-step penalty - rejected: creates a genuine reward-
#       hacking risk. A per-step cost accrues for the FULL episode length regardless
#       of outcome, so "die early to stop paying the per-step cost" becomes
#       attractive unless the penalty is kept so small (well under
#       DEATH_PENALTY/max_steps ~= 0.00067 per step) that it's probably too weak a
#       signal to matter anyway - a bad trade-off between safety and strength.
#   (b) scaling gamma down - rejected: too blunt an instrument. It would discount
#       EVERY future reward (checkpoints, enemy kills, not just the end-trigger)
#       faster, risking destabilizing the already-working checkpoint-collection
#       incentive structure for a signal that's supposed to specifically target
#       end-to-end speed, not general short-termism.
#   (c) a one-time bonus at the end-trigger itself, scaled by how much of the step
#       budget was NOT used - chosen. Zero cost/effect on any episode that doesn't
#       reach the end (can't incentivize dying early, can't touch non-terminal
#       reward shaping at all), continuous (no hard threshold to discover by chance -
#       every speed improvement helps, not just crossing some specific target), and
#       directly targets exactly the metric asked for (time-to-end-trigger).
# SPEED_BONUS_SCALE=10.0 chosen to be comparable to END_REWARD=5.0 itself at
# meaningfully fast clears (e.g. a 1500-step clear, half of max_steps=3000, earns
# 10.0*(1-1500/3000)=5.0 - as much bonus as the whole base end reward) while staying
# well short of the total achievable checkpoint reward (up to 34) - real, motivating,
# not dominant. Verify via a real smoke test: measure time-to-end-trigger with this
# active vs. the 6 real clears already on record, and check n_checkpoints at each
# clear doesn't drop (which would indicate skipped checkpoints / corner-cutting)
# before trusting this.
SPEED_BONUS_SCALE = 10.0
# Mini-batch size cap for sonic_policy_train's internal graph (added after the v3
# two-conv architecture's first real BC-pretraining run crashed Vulkan with
# "Not enough memory for command submission" / "device lost" at ~4049 samples in one
# graph - empirically bisected: 3000 works, 4049 crashes. 2000 gives a comfortable
# margin under that ceiling. Episodes shorter than this just get one minibatch covering
# the whole episode (see train_on_episode()); a full 7200-step episode gets split into
# multiple minibatches/gradient steps per update instead of one giant one.
SAFE_MINIBATCH = 2000
ACTION_LOG_ROOT = "/home/foxo/gateway-vulkan-rl/action_logs"

# Step 15: standing lightweight entropy-collapse health check, added after a real
# collapse went undetected for 300+ episodes (caught only by a human watching the
# agent play live, not by anything in this file).
#
# FIRST version of this check reused the current episode's own just-played frames -
# cheap (no extra env steps), but turned out to be an unreliable blind spot: a real
# smoke test (entropy_coef=0.01) that LOOKED healthy on this self-play-frame check at
# episode 20 (RIGHT mean=0.994, confidently correct) had, by episode 40, actually
# degraded into pinning B/DOWN/C to ~1.0 regardless of the real frame content - but
# the self-play-frame check at episode 40 didn't clearly flag it, because the
# policy's OWN rollout frames are survivorship-biased (correlated with whatever the
# current, possibly-already-degrading policy chooses to do), not an independent test
# of real game states. Only checking against the SAME 5 real human demo frames used
# for BC-pretraining (independent of whatever the current policy does) exposed the
# real problem clearly: overall per-button accuracy on those frames had fallen to
# 0.502 - WORSE than just always guessing the majority class (0.964) - with B/DOWN/C
# predicted at ~1.0 on every single spot-checked frame regardless of the true label.
#
# Fixed version below checks against this FIXED, policy-independent reference batch
# (loaded once, reused every check) INSTEAD of the current episode's own frames.
# STD_FLOOR is not a guess: the known-good BC-pretrained checkpoint showed per-button
# std 0.11-0.33 on these exact frames for the buttons that vary (RIGHT/C/LEFT/DOWN -
# see "Step 13" fit-quality check); both real collapses found this session (the
# original entropy_coef=0.02 one AND this one) showed std well below that on the
# buttons that should vary. ACCURACY_FLOOR is the more direct, robust signal found by
# this investigation: flag if accuracy on the real demo frames falls notably below
# the majority-class baseline computed from the SAME frames (0.9644) - a policy that
# can't even match "always guess the common class" on real, independent frames has
# clearly stopped using the image, regardless of what its own self-play std looks like.
POLICY_HEALTH_CHECK_EVERY = 20  # episodes
# Step 29: for the at-scale 2a/2b/2c comparison, Foxo's explicit ask is to track
# boss-defeat RATE over buckets of episodes (not just a final summary), so a
# long-running (300+ episode) job's progress can be read mid-run by tailing the
# log, and so training-closing-the-gap-with-random-luck can actually be seen
# over time the way checkpoint-progress trends were tracked for 2a's original
# run. Also used as the cadence for a periodic NaN/Inf weights-file check during
# a long run (not just once at the very end) - "verify weights stay clean" per
# Foxo's standing rigor requirement, checked repeatedly, not assumed.
BOSS_DEFEAT_RATE_BUCKET = 50  # episodes
POLICY_HEALTH_STD_FLOOR = 0.02
POLICY_HEALTH_ACCURACY_FLOOR = 0.90  # majority-baseline on these frames is 0.9644

# Full Genesis controller, in the order stable-retro/env.buttons reports it (confirmed
# empirically earlier this project). This is the order env.step() expects a MultiBinary
# vector in.
BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

# v2 action space: independent per-button sigmoid outputs (multi-label Bernoulli policy),
# NOT a categorical choice over a fixed menu of button-bundles. The earlier v1 design
# (DISCRETE_ACTIONS, a hand-picked list of 7 button-bundles incl. a NOOP) was
# structurally incapable of ever producing a combo that wasn't explicitly one of those 7
# - e.g. DOWN+C (spin dash, needed throughout AIZ and later zones) simply could not come
# out no matter how training went. Now every learnable button crosses its own threshold
# independently each step: no-op falls out naturally (nothing crosses), every one of the
# 2^N_BUTTONS possible subsets (including combos never explicitly enumerated) is
# directly reachable, with no architectural ceiling. See sonic_policy_train.cpp (v2) and
# policy_net.py for the matching training/inference implementations - this required real
# ggml graph and REINFORCE-loss changes (categorical softmax -> per-button sigmoid,
# single categorical log-prob -> sum of independent per-button binary log-probs), not
# just a config tweak.
#
# MODE/START are deliberately excluded from the learnable set entirely (not just given
# low prior probability) - there's no gameplay reason the policy should ever be allowed
# to press them, so they're not part of the output space at all, always 0 in the env
# action vector regardless of what the policy does.
LEARNABLE_BUTTONS = ['B', 'A', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
N_BUTTONS = len(LEARNABLE_BUTTONS)
_LEARNABLE_IDX_IN_FULL = [BUTTONS.index(b) for b in LEARNABLE_BUTTONS]


def buttons_to_multibinary(button_vec: np.ndarray) -> np.ndarray:
    """button_vec: (N_BUTTONS,) 0/1 array in LEARNABLE_BUTTONS order (what the policy
    outputs / what a random baseline samples). Returns the full (len(BUTTONS),) 0/1
    array env.step() expects, with MODE/START (and anything else not in
    LEARNABLE_BUTTONS) always 0."""
    v = np.zeros(len(BUTTONS), dtype=np.int8)
    v[_LEARNABLE_IDX_IN_FULL] = button_vec
    return v


def multibinary_to_buttons(full_vec: np.ndarray) -> np.ndarray:
    """Inverse of buttons_to_multibinary: given a (..., len(BUTTONS)) array in the full
    Genesis-controller order (e.g. straight from a stable-retro recording's "actions"
    array, which stores one column per env.buttons entry - confirmed to match BUTTONS
    exactly, no reordering needed), slice out just the LEARNABLE_BUTTONS columns in
    LEARNABLE_BUTTONS order - i.e. drop MODE/START. Used by the demo-recording
    ingestion pipeline (behavior cloning) to turn a human recording's raw button array
    into the same (T, N_BUTTONS) target format the policy/trainer use everywhere else."""
    return np.asarray(full_vec)[..., _LEARNABLE_IDX_IN_FULL]


def downsample_gray(frame: np.ndarray, iw: int, ih: int) -> np.ndarray:
    h, w, _ = frame.shape
    ys = np.linspace(0, h - 1, ih).astype(np.int64)
    xs = np.linspace(0, w - 1, iw).astype(np.int64)
    sub = frame[ys][:, xs]
    return sub.astype(np.float32).mean(axis=2) / 255.0


def save_action_log(actions: np.ndarray, log_dir: str, tag: str):
    """Standing infrastructure (not a one-off): save the exact per-step button vectors
    for one episode to a tiny file, so that episode can be deterministically replayed
    later - with actual rendering on a non-headless machine - by feeding the same button
    presses to a fresh env in the same order, without needing to watch runs visually
    during headless training on Gateway. `actions` is (T, N_BUTTONS) int8, 0/1 per
    learnable button per step - tiny (~70KB for a full 7200-step episode at 10 buttons)
    compared to the frame data this project already knows not to retain long-term (see
    the memory-leak fix above) - cheap enough to log unconditionally for every episode,
    not just flagged/anomalous ones, since you don't know an episode was interesting
    until after the fact (exactly the situation with episode 8's 7367px jump this
    logging capability was requested in response to). The action sequence alone is
    sufficient to replay exactly: stable-retro/the underlying emulator is deterministic
    given a fixed start state and a fixed input sequence, so no policy weights or RNG
    state need to be saved alongside it."""
    os.makedirs(log_dir, exist_ok=True)
    np.save(os.path.join(log_dir, f"{tag}.npy"), actions.astype(np.int8))


def write_run_metadata(log_dir: str, extra: dict):
    """One small sidecar JSON per run (not per episode) recording everything a replay
    script needs to reconstruct the action->button mapping and know which game/weights
    this run used - the action logs themselves are just raw 0/1 button vectors,
    meaningless without this. `buttons` = the full Genesis controller order env.step()
    expects; `learnable_buttons` = which of those the policy actually controls and in
    what column order the saved (T, N_BUTTONS) arrays are in (MODE/START are never in
    this list and are always 0 in the full vector)."""
    os.makedirs(log_dir, exist_ok=True)
    meta = {
        "game": GAME,
        "buttons": BUTTONS,
        "learnable_buttons": LEARNABLE_BUTTONS,
        "weights_file": WEIGHTS_FILE,
        **extra,
    }
    with open(os.path.join(log_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def check_policy_health(policy, frames: np.ndarray, true_actions: np.ndarray = None):
    """Step 15: entropy-collapse/policy-degradation diagnostic. `frames` should be a
    FIXED, policy-independent reference batch (the real human demo frames, loaded
    once - see POLICY_HEALTH_CHECK_EVERY's comment above for why checking against the
    CURRENT episode's own self-play frames turned out to be an unreliable blind spot).
    Returns (per-button mean, per-button std, overall_accuracy_or_None). accuracy is
    only computed if `true_actions` (the demo frames' real recorded actions) is given.
    See AGENTS.md "Step 15" for the full incident this exists to catch earlier next
    time: entropy_coef silently collapsed a correct, sharp BC-pretrained policy into
    image-independent (or wrong-but-confident) behavior over 300+ RL episodes in one
    case and just ~25-40 episodes in two follow-up attempts, undetected by the
    self-play-frame version of this same check."""
    probs = policy.forward(frames)  # (T, n_buttons)
    accuracy = None
    if true_actions is not None:
        pred = (probs > 0.5).astype(np.int8)
        accuracy = (pred == true_actions).mean()
    return probs.mean(axis=0), probs.std(axis=0), accuracy


class PolicyServerClient:
    """Thin client for the persistent sonic_policy_server process (see
    sonic_policy_server.cpp for the full protocol/design rationale). Replaces BOTH the
    old per-step numpy CPU forward pass (rollout action-selection) and the old
    per-episode sonic_policy_train subprocess spawn (gradient update) with calls into
    one long-lived process that keeps the Vulkan device + weight tensors resident for
    the whole run - no process restart, no device/shader re-init, per episode.

    Verified numerically equivalent to both of the designs it replaces BEFORE being
    wired into real training (see verify_server_equivalence.py, run once as a
    standalone check): TRAIN matched the old per-process subprocess trainer to
    max|loss diff|=4.2e-7 across 3 epochs on an identical batch+starting weights, with
    byte-identical final weight files; INFER matched the CPU-only forward_check tool to
    max|prob diff|=2.2e-4, consistent with this project's established ~1e-4
    CPU-vs-Vulkan backend tolerance bar (different backends sum floats in a different
    order - expected noise, not a logic difference)."""

    def __init__(self, server_bin: str, weights_file: str, iw: int, ih: int, n_buttons: int):
        self.n_buttons = n_buttons
        self.proc = subprocess.Popen(
            [server_bin, weights_file, str(iw), str(ih), str(n_buttons)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        )

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("sonic_policy_server closed its stdout unexpectedly "
                                    "(check its stderr log for a crash)")
            buf += chunk
        return buf

    def infer(self, frame: np.ndarray) -> np.ndarray:
        self.proc.stdin.write(b"I" + frame.astype(np.float32).tobytes())
        self.proc.stdin.flush()
        return np.frombuffer(self._read_exact(4 * self.n_buttons), dtype=np.float32).copy()

    def train(self, frames: np.ndarray, mask: np.ndarray, coefficients: np.ndarray,
               minibatch_size: int, nepoch: int, lr: float,
               weight_decay: float) -> np.ndarray:
        """`coefficients` is the per-sample multiplier on bce_action (Step 16: no
        more entropy_coef) - the real standardized REINFORCE advantage for genuine
        episode rows, or the fixed ANCHOR_WEIGHT for concatenated BC-anchor rows. The
        server treats every row identically; the distinction is made entirely by the
        caller when constructing this array (see train_on_episode())."""
        T = frames.shape[0]
        req = (b"T" + struct.pack("<qqq", T, minibatch_size, nepoch)
               + struct.pack("<ff", lr, weight_decay)
               + frames.astype(np.float32).tobytes()
               + mask.astype(np.float32).tobytes()
               + coefficients.astype(np.float32).tobytes())
        self.proc.stdin.write(req)
        self.proc.stdin.flush()
        status = struct.unpack("<i", self._read_exact(4))[0]
        losses = np.frombuffer(self._read_exact(4 * nepoch), dtype=np.float32).copy()
        if status != 0:
            print(f"WARNING: TRAIN call reported a non-finite loss at least once this "
                  f"episode (status={status}) - see server stderr in this same log for "
                  f"the [nan-diag] parameter dump", file=sys.stderr)
        return losses

    def quit(self):
        """Flush weights to disk and let the server exit cleanly. Safe to call more
        than once / on an already-exited process."""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(b"Q")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_episode(env, tracker, policy=None, rng=None, max_steps=1000, pre_visited=None):
    """policy=None -> uniform random action selection (baseline run, no learning).
    policy=a PolicyNet -> each step's action comes from the numpy CPU forward pass
    (see the Step 14 comment above SERVER_BIN for why this deliberately stays on numpy
    CPU rather than the persistent server's GPU-backed INFER opcode - measured slower
    in practice for this tiny model/call pattern, despite being correct and fast in
    isolation).
    Returns dict with frames, actions, rewards (for training runs) plus stats.

    Tracks info['lives'] every step and ends the episode the instant it decreases
    (death/respawn). Before this fix, nothing watched lives at all - an episode could
    only end via the end-trigger box, a hazard box, or the step timeout, so a death
    (or, the user's specific worry, any attract-mode demo input that might fire on a
    title/game-over screen) would just keep being stepped and recorded as if it were
    genuine policy behavior, with the respawn's x/y/score silently folded into the same
    trajectory. This closes that gap unconditionally, regardless of whether that
    specific contamination scenario was actually occurring in past runs.
    NOTE: rl_run1 through rl_run4 and the diagnostic batch all predate this fix -
    lives was never logged in any of them, so there is no way to retroactively check
    whether death/demo-mode contamination affected those already-reported numbers.
    We genuinely don't know, and can't find out after the fact."""
    obs, info = env.reset()
    tracker.reset(pre_visited=pre_visited)
    frames, actions, rewards = [], [], []
    total_reward = 0.0
    outcome = "timeout"
    prev_lives = info.get("lives")
    prev_rings = info.get("rings")
    prev_score = info.get("score")
    ring_reward_total = 0.0   # Step 23 diagnostic - printed per episode to directly
    score_reward_total = 0.0  # verify these are earning/clawing back nonzero amounts
    boss_detector = BossHitDetector()  # Step 24/25 - fresh per episode, started at
    boss_fight_started = False          # the boss_trigger event, stepped every frame after
    for t in range(max_steps):
        gray = downsample_gray(obs, IW, IH)
        if policy is None:
            # random baseline in the multi-label space: independent Bernoulli(0.5) per
            # learnable button - the natural "uniform random" analog now that actions
            # aren't a fixed categorical menu (matches the policy's own action
            # representation exactly, just with untrained/unbiased per-button coin flips).
            button_vec = rng.integers(0, 2, size=N_BUTTONS).astype(np.int8)
        else:
            button_vec = policy.sample_buttons(gray, rng)
        mb = buttons_to_multibinary(button_vec)
        obs, _env_reward, terminated, truncated, info = env.step(mb)
        x, y, score = info.get("x", 0), info.get("y", 0), info.get("score", 0)
        lives = info.get("lives")
        rings = info.get("rings")
        r, cp_terminated, event = tracker.step(x, y, score, t)
        died = prev_lives is not None and lives is not None and lives < prev_lives
        prev_lives = lives
        if died:
            r += DEATH_PENALTY
        # Step 23: ring/score reward shaping - see RING_REWARD_RATE's own comment
        # above for the full reasoning (delta-based, symmetric clawback via the same
        # signed formula, gated on `not died` to avoid double-penalizing a death that
        # already resets rings and already gets DEATH_PENALTY).
        if rings is not None and prev_rings is not None:
            ring_delta = rings - prev_rings
            if ring_delta > 0 or (ring_delta < 0 and not died):
                this_ring_reward = RING_REWARD_RATE * ring_delta
                r += this_ring_reward
                ring_reward_total += this_ring_reward
        prev_rings = rings
        if score is not None and prev_score is not None:
            score_delta = score - prev_score
            if score_delta > 0:
                this_score_reward = SCORE_REWARD_RATE * (score_delta / 100.0)
                r += this_score_reward
                score_reward_total += this_score_reward
        prev_score = score
        if event == "level_end_trigger":
            # Step 18: one-time speed bonus, added directly to this step's reward
            # (the same step the base END_REWARD was already added to by
            # tracker.step()) - continuous in how much step budget remains unused,
            # zero effect on any episode that doesn't reach here. `t` is 0-indexed
            # (t=0 on the first step), so (max_steps - (t+1)) is the actual number of
            # steps NOT used out of the budget.
            speed_bonus = SPEED_BONUS_SCALE * max(0.0, (max_steps - (t + 1)) / max_steps)
            r += speed_bonus
        if event is not None and event.startswith("boss_trigger") and info.get("act") != 1:
            # Step 29 follow-up - real bug caught by Foxo replaying episode 218's
            # actual frames (method 2a, structural_offset=47): the checkpoint
            # tracker's own "interactive-cutscene trigger"/"Survived cutscene"
            # threshold chain (boxes 36/37, both bare "x >= N" checks with no
            # coordinate-epoch memory - see that same caveat already documented in
            # coords.list) can apparently be satisfied by x alone WITHOUT the real
            # scripted Act1->Act2 transition ever actually firing: x can cross
            # 12210 while still fully inside the original pre-warp act (confirmed
            # directly by replaying ep218 and tracing every tracker event - act
            # stayed 0 throughout, x never warped to the small post-cutscene
            # range), which then trivially satisfies box37/71's own low
            # thresholds too, firing "boss_trigger" in completely ordinary
            # mid-level terrain. BossHitDetector then starts scanning real RAM in
            # a nonsense context and - over the remaining THOUSANDS of ordinary
            # gameplay frames in a long (max_steps=7200) 2a episode - finds some
            # unrelated byte that coincidentally matches its warp-then-decrease
            # signature, producing a bogus "boss_defeated" outcome. Confirmed:
            # info['act'] cleanly tracks the REAL transition (0->1, see the
            # Step-27 finding this reuses) and 2b/2c's save-states both already
            # have act=1 from frame 0 (checked directly), so gating on act==1
            # here is a pure tightening - it can never block a legitimate 2b/2c
            # boss_trigger, only reject a bogus premature one under 2a.
            print(f"[boss] REJECTED premature boss_trigger at step {t}: event fired but "
                  f"info['act']={info.get('act')} (expected 1) - the real Act1->Act2 "
                  f"transition hasn't actually happened yet, this is almost certainly "
                  f"the ep218-style tracker-threshold false-fire, not a real boss "
                  f"encounter. NOT starting BossHitDetector.")
        elif event is not None and event.startswith("boss_trigger"):
            # Step 22: sanity-check the hit-counter address at the exact moment the
            # boss fight begins, EVERY time this fires (not just once) - per Foxo's
            # explicit instruction, since this address is only verified correct
            # under one specific spawn-context and could silently be tracking the
            # wrong dynamic-object slot under method 2a's varying routes. Warns via
            # boss_hit_counter.py's own stderr message on failure - not silently
            # trusted either way.
            ram = env.unwrapped.get_ram()
            boss_sanity_ok = check_boss_hit_counter_sanity(ram)
            print(f"[boss] trigger reached at step {t}, hit-counter sanity check: "
                  f"{'OK' if boss_sanity_ok else 'FAILED (see warning above)'}")
            # Opportunistic one-time save-state capture for method 2b - see
            # BOSS_SAVESTATE_PATH's own comment above for why this is file-existence
            # gated rather than re-captured every time.
            if not os.path.exists(BOSS_SAVESTATE_PATH):
                snapshot = env.unwrapped.em.get_state()
                with open(BOSS_SAVESTATE_PATH, "wb") as f_state:
                    f_state.write(snapshot)
                print(f"[boss] Captured save-state at boss trigger -> {BOSS_SAVESTATE_PATH} "
                      f"({len(snapshot)} bytes) - first time this has happened this run.")
            # Step 24/25: start the generalized per-episode detector right here too -
            # does NOT rely on the fixed 0xb278 address (confirmed unreliable under
            # method 2a), scans for the real warp-to-6 signature instead.
            boss_detector.start(ram)
            boss_fight_started = True
        elif boss_fight_started:
            # Keep stepping the detector every frame after the boss fight has
            # started, even on steps where nothing else notable happened this
            # frame (event is None or something unrelated) - the boss can be hit
            # on any frame, not just ones with a checkpoint-box event.
            ram = env.unwrapped.get_ram()
            confirmed_addr = boss_detector.step(ram)
            if confirmed_addr is not None:
                r += BOSS_DEFEATED_REWARD
                outcome = "boss_defeated"
                print(f"[boss] DEFEATED at step {t} - confirmed hit-counter address "
                      f"0x{confirmed_addr:04x} (structural_offset={structural_offset(confirmed_addr)})")
                frames.append(gray)
                actions.append(button_vec)
                rewards.append(r)
                total_reward += r
                break
        frames.append(gray)
        actions.append(button_vec)
        rewards.append(r)
        total_reward += r
        if event == "level_end_trigger":
            outcome = "reached_end"
        elif event is not None and event.startswith("hazard"):
            outcome = "hazard"
        if died:
            outcome = "died"
            break
        if cp_terminated or terminated or truncated:
            if cp_terminated and outcome == "timeout":
                outcome = "cp_terminated"
            elif terminated or truncated:
                outcome = outcome if outcome != "timeout" else "env_done"
            break
    return {
        "frames": np.array(frames, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int8),  # (T, N_BUTTONS) multi-hot, not an index
        "rewards": np.array(rewards, dtype=np.float32),
        "total_reward": total_reward,
        "n_checkpoints": tracker.n_reached,
        "length": len(rewards),
        "outcome": outcome,
        "ring_reward_total": ring_reward_total,    # Step 23 diagnostic
        "score_reward_total": score_reward_total,  # Step 23 diagnostic
    }


def _bce_action_numpy(probs: np.ndarray, actions: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Read-only numpy diagnostic version of the per-sample BCE the server computes
    from logits internally - clips probabilities away from exact 0/1 (acceptable for
    a diagnostic print, NOT used anywhere in the actual gradient/training path, which
    stays on the numerically-stable logits-based formulation in
    sonic_policy_server.cpp). Returns (T,) per-sample bce_action."""
    p = np.clip(probs, eps, 1 - eps)
    return -(actions * np.log(p) + (1 - actions) * np.log(1 - p)).sum(axis=1)


def train_on_episode(ep, server: PolicyServerClient, policy: PolicyNet,
                      anchor_frames: np.ndarray, anchor_actions: np.ndarray, rng: np.random.Generator):
    rewards = ep["rewards"]
    returns = discounted_returns(rewards, GAMMA)
    std = returns.std()
    advantages = (returns - returns.mean()) / (std + 1e-6) if std > 1e-8 else returns - returns.mean()
    advantages = np.clip(advantages, -ADVANTAGE_CLIP, ADVANTAGE_CLIP)

    T = len(rewards)
    # ep["actions"] is already the (T, N_BUTTONS) multi-hot mask the trainer needs - no
    # one-hot construction needed, unlike v1's categorical index -> one-hot conversion.
    mask = ep["actions"].astype(np.float32)

    # Step 16: BC-anchor rows, concatenated into the SAME batch/gradient step as the
    # real episode rows (not a separate sequential call) - a fresh random subsample of
    # the real demo pool every episode, re-sampled each time (not one fixed subset) so
    # the full pool gets covered in expectation over many episodes.
    anchor_idx = rng.choice(anchor_frames.shape[0], size=min(ANCHOR_BATCH_SIZE, anchor_frames.shape[0]), replace=False)
    a_frames = anchor_frames[anchor_idx]
    a_actions = anchor_actions[anchor_idx]
    a_coef = np.full(a_frames.shape[0], ANCHOR_WEIGHT, dtype=np.float32)

    combined_frames = np.concatenate([ep["frames"], a_frames], axis=0).astype(np.float32)
    combined_mask = np.concatenate([mask, a_actions], axis=0).astype(np.float32)
    combined_coef = np.concatenate([advantages, a_coef], axis=0).astype(np.float32)
    T_combined = combined_frames.shape[0]

    # Diagnostic (printed every episode, cheap - one extra small numpy forward pass):
    # confirm the anchor term isn't dominating/distorting the real REINFORCE signal,
    # per the explicit validation requirement this was added under. Uses the CURRENT
    # (pre-this-update) policy, the same read-only numpy forward pass already used for
    # rollout - not part of the actual training graph.
    ep_probs = policy.forward(ep["frames"])
    ep_bce = _bce_action_numpy(ep_probs, mask)
    episode_contribution = float(np.mean(ep_bce * advantages))
    anchor_probs = policy.forward(a_frames)
    anchor_bce = _bce_action_numpy(anchor_probs, a_actions)
    anchor_contribution = float(np.mean(anchor_bce) * ANCHOR_WEIGHT)
    print(f"  [anchor-check] episode mean(bce*advantage)={episode_contribution:+.4f}  "
          f"anchor mean(bce)*weight={anchor_contribution:+.4f}  "
          f"(anchor rows={a_frames.shape[0]}/{T_combined}={a_frames.shape[0]/T_combined:.1%} of combined batch)")

    # Same minibatch-sizing logic as before, now applied to the COMBINED total - see
    # AGENTS.md "Step 12" for the full reasoning (ceil n_chunks, floor minibatch_size,
    # to avoid dropping a full extra chunk via the server's own floor-division).
    n_chunks = max(1, -(-T_combined // SAFE_MINIBATCH))
    minibatch_size = T_combined // n_chunks

    # Sent straight over the server's stdin pipe - no temp batch file needed any more
    # (that was only there to hand data to a fresh subprocess's argv/open() call).
    return server.train(combined_frames, combined_mask, combined_coef,
                         minibatch_size, NEPOCH_PER_UPDATE, LR, WEIGHT_DECAY)


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_baseline = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    fresh_init = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    # Step 22, method 2a (full-level-replay-to-boss): extend_past_end=1 makes the
    # "end"-kind box (x>=12210, the original cutscene trigger) a milestone instead of
    # a terminator, so the episode continues through the merged AIZ1-continuation
    # boxes toward the boss trigger instead of stopping at the original Act1 finish.
    # Default 0 preserves the exact original behavior (every existing non-boss run,
    # and all BC-pretraining tooling, is unaffected).
    extend_past_end = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    # Step 28: start_mode selects the three-way comparison's method - "default" (2a,
    # real ROM, normal Act1 start), "boss" (2b, decomp ROM, AngelIslandZone.Act1Boss
    # save-state), "mid" (2c, decomp ROM, AngelIslandZone.Act1Midpoint save-state).
    start_mode = sys.argv[6] if len(sys.argv) > 6 else "default"
    if start_mode not in START_MODES:
        raise ValueError(f"start_mode must be one of {list(START_MODES)}, got {start_mode!r}")
    mode_cfg = START_MODES[start_mode]
    if start_mode != "default":
        extend_past_end = True  # 2b/2c always need this - they start past the old end-trigger entirely

    boxes = parse_coords_file(COORDS_FILE)
    tracker = CheckpointTracker(boxes, extend_past_end=extend_past_end)
    print(f"Loaded {len(boxes)} checkpoint boxes "
          f"({tracker.n_checkpoints} checkpoint, {tracker.n_enemy} enemy, "
          f"{sum(1 for b in boxes if b.kind=='hazard')} hazard, 1 end-trigger) "
          f"extend_past_end={extend_past_end} start_mode={start_mode}")
    # Step 28: 2b/2c start already past the old pre-warp cutscene-trigger box, which
    # will never re-fire naturally in that episode's (small, post-warp) x range - see
    # CheckpointTracker.reset()'s own pre_visited docstring for why pre-seeding just
    # this one index is sufficient to unlock the rest of the [requires: Survived
    # cutscene] chain, including the boss trigger itself.
    pre_visited = {tracker.end_idx} if start_mode != "default" else None

    # Fixed reference batch for the standing health check (Step 15) AND the BC-anchor
    # loss (Step 16) - loaded once, independent of whatever this run's policy does.
    # Deferred import - see the top-of-file note on why this can't be a module-level
    # import (circular import through ingest_demo).
    from bc_pretrain import build_combined_batch
    health_frames, health_actions, _ = build_combined_batch(HEALTH_CHECK_DEMOS)
    health_majority_baseline = np.maximum(health_actions.mean(axis=0), 1 - health_actions.mean(axis=0)).mean()
    print(f"Health-check reference batch: {health_frames.shape[0]} real demo frames, "
          f"majority-class baseline accuracy={health_majority_baseline:.4f}")

    if start_mode == "default":
        env = retro.make(game=mode_cfg["game"], render_mode=None)
    else:
        # decomp ROM + a specific save-state, needs the ALL integrations search path
        # (same as used to independently verify these save-states load correctly).
        env = retro.make(game=mode_cfg["game"], state=mode_cfg["state"],
                          inttype=retro.data.Integrations.ALL, render_mode=None)
    rng = np.random.default_rng(123)

    if fresh_init and os.path.exists(WEIGHTS_FILE):
        os.remove(WEIGHTS_FILE)  # explicit opt-in: start this run from scratch
    if os.path.exists(WEIGHTS_FILE):
        print(f"Warm-starting from existing weights file: {WEIGHTS_FILE} "
              f"(e.g. a BC-pretrained checkpoint, or a previous run's progress, "
              f"deliberately left in place)")
    else:
        print(f"No existing weights file at {WEIGHTS_FILE} - starting from fresh "
              f"random init (sonic_policy_server will create+init it on startup)")

    # Step 14: one persistent server process for the whole run, used for the TRAIN call
    # every episode (see the comment above SERVER_BIN for why rollout stays on numpy
    # CPU). It loads WEIGHTS_FILE once at startup (or random-inits if absent) and keeps
    # those tensors resident/updated in-process across TRAIN calls, saving to disk after
    # each one - so `policy` below still needs its own policy.load(WEIGHTS_FILE) per
    # episode to pick up what the server just wrote, same as the old design.
    server = PolicyServerClient(SERVER_BIN, WEIGHTS_FILE, IW, IH, N_BUTTONS)
    policy = PolicyNet(N_BUTTONS)

    try:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(ACTION_LOG_ROOT, run_id)
        write_run_metadata(log_dir, {"n_train": n_train, "n_baseline": n_baseline, "max_steps": max_steps})
        print(f"Action-sequence logs for this run -> {log_dir}/ "
              f"(one .npy per episode, replayable later via replay_episode.py)")

        print(f"\n=== Training: {n_train} episodes, max_steps={max_steps}, gamma={GAMMA}, lr={LR} ===")
        print(f"[mem] VmRSS at training start: {vmrss_mb():.1f} MiB")
        train_stats = []
        t0 = time.time()
        for ep_i in range(n_train):
            policy.load(WEIGHTS_FILE)  # pick up whatever the server's last TRAIN call wrote
            ep = run_episode(env, tracker, policy=policy, rng=rng, max_steps=max_steps, pre_visited=pre_visited)
            epoch_losses = train_on_episode(ep, server, policy, health_frames, health_actions, rng)
            # Keep only the lightweight scalars summarize() actually needs - NOT the full
            # episode dict. ep["frames"]/["actions"]/["rewards"] are full per-step arrays
            # (at max_steps=7200, frames alone is ~123MB/episode); retaining all of them
            # across 80+20 episodes was a real bug that caused severe memory pressure/
            # thrashing on a prior run. train_on_episode() has already consumed what it
            # needs by this point, so the heavy arrays are free to be garbage-collected
            # the moment `ep` goes out of scope (no other reference is kept).
            light = {
                "total_reward": ep["total_reward"],
                "n_checkpoints": ep["n_checkpoints"],
                "length": ep["length"],
                "outcome": ep["outcome"],
                "ring_reward_total": ep["ring_reward_total"],
                "score_reward_total": ep["score_reward_total"],
            }
            train_stats.append(light)
            save_action_log(ep["actions"], log_dir, f"train_ep{ep_i+1:03d}_outcome-{light['outcome']}")
            # Step 17: every episode now, not every 10 - a real thrashing incident's
            # own self-reported VmRSS stayed flat/healthy right up to the stalled
            # line, because the spike happened in the gap AFTER a 10-episode-interval
            # print, not within one. Printing every episode plus system-wide
            # MemAvailable (not just this process's own VmRSS) catches a recurrence
            # (this one or a different one) immediately instead of after the fact.
            mem_note = f" [mem] VmRSS={vmrss_mb():.1f} MiB sys_avail={sys_mem_available_mb():.0f} MiB"
            print(f"[train {ep_i+1:3d}/{n_train}] reward={light['total_reward']:+.1f} "
                  f"checkpoints={light['n_checkpoints']:2d}/{tracker.n_checkpoints+tracker.n_enemy+1} "
                  f"len={light['length']:4d} outcome={light['outcome']} "
                  f"ring_r={light['ring_reward_total']:+.3f} score_r={light['score_reward_total']:+.3f} "
                  f"pg_loss={epoch_losses.mean():+.4f}{mem_note}")
            if (ep_i + 1) % BOSS_DEFEAT_RATE_BUCKET == 0:
                bucket = train_stats[-BOSS_DEFEAT_RATE_BUCKET:]
                defeat_rate = np.mean([s["outcome"] == "boss_defeated" for s in bucket])
                bucket_rewards = [s["total_reward"] for s in bucket]
                bucket_cps = [s["n_checkpoints"] for s in bucket]
                print(f"  [defeat-rate bucket] episodes {ep_i + 2 - BOSS_DEFEAT_RATE_BUCKET}-{ep_i + 1}: "
                      f"boss_defeat_rate={defeat_rate:.3f} ({int(defeat_rate * BOSS_DEFEAT_RATE_BUCKET)}/{BOSS_DEFEAT_RATE_BUCKET}) "
                      f"mean_reward={np.mean(bucket_rewards):+.2f} mean_checkpoints={np.mean(bucket_cps):.2f}")
                # Periodic NaN/Inf check on the live weights file - checked repeatedly
                # during a long run, not just assumed clean until the very end.
                try:
                    w = np.fromfile(WEIGHTS_FILE, dtype=np.float32)
                    bad = bool(np.isnan(w).any() or np.isinf(w).any())
                    print(f"  [weights check] {WEIGHTS_FILE}: any_nan_or_inf={bad} "
                          f"({'*** CORRUPTED - STOP AND INVESTIGATE ***' if bad else 'clean'})")
                except OSError as e:
                    print(f"  [weights check] could not read {WEIGHTS_FILE}: {e}")
            if (ep_i + 1) % POLICY_HEALTH_CHECK_EVERY == 0:
                # Step 15: standing entropy-collapse check - see its comment above
                # POLICY_HEALTH_CHECK_EVERY for why this exists. Uses the JUST-UPDATED
                # policy (reloaded next iteration) against THIS episode's own real
                # frames - cheap, no extra env steps.
                mean_p, std_p, acc = check_policy_health(policy, health_frames, health_actions)
                print(f"  [health ep{ep_i+1}] (checked against {health_frames.shape[0]} FIXED real "
                      f"demo frames, not self-play) per-button mean={np.round(mean_p, 3).tolist()} "
                      f"std={np.round(std_p, 3).tolist()} overall_accuracy={acc:.4f} "
                      f"(majority_baseline={health_majority_baseline:.4f}) (buttons={LEARNABLE_BUTTONS})")
                collapse = np.all(std_p < POLICY_HEALTH_STD_FLOOR)
                degraded = acc < POLICY_HEALTH_ACCURACY_FLOOR
                if collapse or degraded:
                    reasons = []
                    if collapse:
                        reasons.append(f"EVERY button's probability std on the real demo "
                                       f"frames is below {POLICY_HEALTH_STD_FLOOR} (ignoring the image)")
                    if degraded:
                        reasons.append(f"overall accuracy {acc:.4f} is below the "
                                        f"{POLICY_HEALTH_ACCURACY_FLOOR} floor (majority-class "
                                        f"baseline alone gets {health_majority_baseline:.4f})")
                    print(f"  *** WARNING: POSSIBLE POLICY DEGRADATION at episode {ep_i+1} - "
                          f"{'; '.join(reasons)}. See AGENTS.md 'Step 15'. Consider stopping "
                          f"and inspecting before continuing. ***", file=sys.stderr)
                sys.stderr.flush()
            sys.stdout.flush()
            del ep
        train_dt = time.time() - t0
        print(f"Training phase took {train_dt:.1f}s for {n_train} episodes")
        print(f"[mem] VmRSS after training phase: {vmrss_mb():.1f} MiB")

        print(f"\n=== Baseline: {n_baseline} random-policy episodes (no learning, no net) ===")
        baseline_stats = []
        for ep_i in range(n_baseline):
            ep = run_episode(env, tracker, policy=None, rng=rng, max_steps=max_steps, pre_visited=pre_visited)
            light = {
                "total_reward": ep["total_reward"],
                "n_checkpoints": ep["n_checkpoints"],
                "length": ep["length"],
                "outcome": ep["outcome"],
            }
            baseline_stats.append(light)
            save_action_log(ep["actions"], log_dir, f"baseline_ep{ep_i+1:03d}_outcome-{light['outcome']}")
            mem_note = f" [mem] VmRSS={vmrss_mb():.1f} MiB sys_avail={sys_mem_available_mb():.0f} MiB"
            print(f"[baseline {ep_i+1:3d}/{n_baseline}] reward={light['total_reward']:+.1f} "
                  f"checkpoints={light['n_checkpoints']:2d} len={light['length']:4d} outcome={light['outcome']}{mem_note}")
            del ep

        print(f"[mem] VmRSS after baseline phase: {vmrss_mb():.1f} MiB")
        env.close()

        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"[mem] Peak VmRSS for this process (getrusage ru_maxrss), whole run: {peak_rss_mb:.1f} MiB")

        # ---- summary ----
        def summarize(stats, label):
            rewards = [s["total_reward"] for s in stats]
            cps = [s["n_checkpoints"] for s in stats]
            print(f"{label}: mean_reward={np.mean(rewards):+.2f} (std={np.std(rewards):.2f}) "
                  f"mean_checkpoints={np.mean(cps):.2f} max_checkpoints={np.max(cps)}")

        print("\n=== SUMMARY ===")
        half = n_train // 2
        if half > 0:
            summarize(train_stats[:half], f"Training episodes 1-{half} (early)")
            summarize(train_stats[half:], f"Training episodes {half+1}-{n_train} (late)")
        summarize(train_stats, "All training episodes")
        summarize(baseline_stats, "Random-policy baseline")

        def defeat_rate(stats):
            return float(np.mean([s["outcome"] == "boss_defeated" for s in stats])) if stats else float("nan")

        print("\n=== BOSS-DEFEAT RATE BY BUCKET (Step 29, Foxo's explicit ask: see over time, not just final) ===")
        n_buckets = max(1, n_train // BOSS_DEFEAT_RATE_BUCKET)
        for b in range(n_buckets):
            lo, hi = b * BOSS_DEFEAT_RATE_BUCKET, min(n_train, (b + 1) * BOSS_DEFEAT_RATE_BUCKET)
            bucket = train_stats[lo:hi]
            print(f"  episodes {lo+1}-{hi}: boss_defeat_rate={defeat_rate(bucket):.3f} "
                  f"({sum(s['outcome']=='boss_defeated' for s in bucket)}/{len(bucket)})")
        remainder = train_stats[n_buckets * BOSS_DEFEAT_RATE_BUCKET:]
        if remainder:
            print(f"  episodes {n_buckets*BOSS_DEFEAT_RATE_BUCKET+1}-{n_train} (partial bucket): "
                  f"boss_defeat_rate={defeat_rate(remainder):.3f} "
                  f"({sum(s['outcome']=='boss_defeated' for s in remainder)}/{len(remainder)})")
        print(f"  OVERALL: train boss_defeat_rate={defeat_rate(train_stats):.3f} "
              f"({sum(s['outcome']=='boss_defeated' for s in train_stats)}/{len(train_stats)}) vs "
              f"RANDOM BASELINE boss_defeat_rate={defeat_rate(baseline_stats):.3f} "
              f"({sum(s['outcome']=='boss_defeated' for s in baseline_stats)}/{len(baseline_stats)})")
    finally:
        # Always flush weights + let the server exit cleanly, even on an exception/
        # KeyboardInterrupt mid-run - this is the ONLY place weights get written now
        # (every TRAIN call inside the server also saves to disk immediately after,
        # same as the old per-episode subprocess always did, so this is a safety net
        # for a clean shutdown, not the only save point).
        print("Shutting down policy server (flushing weights)...")
        server.quit()


if __name__ == "__main__":
    main()
