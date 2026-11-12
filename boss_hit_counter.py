"""AIZ1 miniboss hit-counter utilities (Step 20).

Address 0xb278 was found by Foxo via live empirical RAM-diffing (calibration pass to
build a noise-exclusion set, then diffing full 64KB snapshots before/after each of 6
confirmed landed hits) and cross-validated against the disassembly's
`move.b #6,collision_property(a0)` boss-init instruction - both methods independently
agree the counter starts at 6 and counts down monotonically to 0 on a kill.

CONFIRMED, not just a theoretical caveat (Step 22 update) - this address is NOT a
general technique: 0xb278 sits inside one of 90 `Dynamic_object_RAM` slots
(object_size=$4A in the source). WHICH slot a given boss instance occupies depends
on dynamic-object spawn history leading up to it, not on boss identity alone. A
second live test (keypress-free, logs a watchlist of candidates every frame, full
~96s playthrough reaching the boss via a different/faster path than the first test)
settled this empirically: 0xb278 never changed ONCE across that entire session -
the SAME address that gave a clean 6->0 countdown in the first test. The real
counter that run was somewhere else entirely; nearby candidates near the boss fight
were sawtooth/wrapping animation timers, not a hit counter.
  - Save-state-right-before-the-boss (method 2b): rock-solid. Every episode reloads
    byte-identical emulator state -> identical spawn history -> identical slot,
    guaranteed, every time.
  - Full-level-replay-to-boss (method 2a): CONFIRMED UNRELIABLE, not merely
    unconfirmed. Do not use 0xb278 for any hit-based reward/termination under this
    method - rely on other signals (death, timeout, a flat arrival reward for
    reaching the trigger) instead, unless a generalized all-90-slots detector
    (scanning for whichever slot actually behaves like a monotonic countdown each
    episode) is built later.

Because of this, `check_boss_hit_counter_sanity()` below MUST be called at the
moment the boss fight begins (camera lock / boss trigger - whichever coordinate
Foxo's Debug Mode pass identifies) for EVERY episode, regardless of which training
method is used - not just once/offline. If it ever reads something other than the
expected initial value, the caller must warn/flag, not silently trust stale data
from a different dynamic-object slot.

Also only verified for THIS ONE boss under THIS ONE save-state/spawn-context - not
assumed to transfer to any other boss (AIZ2's, or any later zone's) without redoing
this same live-verification process for that boss specifically.
"""
import sys
import numpy as np

BOSS_HIT_COUNTER_ADDR = 0xB278
BOSS_HIT_COUNTER_INITIAL = 6

# Step 24/25: structural constants from the disassembly, used to check whether a
# detected hit-counter address's offset WITHIN its Dynamic_object_RAM slot is
# constant across episodes even though the absolute address shifts with spawn
# history - `collision_property` is at offset 0x29 within each $4A-byte object slot.
COLLISION_PROPERTY_OFFSET = 0x29
DYNAMIC_OBJECT_SIZE = 0x4A


def structural_offset(addr: int) -> int:
    """(addr - 0x29) mod 0x4A - constant across episodes' winning addresses would be
    strong structural confirmation of the exact mechanism (same relative offset
    within shifting object slots), independent of pinning down Dynamic_object_RAM's
    absolute base address."""
    return (addr - COLLISION_PROPERTY_OFFSET) % DYNAMIC_OBJECT_SIZE


class BossHitDetector:
    """Step 24/25: generalized, per-episode boss-hit-counter detector - does NOT
    trust any fixed address (0xb278 confirmed unreliable under method 2a - see the
    module docstring above). Instead, scans full RAM for the boss-SPAWN signature: a
    byte that WARPS/SNAPS TO EXACTLY 6 right around when the boss trigger activates
    (matching the disassembly's `move.b #6,collision_property(a0)`, which fires once
    at object creation - not a loose "decreases to 0" check, which can false-positive
    on unrelated bytes that happen to cycle through small values, e.g. the
    animation-timer false leads found during manual live-testing). Confirms
    candidacy by requiring the address to ONLY decrease from there, never increase -
    this is what disqualifies sawtooth/wrapping animation/frame timers.

    Step 29 addition - the SINGLETON check: a real, scaled-up (300-episode) run of
    method 2b surfaced a second, substantial cluster of "confirmed" detections at
    structural_offset=60 (32/109 of that run's detections - not a rare fluke),
    alongside the originally-validated structural_offset=42 cluster (54/109).
    Traced the real confirmed addresses behind the offset=60 cluster
    (0xcc2f/0xcc79/0xccc3/0xcd0d - four addresses each exactly DYNAMIC_OBJECT_SIZE
    apart, i.e. four DIFFERENT simultaneously-occupied object slots all showing
    this signature at once) and cross-referenced sonic3k.constants.asm: that
    relative intra-slot offset (worked out from these addresses the same way
    structural_offset() works out collision_property's) lands on `ros_bit = $3B`
    - "the bit to be cleared when an object is destroyed if the ROS flag is set",
    a GENERIC per-object-slot field reused across many ordinary object types (the
    same constants file overloads this same offset region for unrelated purposes
    depending on object type - `tilt`, `routine_secondary`, `shield_plc`,
    `stick_to_convex`), not anything boss-specific. A real boss encounter should
    be the UNIQUE object in the arena exhibiting this exact warp-then-decrease
    signature; an ordinary engine-level field shared by several ordinary objects
    at once is exactly what showed up as the false-positive cluster. Rather than
    hardcoding structural_offset==42 (fragile - a single-value allowlist doesn't
    generalize if the real table-base residual ever shifts), this generalizes the
    fix: any time two or more live candidates share the same structural_offset
    AT THE SAME TIME, that offset is almost certainly a common/generic field, not
    the real singleton hit-counter - disqualify every candidate currently at that
    offset, and reject any FUTURE candidate at that same offset for the rest of
    this episode too (not just the ones already seen), since this is a property
    of the field itself, not a one-off coincidence of this particular frame.

    Usage: call start(ram) once at the exact boss-trigger frame, then step(ram)
    every subsequent frame. step() returns the confirmed hit-counter address once
    some candidate has reached exactly 0 while never having increased AND never
    having shared its structural offset with another simultaneous candidate -
    None until then (or if no such candidate exists this episode)."""

    WARP_WINDOW_FRAMES = 10  # frames around the trigger to scan for a fresh warp-to-value
    # Step 29 follow-up (Foxo's spec): the boss's invincibility window after a hit is
    # ~500ms (~30 frames @60fps) before it can be hit again, plus/minus ~1 frame of
    # collision-detection lag - so consecutive REAL hits can't land closer together
    # than ~29-31 frames. Use 28 (a small margin below that) to avoid false-rejecting
    # on emulator timing jitter while still catching anything decrementing faster
    # than any real hit-counter possibly could. NOTE (verified independently, not
    # just trusted - see the real-timing investigation this same step): several
    # "confirmed" addresses, INCLUDING ones at the originally-validated
    # structural_offset=42, were found decrementing every single frame (gap=1) while
    # x/y/score/rings showed ordinary walking, not combat - these are false
    # positives this filter is specifically meant to catch, and their existence
    # means some of this project's earlier "verified via replay" defeat counts
    # likely still overcounted real wins even after the singleton+act-gate fixes -
    # see the re-verification run after this fix for the corrected numbers.
    MIN_FRAMES_BETWEEN_HITS = 28

    def __init__(self, warp_value: int = 6):
        """warp_value (Step 29 follow-up): the hit-counter's real init value, per-boss
        - was hardcoded to 6 (correct for AIZ1's miniboss specifically, per the
        `move.b #6,collision_property(a0)` init instruction traced from the
        disassembly). Foxo correctly flagged this does NOT generalize: AIZ2's boss
        (and likely others) needs 8 hits, not 6, and a hardcoded 6 would silently
        find ZERO candidates for such a boss - a total miss, not a wrong answer,
        with no warning anything's off. Pass the real init value (found the same way
        AIZ1's was: the disassembly's init code for that specific Obj_* boss
        routine) for any boss other than AIZ1's miniboss. Auto-discovering this
        value instead of requiring it (e.g. watching for ANY small-integer warp,
        not just a specific target) was considered but deliberately not
        implemented - it would reopen the same false-positive-rate questions
        already worked through for value=6 specifically (see the singleton-check
        fix above), for a generalization not yet needed since AIZ1 is still the
        active focus. Revisit if/when this detector is reused on AIZ2's boss."""
        self.warp_value = warp_value
        self.prev_ram = None
        self.candidates = {}      # addr -> last-seen value, for still-plausible candidates
        self.disqualified = set()  # addrs ruled out (increased at some point, or crowded-offset)
        self.crowded_offsets = set()  # structural_offset values seen shared by >=2 simultaneous candidates this episode
        self.last_decrement_frame = {}  # addr -> frames_since_trigger at its most recent real decrement
        self.frames_since_trigger = None
        self.confirmed_addr = None

    def start(self, ram: np.ndarray):
        """Call once, at the exact frame the boss trigger fires."""
        self.prev_ram = ram.copy()
        self.candidates = {}
        self.disqualified = set()
        self.crowded_offsets = set()
        self.last_decrement_frame = {}
        self.frames_since_trigger = 0
        self.confirmed_addr = None

    def step(self, ram: np.ndarray):
        """Call every frame after start(). Returns the confirmed address (int) the
        moment a candidate reaches 0 having never increased, else None."""
        if self.frames_since_trigger is None:
            return None  # start() was never called this episode
        if self.confirmed_addr is not None:
            return self.confirmed_addr  # already confirmed, nothing more to do

        if self.frames_since_trigger <= self.WARP_WINDOW_FRAMES:
            # look for bytes that just became exactly 6 (warped from something else,
            # not already sitting at 6 for unrelated reasons before the trigger)
            warped = np.nonzero((ram == self.warp_value) & (self.prev_ram != self.warp_value))[0]
            newly_added = []
            for addr in warped:
                addr = int(addr)
                if addr not in self.candidates and addr not in self.disqualified:
                    so = structural_offset(addr)
                    if so in self.crowded_offsets:
                        self.disqualified.add(addr)  # this offset is already known-crowded
                        continue
                    self.candidates[addr] = self.warp_value
                    newly_added.append((addr, so))
            # Singleton check: if any two addresses (newly added this frame, OR a
            # newly-added one colliding with an already-tracked one) share the same
            # structural_offset, that offset is crowded - disqualify ALL of them,
            # now and for the rest of this episode.
            all_live = [(a, structural_offset(a)) for a in self.candidates]
            offset_counts = {}
            for a, so in all_live:
                offset_counts.setdefault(so, []).append(a)
            for so, addrs in offset_counts.items():
                if len(addrs) >= 2:
                    self.crowded_offsets.add(so)
                    for a in addrs:
                        del self.candidates[a]
                        self.disqualified.add(a)

        for addr in list(self.candidates.keys()):
            val = int(ram[addr])
            if val > self.candidates[addr]:
                # increased since we started tracking it - not a monotonic hit
                # counter (most likely an animation/frame timer wrapping around)
                del self.candidates[addr]
                self.disqualified.add(addr)
                self.last_decrement_frame.pop(addr, None)
                continue
            if val < self.candidates[addr]:
                # Step 29 follow-up - real gap caught by Foxo/the coordinator
                # re-examining train_ep271 (one of the then-"final 10/109"
                # survivors): its confirmed byte was observed INCREASING for
                # several frames of ordinary walking (zero combat), then touched
                # the warp value once, then was already 0 on the VERY NEXT
                # frame - confirmed via a single lone decrement that skipped
                # straight past 5,4,3,2,1. A real one-hit-per-event counter
                # that genuinely starts at warp_value cannot do this - each real
                # hit can only ever decrement it by exactly 1. The frame-gap
                # check above only compares the gap BETWEEN decrements, so a
                # lone decrement (nothing to compare against) or ANY decrement
                # that skips multiple values at once trivially passed it before
                # this fix. Require the step size itself to be exactly 1:
                if self.candidates[addr] - val != 1:
                    del self.candidates[addr]
                    self.disqualified.add(addr)
                    self.last_decrement_frame.pop(addr, None)
                    continue
                # a real (single-step) decrement - check it isn't closer to the
                # PREVIOUS decrement than a real hit-counter possibly could be
                # (invincibility-frame minimum gap, see MIN_FRAMES_BETWEEN_HITS's
                # own comment). The very first decrement from the initial
                # warp_value has nothing to compare against yet (object creation
                # isn't a "hit"), so it always passes this specific check.
                prev_frame = self.last_decrement_frame.get(addr)
                if prev_frame is not None and (self.frames_since_trigger - prev_frame) < self.MIN_FRAMES_BETWEEN_HITS:
                    del self.candidates[addr]
                    self.disqualified.add(addr)
                    self.last_decrement_frame.pop(addr, None)
                    continue
                self.last_decrement_frame[addr] = self.frames_since_trigger
            self.candidates[addr] = val
            if val == 0:
                self.confirmed_addr = addr
                break

        self.prev_ram = ram.copy()
        self.frames_since_trigger += 1
        return self.confirmed_addr


def read_boss_hit_count(ram) -> int:
    """ram: the (65536,) uint8 array from env.unwrapped.get_ram(). Returns the raw
    byte value at the verified address - caller interprets (6=full health, 0=dead)."""
    return int(ram[BOSS_HIT_COUNTER_ADDR])


def check_boss_hit_counter_sanity(ram, expected: int = BOSS_HIT_COUNTER_INITIAL) -> bool:
    """Call this at the exact moment the boss fight begins (camera lock / boss
    trigger), every episode, regardless of training method. Returns True if the
    byte at the verified address matches the expected initial value (6) - False
    means this address is NOT tracking the boss in this episode (most likely cause:
    method 2a's spawn-history-dependent slot-shift risk - see module docstring).
    Caller MUST warn/flag on False, never silently trust a stale read."""
    actual = read_boss_hit_count(ram)
    if actual != expected:
        print(f"WARNING: boss hit-counter sanity check FAILED at boss-trigger moment - "
              f"expected {expected} at 0x{BOSS_HIT_COUNTER_ADDR:04x}, got {actual}. "
              f"This address may not be tracking the boss in this episode (see "
              f"boss_hit_counter.py's module docstring - likely a spawn-history-"
              f"dependent slot shift, especially under method 2a). NOT trusting "
              f"hit-counter-based reward/termination for this episode.", file=sys.stderr)
        return False
    return True
