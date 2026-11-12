"""Step 29 follow-up (Foxo's idea): an orthogonal, RAM-hit-counter-independent
boss-defeat confirmation signal. Confirmed directly in the decomp source:
BossDefeated (sonic3k.asm ~180910) awards exactly +100 score
(`moveq #100,d0; jsr HUD_AddToScore`) and BossDefeated_StopTimer (~180900,
called by 10+ different bosses throughout the game, not AIZ1-specific) clears
Update_HUD_timer, which empirically (verified against the real human demo
where this exact +100 jump occurs) freezes the Timer-area RAM bytes
immediately afterward.

TIMER_ADDR calibrated directly against the already-confirmed `score` RAM
address (0xfe26, from this project's own data.json) by counting backward
through sonic3k.constants.asm's declarations (Ring_count ds.w1, Timer ds.l1,
then Score ds.l1 - contiguous, no gap) - NOT long-distance hand-counting like
the earlier (unsuccessful) palette-address attempt. Verified empirically: this
byte increments steadily during normal play and freezes for 11+ consecutive
frames starting the exact frame score jumps by +100 in the real human demo.

This confirms DEFEAT (fight concluded successfully) - it does NOT track
per-hit progress/health during a fight, and can't replace the hit-counter work
for that. It sidesteps every RAM-hit-counter noise class found tonight (slot
reuse, sawtooth, single-decrement skip, frame-gap false positives) since it
reads game-state-level data, not a per-object hit-counter guess.
"""
import numpy as np

BOSS_DEFEATED_SCORE_BONUS = 100  # confirmed: BossDefeated routine, moveq #100,d0
TIMER_BYTE_ADDR = 0xfe24  # the specific Timer-area byte confirmed to freeze on defeat
FREEZE_CHECK_FRAMES = 10  # how many frames after the score jump to confirm the freeze holds


def check_timer_freeze_defeat(score_seq, timer_byte_seq):
    """score_seq, timer_byte_seq: same-length sequences (lists/arrays) of per-frame
    values. Returns a list of frame indices (into the sequence) where a score
    jump of exactly BOSS_DEFEATED_SCORE_BONUS co-occurs with the timer byte
    freezing (staying exactly constant) for at least FREEZE_CHECK_FRAMES frames
    starting at/after that same frame - a high-confidence, independent boss-
    defeat confirmation."""
    hits = []
    n = len(score_seq)
    for t in range(1, n - FREEZE_CHECK_FRAMES):
        if score_seq[t] - score_seq[t - 1] == BOSS_DEFEATED_SCORE_BONUS:
            window = timer_byte_seq[t:t + FREEZE_CHECK_FRAMES]
            if all(v == window[0] for v in window):
                hits.append(t)
    return hits
