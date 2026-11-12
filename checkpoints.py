"""
Parser + runtime tracker for the user's hand-traced checkpoint coordinate list
(~/gateway-vulkan-rl/coords.list) for Sonic 3 & Knuckles, Angel Island Zone Act 1.

File format - TWO line shapes are supported per non-comment/non-blank line:

1. Bounding box (the original, still used by most entries):
       ((A,B),(X,Y)) # comment
   Two corners of an axis-aligned box in the game's raw x/y RAM coordinates (same
   fields as info['x']/info['y']). Corners are NOT guaranteed top-left-first - always
   take min/max per axis.

2. Threshold condition, with an optional prerequisite (added for camera-lock-1 and the
   interactive-cutscene-trigger "end" box - see AGENTS.md "Step 10" for the full
   reasoning):
       >= THRESHOLD_X [requires: SOME OTHER BOX'S COMMENT TEXT] # comment
   The bracketed "[requires: ...]" part is optional. Matches when x >= THRESHOLD_X
   AND (no prerequisite given, OR the named prerequisite box has already been visited
   this episode). This exists for genuinely X-triggered game events (not real 2D
   collision boxes) where constraining Y was actively wrong (see the "interactive
   -cutscene trigger" fix), and/or where the earlier rejected idea of a narrow bounding
   box turned out to be "tunnel-able" - a fast-moving agent can skip clean over a few
   -pixel-wide box between two consecutive 60fps frames, but a one-sided ">=" threshold
   has no width to tunnel through: ANY x past the threshold matches, forever after.
   The prerequisite text is matched as a case-insensitive substring against exactly one
   OTHER box's comment in this same file - this is what stops an unrelated bit of level
   geometry that happens to share the same x range (at a very different point in the
   level, or an earlier visit before the real route gets there) from false-triggering a
   threshold condition that was only ever meant to fire once the player has genuinely
   already passed through an earlier, named point.

Classification into checkpoint / conditional-enemy / hazard / end / start-zone-kill is
done by matching specific phrases the user put in the comments (see _classify() for the
exact semantics this implements) - this is intentionally NOT a generic parser for
arbitrary future comment wording; if the user's checkpoint file format or phrasing
changes, update the keyword matches below. This classification is ORTHOGONAL to the
box-vs-threshold line shape above - either shape can in principle be any kind, though in
practice "threshold" is only used for the two specific entries mentioned.

A THIRD optional annotation, `[disable_after: SOME OTHER BOX'S COMMENT TEXT]`, can
follow the closing `))` of a bounding-box line (added for the "start-zone kill
checkpoint" - see AGENTS.md "Step 15"). This is the mirror image of a threshold's
`[requires: ...]`: instead of "only matches once the named box HAS been visited", it
means "stops matching forever, for the rest of the episode, once the named box HAS been
visited" - i.e. the box is a one-shot, episode-start-only trap that must retire itself
the moment the player legitimately passes the point it guards, so a later backtrack
into the same x/y region (if the level geometry ever loops back near the start) can't
retroactively kill an otherwise-successful run. Resolved the same way as `[requires:
...]` - a case-insensitive substring match against exactly one OTHER box's comment,
failing loudly on ambiguity.
"""
import re
from dataclasses import dataclass, field
from typing import Optional


BOX_LINE_RE = re.compile(r"\(\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\)\s*(?:\[\s*disable_after:\s*([^\]]+?)\s*\]\s*)?(?:#\s*(.*))?")
THRESHOLD_LINE_RE = re.compile(r">=\s*(-?\d+)\s*(?:\[\s*requires:\s*([^\]]+?)\s*\]\s*)?#\s*(.*)")


@dataclass
class CheckpointBox:
    x_min: Optional[int]
    x_max: Optional[int]
    y_min: Optional[int]
    y_max: Optional[int]
    comment: str
    kind: str  # "checkpoint" | "enemy" | "hazard" | "end" | "start_zone_kill" |
               # "kill_checkpoint" | "boss_trigger"
    window: Optional[tuple] = None  # (min_frames, max_frames) for "enemy" kind
    condition: str = "box"  # "box" | "threshold"
    threshold_x: Optional[int] = None  # only set when condition == "threshold"
    prerequisite_text: Optional[str] = None  # raw text from "[requires: ...]", pre-resolution
    prerequisite_idx: Optional[int] = None  # resolved box index, filled in after parsing the whole file
    disable_after_text: Optional[str] = None  # raw text from "[disable_after: ...]", pre-resolution
    disable_after_idx: Optional[int] = None  # resolved box index, filled in after parsing the whole file

    def contains(self, x: int, y: int) -> bool:
        """Pure POSITION check - for "box" condition this is the full geometric
        bounding-box test; for "threshold" condition this checks ONLY x >= threshold_x,
        ignoring Y entirely and ignoring any prerequisite/disable_after (this method has
        no `visited` state to check one against - callers that need the full,
        state-aware match must use matches() instead). This is intentional and matches
        what external position-only checks (e.g. ingest_demo.py's cutscene-trigger
        sanity check) actually want to ask: "has the player's raw position reached this
        point", independent of episode-local visited-state."""
        if self.condition == "threshold":
            return x >= self.threshold_x
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def matches(self, x: int, y: int, visited: set) -> bool:
        """Full match check used by CheckpointTracker.step(): the position check PLUS
        any state-aware conditions. For "box" condition and for threshold boxes with no
        prerequisite/disable_after, this is identical to contains()."""
        if not self.contains(x, y):
            return False
        if self.condition == "threshold" and self.prerequisite_idx is not None:
            if self.prerequisite_idx not in visited:
                return False
        if self.disable_after_idx is not None and self.disable_after_idx in visited:
            return False
        return True


def _classify(comment: str):
    c = comment.lower()
    if "cutscene trigger" in c:
        return "end", None
    if "boss trigger" in c:
        return "boss_trigger", None
    if "start-zone kill" in c:
        return "start_zone_kill", None
    if "special stage room" in c:
        return "kill_checkpoint", None
    if "avoid this for now" in c or re.search(r"get.*out of there", c):
        return "hazard", None
    if "assume enemy defeated" in c or ("enemy" in c and "score goes up" in c):
        m = re.search(r"(\d+)\D+(\d+)\s*frames", c)
        window = (int(m.group(1)), int(m.group(2))) if m else (30, 60)
        return "enemy", window
    return "checkpoint", None


def parse_coords_file(path: str) -> list:
    boxes = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("###"):
                continue
            if line.startswith("#"):
                continue

            m_thresh = THRESHOLD_LINE_RE.match(line)
            if m_thresh:
                threshold_x = int(m_thresh.group(1))
                prereq_text = m_thresh.group(2).strip() if m_thresh.group(2) else None
                comment = (m_thresh.group(3) or "").strip()
                kind, window = _classify(comment)
                boxes.append(CheckpointBox(
                    x_min=None, x_max=None, y_min=None, y_max=None,
                    comment=comment, kind=kind, window=window,
                    condition="threshold", threshold_x=threshold_x,
                    prerequisite_text=prereq_text,
                ))
                continue

            m = BOX_LINE_RE.match(line)
            if not m:
                continue
            a, b, x, y = (int(m.group(i)) for i in range(1, 5))
            disable_after_text = m.group(5).strip() if m.group(5) else None
            comment = (m.group(6) or "").strip()
            x_min, x_max = min(a, x), max(a, x)
            y_min, y_max = min(b, y), max(b, y)
            kind, window = _classify(comment)
            boxes.append(CheckpointBox(x_min, x_max, y_min, y_max, comment, kind, window,
                                        disable_after_text=disable_after_text))

    # Resolve prerequisite/disable_after text -> index now that the full list exists.
    # Fail loudly on ambiguity/typos rather than silently matching the wrong box or none
    # at all.
    for idx, bx in enumerate(boxes):
        if bx.condition == "threshold" and bx.prerequisite_text:
            hits = [i for i, other in enumerate(boxes)
                    if i != idx and bx.prerequisite_text.lower() in other.comment.lower()]
            assert len(hits) == 1, (
                f"prerequisite '{bx.prerequisite_text}' for box {idx} ({bx.comment!r}) "
                f"matched {len(hits)} other boxes by comment substring (expected exactly "
                f"1): {hits} - fix the [requires: ...] text in coords.list."
            )
            bx.prerequisite_idx = hits[0]
        if bx.disable_after_text:
            hits = [i for i, other in enumerate(boxes)
                    if i != idx and bx.disable_after_text.lower() in other.comment.lower()]
            assert len(hits) == 1, (
                f"disable_after '{bx.disable_after_text}' for box {idx} ({bx.comment!r}) "
                f"matched {len(hits)} other boxes by comment substring (expected exactly "
                f"1): {hits} - fix the [disable_after: ...] text in coords.list."
            )
            bx.disable_after_idx = hits[0]

    return boxes


class CheckpointTracker:
    """Stateful per-episode tracker. Call reset() at episode start, step(x, y, score,
    frame_idx) on every env step. Sequential checkpoints give +1 once each (anti-farm via
    a visited set); branching top/bottom routes are handled implicitly by not enforcing
    any order at all - whichever boxes the player's path actually touches get credited.
    """

    CHECKPOINT_REWARD = 1.0
    ENEMY_REWARD = 2.0
    HAZARD_PENALTY = -2.0
    END_REWARD = 5.0
    # Movement-delta diagnostic (see AGENTS.md "qualitative diagnostic" section) showed
    # the repeated re-entries into the "first enemy" box are NOT idling (agent is almost
    # always moving, ~3-7% idle steps) and NOT a reward-farming exploit (entering gives
    # 0 reward, only a confirmed kill gives +2) - it's unproductive wandering: the agent
    # runs past the box, drifts back below its x_min, and re-triggers it, sometimes
  # several times per episode, without committing to actually killing the enemy or
    # continuing on. First couple of entries are a free "genuine swing and miss" - no
    # penalty - but repeated failure at the same enemy box gets penalized from the 3rd
    # entry onward, to discourage that wandering loop without punishing a normal retry.
    STAGNATION_FREE_ATTEMPTS = 2  # entries 1 and 2 are free, penalty starts at entry 3
    STAGNATION_PENALTY = -0.5     # modest relative to CHECKPOINT_REWARD=1.0/ENEMY_REWARD=2.0
    # Step 15: "start-zone kill checkpoint" - ends a non-progressing episode immediately
    # instead of burning its whole step budget wandering near spawn (exactly what a
    # collapsed/noisy policy was observed doing in practice). Deliberately NOT as severe
    # as HAZARD_PENALTY=-2.0 - this isn't punishing a real hazard, just cutting a
    # clearly-stalled episode short - so it uses the same modest magnitude already
    # established for a conceptually similar "unproductive, not a real failure" signal
    # (STAGNATION_PENALTY), not a new arbitrary number.
    START_ZONE_KILL_PENALTY = -0.5
    # Real bug caught by an actual smoke test, not reasoned about in advance: the
    # kill-zone box sits directly BETWEEN spawn and the first real checkpoint, which
    # means it sits on the ONLY path forward - a plain position check fires on every
    # single episode that makes ANY forward progress at all, since reaching checkpoint
    # 1 requires passing through this box first. Without a time gate this killed
    # 40/40 training + 5/5 baseline episodes in a smoke test, all near-instantly
    # (reward exactly START_ZONE_KILL_PENALTY, episode lengths 34-368 steps) - it
    # can't tell "stuck near spawn" apart from "normally passing through on the way
    # onward" using position alone. Fix: only fires if the player is STILL in the box
    # after this many frames. Real human demos cross this exact region (spawn to
    # checkpoint 1) in 45-74 steps (measured directly from all 5 demo recordings, not
    # assumed) - 300 gives roughly 4x margin over the slowest real demo, while still
    # being a small fraction (10%) of a full 3000-step episode, so genuine
    # non-progression still gets caught early rather than only at episode's end.
    START_ZONE_KILL_MIN_FRAME = 300
    # Step 21: general-purpose version of the start-zone kill checkpoint, for
    # off-the-main-path areas a policy should simply never be in (first real case:
    # Foxo's own "ENTIRE SPECIAL STAGE ROOM" box in the AIZ1 continuation - entering a
    # special stage requires a deliberate star-post + enough-rings + jump action, so
    # unlike the start-zone box this does NOT sit on the only path forward and does NOT
    # need a time-gate - immediate termination on touch is safe here). Same mild
    # magnitude as START_ZONE_KILL_PENALTY/STAGNATION_PENALTY - cutting a wrong-area
    # episode short, not punishing a real hazard.
    KILL_CHECKPOINT_PENALTY = -0.5

    def __init__(self, boxes: list, extend_past_end: bool = False):
        """extend_past_end=False (default, preserves ALL existing behavior exactly -
        every current call site in ingest_demo.py/bc_pretrain.py/check_bc_fit_real.py/
        sonic_rl_train.py's original Act1-only runs is unaffected): reaching the
        "end"-kind box (the interactive-cutscene trigger) terminates the episode, same
        as always - this box's role as the BC-pretraining demo-truncation point is
        unchanged.
        extend_past_end=True (Step 21, for method 2a's full-level-replay-to-boss):
        reaching the "end"-kind box still grants END_REWARD once (a milestone marking
        "successfully cleared the original Act1 stretch") and marks it visited, but
        does NOT terminate the episode or break the scan loop - the episode continues
        on into the AIZ1-continuation boxes (cutscene survival, the new route, the
        boss trigger) merged into the same coords.list file after it."""
        self.boxes = boxes
        self.extend_past_end = extend_past_end
        self.n_checkpoints = sum(1 for b in boxes if b.kind == "checkpoint")
        self.n_enemy = sum(1 for b in boxes if b.kind == "enemy")
        end_idx = [i for i, b in enumerate(boxes) if b.kind == "end"]
        assert len(end_idx) == 1, f"expected exactly one 'end' box, found {len(end_idx)}"
        self.end_idx = end_idx[0]
        self.reset()

    def reset(self, pre_visited=None):
        """pre_visited (Step 28, for methods 2b/2c - save-state-start boss-training
        comparison): an episode that starts from a save-state ALREADY past the
        Act1->Act2 cutscene (e.g. AngelIslandZone.Act1Boss/Act1Midpoint) will never
        naturally re-fire the old pre-warp "interactive-cutscene trigger" threshold
        box (x>=12210 in the OLD coordinate epoch - this episode's x values live
        entirely in the post-warp space, e.g. 239 or 4161, and never approach
        12210), which would otherwise permanently lock every downstream
        [requires: Survived cutscene]-gated box (including the boss_trigger itself)
        for the whole episode. pre_visited lets the caller pre-seed self.visited
        with that box's index (tracker.end_idx) at episode start - verified
        sufficient (not assumed): box 37 ("Survived cutscene", x>=472) still fires
        naturally once x actually crosses 472 in-episode (immediately for a
        4161-start, after a short climb for a 239-start), and nothing else in the
        2b/2c-relevant reward chain depends on any OTHER pre-warp-only box."""
        self.visited = set(pre_visited) if pre_visited else set()
        self.enemy_pending = None  # (box_idx, score_at_entry, frame_entered)
        self.n_reached = 0  # count of checkpoint/enemy/end boxes credited this episode
        self.enemy_attempts = {}  # box_idx (enemy-kind) -> number of entries so far this episode

    def step(self, x: int, y: int, score: int, frame_idx: int):
        """Returns (reward, terminated, info_str_or_None) for this step."""
        reward = 0.0
        terminated = False
        event = None

        # Step 22 fix, caught by a real verification test before merging the AIZ1
        # continuation: prerequisite checks below use a SNAPSHOT of `visited` taken
        # here, at the START of this step, not the live set being mutated as this
        # same scan progresses. Without this, a single step whose x value happens to
        # numerically satisfy a whole CHAIN of prerequisite-linked thresholds at once
        # (not just a direct jump, but legitimately possible here: the new AIZ1-
        # continuation's thresholds (4465, 4503) are numerically SMALLER than the OLD
        # pre-cutscene space's natural x range, so crossing the OLD end-trigger at
        # x=12210 trivially also satisfies "x>=472"/"x>=4465"/"x>=4503" as bare
        # numbers - only a snapshot, not same-step cascading through `visited`,
        # stops all of them firing together the instant the OLD trigger does,
        # without the player ever having touched the real post-cutscene path at all.
        # Verified functionally identical to the old live-set behavior for every
        # EXISTING prerequisite chain (camera-lock-1) under realistic, gradual
        # per-step movement - the two differ only for this same-step-cascade case.
        visited_at_step_start = set(self.visited)

        # resolve a pending "did the enemy die" window first
        if self.enemy_pending is not None:
            idx, score_at_entry, frame_entered = self.enemy_pending
            dt = frame_idx - frame_entered
            if idx not in self.visited:
                if score > score_at_entry and 0 <= dt <= 60:
                    reward += self.ENEMY_REWARD
                    self.visited.add(idx)
                    self.n_reached += 1
                    event = f"enemy_defeated(box={idx})"
                    self.enemy_pending = None
                elif dt > 60:
                    self.enemy_pending = None  # window expired, no credit

        for idx, b in enumerate(self.boxes):
            if idx in self.visited:
                continue
            if not b.matches(x, y, visited_at_step_start):
                continue
            if b.kind == "end":
                reward += self.END_REWARD
                self.visited.add(idx)
                self.n_reached += 1
                if self.extend_past_end:
                    # Step 21: milestone only - episode continues into the merged
                    # AIZ1-continuation boxes rather than stopping here. Deliberately
                    # NOT breaking the scan loop, so a same-step box right at/past
                    # this x (unlikely but not impossible) still gets evaluated.
                    event = "level_end_trigger_milestone_continuing"
                else:
                    terminated = True
                    event = "level_end_trigger"
                    break  # episode is over, nothing else matters
            elif b.kind == "hazard":
                reward += self.HAZARD_PENALTY
                terminated = True
                self.visited.add(idx)
                event = f"hazard(box={idx}: {b.comment.strip()})"
                break
            elif b.kind == "kill_checkpoint":
                reward += self.KILL_CHECKPOINT_PENALTY
                terminated = True
                self.visited.add(idx)
                event = f"kill_checkpoint(box={idx}: {b.comment.strip()})"
                break
            elif b.kind == "boss_trigger":
                # Step 21: reaching the boss arena is real progress - reward it like a
                # checkpoint. Deliberately NOT terminating and NOT implementing any
                # hit-based reward/termination here - that needs RAM access
                # (env.unwrapped.get_ram()) this class doesn't have, so it's the
                # rollout loop's job (sonic_rl_train.py) to react to this event string
                # by calling boss_hit_counter.py's sanity check / save-state capture.
                # Scope deliberately limited to exactly what was asked.
                reward += self.CHECKPOINT_REWARD
                self.visited.add(idx)
                self.n_reached += 1
                event = f"boss_trigger(box={idx})"
            elif b.kind == "start_zone_kill":
                if frame_idx >= self.START_ZONE_KILL_MIN_FRAME:
                    reward += self.START_ZONE_KILL_PENALTY
                    terminated = True
                    self.visited.add(idx)
                    event = f"start_zone_kill(box={idx})"
                    break
                # else: position matches, but not enough time has passed yet - this is
                # a normal pass-through en route to checkpoint 1, not genuine
                # non-progression. Deliberately do NOT mark visited/break here - keep
                # checking other boxes this same step, and keep re-checking this box on
                # future steps until either the time gate trips or disable_after kicks
                # in once checkpoint 1 is actually reached.
            elif b.kind == "enemy":
                if self.enemy_pending is None:
                    self.enemy_pending = (idx, score, frame_idx)
                    self.enemy_attempts[idx] = self.enemy_attempts.get(idx, 0) + 1
                    attempt_n = self.enemy_attempts[idx]
                    event = f"enemy_zone_entered(box={idx}, attempt={attempt_n})"
                    if attempt_n > self.STAGNATION_FREE_ATTEMPTS:
                        reward += self.STAGNATION_PENALTY
                        event += f"_STAGNATION_PENALTY({self.STAGNATION_PENALTY})"
            else:  # checkpoint
                reward += self.CHECKPOINT_REWARD
                self.visited.add(idx)
                self.n_reached += 1
                event = f"checkpoint(box={idx})"

        return reward, terminated, event


if __name__ == "__main__":
    import sys
    boxes = parse_coords_file(sys.argv[1] if len(sys.argv) > 1 else "coords.list")
    print(f"Parsed {len(boxes)} boxes:")
    for i, b in enumerate(boxes):
        if b.condition == "threshold":
            prereq = f" requires=#{b.prerequisite_idx}({b.prerequisite_text!r})" if b.prerequisite_idx is not None else ""
            pos = f"x>={b.threshold_x}{prereq}"
        else:
            pos = f"x=[{b.x_min},{b.x_max}] y=[{b.y_min},{b.y_max}]"
        disable_after = f" disable_after=#{b.disable_after_idx}({b.disable_after_text!r})" if b.disable_after_idx is not None else ""
        print(f"  [{i}] kind={b.kind:16s} {pos}{disable_after} window={b.window} # {b.comment}")
