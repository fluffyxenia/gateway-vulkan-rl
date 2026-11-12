"""Direct logic test for the death-detection condition in run_episode(), without relying
on an actual in-game death occurring (which turned out not to happen organically in a
quick 10-episode/3000-step random-policy sample - early AIZ1 just doesn't have much to
die from that fast). This isolates exactly the same condition used in sonic_rl_train.py
and feeds it synthetic lives sequences to confirm the logic itself is correct."""

def died_check(prev_lives, lives):
    return prev_lives is not None and lives is not None and lives < prev_lives

cases = [
    (None, 3, False, "first step ever (prev_lives=None) - must not misfire"),
    (3, 3, False, "no change - must not fire"),
    (3, 2, True, "lost a life - MUST fire"),
    (2, 2, False, "steady at 2 - must not fire"),
    (2, 1, True, "lost another - MUST fire"),
    (1, 0, True, "lost last life - MUST fire"),
    (3, None, False, "info missing lives this step - must not crash/misfire"),
    (None, None, False, "info missing lives both steps - must not crash/misfire"),
    (3, 4, False, "lives INCREASED (extra life bonus) - must not fire"),
]

all_ok = True
for prev, cur, expected, desc in cases:
    got = died_check(prev, cur)
    ok = got == expected
    all_ok &= ok
    print(f"{'OK  ' if ok else 'FAIL'} prev_lives={prev!r:>5} lives={cur!r:>5} -> died={got!r:>5} (expected {expected!r})  # {desc}")

print("\nALL PASSED" if all_ok else "\nSOME FAILED")
