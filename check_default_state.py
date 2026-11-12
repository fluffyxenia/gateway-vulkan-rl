import stable_retro as retro

GAME = "SonicAndKnuckles3-Genesis-v0"
env = retro.make(game=GAME, render_mode=None)
print("statename (state file actually loaded):", env.statename)
obs, info = env.reset()
print("info dict as returned by reset() (always {} by design in this lib):", info)
info2 = env.data.lookup_all()
print("env.data.lookup_all() immediately after reset (no step taken):", info2)
print("x =", info2.get("x"))
print("y =", info2.get("y"))
print("screen_x =", info2.get("screen_x"))
print("screen_y =", info2.get("screen_y"))
env.close()
