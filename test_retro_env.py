"""Smoke test: can stable-retro load Sonic 3 & Knuckles, reset, step, and return frames?
Run with the venv's python directly; takes no external args.
"""
import time
import numpy as np
import stable_retro as retro

GAME = "SonicAndKnuckles3-Genesis-v0"

print(f"Making env: {GAME}")
env = retro.make(game=GAME, render_mode=None)
print("Env created:", env)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs, info = env.reset()
print("Reset OK. obs shape:", np.asarray(obs).shape, "dtype:", np.asarray(obs).dtype)
print("Initial info:", info)

t0 = time.time()
n_steps = 300
total_reward = 0.0
for i in range(n_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if i == 0:
        print("First step OK. obs shape:", np.asarray(obs).shape,
              "reward:", reward, "terminated:", terminated, "truncated:", truncated)
        print("Info keys:", list(info.keys()))
    if terminated or truncated:
        print(f"Episode ended at step {i}, resetting.")
        obs, info = env.reset()

dt = time.time() - t0
print(f"Ran {n_steps} random steps in {dt:.2f}s ({n_steps/dt:.1f} steps/sec). total_reward={total_reward}")
env.close()
print("Closed OK. Smoke test PASSED.")
