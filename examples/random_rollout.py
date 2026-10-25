"""A random-action smoke example; not a trained controller."""
import numpy as np
from push_any_letter import PushLetterEnv

env = PushLetterEnv(letter="A", render_mode="human")
obs, info = env.reset(seed=42)
rng = np.random.default_rng(7)
try:
    for step in range(600):
        if step % 20 == 0:
            action = rng.uniform(28, 612, size=2).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            break
finally:
    env.close()
