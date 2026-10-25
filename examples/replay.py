"""Replay an .npz demonstration and check its saved trajectory."""
import argparse
import json
import numpy as np
from push_any_letter import PushLetterEnv
from push_any_letter.recording import flatten
from push_any_letter.fonts import resolve_recording_font

parser = argparse.ArgumentParser()
parser.add_argument("recording")
parser.add_argument("--font", help="Local font matching the dataset SHA-256")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
with np.load(args.recording, allow_pickle=False) as data:
    meta = json.loads(str(data["metadata"]))
    env = PushLetterEnv(letter=meta["letter"], font_path=resolve_recording_font(meta, args.font), render_mode=None if args.headless else "human", max_episode_steps=meta.get("max_episode_steps", 600), **meta.get('physics_parameters', {}))
    env.reset(seed=meta["seed"], options=meta.get("reset_options"))
    max_error = 0.0
    try:
        for action, expected in zip(data["actions"], data["next_observations"]):
            obs, *_ = env.step(action)
            max_error = max(max_error, float(np.abs(flatten(obs)-expected).max()))
        print(f"Replayed {len(data['actions'])} steps; maximum state error: {max_error:.6g}")
        if max_error > 1e-3:
            raise SystemExit("Replay differs: check font, dependency versions, and platform.")
    finally:
        env.close()
