"""Pickle-free demonstration recording, one pre-action observation per step."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def flatten(obs):
    return np.concatenate([obs["agent_pos"], obs["object_pose"], obs["goal_pose"]]).astype(np.float32)


class Recorder:
    def __init__(self, env, seed, source="unspecified"):
        self.meta = {"format_version": 1, "letter": env.letter, "seed": seed,
                     "font_path": env.glyph.font_path, "control_hz": env.control_hz,
                     "physics_hz": env.physics_hz, "max_episode_steps": env.max_episode_steps,
                     "source": source,
                     "state_columns": ["agent_x", "agent_y", "object_x", "object_y", "object_angle", "goal_x", "goal_y", "goal_angle"],
                     "action": "absolute pusher target in pixels; top-left origin; +y down",
                     "angles": "radians, clockwise on screen"}
        self.meta["reset_options"] = getattr(env, "reset_options", {})
        self.meta["physics_parameters"] = {"block_friction": env.block_friction}
        self.observations, self.actions, self.next_observations = [], [], []
        self.rewards, self.terminated, self.truncated = [], [], []

    def append(self, obs, action, next_obs, reward, terminated, truncated):
        self.observations.append(flatten(obs))
        self.actions.append(np.array(action, dtype=np.float32).copy())
        self.next_observations.append(flatten(next_obs))
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def save(self, directory):
        if not self.actions:
            return None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_U{ord(self.meta['letter']):04X}"
        path = directory / (stem + ".npz")
        np.savez_compressed(path, observations=self.observations, actions=self.actions,
                            next_observations=self.next_observations, rewards=self.rewards,
                            terminated=self.terminated, truncated=self.truncated,
                            metadata=json.dumps(self.meta, ensure_ascii=False))
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
