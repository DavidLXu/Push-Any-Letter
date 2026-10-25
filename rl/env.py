"""Shared, geometry-based goal-conditioned pushing task for online PPO."""
import math
import string
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Point
from shapely.ops import nearest_points
from push_any_letter.env import PushLetterEnv
from push_any_letter.geometry import make_glyph, transform

LETTERS = string.ascii_uppercase + string.ascii_lowercase


def shaped_reward(previous, current, success, action, gamma=0.999):
    # True absorbing success has zero potential. Time limits are NOT absorbing.
    return (20.0 * success - 0.01 - 0.001 * float(np.square(action).sum())
            + gamma * (0.0 if success else current) - previous)


class OnlineLetterEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, font_path=None, gamma=0.999, max_episode_steps=2400,
                 curriculum=True, render_mode=None):
        self.base = PushLetterEnv(font_path=font_path, block_friction=1.4,
                                  max_episode_steps=max_episode_steps,
                                  render_mode=render_mode)
        self.render_mode = render_mode
        self.gamma = gamma
        self.curriculum = curriculum
        self.difficulty = 0.0 if curriculum else 1.0
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (72,), np.float32)

    def set_difficulty(self, value):
        self.difficulty = float(np.clip(value, 0, 1))

    def _sample(self, letter, goal_angle, full):
        rng = self.np_random
        glyph = make_glyph(letter, self.base.font_path)
        goal = np.array([320., 320., goal_angle])
        goal_geom = transform(glyph.geometry, goal)
        for _ in range(1000):
            if full:
                angle = rng.uniform(0, 2 * math.pi)
            else:
                angle = goal_angle + rng.uniform(-1, 1) * (math.pi / 12 + self.difficulty * 11 * math.pi / 12)
            rot = transform(glyph.geometry, (0, 0, angle))
            x0, y0, x1, y1 = rot.bounds
            low, high = np.array([60-x0, 60-y0]), np.array([580-x1, 580-y1])
            if full:
                xy = rng.uniform(low, high)
            else:
                radius = (20 + 200 * self.difficulty) * np.sqrt(rng.uniform())
                theta = rng.uniform(0, 2 * math.pi)
                xy = np.clip(goal[:2] + radius * np.array([np.cos(theta), np.sin(theta)]), low, high)
            pose = np.array([*xy, angle])
            geom = transform(glyph.geometry, pose)
            if geom.intersection(goal_geom).area / glyph.geometry.area < 0.95:
                break
        else:
            raise RuntimeError("Unable to sample a non-successful reset")
        for _ in range(1000):
            if full:
                agent = rng.uniform(44, 596, 2)
            else:
                theta = rng.uniform(0, 2 * math.pi)
                far = xy + 500 * np.array([np.cos(theta), np.sin(theta)])
                boundary = np.array(nearest_points(geom, Point(far))[0].coords[0])
                direction = far - boundary
                direction /= np.linalg.norm(direction)
                agent = boundary + direction * (20 + rng.uniform(0, 12 + 88 * self.difficulty))
            if np.all((agent >= 44) & (agent <= 596)) and geom.distance(Point(agent)) >= 20:
                break
        else:
            raise RuntimeError("Unable to sample a separated pusher")
        return dict(letter=letter, initial_object_pose=pose,
                    initial_agent_pos=agent, goal_pose=goal)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        letter = options.get("letter", LETTERS[int(self.np_random.integers(52))])
        angle = float(options.get("goal_angle", self.np_random.uniform(0, 2 * math.pi)))
        self.full_start = bool(options.get("full", not self.curriculum or self.np_random.random() < 0.25))
        reset = options.get("reset_options") or self._sample(letter, angle, self.full_start)
        _, info = self.base.reset(seed=seed, options=reset)
        self.potential = self._potential(info["coverage"])
        self.episode_return = 0.0
        return self._obs(info), info

    def get_reset_record(self):
        return dict(goal_pose=self.base.goal_pose.tolist(), reset_options=self.base.reset_options)

    def _potential(self, coverage):
        b = self.base
        current = transform(b.glyph.geometry, (*b.block.position, b.block.angle))
        # Centroids and areas treat all letters consistently, including holes and symmetry.
        distance = np.linalg.norm(np.array(b.block.position) - b.goal_pose[:2]) / 156
        contact_gap = max(0., current.distance(Point(*b.agent.position)) - b.pusher_radius) / 156
        return float(4 * coverage - 2 * distance - 0.5 * contact_gap)

    def _obs(self, info):
        b = self.base
        ap, op, gp = np.array(b.agent.position), np.array(b.block.position), b.goal_pose[:2]
        angles = [math.sin(b.block.angle), math.cos(b.block.angle),
                  math.sin(b.goal_pose[2]), math.cos(b.goal_pose[2])]
        onehot = np.zeros(52)
        onehot[LETTERS.index(b.letter)] = 1
        values = np.concatenate([onehot, (ap-320)/320, (op-320)/320, (gp-320)/320,
            angles, np.clip(np.array(b.agent.velocity)/260, -5, 5),
            np.clip(np.array(b.block.velocity)/260, -5, 5),
            [np.clip(b.block.angular_velocity/10, -5, 5)], (ap-op)/320, (gp-op)/320,
            [info["coverage"]]])
        return values.astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite normalized waypoint offsets")
        _, _, terminated, truncated, info = self.base.step(np.array(self.base.agent.position) + 60 * action)
        phi = self._potential(info["coverage"])
        reward = shaped_reward(self.potential, phi, terminated, action, self.gamma)
        self.potential = phi
        self.episode_return += reward
        info.update(full_start=self.full_start, potential=phi)
        if terminated or truncated:
            info["episode"] = dict(r=self.episode_return, l=self.base.steps)
        return self._obs(info), reward, terminated, truncated, info

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()
