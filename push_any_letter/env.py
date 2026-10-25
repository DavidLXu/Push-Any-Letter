"""Deterministic fixed-step, CPU-only Gymnasium environment."""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pymunk
from shapely.geometry import Point, Polygon
from .geometry import make_glyph, transform


class PushLetterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    size = 640
    control_hz = 20
    physics_hz = 200
    pusher_radius = 16

    def __init__(self, letter="T", font_path=None, render_mode=None, max_episode_steps=600,
                 block_friction=0.7):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be human, rgb_array, or None")
        self.letter, self.font_path = letter, font_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        if not np.isfinite(block_friction) or block_friction < 0:
            raise ValueError("block_friction must be finite and nonnegative")
        self.block_friction = float(block_friction)
        self.glyph = make_glyph(letter, font_path)
        self.action_space = spaces.Box(28, 612, shape=(2,), dtype=np.float32)
        pose_space = lambda: spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "object_pose": pose_space(), "goal_pose": pose_space(),
        })
        self.window = None
        self.clock = None
        self.space = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        letter = options.get("letter", self.letter)
        glyph = make_glyph(letter, self.font_path)
        self.letter, self.glyph = letter, glyph
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.12
        self.space.iterations = 30
        self.space.collision_slop = 0.05
        for a, b in [((10,10),(630,10)), ((630,10),(630,630)), ((630,630),(10,630)), ((10,630),(10,10))]:
            wall = pymunk.Segment(self.space.static_body, a, b, 2)
            wall.friction = 0.7
            wall.elasticity = 0
            self.space.add(wall)
        mass = 4.0
        moment = sum(pymunk.moment_for_poly(mass * Polygon(t).area / glyph.geometry.area, t) for t in glyph.triangles)
        self.block = pymunk.Body(mass, moment)
        self.space.add(self.block)
        self.block_shapes = []
        for triangle in glyph.triangles:
            shape = pymunk.Poly(self.block, triangle)
            shape.friction, shape.elasticity = self.block_friction, 0
            self.space.add(shape)
            self.block_shapes.append(shape)
        self.agent = pymunk.Body(1.0, float("inf"))
        self.agent_shape = pymunk.Circle(self.agent, self.pusher_radius)
        self.agent_shape.friction = 0.6
        self.agent_shape.elasticity = 0
        self.space.add(self.agent, self.agent_shape)
        # Separated regions guarantee no initial agent/block or wall overlap.
        self.block.position = (self.np_random.uniform(220, 350), self.np_random.uniform(360, 410))
        self.block.angle = self.np_random.uniform(-math.pi, math.pi)
        self.agent.position = (100, 530)
        self.goal_pose = np.array([self.np_random.uniform(320, 420), self.np_random.uniform(200, 260), self.np_random.uniform(-math.pi, math.pi)], dtype=np.float64)
        if "initial_object_pose" in options:
            pose = np.asarray(options["initial_object_pose"], dtype=float)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError("initial_object_pose must contain three finite values")
            self.block.position, self.block.angle = tuple(pose[:2]), float(pose[2])
        if "initial_agent_pos" in options:
            pos = np.asarray(options["initial_agent_pos"], dtype=float)
            if pos.shape != (2,) or not np.isfinite(pos).all():
                raise ValueError("initial_agent_pos must contain two finite values")
            self.agent.position = tuple(pos)
        if "goal_pose" in options:
            self.goal_pose = np.asarray(options["goal_pose"], dtype=float).copy()
            if self.goal_pose.shape != (3,) or not np.isfinite(self.goal_pose).all():
                raise ValueError("goal_pose must contain three finite values")
        self.reset_options = {key: np.asarray(options[key]).tolist() for key in
                              ("initial_object_pose", "initial_agent_pos", "goal_pose") if key in options}
        self.space.reindex_shapes_for_body(self.block)
        self.space.reindex_shapes_for_body(self.agent)
        self.goal_geometry = transform(self.glyph.geometry, self.goal_pose)
        self.steps = 0
        self.last_action = np.array(self.agent.position, dtype=np.float32)
        self.best_coverage = 0.0
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return self._obs(), info

    def _obs(self):
        return {"agent_pos": np.asarray(self.agent.position, dtype=np.float32),
                "object_pose": np.array([*self.block.position, self.block.angle % (2*math.pi)], dtype=np.float32),
                "goal_pose": np.array([*self.goal_pose[:2], self.goal_pose[2] % (2*math.pi)], dtype=np.float32)}

    def _info(self):
        current = transform(self.glyph.geometry, (*self.block.position, self.block.angle))
        coverage = float(np.clip(current.intersection(self.goal_geometry).area / self.glyph.geometry.area, 0, 1))
        self.best_coverage = max(self.best_coverage, coverage)
        return {"coverage": coverage, "is_success": coverage >= 0.95,
                "letter": self.letter, "steps": self.steps, "best_coverage": self.best_coverage}

    def _advance(self, action):
        """Advance the unchanged dynamics without computing observations/reward."""
        if self.space is None:
            raise RuntimeError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be two finite target coordinates.")
        target = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = target.copy()
        dt = 1 / self.physics_hz
        for _ in range(self.physics_hz // self.control_hz):
            # Dynamic, force-limited servo: contacts can stop the pusher at walls.
            error = target - np.asarray(self.agent.position)
            desired_v = error * 9
            speed = np.linalg.norm(desired_v)
            if speed > 260:
                desired_v *= 260 / speed
            force = (desired_v - np.asarray(self.agent.velocity)) * 35
            magnitude = np.linalg.norm(force)
            if magnitude > 9000:
                force *= 9000 / magnitude
            self.agent.force = tuple(force)
            self.space.step(dt)
        self.steps += 1

    def step(self, action):
        self._advance(action)
        info = self._info()
        reward = min(info["coverage"] / 0.95, 1.0)
        terminated = info["is_success"]
        truncated = self.steps >= self.max_episode_steps and not terminated
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        import pygame
        from .render import draw_arena
        if self.space is None:
            raise RuntimeError("Call reset() before render().")
        surface = draw_arena(self)
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((640, 640))
                pygame.display.set_caption("Push Any Letter")
                self.clock = pygame.time.Clock()
            pygame.event.pump()
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.control_hz)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            self.window = None
