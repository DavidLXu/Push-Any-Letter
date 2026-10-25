"""Contact-sampling model predictive controller. Uses simulation state, not pixels.

Candidates are rolled out in independent copies; only actions are applied to the
real environment. No object pose writes or resets are permitted within an episode.
"""
import copy
import math
import numpy as np
import pymunk
from shapely.geometry import LineString, Point
from .geometry import transform


def clone_env(env):
    out = copy.copy(env)
    out.space = env.space.copy()
    dynamic = [b for b in out.space.bodies if b.body_type == pymunk.Body.DYNAMIC]
    out.agent = next(b for b in dynamic if any(isinstance(s, pymunk.Circle) for s in b.shapes))
    out.block = next(b for b in dynamic if b is not out.agent)
    out.block_shapes = list(out.block.shapes)
    out.agent_shape = next(iter(out.agent.shapes))
    out.window = None
    out.render_mode = None
    return out


def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi


class ContactPlanner:
    def __init__(self, env, seed=0, contacts=24):
        self.rng = np.random.default_rng(seed)
        self.contacts = contacts
        self.iteration = 0
        self.symmetries = [0.0]
        for angle in [math.pi/2, math.pi, 3*math.pi/2]:
            rotated = transform(env.glyph.geometry, (0,0,angle))
            if rotated.intersection(env.glyph.geometry).area / env.glyph.geometry.area > 0.999:
                self.symmetries.append(angle)
        # RMS surface displacement gives rotations comparable weight to translation.
        self.radius2 = env.block.moment / env.block.mass

    def cost(self, env):
        delta = np.asarray(env.block.position) - env.goal_pose[:2]
        angle = min(abs(wrap(env.block.angle-env.goal_pose[2]-s)) for s in self.symmetries)
        return float(delta @ delta + 2*self.radius2*(1-math.cos(angle)))

    @staticmethod
    def _run(env, actions):
        for a in actions:
            env._advance(a)

    @staticmethod
    def _move(env, waypoint, max_steps=80):
        """Servo to a free-space waypoint; return actual applied targets."""
        waypoint = np.clip(waypoint, 28, 612).astype(np.float32)
        actions = []
        for _ in range(max_steps):
            if np.linalg.norm(np.asarray(env.agent.position)-waypoint) < 1.0 and env.agent.velocity.length < 8:
                break
            env._advance(waypoint)
            actions.append(waypoint.copy())
        return actions

    def _approach(self, env, contact, outer_radius):
        """Route outside a conservative circle; direct shortcuts are collision tested."""
        center = np.array(env.block.position)
        obstacle = transform(env.glyph.geometry, (*center, env.block.angle)).buffer(env.pusher_radius+1)
        start = np.array(env.agent.position)
        if not LineString([start, contact]).intersects(obstacle):
            return self._move(env, contact)
        # Exit radially, then take the shorter of the two exterior arcs.
        a0 = math.atan2(*(start-center)[::-1])
        a1 = math.atan2(*(contact-center)[::-1])
        delta = wrap(a1-a0)
        circle_points = [center + outer_radius*np.array([math.cos(a0+t*delta), math.sin(a0+t*delta)])
                         for t in np.linspace(0, 1, max(2, int(abs(delta)/0.22)+2))]
        points = [*circle_points, contact]
        if any((p < 29).any() or (p > 611).any() for p in points):
            # Try the other arc when a wall blocks the shorter path.
            delta = delta - math.copysign(2*math.pi, delta)
            points = [center + outer_radius*np.array([math.cos(a0+t*delta), math.sin(a0+t*delta)])
                      for t in np.linspace(0,1,max(2,int(abs(delta)/0.22)+2))] + [contact]
            if any((p < 29).any() or (p > 611).any() for p in points):
                return None
        actions = []
        for point in points:
            actions += self._move(env, point)
        return actions

    def plan(self, env):
        self.iteration += 1
        start_cost = self.cost(env)
        error = math.sqrt(start_cost)
        # Large moves early; small pixel-scale adjustments near the goal.
        lengths = [0.5, 1.5, 4] if error < 5 else ([2, 6, 12] if error < 15 else ([12, 30, 55] if error < 65 else [30, 65, 100]))
        center = np.array(env.block.position)
        shape = transform(env.glyph.geometry, (*center, env.block.angle))
        inflated = shape.buffer(env.pusher_radius + 3)
        radius = max(np.linalg.norm(np.array(p)-center) for p in inflated.convex_hull.exterior.coords) + 15
        best = None
        # Let existing momentum settle before unnecessarily changing contact side.
        coast = clone_env(env)
        hold = np.asarray(coast.agent.position, dtype=np.float32)
        for n in range(1, 21):
            coast._advance(hold)
            cost = self.cost(coast)
            if cost < start_cost - max(0.05, 0.001*start_cost):
                if best is None or cost < best[0]:
                    best = (cost, [hold.copy() for _ in range(n)], cost)
        offset = self.rng.uniform(0, 2*math.pi/self.contacts)
        for angle in np.linspace(0, 2*math.pi, self.contacts, endpoint=False)+offset:
            radial = np.array([math.cos(angle), math.sin(angle)])
            ray = LineString([center+radial*500, center])
            crossing = ray.intersection(inflated.boundary)
            if crossing.is_empty:
                continue
            if crossing.geom_type == 'Point':
                points = [crossing]
            elif crossing.geom_type == 'MultiPoint':
                points = list(crossing.geoms)
            else:
                continue
            contact = np.array(max(points, key=lambda p: np.linalg.norm(np.array(p.coords[0])-center)).coords[0])
            if (contact < 29).any() or (contact > 611).any():
                continue
            approach_env = clone_env(env)
            approach_actions = self._approach(approach_env, contact, radius)
            if approach_actions is None:
                continue
            for skew in [-1.15, -0.6, 0, 0.6, 1.15]:
                direction = -np.array([math.cos(angle+skew), math.sin(angle+skew)])
                # Test push lengths independently to preserve controller dynamics.
                for length in lengths:
                    endpoint = contact + direction*length
                    if (endpoint < 28).any() or (endpoint > 612).any():
                        continue
                    sim = clone_env(approach_env)
                    # A ramped position target avoids violent jumps at contact.
                    n = max(5, math.ceil(length/6))
                    actions = [np.asarray(contact + direction*length*t, dtype=np.float32)
                               for t in np.linspace(0,1,n+1)[1:]]
                    actions += [actions[-1]]*5
                    for k, action in enumerate(actions):
                        sim._advance(action)
                        cost = self.cost(sim)
                        # Keep the best prefix; a long push can overshoot the goal.
                        score = cost + 0.005*len(approach_actions)
                        if best is None or score < best[0]:
                            best = (score, approach_actions + actions[:k+1], cost)
        if best is None:
            return [], {"start_cost": start_cost, "predicted_cost": start_cost}
        return best[1], {"start_cost": start_cost, "predicted_cost": best[2]}
