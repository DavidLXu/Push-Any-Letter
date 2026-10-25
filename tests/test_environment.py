import json
import string
import numpy as np
import pytest
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from gymnasium.utils.env_checker import check_env
from push_any_letter import PushLetterEnv
from push_any_letter.geometry import make_glyph, polygons, transform
from push_any_letter.recording import Recorder, flatten


@pytest.mark.parametrize('letter', string.ascii_letters)
def test_glyph_mesh_and_initial_state(letter):
    env = PushLetterEnv(letter)
    obs, info = env.reset(seed=13)
    assert env.observation_space.contains(obs)
    mesh = unary_union([Polygon(t) for t in env.glyph.triangles])
    assert mesh.symmetric_difference(env.glyph.geometry).area < 1e-5
    current = transform(env.glyph.geometry, (*env.block.position, env.block.angle))
    assert current.distance(Point(env.agent.position)) > env.pusher_radius
    for _ in range(5):
        obs, _, _, _, info = env.step(obs['agent_pos'])
        assert np.isfinite(flatten(obs)).all()
    env.close()


@pytest.mark.parametrize('letter', 'ABDOQRabdegopq')
def test_holes_are_not_collision_solids(letter):
    env = PushLetterEnv(letter)
    env.reset(seed=2)
    holes = [Polygon(r) for p in polygons(env.glyph.geometry) for r in p.interiors]
    assert holes
    for hole in holes:
        p = hole.representative_point()
        world = env.block.local_to_world((p.x, p.y))
        assert all(s.point_query(world).distance > 0 for s in env.block_shapes)


def test_gym_contract():
    check_env(PushLetterEnv(), skip_render_check=True)


def test_contact_pushes_and_rotates():
    env = PushLetterEnv('T')
    env.reset(seed=0)
    env.block.position = (320, 320)
    env.block.angle = 0
    env.agent.position = (180, 275)
    env.space.reindex_shapes_for_body(env.block)
    start = np.array(env.block.position)
    for _ in range(35):
        env.step([440, 275])
    assert np.linalg.norm(np.array(env.block.position) - start) > 30
    assert abs(env.block.angle) > 0.1


def test_coverage_and_time_limit():
    env = PushLetterEnv(max_episode_steps=1)
    env.reset(seed=0)
    _, _, done, truncated, _ = env.step([100, 530])
    assert not done and truncated
    env.reset(seed=0)
    env.block.position = tuple(env.goal_pose[:2])
    env.block.angle = env.goal_pose[2]
    env.space.reindex_shapes_for_body(env.block)
    _, reward, done, truncated, info = env.step([100, 530])
    assert done and not truncated and reward == 1
    assert info['coverage'] > 0.999


def test_seeded_recording_replay(tmp_path):
    env = PushLetterEnv('B')
    obs, _ = env.reset(seed=123)
    recorder = Recorder(env, 123)
    for action in np.random.default_rng(99).uniform(28, 612, size=(60,2)):
        next_obs, reward, done, trunc, _ = env.step(action)
        recorder.append(obs, action, next_obs, reward, done, trunc)
        obs = next_obs
    path = recorder.save(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        other = PushLetterEnv(metadata['letter'])
        other.reset(seed=metadata['seed'])
        for i, action in enumerate(data['actions']):
            actual, *_ = other.step(action)
            np.testing.assert_allclose(flatten(actual), data['next_observations'][i], atol=1e-3)


def test_hard_push_stays_in_arena():
    env = PushLetterEnv('I')
    env.reset(seed=0)
    env.block.position = (550, 320)
    env.block.angle = 0
    env.agent.position = (460,320)
    env.space.reindex_shapes_for_body(env.block)
    for _ in range(240):
        env.step([612,320])
    geom = transform(env.glyph.geometry, (*env.block.position, env.block.angle))
    assert geom.bounds[2] < 632
    assert env.agent.position.x < 632
    assert geom.distance(Point(env.agent.position)) > env.pusher_radius - 2


def test_rgb_observation():
    env = PushLetterEnv('O', render_mode='rgb_array')
    env.reset(seed=7)
    image = env.render()
    assert image.shape == (640, 640, 3) and image.dtype == np.uint8


@pytest.mark.parametrize('action', [[float('nan'), 4], [1,2,3]])
def test_invalid_action(action):
    env = PushLetterEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
