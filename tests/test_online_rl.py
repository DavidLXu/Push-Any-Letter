"""Physical reset, deterministic dynamics and PPO bootstrap contract checks."""
import os
import numpy as np
import pytest
from shapely.geometry import Point
from rl.env import OnlineLetterEnv, LETTERS, shaped_reward
from push_any_letter.geometry import transform

FONT = os.environ.get('PUSH_LETTER_FONT')


def test_all_letters_safe_reset_and_finite_state():
    env = OnlineLetterEnv(font_path=FONT)
    for full in (False, True):
        for i, letter in enumerate(LETTERS):
            obs, info = env.reset(seed=500+i, options={'letter':letter, 'full':full})
            geom = transform(env.base.glyph.geometry, (*env.base.block.position, env.base.block.angle))
            bounds = geom.bounds
            assert bounds[0] >= 60-1e-6 and bounds[1] >= 60-1e-6
            assert bounds[2] <= 580+1e-6 and bounds[3] <= 580+1e-6
            assert geom.distance(Point(*env.base.agent.position)) >= 20-1e-6
            assert info['coverage'] < .95
            assert obs.shape == (72,) and np.isfinite(obs).all()
            for _ in range(3):
                obs, reward, _, _, _ = env.step([.3,-.4])
                assert np.isfinite(reward) and np.isfinite(obs).all()
    env.close()


def test_seed_and_dynamics_determinism():
    a, b = OnlineLetterEnv(font_path=FONT), OnlineLetterEnv(font_path=FONT)
    np.testing.assert_array_equal(a.reset(seed=42)[0], b.reset(seed=42)[0])
    for action in np.random.default_rng(3).uniform(-1,1,(30,2)):
        left, right = a.step(action), b.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:4] == right[1:4]
    a.close(); b.close()


def test_potential_telescope_and_success_boundary():
    gamma = .999
    potentials = [-3., -2., -.2, 1., 3.8]
    rewards = [shaped_reward(potentials[i], potentials[i+1], False, np.zeros(2), gamma)+.01 for i in range(4)]
    assert sum(gamma**i*r for i,r in enumerate(rewards)) == pytest.approx(-potentials[0]+gamma**4*potentials[-1])
    assert shaped_reward(3.8,4.,True,np.zeros(2),gamma) == pytest.approx(20-.01-3.8)


def test_grouped_vec_matches_serial_and_preserves_terminal_observation():
    pytest.importorskip('stable_baselines3')
    from rl.vec_env import GroupedVecEnv
    vec = GroupedVecEnv(4,2,font_path=FONT,max_episode_steps=2)
    serial = [OnlineLetterEnv(font_path=FONT,max_episode_steps=2) for _ in range(4)]
    try:
        vec.seed(100)
        np.testing.assert_array_equal(vec.reset(), np.stack([e.reset(seed=100+i)[0] for i,e in enumerate(serial)]))
        for _ in range(2):
            actions = np.tile([.2,-.1],(4,1))
            obs, rewards, dones, infos = vec.step(actions)
            for i,env in enumerate(serial):
                o,r,t,tr,info = env.step(actions[i])
                assert rewards[i] == pytest.approx(r)
                assert dones[i] == (t or tr)
                if dones[i]:
                    np.testing.assert_array_equal(infos[i]['terminal_observation'],o)
                    assert infos[i]['TimeLimit.truncated'] == (tr and not t)
                else:
                    np.testing.assert_array_equal(obs[i],o)
        vec.env_method('set_difficulty',.5)
        assert vec.get_attr('difficulty') == [.5]*4
    finally:
        vec.close()
        for env in serial: env.close()
