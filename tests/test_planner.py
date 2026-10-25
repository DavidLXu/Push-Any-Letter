import json
import numpy as np
import pytest
from push_any_letter import PushLetterEnv
from push_any_letter.planner import ContactPlanner, clone_env
from push_any_letter.recording import Recorder, flatten
from push_any_letter.dataset import generate_episode, validate_episode


def test_planner_copies_do_not_move_live_scene():
    env=PushLetterEnv('T'); env.reset(seed=42)
    before=flatten(env._obs())
    simulated=clone_env(env)
    simulated._advance([400,300])
    np.testing.assert_array_equal(before,flatten(env._obs()))
    assert simulated.space is not env.space
    assert simulated.agent is not env.agent
    assert simulated.block is not env.block


@pytest.fixture(scope='module')
def successful_episode(tmp_path_factory):
    output=tmp_path_factory.mktemp('planner')
    result=generate_episode('T',42,output,image_size=32)
    assert result['valid'] and result['max_state_error'] == 0
    assert result['final_coverage'] >= 0.95
    return output/result['path']


def test_successful_rollout_has_aligned_images(successful_episode):
    with np.load(successful_episode,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        assert meta['source']=='model_predictive_planner'
        assert meta['privileged_state'] is True
        assert data['images'].shape==(len(data['actions']),32,32,3)
        assert data['images'].dtype==np.uint8
        assert data['terminated'][-1] and not data['terminated'][:-1].any()
        np.testing.assert_array_equal(data['observations'][1:],data['next_observations'][:-1])
        env=PushLetterEnv('T');env.reset(seed=42)
        np.testing.assert_array_equal(data['observations'][0],flatten(env._obs()))


def test_validator_rejects_tampered_states(successful_episode,tmp_path):
    with np.load(successful_episode,allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    arrays['next_observations'][10,2]+=20
    path=tmp_path/'tampered.npz';np.savez_compressed(path,**arrays)
    result=validate_episode(path)
    assert not result['valid'] and result['max_state_error'] >= 19


def test_validator_rejects_incomplete_episode(tmp_path):
    env=PushLetterEnv('A');obs,_=env.reset(seed=42)
    recorder=Recorder(env,42)
    nxt,reward,done,trunc,_=env.step([100,530])
    recorder.append(obs,[100,530],nxt,reward,done,trunc)
    path=recorder.save(tmp_path)
    result=validate_episode(path)
    assert not result['valid'] and not result['success']


def test_high_friction_episode_replays_and_rejects_wrong_physics(tmp_path):
    result = generate_episode('T', 42, tmp_path, image_size=16, block_friction=1.4)
    assert result['valid'] and result['max_state_error'] == 0
    path = tmp_path/result['path']
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    meta = json.loads(str(arrays['metadata']))
    assert meta['physics_parameters']['block_friction'] == 1.4
    meta['physics_parameters']['block_friction'] = .7
    arrays['metadata'] = json.dumps(meta)
    wrong = tmp_path/'wrong_physics.npz'
    np.savez_compressed(wrong, **arrays)
    invalid = validate_episode(wrong)
    assert not invalid['valid'] and invalid['max_state_error'] > 1e-3
