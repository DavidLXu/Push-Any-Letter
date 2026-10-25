import math
import numpy as np
from shapely.geometry import Point
from push_any_letter.batch import sample_start
from push_any_letter.geometry import default_font, make_glyph, transform
from push_any_letter.env import PushLetterEnv
from push_any_letter.recording import Recorder, flatten
from push_any_letter.dataset import validate_episode


def test_all_goal_samplers_keep_full_glyph_clear_of_walls():
    import string
    font = default_font()
    for letter in string.ascii_letters:
        glyph = make_glyph(letter, font)
        for angle in range(0, 360, 30):
            options = sample_start(letter, angle, ord(letter)*1000+angle, font)
            geom = transform(glyph.geometry, options['initial_object_pose'])
            x0, y0, x1, y1 = geom.bounds
            assert min(x0, y0) >= 60-1e-8 and max(x1, y1) <= 580+1e-8
            assert geom.distance(Point(options['initial_agent_pos'])) >= 20
            assert options['goal_pose'] == [320., 320., math.radians(angle)]
            target = transform(glyph.geometry, options['goal_pose'])
            assert geom.intersection(target).area/glyph.geometry.area < .95


def test_custom_reset_is_recorded_and_replayed(tmp_path):
    env = PushLetterEnv('T')
    options = sample_start('T', 120, 17, default_font())
    obs, _ = env.reset(seed=17, options=options)
    recorder = Recorder(env, 17)
    for _ in range(5):
        action = np.array([100, 100], dtype=np.float32)
        nxt, reward, done, trunc, _ = env.step(action)
        recorder.append(obs, action, nxt, reward, done, trunc)
        obs = nxt
    path = recorder.save(tmp_path)
    result = validate_episode(path)
    assert result['max_state_error'] == 0
    assert result['rewards_match'] and result['termination_flags_match']
    assert not result['success']
