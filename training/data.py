"""All demonstrations for training; unseen-angle, new-start test tasks."""
import argparse
import hashlib
import json
import string
from pathlib import Path
import numpy as np

LETTERS = string.ascii_uppercase + string.ascii_lowercase
OBS_DIM = 62  # ten continuous features plus 52-way letter identity


def test_cases(font_path):
    from push_any_letter.batch import sample_start
    rng = np.random.default_rng(20261027)
    angles = []
    while len(angles) < 10:
        angle = float(rng.uniform(0,360))
        # Keep visibly separated from the training grid and other test angles.
        if min(angle % 30, 30-angle % 30) < 3: continue
        if any(abs((angle-a+180)%360-180) < 3 for a in angles): continue
        angles.append(angle)
    cases = []
    for letter_id, letter in enumerate(LETTERS):
        for slot, angle in enumerate(angles):
            seed = 2026102700+letter_id*10+slot
            cases.append({'letter':letter,'letter_id':letter_id,'angle':angle,'slot':slot,
                'metadata':{'letter':letter,'seed':seed,'font_path':font_path,
                    'physics_parameters':{'block_friction':1.4},
                    'reset_options':sample_start(letter,angle,seed,font_path)}})
    return angles, cases


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def encode_state(raw, letter_ids):
    raw = np.asarray(raw, dtype=np.float32)
    xy = raw[..., [0, 1, 2, 3, 5, 6]] / 320. - 1.
    angles = raw[..., [4, 7]]
    identity = np.eye(52, dtype=np.float32)[letter_ids]
    identity = np.broadcast_to(identity, raw.shape[:-1] + (52,))
    return np.concatenate([xy, np.sin(angles), np.cos(angles), identity], axis=-1).astype(np.float32)


def prepare(source, destination, font_path=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (source/'episodes.jsonl').exists():
        source_index = source/'episodes.jsonl'
        manifest = [json.loads(line) for line in source_index.read_text().splitlines()]
        index = [{**r, 'valid': True, 'folder': str(source/'episodes'),
                  'result': {'path': r['member']}} for r in manifest]
    else:
        status = json.loads((source/'status.json').read_text())
        assert status['state'] == 'complete' and status['accepted'] == 6240
        assert not status['worker_errors'] and not status['duplicate_trajectories']
        source_index = source/'index.json'
        index = json.loads(source_index.read_text())
    assert len(index) == 6240 and all(r['valid'] for r in index)
    destination.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source_index.read_bytes()).hexdigest()
    if (destination/'ready.json').exists():
        assert json.loads((destination/'ready.json').read_text())['source_index_sha256'] == source_hash
        return
    groups = {}
    for row in index:
        groups.setdefault((row['letter'], row['angle_degrees']), []).append(row)
    assert len(groups) == 624
    episodes, offset = [], 0
    hashes = set()
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: r['slot'])
        assert len(rows) == 10 and len({r['slot'] for r in rows}) == 10
        for row in rows:
            path = Path(row['folder'])/row['result']['path']
            with np.load(path, allow_pickle=False) as d:
                meta = json.loads(str(d['metadata']))
                from push_any_letter.fonts import resolve_recording_font
                meta['font_path'] = resolve_recording_font(meta, font_path)
                actions = d['actions']
                obs = d['observations']
                assert meta['validation']['valid'] and meta['physics_parameters']['block_friction'] == 1.4
                assert meta['letter'] == key[0] and len(actions) > 0
                assert np.isfinite(obs).all() and np.isfinite(actions).all()
                assert np.allclose(meta['reset_options']['goal_pose'], [320,320,np.deg2rad(key[1])])
                digest = hashlib.sha256(obs.tobytes()+actions.tobytes()).hexdigest()
                assert digest not in hashes, 'Duplicate episode'
                hashes.add(digest)
                n = len(actions)
            episodes.append({'path': str(path), 'offset': offset, 'length': n,
                             'letter_id': LETTERS.index(key[0]), 'letter': key[0], 'angle': key[1],
                             'slot': row['slot'], 'split': 'train', 'metadata': meta})
            offset += n
    # Images are deliberately never decompressed; only state/action members are read.
    obs_out = np.lib.format.open_memmap(destination/'observations.npy', mode='w+', dtype='float32', shape=(offset,8))
    act_out = np.lib.format.open_memmap(destination/'actions.npy', mode='w+', dtype='float32', shape=(offset,2))
    for ep in episodes:
        with np.load(ep['path'], allow_pickle=False) as d:
            start, n = ep['offset'], ep['length']
            obs_out[start:start+n] = d['observations']
            act_out[start:start+n] = d['actions']
    obs_out.flush(); act_out.flush()
    atomic_json(destination/'episodes.json', episodes)
    angles, cases = test_cases(episodes[0]['metadata']['font_path'])
    atomic_json(destination/'test_cases.json', cases)
    atomic_json(destination/'ready.json', {'source_index_sha256': source_hash, 'source': str(source),
        'episodes': len(episodes), 'steps': offset, 'test_seed': 20261027,
        'splits': {s: sum(e['split'] == s for e in episodes) for s in ['train','val','test']},
        'test_cases':len(cases),'test_angles_degrees':angles,
        'checkpoint_selection':'Fixed final training step; no validation or test selection',
        'normalization': 'positions /320 -1; sin/cos angles; 52-way identity; actions (a-320)/292',
        'block_friction': 1.4})


class StateData:
    def __init__(self, root, split):
        root = Path(root)
        assert (root/'ready.json').exists()
        self.obs = np.load(root/'observations.npy', mmap_mode='r')
        self.actions = np.load(root/'actions.npy', mmap_mode='r')
        self.episodes = [e for e in json.loads((root/'episodes.json').read_text()) if e['split'] == split]
        self.offsets = np.array([e['offset'] for e in self.episodes])
        self.lengths = np.array([e['length'] for e in self.episodes])
        self.letters = np.array([e['letter_id'] for e in self.episodes])

    def windows(self, ids, times, horizon):
        offsets, lengths = self.offsets[ids], self.lengths[ids]
        history = np.maximum(times[:,None] + np.arange(-1,1), 0)
        raw = self.obs[offsets[:,None]+history]
        obs = encode_state(raw, self.letters[ids,None])
        future = times[:,None] + np.arange(horizon)
        mask = future < lengths[:,None]
        indices = offsets[:,None] + np.minimum(future, lengths[:,None]-1)
        actions = (self.actions[indices]-320.)/292.
        return obs, actions.astype(np.float32), mask

    def sample(self, rng, count, horizon):
        # Uniform episodes keeps each letter/goal equally weighted despite duration.
        ids = rng.integers(len(self.episodes), size=count)
        times = (rng.random(count)*self.lengths[ids]).astype(np.int64)
        return self.windows(ids, times, horizon)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source'); p.add_argument('destination')
    p.add_argument('--font', help='Local font matching recorded geometry')
    a = p.parse_args(); prepare(a.source, a.destination, a.font)
