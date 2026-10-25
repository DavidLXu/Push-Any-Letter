"""Resumable CPU batch: center targets, random collision-free starts, verified demos."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import time
import traceback

import numpy as np
from shapely.geometry import Point
from .geometry import make_glyph, transform
from .dataset import generate_episode, font_hash


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def sample_start(letter, angle_degrees, seed, font_path, clearance=48):
    rng = np.random.default_rng(seed)
    glyph = make_glyph(letter, font_path)
    goal = [320.0, 320.0, float(np.deg2rad(angle_degrees))]
    target = transform(glyph.geometry, goal)
    for _ in range(10000):
        angle = float(rng.uniform(0, 2*np.pi))
        rotated = transform(glyph.geometry, (0, 0, angle))
        x0, y0, x1, y1 = rotated.bounds
        low = np.array([12+clearance-x0, 12+clearance-y0])
        high = np.array([628-clearance-x1, 628-clearance-y1])
        center = rng.uniform(low, high)
        pose = [float(center[0]), float(center[1]), angle]
        geom = transform(glyph.geometry, pose)
        if geom.intersection(target).area / glyph.geometry.area >= .95:
            continue
        for _ in range(1000):
            agent = rng.uniform(44, 596, size=2)
            if geom.distance(Point(agent)) >= 20:
                return {"initial_object_pose": pose, "initial_agent_pos": agent.tolist(), "goal_pose": goal}
    raise RuntimeError("Unable to sample a collision-free initial state")


def run_slot(job):
    letter, angle, slot, job_index, settings = job
    folder = Path(settings['output']) / 'jobs' / f'U{ord(letter):04X}' / f'angle{angle:03d}' / f'traj{slot:02d}'
    result_path = folder / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for attempt in range(settings['max_attempts']):
        attempt_path = folder / f'attempt_{attempt:03d}.json'
        if attempt_path.exists():
            result = json.loads(attempt_path.read_text())
        else:
            seed = settings['seed'] + job_index * 1000 + attempt
            try:
                reset = sample_start(letter, angle, seed, settings['font_path'])
                result = generate_episode(letter, seed, folder,
                    contacts=settings['contacts'], iterations=settings['iterations'],
                    max_steps=settings['max_steps'], image_size=settings['image_size'],
                    font_path=settings['font_path'], reset_options=reset,
                    block_friction=settings['block_friction'])
                result['initial_state'] = reset
                result['initial_state_sha256'] = hashlib.sha256(json.dumps(reset, sort_keys=True).encode()).hexdigest()
                if result['valid']:
                    with np.load(folder / result['path'], allow_pickle=False) as data:
                        result['trajectory_sha256'] = hashlib.sha256(data['observations'].tobytes()+data['actions'].tobytes()).hexdigest()
            except Exception:
                result = {'valid': False, 'seed': seed, 'error': traceback.format_exc()}
            save_json(attempt_path, result)
        records.append(result)
        if result['valid']:
            break
    summary = {'letter': letter, 'angle_degrees': angle, 'slot': slot,
               'valid': bool(records[-1]['valid']), 'attempts': len(records),
               'result': records[-1], 'folder': str(folder)}
    save_json(result_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--font', required=True)
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--contacts', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--max-steps', type=int, default=2400)
    parser.add_argument('--image-size', type=int, default=96)
    parser.add_argument('--block-friction', type=float, default=0.7)
    parser.add_argument('--limit', type=int, default=0, help='Smoke-test only; 0 runs every slot')
    args = parser.parse_args()
    if not np.isfinite(args.block_friction) or args.block_friction < 0:
        parser.error('block-friction must be finite and nonnegative')
    if min(args.workers, args.max_attempts, args.contacts, args.iterations, args.max_steps) < 1 or args.max_attempts > 1000:
        parser.error('Counts must be positive and max-attempts <= 1000')
    config = json.loads(Path(args.config).read_text())
    if config['center_x'] != [320] or config['center_y'] != [320]:
        parser.error('This generator requires center targets')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent accidental concurrent writers to the same output (Linux/macOS).
    import fcntl
    lock = (output / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = vars(args).copy()
    settings['font_path'] = str(Path(args.font).resolve())
    settings['font_sha256'] = font_hash(settings['font_path'])
    settings['output'] = str(output)
    settings['sampling_config'] = config
    settings['physics_profile'] = f'damping=0.12 block_friction={args.block_friction} agent_friction=0.6'
    settings['success_criterion'] = 'coverage >= 0.95; geometry symmetries accepted'
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Existing output uses a different configuration')
    save_json(manifest, settings)
    jobs = []
    for letter in config['letters']:
        for angle in config['angles_degrees']:
            for slot in range(config['trajectories_per_goal']):
                jobs.append((letter, angle, slot, len(jobs), settings))
    random.Random(args.seed).shuffle(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    started = time.monotonic()
    results = []
    errors = []
    hashes = set()
    duplicates = []
    def report(state):
        accepted = sum(r['valid'] for r in results)
        counts = {}
        for r in results:
            key = f"{r['letter']}:{r['angle_degrees']}"
            counts[key] = counts.get(key, 0) + int(r['valid'])
        status = {'state': state, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'pid': os.getpid(), 'workers': args.workers, 'requested': len(jobs),
                  'finished_slots': len(results), 'accepted': accepted,
                  'exhausted_slots': sum(not r['valid'] for r in results),
                  'attempts_in_finished_slots': sum(r['attempts'] for r in results),
                  'worker_errors': errors, 'duplicate_trajectories': duplicates,
                  'elapsed_seconds': round(time.monotonic()-started, 1), 'per_goal_accepted': counts}
        save_json(output / 'status.json', status)
        print(json.dumps({k:v for k,v in status.items() if k != 'per_goal_accepted'}), flush=True)
    report('running')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(run_slot, job): job[:3] for job in jobs}
        last_report = 0
        while pending:
            finished, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in finished:
                job = pending.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    if result['valid']:
                        digest = result['result']['trajectory_sha256']
                        if digest in hashes:
                            duplicates.append(list(job))
                        hashes.add(digest)
                except Exception:
                    errors.append({'job': job, 'error': traceback.format_exc()})
            if time.monotonic()-last_report >= 30:
                report('running')
                last_report = time.monotonic()
    complete = len(results) == len(jobs) and all(r['valid'] for r in results) and not errors and not duplicates
    save_json(output / 'index.json', results)
    report('complete' if complete else 'incomplete')
    raise SystemExit(0 if complete else 2)


if __name__ == '__main__':
    main()
