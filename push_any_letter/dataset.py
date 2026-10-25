"""Generate and independently replay-validate planner demonstrations."""
import argparse
import hashlib
import json
import platform
import string
import time
from pathlib import Path
import numpy as np
import pygame
import pymunk
from shapely.geometry import Point
from .env import PushLetterEnv
from .planner import ContactPlanner
from .recording import Recorder, flatten
from .geometry import transform
from .fonts import resolve_recording_font


def font_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_episode(path, image_size=0, font_path=None):
    """Replay only saved actions from reset; never restore saved object poses."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        arrays = {k: data[k].copy() for k in data.files}
    meta['font_path'] = resolve_recording_font(meta, font_path)
    env = PushLetterEnv(meta['letter'], font_path=meta['font_path'],
                        **meta.get('physics_parameters', {}),
                        max_episode_steps=meta.get('max_episode_steps', 600))
    obs, info = env.reset(seed=meta['seed'], options=meta.get('reset_options'))
    max_error = 0.0
    max_penetration = 0.0
    in_bounds = True
    images = []
    rewards_ok = True
    flags_ok = True
    early_done = False
    count = len(arrays['actions'])
    if count == 0:
        raise ValueError('Empty episode.')
    if arrays['actions'].shape != (count,2) or arrays['observations'].shape != (count,8) or arrays['next_observations'].shape != (count,8):
        raise ValueError('Invalid observation/action dimensions.')
    if not all(len(arrays[k]) == count for k in ['rewards','terminated','truncated']):
        raise ValueError('Unequal episode array lengths.')
    finite = all(np.isfinite(arrays[k]).all() for k in ['observations','actions','next_observations','rewards'])
    actions_in_bounds = bool(((arrays['actions'] >= 28) & (arrays['actions'] <= 612)).all())
    for i, action in enumerate(arrays['actions']):
        max_error = max(max_error, float(np.abs(flatten(obs)-arrays['observations'][i]).max()))
        if image_size:
            frame = env.render()
            surface = pygame.surfarray.make_surface(np.transpose(frame,(1,0,2)))
            small = pygame.transform.smoothscale(surface, (image_size,image_size))
            images.append(np.transpose(pygame.surfarray.array3d(small),(1,0,2)))
        obs, reward, done, trunc, info = env.step(action)
        max_error = max(max_error, float(np.abs(flatten(obs)-arrays['next_observations'][i]).max()))
        rewards_ok &= abs(reward-float(arrays['rewards'][i])) < 1e-5
        flags_ok &= bool(done) == bool(arrays['terminated'][i]) and bool(trunc) == bool(arrays['truncated'][i])
        early_done |= (done or trunc) and i < count-1
        geom = transform(env.glyph.geometry, (*env.block.position, env.block.angle))
        dist = geom.distance(Point(env.agent.position))
        max_penetration = max(max_penetration, env.pusher_radius-dist)
        x0,y0,x1,y1 = geom.bounds
        ax,ay = env.agent.position
        in_bounds &= min(x0,y0,ax-16,ay-16) >= 9 and max(x1,y1,ax+16,ay+16) <= 631
    result = {'valid': bool(finite and actions_in_bounds and max_error < 1e-3 and rewards_ok and flags_ok and not early_done and in_bounds and max_penetration < 3 and info['is_success']),
              'success': bool(info['is_success']), 'final_coverage': info['coverage'],
              'steps': count, 'duration_seconds': count/env.control_hz,
              'max_state_error': max_error, 'max_contact_penetration_px': max_penetration,
              'finite': bool(finite), 'actions_in_bounds': actions_in_bounds,
              'in_bounds': bool(in_bounds), 'rewards_match': bool(rewards_ok),
              'termination_flags_match': bool(flags_ok), 'continued_after_done': bool(early_done)}
    if image_size and result['valid']:
        arrays['images'] = np.asarray(images, dtype=np.uint8)
        arrays['timestamps'] = np.arange(count, dtype=np.float64)/env.control_hz
        meta['image_observation'] = f'pre-action RGB, {image_size}x{image_size}, uint8, HWC'
        meta['validation'] = result
        arrays['metadata'] = json.dumps(meta, ensure_ascii=False)
        np.savez_compressed(path, **arrays)
        Path(path).with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    env.close()
    return result


def generate_episode(letter, seed, output, contacts=20, iterations=40, max_steps=2400, image_size=96,
                     font_path=None, reset_options=None, block_friction=0.7):
    started = time.monotonic()
    env = PushLetterEnv(letter, font_path=font_path, max_episode_steps=max_steps,
                        block_friction=block_friction)
    obs, info = env.reset(seed=seed, options=reset_options)
    planner = ContactPlanner(env, seed=seed, contacts=contacts)
    recorder = Recorder(env, seed)
    recorder.meta.update({'source': 'model_predictive_planner', 'planner_version': 1,
                          'privileged_state': True, 'font_sha256': font_hash(env.glyph.font_path),
                          'pymunk_version': pymunk.version, 'numpy_version': np.__version__,
                          'platform': platform.platform(), 'contacts': contacts,
                          'max_planning_iterations': iterations, 'success_threshold': 0.95})
    done = trunc = False
    trace = []
    for iteration in range(iterations):
        actions, diagnostic = planner.plan(env)
        if not actions:
            break
        for action in actions:
            next_obs, reward, done, trunc, info = env.step(action)
            recorder.append(obs, action, next_obs, reward, done, trunc)
            obs = next_obs
            if done or trunc:
                break
        trace.append({**diagnostic, 'steps': env.steps, 'coverage': info['coverage']})
        if done or trunc:
            break
    recorder.meta['success'] = bool(done)
    recorder.meta['planning_trace'] = trace
    # Raw attempts are preserved even when they fail validation.
    path = recorder.save(Path(output)/'attempts')
    result = {'letter': letter, 'seed': seed, 'success': bool(done), 'steps': env.steps,
              'final_coverage': info['coverage'], 'valid': False,
              'planning_iterations': len(trace), 'path': None}
    if path:
        validation = validate_episode(path, image_size=image_size if done else 0)
        result.update(validation)
        if validation['valid']:
            destination = Path(output)/'episodes'/f'U{ord(letter):04X}_seed{seed}.npz'
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            path.rename(destination)
            path.with_suffix('.json').rename(destination.with_suffix('.json'))
            path = destination
        result['path'] = str(path.relative_to(output))
    result['wall_seconds'] = round(time.monotonic()-started,3)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--letters', default=string.ascii_uppercase+string.ascii_lowercase)
    parser.add_argument('--episodes',type=int,default=1,help='Validated successful episodes per letter')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--attempts',type=int,default=3,help='Maximum attempts per requested successful episode')
    parser.add_argument('--contacts',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=40)
    parser.add_argument('--max-steps',type=int,default=2400)
    parser.add_argument('--image-size',type=int,default=96,help='0 disables RGB storage')
    parser.add_argument('--output',default='datasets/planner_v1')
    parser.add_argument('--font', help='Local recording font for validation or generation')
    parser.add_argument('--block-friction', type=float, default=0.7)
    parser.add_argument('--validate',help='Validate one existing NPZ and exit')
    args=parser.parse_args()
    if args.validate:
        result=validate_episode(args.validate, font_path=args.font)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['valid'] else 1)
    if min(args.episodes,args.attempts,args.contacts,args.iterations,args.max_steps) <= 0 or args.image_size < 0:
        parser.error('Counts must be positive; image size must be nonnegative.')
    letters=''.join(dict.fromkeys(args.letters))
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    report_path=output/'report.json'
    config={k:v for k,v in vars(args).items() if k not in ('validate','output')}
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['config'] != config:
            parser.error('Output has a different configuration; use a new --output directory.')
    else:
        report={'source':'model_predictive_planner','config':config,'attempts':[],
                'notes':'Accepted episodes only. Attempt success rate is not a held-out policy evaluation.'}
    def write_report():
        report['accepted_episodes']=sum(a['valid'] for a in report['attempts'])
        report['attempted_episodes']=len(report['attempts'])
        report['per_letter']={c:sum(a['valid'] and a['letter']==c for a in report['attempts']) for c in letters}
        report['complete']=all(n>=args.episodes for n in report['per_letter'].values())
        tmp=report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        tmp.replace(report_path)
    for letter in letters:
        previous=[a for a in report['attempts'] if a['letter']==letter]
        successes=sum(a['valid'] for a in previous)
        for index in range(len(previous),args.episodes*args.attempts):
            if successes>=args.episodes:
                break
            seed=args.seed+index
            result=generate_episode(letter,seed,output,args.contacts,args.iterations,args.max_steps,args.image_size,font_path=args.font,block_friction=args.block_friction)
            report['attempts'].append(result)
            successes+=int(result['valid'])
            write_report()
            print(f"{letter} seed={seed} {'ACCEPT' if result['valid'] else 'REJECT'} coverage={result['final_coverage']:.4f} steps={result['steps']} time={result['wall_seconds']:.1f}s",flush=True)
    write_report()
    print(f"Validated {report['accepted_episodes']} / {report['attempted_episodes']} attempts. Report: {report_path}",flush=True)
    raise SystemExit(0 if report['complete'] else 2)


if __name__=='__main__':
    main()
