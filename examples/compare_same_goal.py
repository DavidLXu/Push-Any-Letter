"""Validate downloaded episodes and render a synchronized same-goal comparison."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pygame
from push_any_letter.env import PushLetterEnv
from push_any_letter.geometry import default_font
from push_any_letter.dataset import validate_episode
from push_any_letter.render import draw_arena

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory')
args = parser.parse_args()
root = Path(args.directory).resolve()
selection = json.loads((root/'selection.json').read_text())
assert len(selection) == 4
(root/'episodes').mkdir(exist_ok=True)
scenes, report = [], []
font_path = default_font()
font_hash = hashlib.sha256(Path(font_path).read_bytes()).hexdigest()
for selected in selection:
    source = Path(selected['local_original'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == selected['sha256']
    with np.load(source, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    meta = json.loads(str(arrays['metadata']))
    assert meta['font_sha256'] == font_hash
    meta['source_font_path'] = meta['font_path']
    meta['font_path'] = font_path
    arrays['metadata'] = json.dumps(meta)
    destination = root/'episodes'/source.name
    np.savez_compressed(destination, **arrays)
    destination.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    validation = validate_episode(destination)
    assert validation['valid'], validation
    env = PushLetterEnv(meta['letter'], font_path=font_path,
                       max_episode_steps=meta['max_episode_steps'], **meta['physics_parameters'])
    env.reset(seed=meta['seed'], options=meta['reset_options'])
    scenes.append((env, arrays['actions'], selected['slot'], meta))
    report.append(dict(validation, slot=selected['slot'], seed=meta['seed'],
                       initial_state=meta['reset_options']))
assert all(env.letter == scenes[0][0].letter and np.array_equal(env.goal_pose, scenes[0][0].goal_pose)
           for env, *_ in scenes)
(root/'local_validation.json').write_text(json.dumps(report, indent=2))
pygame.font.init()
heading = pygame.font.Font(None, 28)
label = pygame.font.Font(None, 24)
small = pygame.font.Font(None, 20)
width, height = 800, 976
trails = [[] for _ in scenes]
length = max(len(actions) for _, actions, _, _ in scenes)
goal = scenes[0][0].goal_pose

def frame():
    canvas = pygame.Surface((width, height)); canvas.fill((18,29,36))
    canvas.blit(heading.render(f'{scenes[0][0].letter} | SAME GOAL: CENTER, {np.degrees(goal[2]):.0f} deg', True, (240,244,240)), (16,12))
    canvas.blit(small.render('Different starts / planner seeds | 2x speed | finish positions held', True, (170,194,190)), (16,43))
    for i, (env, actions, slot, meta) in enumerate(scenes):
        x, y = (i%2)*400, 76+(i//2)*450
        canvas.blit(pygame.transform.smoothscale(draw_arena(env,trails[i]),(400,400)), (x,y))
        info = env._info()
        caption = f'Trial {slot:02d} | step {env.steps}/{len(actions)} | {info["coverage"]:.1%}'
        canvas.blit(label.render(caption, True, (240,244,240)), (x+12,y+405))
        state = 'SUCCESS' if info['is_success'] else 'PUSHING'
        canvas.blit(small.render(f'{state} | seed {meta["seed"]}', True, (170,194,190)), (x+12,y+429))
    return canvas

proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                         '-s',f'{width}x{height}','-r','20','-i','-','-an','-c:v','libx264',
                         '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'same_goal_comparison.mp4')], stdin=subprocess.PIPE)
try:
    initial = frame()
    pygame.image.save(initial, str(root/'initial_states.png'))
    for _ in range(30): proc.stdin.write(pygame.image.tostring(initial,'RGB'))
    for step in range(0,length,2):
        for i,(env,actions,_,_) in enumerate(scenes):
            for j in range(step,min(step+2,len(actions))):
                env.step(actions[j]);trails[i].append(tuple(env.agent.position))
        proc.stdin.write(pygame.image.tostring(frame(),'RGB'))
    final = frame()
    pygame.image.save(final, str(root/'final_states.png'))
    for _ in range(40): proc.stdin.write(pygame.image.tostring(final,'RGB'))
finally:
    proc.stdin.close()
assert proc.wait() == 0
print(json.dumps(report,indent=2))
print(root/'same_goal_comparison.mp4')
