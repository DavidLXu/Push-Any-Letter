"""Build a validated-dataset contact sheet and optional MP4 (requires ffmpeg)."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pygame
from push_any_letter import PushLetterEnv
from push_any_letter.render import draw_arena

parser=argparse.ArgumentParser()
parser.add_argument('directory')
parser.add_argument('--video',action='store_true')
parser.add_argument('--letters',default='ABOT')
args=parser.parse_args()
root=Path(args.directory).resolve()
report=json.loads((root/'report.json').read_text())
accepted=[a for a in report['attempts'] if a['valid']]
pygame.font.init()
font=pygame.font.Font(None,22)
small=pygame.font.Font(None,18)
title=pygame.font.Font(None,36)
BG=(18,29,36);TEXT=(232,240,239);MUTED=(145,166,174)

def load(episode):
    with np.load(root/episode['path'],allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']));actions=d['actions'].copy()
    env=PushLetterEnv(meta['letter'],font_path=meta['font_path'],max_episode_steps=meta['max_episode_steps'], **meta.get('physics_parameters', {}))
    env.reset(seed=meta['seed'], options=meta.get('reset_options'))
    return env,actions

cols=8;tile=160;rows=(len(accepted)+cols-1)//cols
sheet=pygame.Surface((cols*tile,rows*190+100));sheet.fill(BG)
sheet.blit(title.render('PUSH / ANY LETTER - PLANNER DEMONSTRATIONS',True,TEXT),(16,15))
sheet.blit(font.render(f'{len(accepted)} replay-verified successes | coverage >= 95% | model-based, privileged state',True,MUTED),(16,55))
for i,a in enumerate(accepted):
    env,actions=load(a);trail=[]
    for action in actions:
        env.step(action);trail.append(tuple(env.agent.position))
    x=(i%cols)*tile;y=100+(i//cols)*190
    frame=pygame.transform.smoothscale(draw_arena(env,trail),(150,150))
    sheet.blit(frame,(x+5,y))
    sheet.blit(font.render(f"{a['letter']}   {100*a['final_coverage']:.1f}%",True,TEXT),(x+10,y+155))
    sheet.blit(small.render(f"{a['steps']} steps / seed {a['seed']}",True,MUTED),(x+10,y+175))
pygame.image.save(sheet,str(root/'overview.png'))
print(root/'overview.png',flush=True)
if args.video:
    chosen=[next(a for a in accepted if a['letter']==c) for c in args.letters]
    scenes=[load(a) for a in chosen]
    cols=2;rows=(len(chosen)+1)//2
    width,height=cols*320,rows*365+60
    destination=root/'planner_preview.mp4'
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','20','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    frames=max(len(actions) for env,actions in scenes)
    trails=[[] for _ in scenes]
    try:
        for step in range(0,frames+40,2):
            canvas=pygame.Surface((width,height));canvas.fill(BG)
            canvas.blit(title.render('PUSH / ANY LETTER',True,TEXT),(14,8))
            canvas.blit(small.render('Planner demonstrations | 2x playback | replayed actions only',True,MUTED),(14,40))
            for i,(env,actions) in enumerate(scenes):
                x=i%cols*320;y=60+i//cols*365
                for j in range(step,min(step+2,len(actions))):
                    env.step(actions[j]);trails[i].append(tuple(env.agent.position))
                canvas.blit(pygame.transform.smoothscale(draw_arena(env,trails[i][-120:]),(320,320)),(x,y))
                info=env._info()
                text=f"{env.letter}   {100*info['coverage']:.1f}%   {'SUCCESS' if info['is_success'] else 'PUSHING'}"
                canvas.blit(font.render(text,True,TEXT),(x+12,y+328))
            proc.stdin.write(pygame.image.tostring(canvas,'RGB'))
    finally:
        proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    print(destination,flush=True)
