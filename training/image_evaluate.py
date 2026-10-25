"""Closed-loop unseen-angle evaluation; no planner or recorded action assistance."""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import torch
from push_any_letter.env import PushLetterEnv
from push_any_letter.recording import flatten
from .data import encode_state, atomic_json, StateData
from .image_models import ImagePolicy
from .image_data import encode_aux, render_observation


@torch.no_grad()
def evaluate(checkpoint, data_root, output, device, limit=0, cases=None, scope=None):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    saved = torch.load(checkpoint,map_location=device,weights_only=False)
    config = saved['config']
    assert config['modality']=='rgb_proprio_task'
    model = ImagePolicy(config['algo']).to(device).eval()
    model.load_state_dict(saved['model'])
    torch.manual_seed(20261026)
    episodes = cases if cases is not None else json.loads((Path(data_root)/'test_cases.json').read_text())
    if limit: episodes = episodes[:limit]
    results = []
    started = time.monotonic()
    log = (output/'episodes.jsonl').open('w',buffering=1)
    for begin in range(0,len(episodes),16):
        active = []
        for ep in episodes[begin:begin+16]:
            meta = ep['metadata']
            env = PushLetterEnv(meta['letter'],font_path=meta['font_path'],max_episode_steps=2400,
                               **meta['physics_parameters'])
            obs,_ = env.reset(seed=meta['seed'],options=meta['reset_options'])
            image=render_observation(env)
            active.append({'env':env,'episode':ep,'history':deque([flatten(obs),flatten(obs)],maxlen=2),
                           'images':deque([image,image],maxlen=2)})
        while active:
            obs={'images':np.stack([np.stack(r['images']) for r in active]),
                 'aux':np.stack([encode_aux(np.stack(r['history']),r['episode']['letter_id']) for r in active])}
            predictions=model.sample({k:torch.as_tensor(v,device=device) for k,v in obs.items()}).cpu().numpy()*292+320
            remaining = []
            for r, sequence in zip(active,predictions):
                env = r['env']
                done = trunc = False
                for action in sequence[:config['execution_horizon']]:
                    obs,_,done,trunc,info = env.step(action)
                    r['history'].append(flatten(obs))
                    r['images'].append(render_observation(env))
                    if done or trunc: break
                if done or trunc:
                    ep = r['episode']
                    row = {'letter':ep['letter'],'angle':ep['angle'],'slot':ep['slot'],
                           'success':bool(done),'coverage':info['coverage'],
                           'best_coverage':info['best_coverage'],'steps':env.steps}
                    results.append(row); log.write(json.dumps(row)+'\n'); env.close()
                else: remaining.append(r)
            active = remaining
        atomic_json(output/'progress.json',{'evaluated':len(results),'total':len(episodes),
                    'successes':sum(r['success'] for r in results)})
    log.close()
    summary = {'evaluated':len(results),'successes':sum(r['success'] for r in results),
               'success_rate':float(np.mean([r['success'] for r in results])),
               'mean_final_coverage':float(np.mean([r['coverage'] for r in results])),
               'mean_steps':float(np.mean([r['steps'] for r in results])),
               'elapsed_seconds':round(time.monotonic()-started,1),
               'checkpoint_step':saved['step'], 'max_episode_steps':2400,
               'scope':scope or 'Ten unseen target angles shared across 52 letters, one fresh start per letter/angle, one policy seed',
               'per_letter':{c:float(np.mean([r['success'] for r in results if r['letter']==c]))
                             for c in sorted({r['letter'] for r in results})}}
    atomic_json(output/'summary.json',summary)
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('data');p.add_argument('output')
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.device=='cuda': torch.cuda.set_per_process_memory_fraction(.2)
    print(evaluate(args.checkpoint,args.data,args.output,torch.device(args.device),args.limit))
