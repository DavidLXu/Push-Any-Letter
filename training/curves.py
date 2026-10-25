"""Fixed-subset evaluation plan, background workers, and publication-friendly plots."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import atomic_json, LETTERS

ALGOS=['bc','diffusion','flow','act']
LABELS=['Vanilla BC','Diffusion Policy','Flow Matching','ACT']
COLORS=['#2378b5','#e58631','#279e75','#9b60ad']


def make_plan(data, runs):
    path=runs/'curve_plan.json'
    if path.exists():return json.loads(path.read_text())
    episodes=json.loads((data/'episodes.json').read_text())
    tests=json.loads((data/'test_cases.json').read_text())
    rng=np.random.default_rng(20261028)
    plan={'seed':20261028,'train':[],'test':[], 'interval_updates':10000,
          'notes':'Fixed 52 cases per split. Test curves are monitoring only, no checkpoint selection.'}
    for i,letter in enumerate(LETTERS):
        candidates=[e for e in episodes if e['letter']==letter]
        plan['train'].append(candidates[int(rng.integers(len(candidates)))])
        candidates=sorted([e for e in tests if e['letter']==letter],key=lambda e:e['slot'])
        plan['test'].append(candidates[i%10])
    atomic_json(path,plan)
    return plan


def rows(path):
    if not path.exists():return []
    output=[]
    for line in path.read_text().splitlines():
        try:output.append(json.loads(line))
        except json.JSONDecodeError:pass  # Writer may be midway through its last line.
    return output


def plot(runs):
    destination=runs/'curves';destination.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for algo,label,color,ax in zip(ALGOS,LABELS,COLORS,axes.flat):
        # After checkpoint resume, use the most recent logged value for each step.
        values={r['step']:r['loss'] for r in rows(runs/algo/'metrics.jsonl') if 'loss' in r}
        steps=sorted(values);loss=np.array([values[s] for s in steps])
        if len(loss):
            ax.plot(steps,loss,color=color,alpha=.25,linewidth=.7,label='Logged loss')
            smooth=[];v=loss[0]
            for value in loss:v=.9*v+.1*value;smooth.append(v)
            ax.plot(steps,smooth,color=color,linewidth=1.5,label='Smoothed (0.9 EMA)')
            ax.set_yscale('log');ax.legend(fontsize=8)
        ax.set(title=label,xlabel='Optimizer updates',ylabel='Training objective (log scale)')
        ax.grid(alpha=.2)
    fig.suptitle('Training losses — objectives have different scales')
    fig.savefig(destination/'loss_curves.png',dpi=170);fig.savefig(destination/'loss_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    all_rows=[]
    for algo,label,color in zip(ALGOS,LABELS,COLORS):
        measurements=[json.loads(p.read_text()) for p in sorted((runs/algo/'learning_curve').glob('step_*/metrics.json'))]
        all_rows.extend(measurements)
        for split,ax in zip(['train','test'],axes):
            if measurements:
                ax.plot([r['step'] for r in measurements],[100*r[f'{split}_success_rate'] for r in measurements],
                        'o-',color=color,label=label,markersize=4)
    for split,ax in zip(['train','test'],axes):
        ax.set(title='Training starts (52 fixed)' if split=='train' else 'Unseen-angle test starts (52 fixed)',
               xlabel='Optimizer updates',ylabel='Closed-loop success (%)',ylim=(-2,102))
        ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=9)
        else:ax.text(.5,.5,'First evaluation in progress',transform=ax.transAxes,ha='center')
    fig.suptitle('Success curves — 95% coverage threshold, 2400-step limit')
    fig.savefig(destination/'success_curves.png',dpi=170);fig.savefig(destination/'success_curves.pdf');plt.close(fig)
    atomic_json(destination/'success_rates.json',all_rows)
    if all_rows:
        with (destination/'success_rates.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--runs',required=True)
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--image',action='store_true')
    args=p.parse_args();data=Path(args.data).resolve();runs=Path(args.runs).resolve()
    if args.plot_only:plot(runs);return
    import fcntl
    lock=(runs/'curves.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    make_plan(data,runs)
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={int(a.strip()):b.strip() for a,b in (r.split(',') for r in query.splitlines())}
    processes={};logs=[]
    for gpu,algo in enumerate(ALGOS):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=mapping[gpu],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        log=(runs/algo/'curve_worker.log').open('a');logs.append(log)
        processes[algo]=subprocess.Popen([sys.executable,'-u','-m','training.curve_worker',
            '--algo',algo,'--data',str(data),'--runs',str(runs)]+(['--image'] if args.image else []),env=env,stdout=log,stderr=subprocess.STDOUT)
    while True:
        plot(runs)
        codes={a:p.poll() for a,p in processes.items()}
        atomic_json(runs/'curve_service_status.json',{'state':'complete' if all(c==0 for c in codes.values()) else
            'failed' if any(c is not None and c!=0 for c in codes.values()) else 'running',
            'workers':{a:{'pid':processes[a].pid,'exit_code':c} for a,c in codes.items()},
            'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        if all(c is not None for c in codes.values()):break
        time.sleep(30)
    for log in logs:log.close()
    if any(c!=0 for c in codes.values()):raise SystemExit(1)


if __name__=='__main__':main()
