"""Evaluate immutable training snapshots without blocking the optimizer process."""
import argparse
import json
from pathlib import Path
import time
import torch
from .data import atomic_json
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algo',required=True);p.add_argument('--data',required=True);p.add_argument('--runs',required=True)
    p.add_argument('--image',action='store_true')
    args=p.parse_args()
    evaluator=evaluate
    if args.image:
        from .image_evaluate import evaluate as evaluator
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.10)
    root=Path(args.runs)/args.algo
    output=root/'learning_curve';output.mkdir(exist_ok=True)
    import fcntl
    lock=(output/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((Path(args.runs)/'curve_plan.json').read_text())
    while True:
        checkpoints=sorted((root/'checkpoints').glob('step_*.pt'))
        for checkpoint in checkpoints:
            step=int(checkpoint.stem.split('_')[1])
            dest=output/f'step_{step:06d}'
            row_path=dest/'metrics.json'
            if row_path.exists():continue
            dest.mkdir(exist_ok=True)
            results={}
            for split in ['train','test']:
                split_dest=dest/split
                atomic_json(output/'status.json',{'state':'evaluating','step':step,'split':split})
                existing=split_dest/'summary.json'
                if existing.exists():
                    summary=json.loads(existing.read_text())
                else:
                    summary=evaluator(checkpoint,args.data,split_dest,torch.device('cuda'),
                        cases=plan[split],scope=f'Fixed 52-episode {split} subset, one start per letter, learning curve only')
                assert summary['checkpoint_step']==step and summary['evaluated']==52
                results[split]=summary
            row={'step':step,'algo':args.algo,
                 **{f'{split}_{key}':results[split][key]
                    for split in ['train','test'] for key in ['success_rate','successes','evaluated','mean_final_coverage']}}
            atomic_json(row_path,row)
            print(json.dumps(row),flush=True)
        train_status=root/'status.json'
        if train_status.exists():
            state=json.loads(train_status.read_text())
            if state['state']=='complete':
                # A final checkpoint must exist and be evaluated before claiming completion.
                final_step=state['step']
                if (output/f'step_{final_step:06d}/metrics.json').exists():
                    atomic_json(output/'status.json',{'state':'complete','step':final_step});return
        suite=Path(args.runs)/'suite_status.json'
        if suite.exists():
            job=json.loads(suite.read_text()).get('jobs',{}).get(args.algo,{})
            if job.get('state')=='failed':raise RuntimeError(f'{args.algo} training failed; available snapshots were evaluated')
        atomic_json(output/'status.json',{'state':'waiting_for_checkpoint'})
        time.sleep(15)


if __name__=='__main__':main()
