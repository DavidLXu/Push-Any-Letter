"""Equal-update baselines, all demonstrations for training, fixed final EMA evaluation."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .data import StateData, atomic_json
from .models import MODELS


def save_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state,temp); temp.replace(path)


def tensors(batch, device):
    return tuple(torch.as_tensor(x,device=device) for x in batch)


@torch.no_grad()
def training_diagnostic(model, data, device):
    model.eval()
    rng = np.random.default_rng(9867)
    total = 0.
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(9867)
        for _ in range(8):
            obs, actions, mask = tensors(data.sample(rng,64,model.horizon),device)
            pred = model.sample(obs)
            total += float(((pred[:,0]-actions[:,0])*292).square().mean())
    return total/8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algo',choices=MODELS,required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--save-every',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--skip-eval',action='store_true',help='For smoke tests only')
    args = p.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (root/'train.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.20)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = StateData(args.data,'train')
    model = MODELS[args.algo]().to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    config = vars(args).copy()
    config.update(parameters=sum(p.numel() for p in model.parameters()), history=2,
        horizon=model.horizon, execution_horizon=1 if args.algo=='bc' else 4,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), block_friction=1.4,
        data_manifest_sha256=hashlib.sha256((Path(args.data)/'ready.json').read_bytes()).hexdigest(),
        learning_rate=1e-4, ema_decay=.995, checkpoint_selection='Fixed final EMA at 100000 updates; no test tuning',
        device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU')
    config_path = root/'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text()) == config, 'Configuration changed'
    atomic_json(config_path,config)
    start = 0
    checkpoint = root/'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    snapshots = root/'checkpoints'
    snapshots.mkdir(exist_ok=True)
    # Preserve the first available state too: earlier states cannot be reconstructed.
    initial_snapshot = snapshots/f'step_{start:06d}.pt'
    if not initial_snapshot.exists():
        save_checkpoint(initial_snapshot,{'model':ema.state_dict(),'step':start,'config':config})
    started = time.monotonic()
    metrics = (root/'metrics.jsonl').open('a',buffering=1)
    for step in range(start+1,args.steps+1):
        model.train()
        obs, actions, mask = tensors(data.sample(rng,args.batch_size,model.horizon),device)
        warmup = min(1000,max(1,args.steps//10))
        lr = 1e-4*min(step/warmup,1)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.loss(obs,actions,mask)
        if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for e, m in zip(ema.parameters(),model.parameters()): e.lerp_(m,.005)
        if step%100 == 0 or step == 1 or step == args.steps:
            row = {'step':step,'loss':float(loss.detach()),'lr':lr,
                   'elapsed_seconds':round(time.monotonic()-started,1),
                   **{k:float(v) for k,v in components.items()}}
            if device.type=='cuda': row['peak_allocated_GiB'] = torch.cuda.max_memory_allocated()/1024**3
            metrics.write(json.dumps(row)+'\n')
            atomic_json(root/'status.json',dict(row,state='training',total_steps=args.steps))
            print(json.dumps(row),flush=True)
        if step%args.save_every == 0 or step == args.steps:
            mse = training_diagnostic(ema,data,device)
            metrics.write(json.dumps({'step':step,'train_first_action_mse_px2':mse})+'\n')
            save_checkpoint(root/'policy.pt',{'model':ema.state_dict(),'step':step,'config':config})
            save_checkpoint(checkpoint,{'model':model.state_dict(),'ema':ema.state_dict(),
                'optimizer':optimizer.state_dict(),'step':step,'config':config,
                'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
            if step%10000 == 0 or step == args.steps:
                save_checkpoint(snapshots/f'step_{step:06d}.pt',
                                {'model':ema.state_dict(),'step':step,'config':config})
    atomic_json(root/'status.json',{'state':'trained' if args.skip_eval else 'evaluating',
        'step':args.steps,'total_steps':args.steps})
    if not args.skip_eval:
        from .evaluate import evaluate
        summary = evaluate(root/'policy.pt',args.data,root/'evaluation',device)
        atomic_json(root/'status.json',{'state':'complete','step':args.steps,'test':summary})
    metrics.close()


if __name__ == '__main__': main()
