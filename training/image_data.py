"""Exact pre-action RGB observations, disk-backed cache, no object-state policy inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import numpy as np
from .data import StateData, atomic_json


def encode_aux(raw, letter_ids):
    raw=np.asarray(raw,dtype=np.float32)
    xy=raw[..., [0,1,5,6]]/320.-1.
    angle=raw[...,7:8]
    identity=np.broadcast_to(np.eye(52,dtype=np.float32)[letter_ids],raw.shape[:-1]+(52,))
    return np.concatenate([xy,np.sin(angle),np.cos(angle),identity],axis=-1).astype(np.float32)


def write_episode(job):
    cache,ep=job
    with np.load(ep['path'],allow_pickle=False) as data:
        frames=data['images']
        assert frames.dtype==np.uint8 and frames.shape==(ep['length'],96,96,3)
        target=np.load(cache,mmap_mode='r+')
        target[ep['offset']:ep['offset']+ep['length']]=frames
        # OS flush at completion of all writers; assignments target disjoint ranges.
    return ep['length']


def prepare_images(state_root, image_root, workers=8):
    state_root,image_root=Path(state_root).resolve(),Path(image_root).resolve()
    image_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(image_root/'prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_hash=hashlib.sha256((state_root/'ready.json').read_bytes()).hexdigest()
    ready=image_root/'ready.json'
    if ready.exists():
        assert json.loads(ready.read_text())['state_manifest_sha256']==source_hash
        return
    episodes=json.loads((state_root/'episodes.json').read_text())
    count=sum(ep['length'] for ep in episodes)
    required=count*96*96*3
    assert shutil.disk_usage(image_root).free>required+10*1024**3,'Insufficient disk for image cache'
    cache=image_root/'images.npy'
    target=np.lib.format.open_memmap(cache,mode='w+',dtype='uint8',shape=(count,96,96,3))
    del target
    completed=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for n in pool.map(write_episode,[(str(cache),ep) for ep in episodes],chunksize=1):
            completed+=n
            if completed//100000 != (completed-n)//100000:
                atomic_json(image_root/'progress.json',{'state':'preparing','frames':completed,'total':count})
    np.load(cache,mmap_mode='r+').flush()
    assert completed==count
    atomic_json(ready,{'state_root':str(state_root),'state_manifest_sha256':source_hash,
        'frames':count,'bytes':required,'shape':[count,96,96,3],
        'observation':'pre-action RGB; HWC uint8; two-frame boundary-clamped history',
        'auxiliary':'pusher XY, goal XY/sin/cos, letter identity; no object pose or velocity'})
    atomic_json(image_root/'progress.json',{'state':'complete','frames':completed,'total':count})


class ImageData(StateData):
    def __init__(self, state_root, image_root):
        super().__init__(state_root,'train')
        ready=json.loads((Path(image_root)/'ready.json').read_text())
        assert ready['state_manifest_sha256']==hashlib.sha256((Path(state_root)/'ready.json').read_bytes()).hexdigest()
        self.images=np.load(Path(image_root)/'images.npy',mmap_mode='r')

    def windows(self,ids,times,horizon):
        offsets,lengths=self.offsets[ids],self.lengths[ids]
        hist=np.maximum(times[:,None]+np.arange(-1,1),0)
        indices=offsets[:,None]+hist
        aux=encode_aux(self.obs[indices],self.letters[ids,None])
        frames=np.ascontiguousarray(self.images[indices].transpose(0,1,4,2,3))
        future=times[:,None]+np.arange(horizon)
        mask=future<lengths[:,None]
        actions=(self.actions[offsets[:,None]+np.minimum(future,lengths[:,None]-1)]-320.)/292.
        return {'images':frames,'aux':aux},actions.astype(np.float32),mask


def render_observation(env):
    import pygame
    frame=env.render()
    surface=pygame.surfarray.make_surface(np.transpose(frame,(1,0,2)))
    small=pygame.transform.smoothscale(surface,(96,96))
    return np.transpose(pygame.surfarray.array3d(small),(2,1,0)).copy()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state_root');p.add_argument('image_root')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare_images(a.state_root,a.image_root,a.workers)
