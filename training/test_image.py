import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from .image_models import ImagePolicy
from .image_data import ImageData, encode_aux, prepare_images, render_observation

torch.set_num_threads(2)


@pytest.mark.parametrize('algo',['bc','diffusion','flow','act'])
def test_visual_gradient_sampling_checkpoint(algo,tmp_path):
    torch.manual_seed(42)
    model=ImagePolicy(algo)
    obs={'images':torch.randint(256,(2,2,3,96,96),dtype=torch.uint8),'aux':torch.randn(2,2,58)}
    actions=torch.rand(2,model.horizon,2)*2-1
    mask=torch.ones(2,model.horizon,dtype=torch.bool)
    loss,_=model.loss(obs,actions,mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.cnn[0].weight.grad.abs().sum()>0
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    model.eval()
    with torch.no_grad():
        torch.manual_seed(99);pred=model.sample(obs)
        assert pred.shape==actions.shape and torch.isfinite(pred).all()
        changed=dict(obs,images=torch.zeros_like(obs['images']))
        torch.manual_seed(99);other=model.sample(changed)
        assert not torch.allclose(pred,other,atol=1e-7)
        path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
        restored=ImagePolicy(algo).eval();restored.load_state_dict(torch.load(path,weights_only=True))
        torch.manual_seed(99);actual=restored.sample(obs)
        torch.testing.assert_close(pred,actual,rtol=0,atol=0)


def test_object_pose_is_excluded():
    raw=np.arange(16,dtype=np.float32).reshape(2,8)
    changed=raw.copy();changed[:,2:5]=99999
    np.testing.assert_array_equal(encode_aux(raw,2),encode_aux(changed,2))
    assert encode_aux(raw,2).shape==(2,58)


def test_image_cache_and_history_alignment(tmp_path):
    state=tmp_path/'state';state.mkdir();cache=tmp_path/'cache'
    frames=np.stack([np.full((96,96,3),v,dtype=np.uint8) for v in [10,20,100,110,120]])
    eps=[]
    for i,(start,length) in enumerate([(0,2),(2,3)]):
        path=tmp_path/f'ep{i}.npz';np.savez_compressed(path,images=frames[start:start+length])
        eps.append(dict(path=str(path),offset=start,length=length,letter_id=i,split='train'))
    (state/'episodes.json').write_text(json.dumps(eps));(state/'ready.json').write_text('{}')
    np.save(state/'observations.npy',np.zeros((5,8),dtype=np.float32))
    np.save(state/'actions.npy',np.full((5,2),320,dtype=np.float32))
    prepare_images(state,cache,workers=1)
    data=ImageData(state,cache)
    obs,actions,mask=data.windows(np.array([0,1]),np.array([1,0]),16)
    assert obs['images'].shape==(2,2,3,96,96)
    assert obs['images'][:, :, 0,0,0].tolist()==[[10,20],[100,100]]
    assert mask.sum(1).tolist()==[1,3]
    assert (actions==0).all()


def test_saved_pixels_match_live_rendering():
    from push_any_letter.env import PushLetterEnv
    path=os.environ.get('IMAGE_TEST_EPISODE')
    if not path:pytest.skip('Set IMAGE_TEST_EPISODE to verify a recorded trajectory')
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']));frames=data['images'];actions=data['actions']
    from push_any_letter.fonts import resolve_recording_font
    env=PushLetterEnv(meta['letter'],font_path=resolve_recording_font(meta),max_episode_steps=meta['max_episode_steps'],
                      **meta['physics_parameters'])
    env.reset(seed=meta['seed'],options=meta['reset_options'])
    for i in range(min(12,len(actions))):
        np.testing.assert_array_equal(render_observation(env),frames[i].transpose(2,0,1))
        env.step(actions[i])
    env.close()
