"""Boundary, objective, sampling, checkpoint, and conditioning regression checks."""
import json
import numpy as np
import pytest
import torch
from .data import StateData, encode_state, test_cases as make_test_cases
from .models import MODELS, masked_mean

torch.set_num_threads(2)


@pytest.mark.parametrize('algo',list(MODELS))
def test_algorithms_train_sample_and_restore(algo,tmp_path):
    torch.manual_seed(42)
    model=MODELS[algo]()
    obs=torch.randn(4,2,62)
    actions=torch.rand(4,model.horizon,2)*2-1
    mask=torch.ones(4,model.horizon,dtype=torch.bool)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loss,_=model.loss(obs,actions,mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step();model.eval()
    with torch.no_grad():
        torch.manual_seed(91);first=model.sample(obs)
        assert first.shape==(4,model.horizon,2) and torch.isfinite(first).all()
        assert first.min()>=-1 and first.max()<=1
        path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
        restored=MODELS[algo]().eval();restored.load_state_dict(torch.load(path,weights_only=True))
        torch.manual_seed(91);second=restored.sample(obs)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        torch.manual_seed(91);different=restored.sample(obs+1)
        assert not torch.allclose(first,different)


def test_windows_do_not_cross_episode_boundaries(tmp_path):
    raw=np.zeros((5,8),dtype=np.float32);raw[:2,:]=100;raw[2:,:]=500
    np.save(tmp_path/'observations.npy',raw)
    np.save(tmp_path/'actions.npy',np.array([[100,100]]*2+[[500,500]]*3,dtype=np.float32))
    episodes=[dict(offset=0,length=2,letter_id=0,split='train'),dict(offset=2,length=3,letter_id=1,split='train')]
    (tmp_path/'episodes.json').write_text(json.dumps(episodes));(tmp_path/'ready.json').write_text('{}')
    data=StateData(tmp_path,'train')
    obs,actions,mask=data.windows(np.array([0,1]),np.array([1,0]),16)
    assert mask.sum(1).tolist()==[1,3]
    np.testing.assert_array_equal(obs[1,0],obs[1,1])
    assert (actions[0]<0).all() and (actions[1]>0).all()
    assert obs[0,0,10]==1 and obs[1,0,11]==1
    error=torch.tensor([[[1.,1.],[999.,999.]]])
    assert masked_mean(error,torch.tensor([[True,False]]))==1


def test_unseen_angles_and_collision_free_new_starts():
    from push_any_letter.geometry import default_font,make_glyph,transform
    from shapely.geometry import Point
    angles,cases=make_test_cases(default_font())
    assert len(angles)==10 and len(cases)==520
    assert all(min(a%30,30-a%30)>=3 for a in angles)
    assert len({r['metadata']['seed'] for r in cases})==520
    for case in cases:
        reset=case['metadata']['reset_options']
        glyph=make_glyph(case['letter'],default_font())
        geom=transform(glyph.geometry,reset['initial_object_pose'])
        x0,y0,x1,y1=geom.bounds
        assert min(x0,y0)>=60-1e-6 and max(x1,y1)<=580+1e-6
        assert geom.distance(Point(reset['initial_agent_pos']))>=20


def test_angle_encoding_wraps():
    raw=np.zeros((2,8),dtype=np.float32)
    raw[1,[4,7]]=2*np.pi
    features=encode_state(raw,0)
    np.testing.assert_allclose(features[0],features[1],atol=1e-6)
