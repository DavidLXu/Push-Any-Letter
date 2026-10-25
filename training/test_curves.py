import json
from pathlib import Path
import numpy as np
from .curves import make_plan, plot
from .data import LETTERS


def test_balanced_fixed_plan_and_real_metric_plot(tmp_path):
    data=tmp_path/'data';data.mkdir()
    runs=tmp_path/'runs';runs.mkdir()
    episodes=[{'letter':c,'slot':i,'split':'train'} for c in LETTERS for i in range(120)]
    tests=[{'letter':c,'slot':i} for c in LETTERS for i in range(10)]
    (data/'episodes.json').write_text(json.dumps(episodes))
    (data/'test_cases.json').write_text(json.dumps(tests))
    first=make_plan(data,runs);second=make_plan(data,runs)
    assert first==second
    assert len(first['train'])==len(first['test'])==52
    assert len({e['letter'] for e in first['train']})==52
    assert len({e['letter'] for e in first['test']})==52
    counts=np.bincount([e['slot'] for e in first['test']])
    assert counts.max()-counts.min()<=1
    root=runs/'bc';root.mkdir()
    (root/'metrics.jsonl').write_text('{"step":100,"loss":0.1}\n{"step":200,"loss":0.05}\n')
    row={'step':200,'algo':'bc','train_success_rate':.5,'test_success_rate':.25}
    dest=root/'learning_curve/step_000200';dest.mkdir(parents=True)
    (dest/'metrics.json').write_text(json.dumps(row))
    plot(runs)
    assert json.loads((runs/'curves/success_rates.json').read_text())==[row]
    assert (runs/'curves/loss_curves.png').stat().st_size>1000
    assert (runs/'curves/success_curves.png').stat().st_size>1000
