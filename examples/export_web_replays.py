"""Export one verified recorded rollout per letter/angle for the project page."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from push_any_letter.geometry import make_glyph, polygons

p=argparse.ArgumentParser()
p.add_argument('dataset',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--font',required=True)
p.add_argument('--public-index',type=Path,help='Optional published episodes.jsonl for portable NPZ hashes')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[r for r in json.loads((a.dataset/'index.json').read_text()) if r['slot']==0]
assert len(rows)==624 and len({(r['letter'],r['angle_degrees']) for r in rows})==624
manifest={'source':'https://huggingface.co/datasets/DavidLXu/Push-Any-Letter','slot':0,'dt':.05,'episodes':{},'glyphs':{}}
bundles={}
for r in rows:
    assert r['valid'] and r['result']['success'] and r['result']['max_state_error']<=.001
    letter=r['letter'];angle=r['angle_degrees'];key=f'U{ord(letter):04X}_{angle:03d}'
    source=Path(r['folder'])/r['result']['path']
    with np.load(source,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        assert hashlib.sha256(Path(a.font).read_bytes()).hexdigest()==meta['font_sha256']
        states=np.concatenate([data['observations'][:,:5],data['next_observations'][-1:,:5]],axis=0).astype(float)
        assert len(states)==r['result']['steps']+1 and np.isfinite(states).all()
        states[:,:4]=np.round(states[:,:4],3);states[:,4]=np.round(states[:,4],5)
        goal=data['observations'][0,5:8].astype(float).tolist()
    payload={'letter':letter,'angle':angle,'dt':.05,'goal':goal,'states':states.tolist()}
    name=key[:5]+'.json';bundles.setdefault(name,{})[key]=payload
    manifest['episodes'][key]={'file':name,'steps':len(states)-1,'seed':r['result']['seed'],'coverage':r['result']['final_coverage'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'member':f'U{ord(letter):04X}/angle{angle:03d}/traj00.npz'}
    if letter not in manifest['glyphs']:
        manifest['glyphs'][letter]=[[[round(x,4),round(y,4)] for x,y in ring.coords] for poly in polygons(make_glyph(letter,a.font).geometry) for ring in [poly.exterior,*poly.interiors]]
for name,payload in bundles.items():
    (a.output/name).write_text(json.dumps(payload,separators=(',',':')))
if a.public_index:
    public={r['member']:r for r in map(json.loads,a.public_index.read_text().splitlines())}
    for row in manifest['episodes'].values():
        published=public[row['member']]
        assert row['seed']==published['seed'] and row['steps']==published['steps']
        row['public_sha256']=published['sha256']
manifest['note']='Recorded pre-action states plus final next-state; positions rounded to .001 px, angles to .00001 rad for display. No resimulation.'
(a.output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':')))
print(f'Exported {len(rows)} verified trajectories',flush=True)
