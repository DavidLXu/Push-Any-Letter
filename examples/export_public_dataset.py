"""Export portable, checksummed per-letter tar shards without font binaries/host paths."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
import zipfile
import numpy as np


def portable_npz(source, destination):
    with np.load(source,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
    meta['font_path']='Arial-Bold.ttf'
    meta['font_name']='Arial Bold'
    meta['release']='center12-friction14-v1'
    encoded=json.dumps(meta,ensure_ascii=False)
    assert '/home/' not in encoded and '/Users/' not in encoded
    buffer=io.BytesIO();np.save(buffer,np.array(encoded),allow_pickle=False)
    # Keep compressed array payloads byte-for-byte, replacing only metadata.npy.
    # NumPy's seekable ZIP writer stores sizes in local headers (no data descriptors).
    with zipfile.ZipFile(source) as original,open(source,'rb') as raw,zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr('metadata.npy',buffer.getvalue())
        for info in original.infolist():
            if info.filename=='metadata.npy':continue
            assert not info.flag_bits&8,'Unexpected ZIP data descriptor'
            raw.seek(info.header_offset)
            header=raw.read(30);assert header[:4]==b'PK\x03\x04'
            filename_size,extra_size=struct.unpack_from('<HH',header,26)
            length=30+filename_size+extra_size+info.compress_size
            raw.seek(info.header_offset)
            copied=copy.copy(info);copied.header_offset=output.fp.tell()
            remaining=length
            while remaining:
                chunk=raw.read(min(1024*1024,remaining));assert chunk
                output.fp.write(chunk);remaining-=len(chunk)
            output.filelist.append(copied);output.NameToInfo[copied.filename]=copied
            output.start_dir=output.fp.tell();output._didModify=True
    with zipfile.ZipFile(source) as a,zipfile.ZipFile(destination) as b:
        assert sorted(a.namelist())==sorted(b.namelist())
        for info in a.infolist():
            if info.filename!='metadata.npy':
                target=b.getinfo(info.filename)
                assert (info.CRC,info.file_size,info.compress_size)==(target.CRC,target.file_size,target.compress_size)
    return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.output).resolve();(out/'data').mkdir(parents=True,exist_ok=True)
    status=json.loads((source/'status.json').read_text());assert status['state']=='complete' and status['accepted']==6240
    index=json.loads((source/'index.json').read_text())
    groups={}
    for row in index:groups.setdefault(row['letter'],[]).append(row)
    all_rows=[];checksums={};steps=[];coverage=[];font_hash=None;max_error=0.;duplicates=set()
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp=Path(tmp)
        for letter in sorted(groups,key=ord):
            shard=out/'data'/f'U{ord(letter):04X}.tar'
            assert len(groups[letter])==120
            with tarfile.open(shard,'w') as archive:
                for row in sorted(groups[letter],key=lambda r:(r['angle_degrees'],r['slot'])):
                    assert row['valid']
                    src=Path(row['folder'])/row['result']['path']
                    dest=tmp/'episode.npz'
                    meta=portable_npz(src,dest)
                    digest=hashlib.sha256(dest.read_bytes()).hexdigest();assert digest not in duplicates;duplicates.add(digest)
                    name=f'U{ord(letter):04X}/angle{row["angle_degrees"]:03d}/traj{row["slot"]:02d}.npz'
                    tarinfo=archive.gettarinfo(str(dest),name);tarinfo.uid=tarinfo.gid=0;tarinfo.uname=tarinfo.gname='';tarinfo.mtime=0
                    with dest.open('rb') as f:archive.addfile(tarinfo,f)
                    v=meta['validation'];assert v['valid']
                    n=v['steps'];steps.append(n);coverage.append(v['final_coverage']);max_error=max(max_error,v['max_state_error'])
                    font_hash=meta['font_sha256'] if font_hash is None else font_hash;assert font_hash==meta['font_sha256']
                    all_rows.append({'letter':letter,'angle_degrees':row['angle_degrees'],'slot':row['slot'],
                        'seed':meta['seed'],'steps':n,'coverage':v['final_coverage'],'shard':str(shard.relative_to(out)),
                        'member':name,'sha256':digest,'bytes':dest.stat().st_size,'reset_options':meta['reset_options']})
                # Read every array of one episode per shard, including pixels, to check ZIP integrity.
                with np.load(dest,allow_pickle=False) as data:
                    for key in data.files:assert data[key].dtype!=object
            checksums[str(shard.relative_to(out))]=hashlib.sha256(shard.read_bytes()).hexdigest()
            print(json.dumps({'letter':letter,'episodes':len(all_rows),'shard_bytes':shard.stat().st_size}),flush=True)
    with (out/'episodes.jsonl').open('w') as f:
        for row in all_rows:f.write(json.dumps(row)+'\n')
    summary={'episodes':len(all_rows),'letters':52,'goal_angles_degrees':list(range(0,360,30)),
        'trajectories_per_goal':10,'control_hz':20,'physics_hz':200,'steps':sum(steps),
        'steps_min':min(steps),'steps_max':max(steps),'steps_mean':float(np.mean(steps)),
        'coverage_min':min(coverage),'coverage_mean':float(np.mean(coverage)),'max_replay_state_error':max_error,
        'attempts':status['attempts_in_finished_slots'],'block_friction':1.4,'agent_friction':.6,'space_damping':.12,
        'font_name':'Arial Bold','font_sha256':font_hash,'source':'model_predictive_planner',
        'recording_format':'NPZ episodes in 52 uncompressed TAR shards; array members compressed inside NPZ',
        'dataset_bytes':sum(r['bytes'] for r in all_rows),'license':'CC-BY-4.0','author':'Lixin Xu',
        'source_index_sha256':hashlib.sha256((source/'index.json').read_bytes()).hexdigest()}
    assert len(all_rows)==6240 and summary['steps']==4736343
    (out/'statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ['episodes.jsonl','statistics.json']:checksums[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    (out/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(checksums.items())))


if __name__=='__main__':main()
