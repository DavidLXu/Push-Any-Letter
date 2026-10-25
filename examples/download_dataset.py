"""Download, verify and unpack the public episode shards (about 3.2 GB)."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(root):
    root = Path(root).resolve()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (root/name.strip()).resolve()
        if root not in path.parents or sha256(path) != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    manifest = [json.loads(s) for s in (root/'episodes.jsonl').read_text().splitlines()]
    expected = {row['member']: row for row in manifest}
    if len(manifest) != 6240 or len(expected) != 6240:
        raise ValueError('Expected 6240 unique episodes')
    output = root/'episodes'
    seen = set()
    for shard in sorted((root/'data').glob('*.tar')):
        with tarfile.open(shard) as archive:
            for member in archive:
                path = (output/member.name).resolve()
                if (not member.isfile() or output not in path.parents or
                        member.name not in expected or member.name in seen):
                    raise ValueError(f'Unexpected archive member: {member.name}')
                row = expected[member.name]
                if row['shard'] != str(shard.relative_to(root)) or row['bytes'] != member.size:
                    raise ValueError(f'Manifest mismatch: {member.name}')
                seen.add(member.name)
                if path.exists() and sha256(path) == row['sha256']:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.partial')
                with archive.extractfile(member) as source, temp.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if sha256(temp) != row['sha256']:
                    raise ValueError(f'Episode checksum mismatch: {member.name}')
                temp.replace(path)
        print(f'Verified {shard.name}: {len(seen)}/6240 episodes', flush=True)
    if seen != set(expected):
        raise ValueError('Missing episodes')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='datasets/push-any-letter')
    parser.add_argument('--revision', default='main', help='Use a commit hash for a pinned copy')
    parser.add_argument('--local-only', action='store_true', help='Verify already downloaded shards')
    args = parser.parse_args()
    if not args.local_only:
        from huggingface_hub import snapshot_download
        snapshot_download('DavidLXu/Push-Any-Letter', repo_type='dataset',
                          revision=args.revision, local_dir=args.output)
    print(unpack(args.output))
