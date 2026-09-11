"""Publish a complete artifact inventory after the detached training process ends."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crosscut.storage import atomic_json, process_lock


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def recover_video(train):
    """On failure convert every committed frame without replaying the game."""
    import numpy as np
    recovery = train / 'recovered'
    recovery.mkdir(exist_ok=True)
    db = sqlite3.connect(f'file:{(train / "frames.sqlite").as_posix()}?mode=ro', uri=True)
    try:
        cursor = db.execute('SELECT step,before_rgb,after_rgb,action,reward,done FROM transitions ORDER BY step')
        while True:
            records = cursor.fetchmany(512)
            if not records:
                break
            before = np.stack([np.frombuffer(zlib.decompress(r[1]), dtype=np.uint8).reshape(64,64,3) for r in records])
            after = np.stack([np.frombuffer(zlib.decompress(r[2]), dtype=np.uint8).reshape(64,64,3) for r in records])
            archive = recovery / f'trajectory-{records[0][0]:07d}-{records[-1][0]:07d}.npz'
            np.savez_compressed(archive, observations=before, next_observations=after,
                observation_sha256=np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in before], dtype='S64'),
                next_observation_sha256=np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in after], dtype='S64'),
                actions=np.array([r[3] for r in records]), rewards=np.array([r[4] for r in records]),
                dones=np.array([r[5] for r in records]))
    finally:
        db.close()
    if any(recovery.glob('trajectory-*.npz')) and not (recovery / 'training_videos.json').exists():
        sys.path.insert(0, str(ROOT / 'experiments/LLM4Teach'))
        from render_training import render_training
        render_training(recovery)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    with process_lock(out / 'artifact-watcher.lock'):
        while True:
            suite = json.loads((out / 'suite-status.json').read_text())
            if suite['phase'] in ('complete', 'failed'):
                break
            time.sleep(5)
        atomic_json(out / 'artifact-status.json', {'phase': 'preparing', 'updated_at': time.time()})
        if suite['phase'] == 'failed' and (out / 'train/frames.sqlite').exists():
            recover_video(out / 'train')
        allowed = {'.json', '.jsonl', '.sqlite', '.pt', '.mp4', '.npz', '.log'}
        paths = [p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in allowed
                 and p.name not in ('artifact-manifest.json', 'artifact-status.json')]
        manifest = dict(run_id=suite['run_id'], phase=suite['phase'], created_at=time.time(),
            files=[dict(path=p.relative_to(out).as_posix(), size=p.stat().st_size, sha256=sha256(p)) for p in paths])
        atomic_json(out / 'artifact-manifest.json', manifest)
        atomic_json(out / 'artifact-status.json', {'phase': 'ready', 'files': len(paths), 'updated_at': time.time()})


if __name__ == '__main__':
    main()
