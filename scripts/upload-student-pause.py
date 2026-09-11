"""Sync a paused run's snapshot and teacher audit using a memory-only W&B key."""
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crosscut.storage import atomic_json


def main():
    import wandb
    output = ROOT / 'results/exp002-astra-seed0'
    data = ROOT / 'data/local/exp002'
    monitor = json.loads((data / 'monitor-status.json').read_text())
    if monitor['status'] != 'stopped':
        raise RuntimeError('Finish the existing monitor before resuming the W&B run for upload')
    snapshot = output / 'pause-step-0000095.tar.gz'
    expected = 'dfdfe4568bf695fa5c7a60ca29e089f1f4e60b527a1f4c834ec353498a9da3d0'
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
        raise ValueError('Pause snapshot hash mismatch')
    teacher_archive = data / 'paused-teacher-audit.tar.gz'
    teacher_root = data / 'teacher'
    with tarfile.open(teacher_archive, 'w:gz') as bundle:
        for path in sorted((teacher_root / 'requests').rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.png') and 'workspace' not in path.parts:
                bundle.add(path, arcname=path.relative_to(teacher_root).as_posix())
        bundle.add(ROOT / 'prompts/student-teacher.md', arcname='student-teacher.md')
    os.environ['WANDB_API_KEY'] = getpass.getpass('W&B API key (pause upload only): ')
    run = None
    try:
        run = wandb.init(entity='nileshsarkar-ai', project='crosscut-llm4teach',
            id='exp002-astra-seed0', resume='allow', dir=str(data))
        artifact = wandb.Artifact('exp002-astra-seed0-pause-step95', type='paused-experiment',
            metadata={'env_step':95, 'optimizer_updates':0, 'phase':'paused', 'snapshot_sha256':expected})
        for path in (snapshot, output / 'pause-step-0000095.json', output / 'PAUSED.md', teacher_archive):
            artifact.add_file(str(path), name=path.name)
        logged = run.log_artifact(artifact)
        logged.wait()
        receipt = {'run_id':run.id, 'run_url':run.url, 'artifact':logged.qualified_name,
            'phase':'paused', 'env_step':95, 'optimizer_updates':0, 'snapshot_sha256':expected,
            'teacher_audit_sha256':hashlib.sha256(teacher_archive.read_bytes()).hexdigest(),
            'artifact_upload_acknowledged':True}
        run.summary.update({'suite_phase':'paused', 'paused_at_env_step':95,
                            'pause_artifact':logged.qualified_name, 'optimizer_updates':0})
        run.finish()
        run = None
        receipt['tracking_finish_acknowledged'] = True
        atomic_json(output / 'pause-wandb-delivery.json', receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        if run is not None:
            run.finish()
        os.environ.pop('WANDB_API_KEY', None)


if __name__ == '__main__':
    main()
