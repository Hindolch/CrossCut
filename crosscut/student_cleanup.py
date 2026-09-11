"""Remove this experiment's temporary SSH access after acknowledged delivery."""
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen
from .storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def cleanup():
    data = ROOT / 'data/local/exp002'
    key = ROOT / 'data/session-access-exp002/id_ed25519'
    public = key.with_suffix('.pub')
    report = ROOT / 'results/exp002-astra-seed0/cleanup.json'
    if not report.exists():
        delivery = json.loads((data / 'artifact-delivery-status.json').read_text())
        monitor = json.loads((data / 'monitor-status.json').read_text())
        if delivery.get('status') != 'complete' or monitor.get('status') != 'cleaning_up':
            raise RuntimeError('Cleanup requires verified artifact delivery, Git push, and finished tracking')
        checkpoint_path = data / 'cleanup-checkpoint.json'
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
        if checkpoint.get('phase') != 'remote_removed':
            fields = public.read_text().strip().split()
            if len(fields) < 2 or fields[0] != 'ssh-ed25519':
                raise ValueError('Unexpected temporary key format')
            target = ' '.join(fields[:2])
            atomic_json(checkpoint_path, {'phase': 'removing_remote', 'public_key': target})
        else:
            target = checkpoint['public_key']
        program = """from pathlib import Path
import json
target = TARGET
path = Path('/root/.ssh/authorized_keys')
lines = path.read_text().splitlines()
kept = [line for line in lines if ' '.join(line.split()[:2]) != target]
temporary = path.with_name('authorized_keys.crosscut-exp002-tmp')
temporary.write_text('\\n'.join(kept) + '\\n')
temporary.chmod(0o600)
temporary.replace(path)
receipt = {'temporary_server_keys_removed':len(lines)-len(kept), 'remaining_server_keys':len(kept), 'public_key':target}
Path('/home/CrossCut/data/experiments/exp002-astra-seed0/cleanup-receipt.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
""".replace('TARGET', repr(target))
        command = ['ssh', '-i', str(key), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                   '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                   'root@217.18.55.48', 'python3 -']
        if checkpoint.get('phase') == 'remote_removed':
            evidence = checkpoint['evidence']
        else:
            try:
                result = subprocess.run(command, input=program, text=True, capture_output=True, check=True, timeout=40)
                evidence = json.loads(result.stdout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # The existing tunnel can recover the receipt if the process died
                # after remote key removal but before storing the local checkpoint.
                with urlopen('http://127.0.0.1:8768/cleanup-receipt.json', timeout=15) as response:
                    evidence = json.load(response)
            if evidence.get('public_key') != target:
                raise RuntimeError('Cleanup receipt key identity mismatch')
            atomic_json(checkpoint_path, {'phase': 'remote_removed', 'public_key': target, 'evidence': evidence})
        (data / 'TUNNEL_STOP').touch()
        (data / 'teacher/STOP').touch()
        key.unlink(missing_ok=True)
        public.unlink(missing_ok=True)
        evidence.pop('public_key', None)
        atomic_json(report, dict(evidence, temporary_local_key_deleted=True,
            temporary_jarvis_account_key='5U8AHrsH removed during setup; API client closed',
            wandb='Tracking and artifact upload acknowledged before cleanup; process exits after this step',
            provider_api_tokens='Not revoked at provider; session credentials were held in memory only'))
    relative = report.relative_to(ROOT).as_posix()
    git = ['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT)]
    subprocess.run(git + ['add', '--', relative], check=True, capture_output=True)
    changed = subprocess.run(git + ['diff', '--cached', '--quiet', '--', relative], capture_output=True)
    if changed.returncode == 1:
        subprocess.run(git + ['commit', '--only', '-m', 'Record experiment 2 temporary access cleanup', '--', relative], check=True, capture_output=True)
    elif changed.returncode:
        raise RuntimeError('Could not inspect cleanup result staging')
    subprocess.run(git + ['push'], check=True, capture_output=True)


if __name__ == '__main__':
    cleanup()
