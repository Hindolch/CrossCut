"""Remove this experiment's temporary SSH access after acknowledged delivery."""
import json
import subprocess
from pathlib import Path
from .storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def cleanup():
    data = ROOT / 'data/local/exp002'
    key = ROOT / 'data/session-access-exp002/id_ed25519'
    public = key.with_suffix('.pub')
    report = ROOT / 'results/exp002-astra-seed0/cleanup.json'
    if not report.exists():
        fields = public.read_text().strip().split()
        if len(fields) < 2 or fields[0] != 'ssh-ed25519':
            raise ValueError('Unexpected temporary key format')
        target = ' '.join(fields[:2])
        program = """from pathlib import Path
import json
target = TARGET
path = Path('/root/.ssh/authorized_keys')
lines = path.read_text().splitlines()
kept = [line for line in lines if ' '.join(line.split()[:2]) != target]
path.write_text('\\n'.join(kept) + '\\n')
path.chmod(0o600)
Path('/home/CrossCut/data/experiments/exp002-astra-seed0/HTTP_STOP').touch()
print(json.dumps({'temporary_server_keys_removed':len(lines)-len(kept), 'remaining_server_keys':len(kept)}))
""".replace('TARGET', repr(target))
        command = ['ssh', '-i', str(key), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                   '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                   'root@217.18.55.48', 'python3 -']
        result = subprocess.run(command, input=program, text=True, capture_output=True, check=True, timeout=40)
        evidence = json.loads(result.stdout)
        (data / 'TUNNEL_STOP').touch()
        (data / 'teacher/STOP').touch()
        key.unlink()
        public.unlink()
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
