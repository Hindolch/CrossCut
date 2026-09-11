"""Operational adapter around the unchanged imported CNN/GRU and PPO trainer.

No policy, reward, rollout, optimizer, or teacher schedule implementation is
replaced. Astra labels arrive from the local controller over an SSH tunnel.
"""
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
import zlib

import numpy as np
import torch
from PIL import Image
import common
import ppo
import tracking

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crosscut.storage import atomic_json


class Runtime:
    def __init__(self, output, config):
        self.out = Path(output)
        self.config = config
        self.step = self.updates = self.episode = self.labels = 0
        self.diamonds = 0
        self.started = time.time()
        self.db = None

    def status(self, phase, **extra):
        atomic_json(self.out / 'status.json', dict(
            phase=phase, env_step=self.step, updates=self.updates,
            episode=self.episode, teacher_labels=self.labels,
            diamond_episodes=self.diamonds, total_steps=self.config['total_steps'],
            device=self.config['device'], updated_at=time.time(),
            wall_seconds=time.time()-self.started, **extra))

    def provenance(self, output, config):
        output.mkdir(parents=True, exist_ok=False)
        common.write_json(output / 'config.json', config)
        git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True).strip()
        import crafter
        package = Path(crafter.__file__).resolve().parent
        common.write_json(output / 'provenance.json', {
            'command': sys.argv, 'python': sys.version,
            'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
            'crafter_version': importlib.metadata.version('crafter'),
            'crafter_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(package.glob('*.py'))},
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'cuda': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'upstream_unchanged_sha256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                         for name in ('train.py', 'student.py', 'ppo.py', 'schedules.py')},
            'teacher_target': 'Action label smoothed with epsilon=0.05; no Astra log probabilities',
            'evaluation': 'No extra environment steps beyond the 1,000,000 training cap are launched.'})
        for name in ('metrics.jsonl', 'episodes.jsonl', 'teacher_labels.jsonl'):
            (output / name).touch()
        self.db = sqlite3.connect(output / 'frames.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE transitions (step INTEGER PRIMARY KEY, episode INTEGER, before_rgb BLOB, after_rgb BLOB, action INTEGER, reward REAL, done INTEGER, info TEXT)')
        self.db.execute('CREATE TABLE resets (step INTEGER PRIMARY KEY, episode INTEGER, rgb BLOB)')
        self.db.commit()
        self.status('initializing')

    def environment(self, seed, horizon):
        import crafter
        env = crafter.Env(seed=seed, length=horizon, reward=True, size=(64, 64))
        assert list(env.action_names) == common.ACTIONS
        runtime = self

        class RecordedEnvironment:
            def reset(self):
                self.obs = env.reset()
                self.has_diamond = False
                runtime.db.execute('INSERT INTO resets VALUES (?,?,?)',
                    (runtime.step, runtime.episode, zlib.compress(self.obs.tobytes())))
                runtime.db.commit()
                return self.obs

            def step(self, action):
                if runtime.step >= runtime.config['total_steps']:
                    raise RuntimeError('Environment step budget exhausted')
                before = self.obs
                after, reward, done, info = env.step(action)
                runtime.step += 1
                record = {k: info[k] for k in ('discount', 'achievements', 'inventory') if k in info}
                runtime.db.execute('INSERT INTO transitions VALUES (?,?,?,?,?,?,?,?)',
                    (runtime.step, runtime.episode, zlib.compress(before.tobytes()),
                     zlib.compress(after.tobytes()), int(action), float(reward), int(done), json.dumps(record)))
                runtime.db.commit()
                if not self.has_diamond and info['achievements']['collect_diamond'] > 0:
                    self.has_diamond = True
                    runtime.diamonds += 1
                    common.append_json(runtime.out / 'diamonds.jsonl',
                        dict(env_step=runtime.step, episode=runtime.episode, achievements=info['achievements']))
                self.obs = after
                if done:
                    runtime.episode += 1
                runtime.status('collecting')
                return after, reward, done, info

        return RecordedEnvironment()

    def make_teacher(self, config):
        runtime = self

        class AstraTeacher:
            revision = 'gpt-6-astra via local Codex; resolved weight revision unavailable'
            prompt = (ROOT / 'prompts/student-teacher.md').read_text()

            def probabilities(self, frames):
                request_id = f"{config['run_id']}-step-{runtime.step:07d}"
                encoded = []
                for frame in frames:
                    buffer = io.BytesIO()
                    # Pixel-preserving enlargement improves image presentation only.
                    Image.fromarray(frame).resize((512, 512), Image.Resampling.NEAREST).save(buffer, format='PNG')
                    encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
                body = json.dumps(dict(request_id=request_id, model='gpt-6-astra', frames=encoded)).encode()
                retries = 0
                while True:
                    runtime.status('waiting_teacher', request_id=request_id, network_retries=retries)
                    try:
                        request = urllib.request.Request(config['teacher_url']+'/predict', data=body,
                            headers={'Content-Type': 'application/json'}, method='POST')
                        with urllib.request.urlopen(request, timeout=660) as response:
                            result = json.load(response)
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code not in (502, 503, 504):
                            raise RuntimeError(f'Teacher rejected request {request_id}: HTTP {exc.code}') from exc
                    except (urllib.error.URLError, TimeoutError, ConnectionError):
                        pass
                    retries += 1
                    time.sleep(min(30, retries * 2))
                if result.get('model') != 'gpt-6-astra' or result.get('request_id') != request_id:
                    raise ValueError('Teacher response identity mismatch')
                if result.get('target_kind') != 'smoothed_teacher_action' or result.get('epsilon') != 0.05:
                    raise ValueError('Teacher target protocol mismatch')
                target = torch.tensor(result['probabilities'], dtype=torch.float32)
                if target.shape != (17,) or not torch.isfinite(target).all() or (target < 0).any() or abs(target.sum().item()-1) > 1e-5:
                    raise ValueError('Invalid teacher target')
                result.update(env_step=runtime.step, network_retries=retries)
                common.append_json(runtime.out / 'teacher_labels.jsonl', result)
                runtime.labels += 1
                return target

        return AstraTeacher()

    def install(self):
        common.write_json = atomic_json
        common.provenance = self.provenance
        common.environment = self.environment
        sys.modules['teacher'] = types.SimpleNamespace(Teacher=self.make_teacher)
        # W&B credentials remain on the laptop. Its monitor streams these exact logs.
        run = types.SimpleNamespace(log=lambda *a, **k: None,
            define_metric=lambda *a, **k: None, finish=lambda: None)
        sys.modules['wandb'] = types.SimpleNamespace(init=lambda **kwargs: run)
        tracking.record_run = lambda run, out: atomic_json(out / 'wandb_run.json', {
            'id': self.config['run_id'], 'url': self.config['wandb_url'],
            'transport': 'Local authenticated monitor over SSH; no credential on GPU'})
        tracking.upload_results = lambda *a: None
        original_update = ppo.update

        def update(student, optimizer, rollout, config):
            self.status('updating')
            before = torch.cat([p.detach().flatten() for p in student.parameters()])
            metrics = original_update(student, optimizer, rollout, config)
            after = torch.cat([p.detach().flatten() for p in student.parameters()])
            metrics['parameter_delta_l2'] = torch.linalg.vector_norm(after-before).item()
            metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated()/1024**2
            self.updates += 1
            metrics['updates'] = self.updates
            path = self.out / 'checkpoint-latest.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(dict(student=student.state_dict(), optimizer=optimizer.state_dict(),
                env_step=self.step, updates=self.updates, config=config,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state()), temporary)
            os.replace(temporary, path)
            self.status('collecting', last_update=metrics)
            return metrics

        ppo.update = update


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    runtime = Runtime(args.output, config)
    runtime.install()
    import train
    try:
        train.main()
        runtime.status('complete')
    except BaseException as exc:
        if runtime.out.exists():
            runtime.status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if runtime.db is not None:
            runtime.db.close()


if __name__ == '__main__':
    main()
