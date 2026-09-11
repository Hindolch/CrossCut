"""Pinned environment, reproducibility records, and episode metrics."""
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'environments/crafter'))
import crafter

ACTIONS = ('noop move_left move_right move_up move_down do sleep place_stone '
           'place_table place_furnace place_plant make_wood_pickaxe make_stone_pickaxe '
           'make_iron_pickaxe make_wood_sword make_stone_sword make_iron_sword').split()


def environment(seed, horizon):
    env = crafter.Env(seed=seed, length=horizon, reward=True, size=(64, 64))
    assert list(env.action_names) == ACTIONS
    assert Path(crafter.__file__).resolve().is_relative_to(ROOT / 'environments/crafter')
    return env


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this configuration.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')


def provenance(output, config):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    write_json(output / 'provenance.json', {
        'command': sys.argv, 'python': sys.version, 'git_commit': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--short'),
        'crafter_commit': subprocess.check_output(['git', '-C', str(ROOT / 'environments/crafter'),
                                                 'rev-parse', 'HEAD'], text=True).strip(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'cuda': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})


def summary(episodes, horizon):
    first = [e['first_diamond'] for e in episodes if e['first_diamond'] is not None]
    names = sorted(episodes[0]['achievements'])
    rates = {k: float(np.mean([e['achievements'][k] > 0 for e in episodes])) for k in names}
    return {
        'episodes': len(episodes), 'diamond_success_rate': len(first) / len(episodes),
        'first_diamond_mean_successes': float(np.mean(first)) if first else None,
        'first_diamond_median_successes': float(np.median(first)) if first else None,
        'failure_inclusive_time': float(np.mean([e['first_diamond'] if e['first_diamond'] is not None
                                               else horizon for e in episodes])),
        'cumulative_success': {str(t): sum(s <= t for s in first) / len(episodes)
                               for t in (1000, 2000, 4000, 6000, 8000, 10000)},
        'achievement_rates': rates,
        'crafter_score': float(np.expm1(np.mean(np.log1p(100 * np.array(list(rates.values())))))),
        'mean_reward': float(np.mean([e['reward'] for e in episodes]))}
