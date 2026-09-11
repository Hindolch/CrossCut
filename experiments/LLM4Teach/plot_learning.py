"""Plot episodic return against environment steps and mark teacher removal."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Training output directory.')
    parser.add_argument('--window', type=int, default=100, help='Completed-episode rolling window.')
    args = parser.parse_args()
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    episodes = [json.loads(line) for line in (run / 'episodes.jsonl').read_text().splitlines() if line]
    if not episodes:
        raise RuntimeError('No completed episodes in episodes.jsonl')

    steps = np.asarray([row['env_step'] for row in episodes])
    returns = np.asarray([row['reward'] for row in episodes], dtype=float)
    cutoff = config['teacher_cutoff']

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.scatter(steps, returns, s=7, alpha=0.18, color='#168c95', label='Completed-episode return')
    for phase, mask in enumerate((steps <= cutoff, steps > cutoff)):
        phase_returns, phase_steps = returns[mask], steps[mask]
        if len(phase_returns) == 0:
            continue
        window = min(args.window, len(phase_returns))
        rolling = np.convolve(phase_returns, np.ones(window) / window, mode='valid')
        axis.plot(phase_steps[window - 1:], rolling, linewidth=2.2, color='#082b66',
                  label=f'Rolling mean ({args.window} episodes)' if phase == 0 else None)
    axis.axvline(cutoff, color='#7a2cb7', linestyle='--', linewidth=2,
                 label=f'Teacher removed ({cutoff:,} steps)')
    if config.get('teacher_schedule') == 'staged':
        decay_end = config['teacher_decay_end']
        axis.axvline(decay_end, color='#7a2cb7', linestyle=':', linewidth=2,
                     label=f'KL plateau begins ({decay_end:,} steps)')
    axis.axvspan(0, cutoff, color='#7a2cb7', alpha=0.05, label='PPO + teacher KL')
    axis.axvspan(cutoff, config['total_steps'], color='#ee7b22', alpha=0.05, label='PPO only')
    axis.set(xlabel='Environment step', ylabel='Episode return J',
             title='Crafter learning return before and after teacher removal')
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(run / 'learning-return.png', dpi=200)
    figure.savefig(run / 'learning-return.pdf')


if __name__ == '__main__':
    main()
