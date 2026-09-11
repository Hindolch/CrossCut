"""Render saved lossless observations; never guess actions or re-simulate gameplay."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from common import write_json


def render_training(run_dir, fps=10, part_steps=10000):
    run_dir = Path(run_dir)
    parts, writer = [], None
    step, previous_done = 0, True
    try:
        for archive in sorted(run_dir.glob('trajectory-*.npz')):
            with np.load(archive, allow_pickle=False) as data:
                for index, (before, after, done) in enumerate(zip(
                        data['observations'], data['next_observations'], data['dones'])):
                    assert hashlib.sha256(before.tobytes()).hexdigest() == data['observation_sha256'][index].decode()
                    assert hashlib.sha256(after.tobytes()).hexdigest() == data['next_observation_sha256'][index].decode()
                    if step % part_steps == 0:
                        if writer is not None:
                            writer.close()
                        path = run_dir / f'training-part-{len(parts):04d}.mp4'
                        if path.exists():
                            raise FileExistsError(path)
                        writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                                   pixelformat='yuv420p', quality=8)
                        parts.append({'filename': path.name, 'first_step': step + 1,
                                      'frames': 0, 'fps': fps})
                        writer.append_data(before.repeat(8, axis=0).repeat(8, axis=1))
                        parts[-1]['frames'] += 1
                    elif previous_done:
                        writer.append_data(before.repeat(8, axis=0).repeat(8, axis=1))
                        parts[-1]['frames'] += 1
                    writer.append_data(after.repeat(8, axis=0).repeat(8, axis=1))
                    step += 1
                    parts[-1]['last_step'] = step
                    parts[-1]['frames'] += 1
                    previous_done = bool(done)
    finally:
        if writer is not None:
            writer.close()
    if not parts:
        raise FileNotFoundError('No recorded trajectory archives; reconstruction is unavailable.')
    manifest = {'env_steps': step, 'parts': parts,
                'source': 'Lossless pre-action and post-action RGB arrays in trajectory-*.npz',
                'selection': 'Every recorded transition in chronological order; reset/terminal frames included',
                'encoding': 'H.264 compressed playback; NPZ source observations remain pixel-exact'}
    write_json(run_dir / 'training_videos.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--part-steps', type=int, default=10000)
    args = p.parse_args()
    assert args.fps > 0 and args.part_steps > 0
    print(json.dumps(render_training(args.run_dir, args.fps, args.part_steps), indent=2))
