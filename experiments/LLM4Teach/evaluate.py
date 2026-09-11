"""Complete fixed-seed episodes: student-only evaluation or teacher-only diagnostic."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import imageio.v2 as imageio
from torch.distributions import Categorical
from common import append_json, environment, provenance, seed_all, summary, write_json
from student import Student
from tracking import record_run, upload_results


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--teacher-config', help='Diagnostic only; teacher acts directly, no training.')
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--video-episodes', type=int, default=None, help='Record all episodes by default; 0 disables video.')
    p.add_argument('--video-fps', type=int, default=10)
    args = p.parse_args()
    assert (args.video_episodes is None or args.video_episodes >= 0) and args.video_fps > 0
    assert args.episodes > 0
    seed_all(args.seed, args.device)
    teacher = None
    if args.checkpoint:
        # Load only trusted checkpoints produced by this experiment.
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        c = checkpoint['config']
        student = Student().to(args.device).eval()
        student.load_state_dict(checkpoint['student'])
    else:
        from teacher import Teacher
        c = json.loads(Path(args.teacher_config).read_text())
        teacher = Teacher(c)
    out = Path(args.output)
    provenance(out, dict(c, evaluation=vars(args)))
    import wandb
    run = wandb.init(project=c['wandb_project'], name=out.name, job_type='evaluation',
                     config=dict(c, evaluation=vars(args)), mode=c['wandb_mode'], dir=str(out))
    record_run(run, out)
    if teacher:
        write_json(out / 'teacher.json', {'resolved_revision': teacher.revision, 'prompt': teacher.prompt})
    records, videos, trajectories = [], [], []
    started = time.monotonic()
    for episode in range(args.episodes):
        env = environment(args.seed + episode, c['horizon'])
        obs = env.reset()
        writer = None
        if args.video_episodes is None or episode < args.video_episodes:
            video_path = out / f'gameplay-episode-{episode:03d}.mp4'
            writer = imageio.get_writer(video_path, fps=args.video_fps, codec='libx264',
                                        pixelformat='yuv420p', quality=8)
            writer.append_data(obs.repeat(8, axis=0).repeat(8, axis=1))
        hidden = torch.zeros(1, 512, device=args.device)
        history = deque(maxlen=c['history_frames'])
        observations, next_observations = [], []
        actions, rewards, dones = [], [], []
        total_reward, first = 0.0, None
        for step in range(1, c['horizon'] + 1):
            observations.append(obs.copy())
            with torch.inference_mode():
                if teacher:
                    history.append(obs.copy())
                    probabilities = teacher.probabilities(list(history))
                    action = probabilities.argmax().item()
                    append_json(out / 'teacher_actions.jsonl', {'episode': episode, 'step': step,
                        'probabilities': probabilities.tolist(), 'action': action})
                else:
                    logits, _, hidden = student(torch.as_tensor(obs, device=args.device).unsqueeze(0), hidden)
                    action = Categorical(logits=logits).sample().item()
            obs, reward, done, info = env.step(action)
            next_observations.append(obs.copy())
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            if writer is not None:
                writer.append_data(obs.repeat(8, axis=0).repeat(8, axis=1))
            total_reward += reward
            if first is None and info['achievements']['collect_diamond'] > 0:
                first = step
            if done:
                break
        trajectory_path = out / f'evaluation-episode-{episode:03d}.npz'
        np.savez_compressed(
            trajectory_path,
            observations=np.stack(observations),
            next_observations=np.stack(next_observations),
            observation_sha256=np.array(
                [hashlib.sha256(x.tobytes()).hexdigest() for x in observations], dtype='S64'),
            next_observation_sha256=np.array(
                [hashlib.sha256(x.tobytes()).hexdigest() for x in next_observations], dtype='S64'),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_))
        trajectory = {
            'episode': episode,
            'filename': trajectory_path.name,
            'transitions': step,
            'sha256': hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        }
        trajectories.append(trajectory)
        write_json(out / 'evaluation_trajectories.json', {
            'episodes': trajectories,
            'observation_shape': [64, 64, 3],
            'lossless': True,
            'coverage': 'Every transition of every fixed-seed evaluation episode',
        })
        record = {'episode': episode, 'seed': args.seed + episode, 'length': step,
                  'reward': total_reward, 'first_diamond': first, 'achievements': info['achievements'],
                  'trajectory_file': trajectory_path.name, 'trajectory_sha256': trajectory['sha256']}
        if writer is not None:
            writer.close()
            video = dict(record, filename=video_path.name, fps=args.video_fps,
                         frames=step + 1, selection='all fixed-seed evaluation episodes' if args.video_episodes is None else 'first fixed-seed evaluation episodes',
                         encoding='H.264; every environment frame, including reset and terminal')
            videos.append(video)
            run.log({f'gameplay/episode_{episode:03d}': wandb.Video(str(video_path),
                     caption=f'Seed {record["seed"]}; reward {total_reward:.2f}; diamond {first}',
                     format='mp4')}, step=episode, commit=False)
        records.append(record)
        append_json(out / 'episodes.jsonl', record)
        run.log({'episode_reward': total_reward, 'episode_length': step,
                 'diamond': float(first is not None)}, step=episode)
    result = summary(records, c['horizon'])
    result['gameplay_videos'] = videos
    write_json(out / 'videos.json', videos)
    result['wall_seconds'] = time.monotonic() - started
    write_json(out / 'summary.json', result)
    run.summary.update(result)
    upload_results(run, out, 'evaluation-results')
    run.finish()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
