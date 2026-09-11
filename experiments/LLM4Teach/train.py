"""Single-stream on-policy Crafter PPO and frozen visual distillation."""
import argparse
import hashlib
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.distributions import Categorical
from common import append_json, environment, provenance, seed_all, write_json
from ppo import update
from schedules import teacher_weight
from student import Student
from tracking import record_run, upload_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int, help='Explicitly overrides training budget; recorded in config.')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.seed is not None:
        c['seed'] = args.seed
    if args.steps is not None:
        c['total_steps'] = args.steps
    assert c['environment'] == 'crafter'
    assert c['history_frames'] in (1, 4), 'Persistent KV history is a deferred ablation.'
    assert c['temperature'] > 0 and c['total_steps'] > 0 and c['teacher_cutoff'] >= 0
    seed_all(c['seed'], c['device'])
    out = Path(args.output)
    provenance(out, c)
    import wandb
    run = wandb.init(project=c['wandb_project'], name=out.name, config=c, mode=c['wandb_mode'], dir=str(out))
    run.define_metric('env_step')
    run.define_metric('*', step_metric='env_step')
    record_run(run, out)
    started = time.monotonic()
    student = Student().to(c['device'])
    optimizer = torch.optim.Adam(student.parameters(), lr=c['learning_rate'], eps=1e-5)
    write_json(out / 'student.json', {'parameters': sum(p.numel() for p in student.parameters())})
    teacher = None
    if c['teacher_model'] is not None and c['teacher_cutoff'] > 0:
        from teacher import Teacher
        teacher = Teacher(c)
        write_json(out / 'teacher.json', {'model': c['teacher_model'], 'resolved_revision': teacher.revision,
                                         'prompt': teacher.prompt})
    env = environment(c['seed'], c['horizon'])
    obs = env.reset()
    hidden = torch.zeros(1, 512, device=c['device'])
    history = deque(maxlen=c['history_frames'])
    step = episode = episode_step = 0
    episode_reward = 0.0
    first_diamond = None
    teacher_seconds = 0.0
    teacher_labels = 0
    next_checkpoint = c['checkpoint_interval']
    trajectory_index = []
    record_frames = c.get('record_training_frames', False)
    while step < c['total_steps']:
        r = {k: [] for k in ('image', 'hidden', 'action', 'logprob', 'value', 'reward',
                             'bootstrap', 'done', 'teacher', 'weight')}
        teacher_entropy = []
        teacher_student_agreement = []
        post_observations = []
        for _ in range(min(c['rollout_steps'], c['total_steps'] - step)):
            image = torch.as_tensor(obs.copy(), device=c['device'])
            history.append(obs.copy())
            with torch.no_grad():
                logits, value, next_hidden = student(image.unsqueeze(0), hidden)
                pdf = Categorical(logits=logits)
                action = pdf.sample()
            # Teacher has access only to pre-action raw images. Student is the sole actor.
            weight = teacher_weight(c, step, teacher is not None)
            target = torch.zeros(17)
            if weight > 0:
                label_start = time.monotonic()
                target = teacher.probabilities(list(history))
                teacher_seconds += time.monotonic() - label_start
                teacher_labels += 1
                teacher_entropy.append(float(-(target * target.clamp_min(1e-30).log()).sum()))
                teacher_student_agreement.append(
                    float(target.argmax().item() == logits.squeeze(0).argmax().item()))
            next_obs, reward, done, info = env.step(action.item())
            if record_frames:
                post_observations.append(next_obs.copy())
            with torch.no_grad():
                # Bootstrap true time limits from final observation; death has zero discount.
                _, next_value, _ = student(torch.as_tensor(next_obs, device=c['device']).unsqueeze(0), next_hidden)
                bootstrap = next_value.squeeze(0) * info['discount']
            values = {'image': image.cpu(), 'hidden': hidden.squeeze(0).cpu(), 'action': action.squeeze(0).cpu(),
                      'logprob': pdf.log_prob(action).squeeze(0).cpu(), 'value': value.squeeze(0).cpu(),
                      'reward': torch.tensor(float(reward)), 'bootstrap': bootstrap.cpu(),
                      'done': torch.tensor(float(done)), 'teacher': target, 'weight': torch.tensor(weight)}
            for key, val in values.items():
                r[key].append(val)
            step += 1
            episode_step += 1
            episode_reward += reward
            if first_diamond is None and info['achievements']['collect_diamond'] > 0:
                first_diamond = episode_step
            obs, hidden = next_obs, next_hidden
            if done:
                record = {'episode': episode, 'env_step': step, 'length': episode_step,
                          'reward': episode_reward, 'first_diamond': first_diamond,
                          'achievements': info['achievements']}
                append_json(out / 'episodes.jsonl', record)
                run.log({'env_step': step, 'episode_reward': episode_reward, 'episode_length': episode_step,
                         'diamond': float(first_diamond is not None)})
                episode += 1
                episode_step, episode_reward, first_diamond = 0, 0.0, None
                obs = env.reset()
                hidden = torch.zeros_like(hidden)
                history.clear()
        if record_frames:
            archive = out / f'trajectory-{step-len(post_observations)+1:07d}-{step:07d}.npz'
            np.savez_compressed(archive,
                observations=torch.stack(r['image']).numpy(),
                next_observations=np.stack(post_observations),
                observation_sha256=np.array([hashlib.sha256(x.numpy().tobytes()).hexdigest()
                                             for x in r['image']], dtype='S64'),
                next_observation_sha256=np.array([hashlib.sha256(x.tobytes()).hexdigest()
                                                  for x in post_observations], dtype='S64'),
                actions=torch.stack(r['action']).numpy(),
                rewards=torch.stack(r['reward']).numpy(),
                dones=torch.stack(r['done']).numpy())
            trajectory_index.append({'filename': archive.name,
                'first_step': step-len(post_observations)+1, 'last_step': step,
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
            write_json(out / 'trajectories.json', {'chunks': trajectory_index,
                'observation_shape': [64, 64, 3], 'lossless': True,
                'coverage': 'All transitions from this training process start'})
        metrics = update(student, optimizer, r, c)
        # Unload teacher after the cutoff; subsequent behavior and optimization use PPO only.
        if teacher is not None and step >= c['teacher_cutoff']:
            del teacher
            teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elapsed = time.monotonic() - started
        metrics.update(env_step=step, teacher_labels=teacher_labels, teacher_seconds=teacher_seconds,
                       teacher_labels_per_second=teacher_labels/teacher_seconds if teacher_seconds else None,
                       teacher_entropy=float(np.mean(teacher_entropy)) if teacher_entropy else None,
                       teacher_student_argmax_agreement=float(np.mean(teacher_student_agreement))
                       if teacher_student_agreement else None,
                       wall_seconds=elapsed, allocated_gpu_hours=elapsed/3600*c['allocated_gpus'])
        append_json(out / 'metrics.jsonl', metrics)
        run.log({k: v for k, v in metrics.items() if v is not None})
        print(json.dumps(metrics), flush=True)
        if step >= next_checkpoint or step == c['total_steps']:
            path = out / f'checkpoint-{step}.pt'
            torch.save({'student': student.state_dict(), 'optimizer': optimizer.state_dict(),
                        'env_step': step, 'config': c, 'torch_rng': torch.get_rng_state(),
                        'numpy_rng': np.random.get_state()}, path)
            next_checkpoint += c['checkpoint_interval']
    write_json(out / 'completion.json', {'env_steps': step, 'completed_episodes': episode,
        'unfinished_episode_steps': episode_step, 'wall_seconds': time.monotonic()-started})
    if record_frames:
        from render_training import render_training
        videos = render_training(out, fps=10, part_steps=10000)
        completed = json.loads((out / 'completion.json').read_text())
        completed['training_gameplay'] = videos
        write_json(out / 'completion.json', completed)
    upload_results(run, out, 'training-results')
    run.finish()


if __name__ == '__main__':
    main()
