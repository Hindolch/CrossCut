"""Adapted from ZJLAB-AMMI/LLM4Teach algos/ppo.py (see docs/UPSTREAM.md).

Retains clipped PPO, value clipping, entropy, and soft teacher cross-entropy;
uses contiguous recurrent chunks and per-transition environment-step annealing.
"""
import numpy as np
import torch
from torch.distributions import Categorical


def update(student, optimizer, rollout, config):
    device = config['device']
    tensors = {k: torch.stack(v).to(device) for k, v in rollout.items()}
    r = tensors
    advantage = torch.zeros_like(r['reward'])
    gae = torch.zeros((), device=device)
    for t in reversed(range(len(advantage))):
        delta = r['reward'][t] + config['gamma'] * r['bootstrap'][t] - r['value'][t]
        gae = delta + config['gamma'] * config['gae_lambda'] * (1 - r['done'][t]) * gae
        advantage[t] = gae
    returns = advantage + r['value']
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(config['epochs']):
        starts = np.random.permutation(range(0, len(advantage), config['sequence_length']))
        for start in starts:
            end = min(start + config['sequence_length'], len(advantage))
            # Stored pre-observation behavior memory; truncated BPTT within each chunk.
            hidden = r['hidden'][start].unsqueeze(0)
            logits, values = [], []
            for t in range(start, end):
                if t > start:
                    hidden = hidden * (1 - r['done'][t - 1])
                logit, value, hidden = student(r['image'][t].unsqueeze(0), hidden)
                logits.append(logit.squeeze(0))
                values.append(value.squeeze(0))
            logits, values = torch.stack(logits), torch.stack(values)
            sl = slice(start, end)
            pdf = Categorical(logits=logits)
            ratio = (pdf.log_prob(r['action'][sl]) - r['logprob'][sl]).exp()
            policy_loss = -torch.minimum(ratio * advantage[sl],
                ratio.clamp(1-config['clip'], 1+config['clip']) * advantage[sl]).mean()
            clipped = r['value'][sl] + (values-r['value'][sl]).clamp(-config['clip'], config['clip'])
            value_loss = torch.maximum((values-returns[sl]).square(), (clipped-returns[sl]).square()).mean()
            student_logp = (logits / config['temperature']).log_softmax(-1)
            teacher = r['teacher'][sl]
            kl = (teacher * (teacher.clamp_min(1e-30).log() - student_logp)).sum(-1)
            kd = config['temperature']**2 * kl
            entropy = pdf.entropy().mean()
            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy
            loss = loss + (r['weight'][sl] * kd).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config['max_grad_norm'])
            optimizer.step()
            valid = r['weight'][sl] > 0
            losses.append([loss.item(), entropy.item(), kl[valid].sum().item(), valid.sum().item()])
    a = np.array(losses)
    return {'loss': float(a[:, 0].mean()), 'student_entropy': float(a[:, 1].mean()),
            'teacher_student_kl': float(a[:, 2].sum()/a[:, 3].sum()) if a[:, 3].sum() else None}
