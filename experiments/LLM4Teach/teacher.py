"""Frozen vision-only candidate likelihoods. No reward/state/action-history input."""
import torch
from PIL import Image
from common import ACTIONS


class Teacher:
    def __init__(self, config):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.temperature = config['temperature']
        self.device = config['teacher_device']
        self.prefix_cache = config.get('teacher_prefix_cache', False)
        self.dtype = getattr(torch, config.get('teacher_dtype', 'bfloat16'))
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config.get(
            'teacher_bf16_reduced_precision_reduction', True)
        self.processor = AutoProcessor.from_pretrained(config['teacher_model'], revision=config['teacher_revision'])
        self.model = AutoModelForMultimodalLM.from_pretrained(
            config['teacher_model'], revision=config['teacher_revision'],
            dtype=self.dtype, device_map=self.device).eval().requires_grad_(False)
        self.revision = self.model.config._commit_hash
        self.prompt = ('Mine a diamond as quickly as possible. Images are ordered oldest to newest. '
                       'Choose the next action for the newest image. Answer with exactly one allowed action name: '
                       + ', '.join(ACTIONS))

    @torch.inference_mode()
    def probabilities(self, frames):
        content = [{'type': 'image', 'image': Image.fromarray(frame)} for frame in frames]
        content.append({'type': 'text', 'text': self.prompt})
        prefix = self.processor.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt')
        prefix = {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                  if isinstance(v, torch.Tensor) else v for k, v in prefix.items()}
        if self.prefix_cache:
            return self.cached_probabilities(prefix)
        scores = []
        # Sequential candidates bound peak memory. No free-text generation or top-k approximation.
        for action in ACTIONS:
            ids = self.processor.tokenizer(action, add_special_tokens=False, return_tensors='pt')['input_ids'].to(self.device)
            inputs = dict(prefix)
            inputs['input_ids'] = torch.cat((prefix['input_ids'], ids), dim=1)
            inputs['attention_mask'] = torch.cat((prefix['attention_mask'], torch.ones_like(ids)), dim=1)
            for key in ('token_type_ids', 'mm_token_type_ids'):
                if key in inputs:
                    inputs[key] = torch.cat((inputs[key], torch.zeros_like(ids)), dim=1)
            # Only candidate-prediction positions need vocabulary logits.
            logits = self.model(**inputs, use_cache=False, logits_to_keep=ids.shape[1] + 1).logits
            logp = logits[:, -ids.shape[1]-1:-1].float().log_softmax(-1)
            scores.append(logp.gather(-1, ids.unsqueeze(-1)).mean())
        probs = torch.stack(scores).div(self.temperature).softmax(-1)
        if not torch.isfinite(probs).all():
            raise RuntimeError('Nonfinite teacher probabilities')
        return probs.cpu()

    def cached_probabilities(self, prefix):
        # Cache is scoped to this observation, never carried across environment steps.
        prefill = self.model(**prefix, use_cache=True, logits_to_keep=1)
        candidates = [self.processor.tokenizer(action, add_special_tokens=False,
                      return_tensors='pt')['input_ids'].to(self.device) for action in ACTIONS]
        count, length = len(candidates), max(ids.shape[1] for ids in candidates)
        ids = torch.full((count, length), self.processor.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, candidate in enumerate(candidates):
            ids[row, :candidate.shape[1]] = candidate[0]
            mask[row, :candidate.shape[1]] = 1
        first = prefill.logits[:, -1:].float().log_softmax(-1).expand(count, -1, -1)
        token_logp = first.gather(-1, ids[:, :1].unsqueeze(-1)).squeeze(-1)
        if length > 1:
            cache = prefill.past_key_values
            cache.batch_repeat_interleave(count)
            attention_mask = torch.cat((prefix['attention_mask'].repeat(count, 1), mask[:, :-1]), dim=1)
            continuation = self.model(input_ids=ids[:, :-1], attention_mask=attention_mask,
                past_key_values=cache, use_cache=True, logits_to_keep=length - 1).logits
            rest = continuation.float().log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            token_logp = torch.cat((token_logp, rest), dim=1)
        scores = (token_logp * mask).sum(-1) / mask.sum(-1)
        probs = scores.div(self.temperature).softmax(-1)
        if not torch.isfinite(probs).all():
            raise RuntimeError('Nonfinite teacher probabilities')
        return probs.cpu()
