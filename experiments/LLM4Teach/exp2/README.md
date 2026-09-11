# Experiment 2: staged teacher decay

## Research question

Does retaining a low, constant amount of Gemma guidance before hard removal improve final student-only Crafter performance compared with Experiment 1's continuous linear decay to zero?

This is a single-variable ablation. Teacher model, raw-image input, prompt, 17-action likelihood pseudo-policy, student, PPO implementation, KL direction, environment, seeds, training budget, checkpoints, evaluation and recording are identical to Experiment 1. Only the scalar teacher-loss schedule changes.

## Schedule

Let `s` be the global environment-action count:

```text
0 <= s < 180,000:       lambda(s) = 1.0 - 0.8 * s / 180,000
180,000 <= s < 300,000: lambda(s) = 0.2
300,000 <= s <= 1M:     lambda(s) = 0; teacher unloaded
```

Thus the student uses PPO plus decreasing KL, then PPO plus a 0.2 KL plateau,
then PPO alone. The 180K boundary and 0.2 plateau are the predefined Experiment
2 intervention. Keeping the hard-removal point at 300K makes the comparison with
Experiment 1 clean. The absolute initial weight remains 1.0, so only the schedule
shape changes.

Configuration: [`config.json`](config.json). Experiment 1 reference: [`../exp1/`](../exp1/).
GPU selection and measured projections: [`GPU_SELECTION.md`](GPU_SELECTION.md).
Architecture: [`architecture.png`](architecture.png) • editable source: [`architecture.svg`](architecture.svg) • [`A4 PDF`](architecture-a4-landscape.pdf). The existing `docs/architecture copy.png` path is retained as a compatibility copy.

## Execution

Run from `experiments/LLM4Teach` after validation:

```bash
bash run_jarvis.sh exp2 0 exp2-staged-gemma12b-seed0
```

Final evaluation remains student-only over 100 complete episodes using seeds 10000–10099. Run training seeds 0, 1 and 2 for the planned comparison; do not interpret a single seed as the final ablation result.
