# Experiment 1: direct-vision Gemma 4 12B teacher

## Research question

Does a frozen Gemma 4 12B vision-language teacher improve a recurrent PPO
student's ability to acquire Crafter achievements, especially a diamond, when
the teacher supplies a soft pseudo-policy over the 17 primitive actions during
the first 300,000 of 1,000,000 environment steps?

The teacher pseudo-policy is induced from length-normalized conditional
action-string likelihoods. It is not assumed to be Gemma's calibrated native
action probability.

## Fixed protocol

- Environment: pinned, unmodified `CrafterReward-v1` implementation.
- Teacher: frozen `google/gemma-4-12B-it`, BF16, one current RGB frame.
- Student: 2,185,906-parameter FP32 CNN-GRU actor-critic.
- Objective: PPO plus forward KL from teacher to student while teacher weight is
  positive.
- Schedule: teacher weight decreases linearly from 1 at step 0 to 0 at step
  300,000; PPO continues alone through step 1,000,000.
- Schedule identifier: `teacher_schedule: "linear"`.
- Episode horizon: at most 10,000 actions; death may terminate earlier.
- Final evaluation: frozen student, 100 complete episodes, constructor seeds
  10000–10099, full videos and lossless trajectories.
- Planned training seeds: 0, 1 and 2. The active allocation runs seed 0 only.

The executable settings are exposed inside this folder as [`config.json`](config.json), which points to the compatibility path [`../configs/gemma12b.json`](../configs/gemma12b.json) used by the active runner.
The run-specific operational record is [`../docs/RUN_20260911.md`](../docs/RUN_20260911.md).
The full scientific specification is [`../docs/EXPERIMENT_SPEC.md`](../docs/EXPERIMENT_SPEC.md).

## Commands

Run from `experiments/LLM4Teach`:

```bash
bash run_jarvis.sh gemma12b 0 exp01-gemma12b-seed0
python plot_learning.py --run outputs/exp01-gemma12b-seed0
```

The current remote run predates this registry and retains its original output
prefix recorded in `manifest.json`. The registry does not change or restart it.

## Primary measurements

- Diamond success rate and first-diamond time in final student-only evaluation.
- All 22 Crafter achievement rates, Crafter score and mean reward.
- Completed-episode return against global environment step before and after the
  300K teacher-removal boundary.
- Teacher entropy, student entropy, KL divergence and teacher-student argmax
  agreement while teacher labels are available.

Only final fixed-seed evaluation supports the main performance comparison.
Training curves and teacher diagnostics explain behavior but are not independent
test results.
