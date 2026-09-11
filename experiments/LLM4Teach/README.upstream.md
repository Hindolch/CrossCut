# Vision-LLM4Teach for Crafter

![Experiment architecture](docs/architecture.png)

**Experiment folders:** [Experiment 1 — active linear-decay Gemma 12B pipeline](exp1/README.md) · [Experiment 2 — staged-decay Gemma 12B ablation](exp2/README.md).
[Documentation index](docs/README.md).
[Automated detached execution and result retrieval](docs/AUTOMATION.md).
[GPU requirements and proposed query schedule](docs/GPU_AND_QUERY_PLAN.md).

Crafter-only adaptation of the [official LLM4Teach implementation](https://github.com/ZJLAB-AMMI/LLM4Teach),
pinned at `d8b537d0ea1e5d5437c9dce799d84a5f2239e90c` (verified against official HEAD on 2026-09-11).
The pristine original source is in `upstream/`; the active Crafter adaptation is
in this directory. See [source provenance and changes](docs/UPSTREAM.md) and the
[user-supplied experiment specification](docs/EXPERIMENT_SPEC.md).

`train.py` samples every executed action from a 2,185,906-parameter CNN–GRU
student. `ppo.py` adapts upstream's clipped PPO and soft-teacher objective.
`teacher.py` scores all 17 canonical action strings from raw images with a frozen
Gemma model. It never receives simulator state, rewards, achievements, captions,
inventory text, or prior actions. Those signals do not enter the teacher API.
The teacher's mean token log-likelihoods are normalized at temperature tau;
training adds tau-squared forward KL to PPO. Teacher weights decay per transition
from lambda=1 at step zero to zero at 300K steps. The teacher is unloaded after
that cutoff. The final budget is exactly 1M student-selected environment actions.

The environment is the pinned `environments/crafter` source. We instantiate
`crafter.Env(reward=True, size=(64,64), length=10000)` directly, matching
`CrafterReward-v1` without an additional Gym time-limit wrapper. Death terminates;
10K steps truncates. The value function bootstraps at time limits, not deaths;
GAE and GRU memory never cross episode boundaries. Diamond collection is logged
and does not terminate an episode. Craftax and MiniGrid are not runnable conditions.

Shared executable settings are under `configs/`. Experiment 01 uses
`configs/gemma12b.json`; `ppo.json` remains the comparison baseline and
`gemma31b.json` is retained for a possible later experiment. The baseline
never imports Transformers or downloads a teacher. `history_frames` accepts 1
(current image) or 4 (current plus up to three immediately prior images).
History is reset on each episode. The student's memory is independent.

## JarvisLabs setup (remote GPU only)

Use an existing CUDA GPU instance; this preparation does not provision or start
one. The specification proposes a single H200 for sequential BF16 teacher runs.
Actual memory usage and throughput must be measured remotely, especially before
the 31B run. This implementation places a complete teacher on one GPU, with no
tensor parallelism or pooled multi-GPU memory. Set `allocated_gpus` to the actual
number of reserved GPUs for the GPU-hour estimate.

Clone the complete repository, including its pinned Crafter submodule:

```bash
cd /home
git clone --recurse-submodules https://github.com/nileshsarkar-ai/Game-Playing-Agents-Crafter.git
cd Game-Playing-Agents-Crafter
git submodule update --init environments/crafter
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -c 'import torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install -r experiments/LLM4Teach/requirements.txt
wandb login
export HF_HOME=/home/huggingface-cache
```

Requires Python 3.11+ and a compatible CUDA PyTorch/torchvision pair provided by
the instance image. Transformers is pinned; resolved installed package versions
are captured with every run. Hugging Face downloads happen only when invoking a
teacher condition on the remote. If model access requires authentication, use
`hf auth login` there. Never add credentials to this repository.
Before final comparisons replace `teacher_revision: "main"` with the desired
model commit in each teacher config. The resolved model revision and prompt are
also recorded automatically. The processor is loaded from that same revision.

Start each long workload in tmux (run from the experiment directory after
activating the venv). Use a different run name for each attempt:

```bash
cd experiments/LLM4Teach
tmux new-session -d -s llm4teach-12b 'bash run_jarvis.sh gemma12b 0 gemma12b-seed0 > /home/llm4teach-12b.log 2>&1'
```

The script trains and then evaluates the final checkpoint without Gemma. Repeat
with seeds 1 and 2 for the planned multi-seed comparison. A one-seed run is
pipeline validation, not a scientific conclusion.
W&B online tracking is the default; use `wandb_mode: "offline"` explicitly if
network access is unavailable and sync the saved run afterward.

Short remote validation and a teacher-only competence diagnostic:

```bash
python train.py --config configs/ppo.json --steps 512 --output outputs/ppo-validation
python evaluate.py --teacher-config configs/gemma12b.json --episodes 5 --seed 20000 --output outputs/teacher12b-diagnostic
python train.py --config configs/gemma12b.json --steps 512 --output outputs/teacher-validation
```

The teacher-only diagnostic uses argmax and saves all 17 probabilities for every
action. It is explicitly separate from training. Student evaluation samples
from its untempered policy, uses no teacher, and always finishes all requested
episodes. Diagnostic seeds differ from final evaluation seeds.

## Outputs and research protocol

Each output directory must be new. It contains configuration, command, repository
and Crafter commits, dirty status, dependency versions, GPU details, W&B run/artifact links and files,
JSONL training metrics, completed episode records, and periodic student/optimizer
checkpoints. Incomplete training episodes are reported separately and never
counted as completed successes. Raw evaluation records preserve all achievement
counts, rewards, episode lengths and first-diamond times.

Final selection is the checkpoint at 1M steps, not the best observed checkpoint.
Evaluation uses 100 complete episodes with explicit environment constructor seeds
10000–10099 for each trained student. Crafter additionally derives its map seed
from its internal episode counter; each evaluation environment starts at reset 1.
Report per-training-seed results and the mean and sample standard deviation across
training seeds; do not select successful seeds or pool episodes as independent
training replicates. No training is run in this code-preparation task.

`evaluate.py` reports diamond success, conditional mean/median first-diamond
steps, cumulative success at 1K/2K/4K/6K/8K/10K, all 22 achievement rates, mean
reward and the standard score `exp(mean(log(1 + achievement_percent))) - 1`.
The failure-inclusive time assigns the full 10K horizon to every failed episode,
including early deaths. This is a specified penalized time, not a survival-model
estimate. Success-conditional times are null when no episode succeeds.

Training records teacher–student KL on labeled observations, teacher and student
entropy, label count and throughput, elapsed wall time, and reserved GPU-hours
(elapsed time times configured allocated GPUs). GPU-hours include model loading
and PPO work, not GPU utilization or a measured billing charge. W&B initialization
is outside this timer. One label requires 17 full candidate forward passes.

## Implementation limits

This is a direct-vision primitive-action adaptation, not an exact reproduction of
LLM4Teach's high-level option architecture. Hyperparameters besides the supplied
budget/architecture/schedule are provisional choices, held equal across conditions.
PPO uses 32-step truncated backpropagation through the GRU, initialized from stored
behavior-policy memory, and four epochs over 512-step rollouts. This introduces
the usual stale chunk-initial memory approximation after an update.

Candidate scoring is sequential and recomputes the image prefix to keep the
implementation auditable and bound peak memory. It is likely expensive at 300K
labels; measure throughput before a full run. No batched inference, persistent
rolling KV-cache history, distributed training, or reward-aware teacher ablation
is claimed. Persistent history and reward feedback in the specification remain
later controlled ablations; the first experiment supports current/four-frame
vision-only conditions. Compute per label is 17 model forwards; rollout storage
is linear in rollout length and pixel count, with recurrent backpropagation
memory bounded by the configured chunk length.

Checkpoints support evaluation and preserve optimizer/RNG information for
inspection. Exact interrupted-run resume is not implemented: Crafter world state,
rollout contents and recurrent state are not serialized. Start a new output
directory after interruption; do not combine partial runs into a nominal 1M run.
CUDA kernels and library versions can affect bitwise reproducibility.

Teacher inference and full GPU training require remote validation. Local checks
must not be interpreted as evidence of teacher competence or diamond success.

## References

- [Official LLM4Teach code](https://github.com/ZJLAB-AMMI/LLM4Teach), [paper](https://arxiv.org/abs/2311.13373).
- [Crafter environment](https://github.com/danijar/crafter).
- [Gemma 4 12B model card](https://huggingface.co/google/gemma-4-12B-it).
- [Gemma 4 Transformers API](https://huggingface.co/docs/transformers/model_doc/gemma4), [Unified API](https://huggingface.co/docs/transformers/model_doc/gemma4_unified).
- [Knowledge distillation](https://arxiv.org/abs/1503.02531), [PPO](https://arxiv.org/abs/1707.06347).

Aggregate independent training seeds within one condition:

```bash
python aggregate.py outputs/ppo-seed0-eval outputs/ppo-seed1-eval outputs/ppo-seed2-eval --output outputs/ppo-across-seeds.json
```

This rejects duplicate training seeds and differing configurations/evaluation
protocols. Undefined success-conditional times are reported with their count of
defined seeds; failure-inclusive metrics always retain every seed.
During teacher-guided rollouts, W&B and `metrics.jsonl` record teacher entropy and
teacher–student argmax agreement. Agreement is
`Pr[argmax pi_T = argmax pi_student]` over the observations in that rollout.
These metrics are undefined after the teacher is unloaded at 300K steps. Plot
completed-episode return on both sides of that boundary with:

```bash
python plot_learning.py --run outputs/<training-run>
```

The plot reports episodic return against global environment steps and does not
smooth across the teacher-removal boundary conceptually; the displayed rolling
mean is descriptive rather than an independent evaluation.
