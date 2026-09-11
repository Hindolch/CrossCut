# Experiment 2: Astra teaches the CNN+GRU student

The selected source is nileshsarkar-ai/Game-Playing-Agents-Crafter at commit
08961111faa8caac1e721b7e27df347c4e73d2eb, specifically experiments/LLM4Teach/exp2.
Only the 25 required source/configuration/documentation files were imported
(77,163 bytes). IMPORT_MANIFEST.json records every original Git blob and digest.
The original overview is README.upstream.md. The source's exp1 is a different
student ablation; it is not CrossCut's direct-Astra Experiment 1 baseline.

The original student.py, ppo.py, schedules.py, and train.py are unchanged.
astra_runtime.py adapts environment package loading, teacher transport,
recording, checkpoints, and telemetry around the original training loop.

The student has the original three-layer CNN, 512-unit GRU, actor and critic
(2,185,906 parameters). It observes only 64x64 RGB frames and selects every
environment action. The native Crafter reward is unchanged. It starts from
random initialization; there is no pretrained student or demonstration warmup.

A single local GPT-6 Astra process labels each student observation until step
300,000. It chooses an action using the current raw image alone (nearest-neighbor
enlarged for presentation). The adapter constructs a target with epsilon=0.05:
q(a)=0.95 for the chosen action plus 0.05/17 for each action. These are fixed
smoothed action labels, not Astra token probabilities or calibrated confidence.
The source forward KL teaches the student while PPO uses the student's own
on-policy log probabilities, native reward, clipped objective and recurrent BPTT.
Teacher weight falls from 1.0 to 0.2 by 180,000 steps, remains 0.2 until 300,000,
then becomes zero. This retains the source exp2 schedule, but the teacher target
is explicitly an adaptation of the original soft-logit distillation protocol.

The strict run cap is 1,000,000 total environment steps. No evaluation actions
are automatically added beyond that cap. Training continues after a diamond so
the student receives the full learning budget. The final 700,000 steps use no
teacher. Training performance is recorded, but it is not a held-out comparison
and cannot establish superiority to the 137-step, seed-42 direct Astra baseline.
The supplied frozen-policy evaluator is preserved for a separately budgeted study.

Every transition is committed to frames.sqlite immediately, including its
before/after RGB frames, reward, action and native achievement counters. Each
rollout also produces the source lossless NPZ archive. A checkpoint is written
after every PPO update in addition to the source's 100,000-step checkpoints.
On normal completion the source renderer automatically converts all trajectories
to chronological MP4 parts with reset and terminal frames. An interrupted process
retains its SQLite frames; it is not automatically restarted with a fresh budget.

Run with the Jarvis CUDA Python environment:

    python scripts/run-student-suite.py --output /home/CrossCut/data/experiments/exp002-astra-seed0

Use tmux for the GPU suite. The local Astra worker, SSH supervisor and W&B/Git
monitor have hidden PowerShell launchers in scripts/. Keep the laptop online and
awake during the teacher phase. A disconnected teacher causes waiting, never a
substitute teacher or unlabelled fallback. All inference credentials stay local;
W&B runs locally under the experiment-only key. Configuration, source provenance,
metrics and immutable progress commits are published to the experiment branch.
Raw videos and model checkpoints are experiment artifacts, not automatic Git blobs.

scripts/student-artifacts.py runs alongside training. It creates a consistent
SQLite backup (including committed WAL records), recovers partial-run video if
needed, and publishes a SHA-256 inventory. The local monitor downloads and
verifies every artifact, uploads the full collection to W&B, and pushes a compact
delivery receipt to Git. Successful completion then removes this experiment's
temporary SSH access and closes the tracking process. Failed runs retain access
for repair. The temporary Jarvis account SSH key was already removed and the API
client closed after setup; the provider API tokens themselves are not revoked.

The teacher uses the newer desktop-bundled Codex executable when available
(override with ASTRA_CODEX_EXE). During the real launch the older PATH CLI 0.142.5
was rejected by Astra; the installed desktop CLI 0.153.4 served the live request.

This is a real run, not a smoke test. Verification uses live teacher responses,
recorded environment actions, CUDA residency, PPO losses and parameter changes.
