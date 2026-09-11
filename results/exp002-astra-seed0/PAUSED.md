# Experiment 2 paused at step 95

The game is stopped. Native trainer PID 342 on Jarvis is suspended with SIGSTOP.
Its environment, tensors, RNG and pending rollout remain in memory. The local
teacher and telemetry STOP files are present, and the scheduled monitor is paused.
The Jarvis instance remains running to retain that process; instance charges can
continue. Do not reboot or pause the instance when relying on the in-memory state.

The snapshot archive contains every recorded transition, a complete 96-frame
video (reset plus 95 actions), SQLite backup, teacher labels, configuration,
provenance, and a policy/partial-rollout checkpoint. No optimizer update has
occurred. The initial policy and sampling RNG were reconstructed from the original
seed; all 95 sampled actions matched the actual recorded actions. That operation
performed no new environment actions or optimizer steps.

Archive SHA-256:
dfdfe4568bf695fa5c7a60ca29e089f1f4e60b527a1f4c834ec353498a9da3d0

Resume only on a new user instruction: restore the local teacher, tracking and
tunnel first, then SIGCONT the retained trainer. Never launch the original trainer
as a fresh run at step zero. If the process is lost, the stored seed/actions/frames
support verified environment reconstruction, and the saved policy/rollout preserves
the pre-update learning state; a disk-recovery entry point is not implemented yet.

Token-efficiency work is saved but not deployed: the source contains a candidate
eight-frame batch teacher endpoint and a fixed CLI working directory for better
prefix-cache reuse. The running experiment still uses the original single-frame
teacher and 512-step rollout. Batch integration and live verification remain
unfinished. These source edits must not be described as a demonstrated speedup.
