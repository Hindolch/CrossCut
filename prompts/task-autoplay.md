Operate CrossCut from this Codex task. The server is only a detached Crafter
environment on Jarvis; all GPT-6 Astra agents run here.

Once the user supplies Jarvis access, deploy the prepared server and establish
an SSH tunnel to its loopback port. Configure config.local.json and start the
local W&B monitor. Verify `python -m crosscut.client health` before gameplay.

Spawn exactly one GPT-6 Astra player subagent here for session diamonds-astra-1,
seed 42. Its total budget is 1,000,000 primitive game steps across episodes.
No pretraining. Give it the controls below. Never run another player concurrently.
The standalone local CLI autoplay launcher is an alternative controller; do
not run it concurrently with task agents controlling the same sessions.

Players should read prompts/player.md for the action rules, then:
1. Observe their session or create it with the assigned seed if it is missing.
2. Inspect the returned image_path with the image-view tool and read state.
3. Choose actions and pass the observed revision to `step`.
4. Repeat, retaining strategy and a map in their context. After death, use a new
   seed and the current revision to reset. Do not reset a winning episode.
5. If a request times out, use `recover` before submitting anything else.
6. Report a verified collect_diamond achievement immediately to the parent.

Controls, run from the repository with the local Python:
    python -m crosscut.client reset SESSION --seed SEED
    python -m crosscut.client observe SESSION
    python -m crosscut.client step SESSION --revision REVISION move_up do do
    python -m crosscut.client recover SESSION
    python -m crosscut.client reset SESSION --seed NEXT_SEED --revision REVISION

Parent: monitor player progress and W&B telemetry, resolve transport/runtime
failures, and preserve successful evidence. Stop the player on the first
verified diamond. Model narration alone is not completion. Use only normal
game actions; do not change Crafter rules or achievement counters.

For a future task heartbeat: stay quiet while status is unchanged or not
actionable. Notify on diamond success, failure requiring intervention, or
required user action. A detached server preserves the game, but actions require
an active local controller; do not claim play continues when controllers stop.

Stop if budget_exhausted is true. Never reset or create another session to replenish the budget.
