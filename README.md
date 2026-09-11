# CrossCut

**One GPT-6 Astra agent plays Python Crafter to mine and collect a diamond.**
No pretraining. Maximum **1,000,000 primitive game steps total**, across deaths,
resets, and server restarts. Success requires Crafter's collect_diamond achievement.

Crafter runs detached on Jarvis. The Astra player, Codex authentication, W&B
credentials, telemetry, and Git operations remain on the local computer.

```text
Local GPT-6 Astra player -> SSH tunnel -> detached Python Crafter on Jarvis
         |                               loopback HTTP + replay journal
         +-> local frames / W&B
         +-> Git autocommit + autopush
```

## Local setup

Use Python 3.13 and an existing local Codex login for the standalone launcher.
Agents in this Codex task can instead control the same bridge directly.

```powershell
.\scripts\setup-local.ps1
.\.venv-local\Scripts\python.exe -m crosscut.autoplay --dry-run
```

Setup copies config.example.json to the ignored config.local.json. Defaults:
exactly one gpt-6-astra player, high reasoning, autocommit and autopush enabled,
and W&B offline until its destination/authentication are available.
No model or gameplay starts during setup.

## Jarvis deployment

The supplied instance is root@217.18.55.48. An authorized SSH identity is required.
The game server needs Python 3.11-3.13 and only requirements-server.txt.
No OpenAI, Codex, or W&B login is needed there.

```bash
git clone --branch codex/crafter-setup https://github.com/Hindolch/CrossCut.git
cd CrossCut
bash scripts/start-server.sh
```

Set CROSSCUT_PYTHON=python3.13 if needed. The server binds to 127.0.0.1:8765,
survives SSH disconnects, and saves committed actions under data/server.
After an instance reboot, rerun the start command to reconstruct sessions.
Keep Python and dependency versions identical for deterministic restoration.
The journal counts logical game actions once; restoration replays the saved
history without consuming the player's remaining step budget.

Open an SSH tunnel on this computer:

```powershell
.\scripts\tunnel.ps1 -SshHost root@217.18.55.48
.\.venv-local\Scripts\python.exe -m crosscut.client health
```

A configured SSH alias can supply a private-key path or proxy. Keep the tunnel
running. The server needs no public HTTP port; game time advances only on actions.

## Play

The preferred mode for this task is exactly one GPT-6 Astra subagent following
prompts/task-autoplay.md, assigned to diamonds-astra-1. Run the local monitor.
Do not simultaneously use another controller for that session.

The optional standalone launcher also runs entirely on this computer:

```powershell
.\scripts\start-local.ps1 -Detached
```

This starts a hidden local Codex CLI supervisor and W&B monitor. It uses the
existing local login, refuses model fallback, and retains compact strategy memory.
Death starts a new episode without replenishing the total step budget. Success
or budget exhaustion stops the player. The local computer must stay powered and
connected; detaching only the game server does not keep model decisions running.

To request a clean stop:

```powershell
New-Item -ItemType File -Path data/local/diamonds/STOP
```

A STOP file prevents restarting autoplay until it is removed. The monitor remains
active to flush telemetry. Process IDs and logs are under data/.

Direct controls:

```powershell
python -m crosscut.client reset diamonds-astra-1 --seed 42
python -m crosscut.client observe diamonds-astra-1
python -m crosscut.client step diamonds-astra-1 --revision 1 move_up do do
python -m crosscut.client recover diamonds-astra-1
```

Use the revision returned by the latest observation. A step batch contains at
most 32 actions and stops early on death, diamond, or exhausted budget. Open
the returned absolute image_path to inspect the frame. local_map contains the
visible 9x7 window; the player is row 3, column 4, with y increasing downward.

Requests have idempotency IDs and revision checks to prevent duplicate/stale
actions after lost responses. recover retries an uncertain operation. Winning
episodes cannot be reset. Polling uses cached frames without changing game RNG.

## W&B, autocommit, and autopush

```powershell
python -m crosscut.monitor
python -m crosscut.monitor --once --no-wandb
```

W&B runs locally. Set wandb.entity, wandb.project, and mode online in the ignored
local config when the destination is known. Use existing local W&B authentication;
never commit keys. Offline testing needs no login and can later be uploaded with
wandb sync. Events also remain in the local spool for online replay.

The monitor logs game steps, inventory, achievements, episode outcomes, and frames.
Immutable progress milestones go under progress/<campaign>/<session>/<revision>.json.
Autocommit stages and commits only the named progress artifact; unrelated staged
work stays untouched. Autopush uses the configured branch upstream. Failed
commit/push deliveries are retried on monitor restart. monitor-status.json records
failures. Use a new campaign identifier for a new experiment; never create a new
session to evade the million-step budget.

The standalone controller writes SUCCESS.json when a diamond is verified and
BUDGET_EXHAUSTED.json if the limit is reached. The server's achievement counter and
retained action journal provide evidence; model narration is not a success signal.

## Keep awake

crosscut.awake uses a Windows execution-state request to prevent idle sleep while
its process remains alive. It is started locally for the task. The user's
requested automatic sleep/hibernate settings were also set to Never for both AC
and battery; the prior power settings are saved in ignored data/power-settings-before.txt.

## Tests

```powershell
py -3.13 -m venv .venv-test
$env:PYTHONUTF8 = '1'
.\.venv-test\Scripts\python.exe -m pip install -r requirements-server.txt -r requirements-local.txt
.\.venv-test\Scripts\python.exe -m unittest discover -s tests -v
```

The UTF-8 setting avoids an upstream Crafter packaging issue on Windows. Tests
exercise real Crafter through HTTP, reproducible replay, RNG-invariant polling,
retry/revision handling, budget enforcement, and telemetry/Git delivery failures.

## References

- [Crafter 1.8.3](https://pypi.org/project/crafter/1.8.3/) and [upstream source](https://github.com/danijar/crafter)
- [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [W&B Run API](https://docs.wandb.ai/models/ref/python/experiments/run)

## Complete recording

The server records the initial frame and every primitive action frame as PNG,
with a corresponding action/state JSON record. This includes terminal frames.
Replay creates no duplicate records. After the experiment, export lossless
per-episode MP4 videos, full JSONL action logs, and a verified session manifest:

```bash
python -m crosscut.recording --data-dir data/server --session diamonds-astra-1 --output data/results
```

Export includes only committed actions and verifies frame hashes. Videos use
10 presentation frames per second; this is a complete action-by-action recording,
not a recording of wall-clock waiting between model decisions. Original PNGs
remain available. Download the raw recordings and exported results before any
instance cleanup. Temporary API credentials are accepted through a hidden prompt
by scripts/jarvis-session.py and are never saved in a configuration file.


Live monitoring uses scripts/start-monitor-session.ps1 to accept a masked key
and launch crosscut.monitor_supervisor without saving it. The supervisor retries
tracking failures and preserves queued events. crosscut.tunnel reconnects after
network drops. The server start script also launches crosscut.finish, which
automatically creates full-session.mp4 after diamond success or budget exhaustion.
Set data/local/diamonds/MONITOR_STOP to flush and stop tracking after results;
TUNNEL_STOP stops the tunnel.
