"""Local telemetry spool -> W&B and scoped, retryable progress commits."""

import argparse
import json
import subprocess
import time

from .client import Client, ROOT, load_config
from .storage import atomic_json, process_lock


def progress_commit(config, state):
    # Immutable milestone files let a retry identify exactly the snapshot whose
    # commit or push failed, even after the working file has already been written.
    directory = ROOT / "progress" / config["campaign"] / state["session"]
    destination = directory / f"{state['revision']:012d}.json"
    if not destination.exists():
        previous_paths = sorted(path for path in directory.glob("*.json") if path.name < destination.name)
        previous = json.loads(previous_paths[-1].read_text(encoding="utf-8")) if previous_paths else None
        milestone = (previous is None or previous["episode"] != state["episode"]
                     or state["done"] or state["success"]
                     or previous["achievements"] != state["achievements"])
        if not milestone:
            return
        summary = {k: v for k, v in state.items() if k not in ("image_path", "observed_at", "local_map")}
        summary.update(model=config["model"], objective="collect_diamond",
                       evidence="Crafter engine achievement counter; action journal retained on game server")
        atomic_json(destination, summary)
    if not config.get("autocommit", True):
        return
    relative = destination.relative_to(ROOT).as_posix()
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    subprocess.run(command + ["add", "--", relative], check=True, capture_output=True)
    changed = subprocess.run(command + ["diff", "--cached", "--quiet", "--", relative], capture_output=True)
    if changed.returncode == 1:
        subprocess.run(command + ["commit", "--only", "-m",
                       f"Crafter {state['session']}: episode {state['episode']}, step {state['steps']}",
                       "--", relative], check=True, capture_output=True)
    elif changed.returncode != 0:
        raise RuntimeError("Could not inspect staged progress")
    # Push is retried even when the preceding commit already succeeded.
    if config.get("autopush", False):
        subprocess.run(command + ["push"], check=True, capture_output=True)


def delivery_cursor(path):
    cursor = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    # Legacy cursors cannot prove W&B delivery: --no-wandb also advanced them.
    return {"version": 2, "progress": cursor.get("progress", {}) if cursor.get("version") == 2 else {},
            "wandb": cursor.get("wandb", {}), "success": cursor.get("success", False)}


def run_monitor(config, once=False, no_wandb=False):
    client = Client(config)
    root = client.root
    with process_lock(root / "monitor.lock"):
        cursor_path = root / "monitor-cursor.json"
        status_path = root / "monitor-status.json"
        cursor = delivery_cursor(cursor_path)
        progress_sink = json.dumps([config.get("autocommit", True), config.get("autopush", False)])
        progressed = cursor["progress"].setdefault(progress_sink, {})
        initial_wandb = json.loads(json.dumps(cursor["wandb"]))
        run = None
        phase = "starting"
        try:
            atomic_json(status_path, {"status": "running", "phase": phase, "updated_at": time.time()})
            if not no_wandb:
                import wandb
                settings = config["wandb"]
                # Test/offline runs must never suppress subsequent online delivery.
                sink = json.dumps([settings["mode"], settings.get("entity"), settings["project"]])
                delivered = cursor["wandb"].setdefault(sink, {})
                options = dict(project=settings["project"], entity=settings.get("entity"), mode=settings["mode"],
                               config={"model": config["model"], "objective": "collect_diamond", "agents": config["agents"]},
                               dir=str(root), name=config["campaign"])
                if settings["mode"] == "online":
                    options.update(id=config["campaign"], resume="allow")
                phase = "wandb_init"
                run = wandb.init(**options)
                registered_sessions = set()
            while True:
                for path in sorted((root / "events").glob("*/*.json")):
                    state = json.loads(path.read_text(encoding="utf-8"))
                    session, revision = state["session"], state["revision"]
                    if revision > progressed.get(session, 0):
                        phase = "progress_commit"
                        progress_commit(config, state)
                        progressed[session] = revision
                        cursor["success"] = bool(cursor["success"] or state["success"])
                        atomic_json(cursor_path, cursor)
                    if run and revision > delivered.get(session, 0):
                        phase = "wandb_log"
                        if session not in registered_sessions:
                            run.define_metric(f"{session}/total_steps")
                            run.define_metric(f"{session}/*", step_metric=f"{session}/total_steps")
                            registered_sessions.add(session)
                        metrics = {f"{session}/{key}": state[key] for key in
                                   ("steps", "episode", "revision", "reward", "total_reward", "success", "done")}
                        metrics.update({
                            f"{session}/total_steps": state.get("total_steps", state["steps"]),
                            f"{session}/step_budget": state.get("step_budget", 10000),
                            f"{session}/budget_exhausted": state.get("budget_exhausted", False)})
                        for group in ("inventory", "achievements"):
                            metrics.update({f"{session}/{group}/{k}": v for k, v in state[group].items()})
                        if state.get("image_path"):
                            metrics[f"{session}/frame"] = wandb.Image(state["image_path"])
                        run.log(metrics)
                        run.summary["diamond_obtained"] = bool(cursor["success"] or state["success"])
                        delivered[session] = revision
                        atomic_json(cursor_path, cursor)
                phase = "watching"
                atomic_json(status_path, {"status": "running", "phase": phase, "updated_at": time.time(),
                                         "diamond_obtained": cursor["success"]})
                if once or (root / "MONITOR_STOP").exists():
                    break
                time.sleep(5)
            if run:
                phase = "wandb_finish"
                run.finish()
                run = None
            atomic_json(status_path, {"status": "stopped", "phase": "complete", "updated_at": time.time(),
                                     "diamond_obtained": cursor["success"]})
        except BaseException as exc:
            cursor["wandb"] = initial_wandb
            atomic_json(cursor_path, cursor)
            error = str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                error += ": " + (exc.stderr.decode("utf-8", errors="replace")
                                  if isinstance(exc.stderr, bytes) else exc.stderr)
            atomic_json(status_path, {"status": "failed", "phase": phase, "error": error,
                                     "updated_at": time.time()})
            raise
        finally:
            if run:
                try:
                    run.finish()
                except Exception:
                    pass  # Preserve the original failure recorded above.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-wandb", action="store_true", help="Only local progress summaries and commits")
    args = parser.parse_args()
    run_monitor(load_config(args.config), args.once, args.no_wandb)


if __name__ == "__main__":
    main()
