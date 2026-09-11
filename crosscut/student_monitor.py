"""Local telemetry and scoped progress delivery for detached student training."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .storage import atomic_json, process_lock

ROOT = Path(__file__).resolve().parents[1]
PATHS = ("/suite-status.json", "/train/status.json", "/train/metrics.jsonl",
         "/train/episodes.jsonl", "/eval/episodes.jsonl")


def fetch(url):
    with urlopen(url, timeout=15) as response:
        return response.read().decode("utf-8")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def poll(base, root, transport=fetch):
    result = {}
    for endpoint in PATHS:
        try:
            raw = transport(base.rstrip("/") + endpoint)
        except HTTPError as exc:
            if exc.code != 404 or not endpoint.endswith("jsonl"):
                raise
            atomic_json(root / "spool" / (digest(endpoint + "404") + ".json"),
                        {"endpoint": endpoint, "http_status": 404})
            result[endpoint] = []
            continue
        # Store exact decoded responses, including JSONL whitespace and newlines.
        atomic_json(root / "spool" / (digest(endpoint + raw) + ".json"),
                    {"endpoint": endpoint, "text": raw})
        # The HTTP server can observe the final line while the trainer appends it.
        complete = raw if raw.endswith("\n") else raw[:raw.rfind("\n") + 1]
        result[endpoint] = ([json.loads(line) for line in complete.splitlines() if line.strip()]
                            if endpoint.endswith("jsonl") else json.loads(raw))
    return result


def progress(snapshot, run_id, push=True):
    marker = snapshot["marker"]
    path = ROOT / "progress" / run_id / (digest(marker)[:24] + ".json")
    if not path.exists():
        atomic_json(path, snapshot)
    relative = path.relative_to(ROOT).as_posix()
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    subprocess.run(git + ["add", "--", relative], check=True, capture_output=True)
    changed = subprocess.run(git + ["diff", "--cached", "--quiet", "--", relative], capture_output=True)
    if changed.returncode == 1:
        subprocess.run(git + ["commit", "--only", "-m", f"Student {run_id}: {marker}",
                              "--", relative], check=True, capture_output=True)
    elif changed.returncode:
        raise RuntimeError("Could not inspect progress staging")
    if push:
        subprocess.run(git + ["push"], check=True, capture_output=True)


def numeric(value):
    return {key: item for key, item in value.items()
            if isinstance(item, (int, float)) and math.isfinite(item)}


def run(args, transport=fetch, wandb_module=None):
    from .server import identifier
    identifier(args.run_id)
    root = Path(args.data_dir).resolve()
    with process_lock(root / "student-monitor.lock"):
        cursor_path = root / "delivery.json"
        cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else {"git": {}, "wandb": {}}
        if isinstance(cursor["git"], list):
            cursor["git"] = {"commit": cursor["git"]}
        git_delivered = cursor["git"].setdefault("push" if args.push else "commit", [])
        sink = json.dumps([args.mode, args.entity, args.project, args.run_id])
        acknowledged = set(cursor["wandb"].get(sink, []))
        pending = set(acknowledged)
        status_path = root / "monitor-status.json"
        tracking = None
        suite = {}

        def status(phase, **extra):
            atomic_json(status_path, dict(run_id=args.run_id, status=phase,
                                         updated_at=time.time(), suite_phase=suite.get("phase"), **extra))

        try:
            status("starting")
            if wandb_module is None:
                import wandb as wandb_module
            tracking = wandb_module.init(project=args.project, entity=args.entity, id=args.run_id,
                                         name=args.run_id, resume="allow", mode=args.mode, dir=str(root))
            tracking.define_metric("env_step")
            tracking.define_metric("*", step_metric="env_step")
            while not (root / "STOP").exists():
                try:
                    values = poll(args.server_url, root, transport)
                    suite, train = values[PATHS[0]], values[PATHS[1]]
                    if suite.get("run_id") not in (None, args.run_id):
                        raise ValueError("Suite run_id does not match monitor run_id")
                    # Teacher waiting/collecting transitions do not create commits per poll.
                    marker = f"{suite.get('phase', 'unknown')}-update-{train.get('updates', 'unknown')}"
                    if marker not in git_delivered:
                        progress(dict(marker=marker, suite=suite, train=train), args.run_id, args.push)
                        git_delivered.append(marker)
                        atomic_json(cursor_path, cursor)
                    for endpoint in PATHS[2:]:
                        prefix = endpoint.strip("/").replace(".jsonl", "")
                        for index, event in enumerate(values[endpoint]):
                            event_id = digest(endpoint + str(index) + json.dumps(event, sort_keys=True))
                            if event_id in pending:
                                continue
                            fields = {f"{prefix}/{key}": value for key, value in numeric(event).items()}
                            step = event.get("env_step", train.get("env_step"))
                            if step is not None:
                                fields["env_step"] = step
                            if fields:
                                tracking.log(fields)
                            pending.add(event_id)
                    fields = {f"train/status/{key}": value for key, value in numeric(train).items()
                              if key != "updated_at"}
                    if "env_step" in train:
                        fields["env_step"] = train["env_step"]
                    try:
                        raw = transport(args.teacher_url.rstrip("/") + "/status")
                        atomic_json(root / "teacher-status.json", {"text": raw})
                        fields.update({f"teacher/{key}": value for key, value in numeric(json.loads(raw)).items()})
                    except (URLError, TimeoutError, ConnectionError, ValueError):
                        pass  # Teacher statistics are optional; missing values remain absent.
                    if fields:
                        tracking.log(fields)
                    tracking.summary["suite_phase"] = suite.get("phase")
                    if suite.get("config"):
                        tracking.config.update(suite["config"], allow_val_change=False)
                    status("watching", env_step=train.get("env_step"), updates=train.get("updates"))
                    if suite.get("phase") in ("complete", "failed"):
                        from .student_delivery import deliver
                        status("delivering_artifacts")
                        # Missing manifest means the renderer/inventory is still working.
                        deliver(tracking, root, args.server_url)
                        break
                except (URLError, TimeoutError, ConnectionError, subprocess.CalledProcessError) as exc:
                    status("retrying", error=str(exc))
                # Poll STOP each second so clean shutdown does not wait a full interval.
                for _ in range(args.interval):
                    if (root / "STOP").exists():
                        break
                    time.sleep(1)
            status("finishing")
            tracking.finish()
            tracking = None
            # SDK finish must succeed before a W&B sink is durably acknowledged.
            # Abrupt restarts can replay events; the source spool is never discarded.
            cursor["wandb"][sink] = sorted(pending)
            atomic_json(cursor_path, cursor)
            if suite.get("phase") == "complete":
                from .student_cleanup import cleanup
                status("cleaning_up")
                cleanup()
            status("sync_complete" if suite.get("phase") in ("complete", "failed") else "stopped",
                   artifacts_ready=suite.get("phase") in ("complete", "failed"),
                   next_action="Collect final artifacts and perform authorized cleanup")
        except BaseException as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if tracking is not None:
                try:
                    tracking.finish()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8768")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8767")
    parser.add_argument("--data-dir", default="data/local/exp002")
    parser.add_argument("--entity", default="nileshsarkar-ai")
    parser.add_argument("--project", default="crosscut-llm4teach")
    parser.add_argument("--run-id", default="exp002-astra-seed0")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--no-push", dest="push", action="store_false")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    run(args)


if __name__ == "__main__":
    main()
