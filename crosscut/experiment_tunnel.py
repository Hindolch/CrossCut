"""Reconnect the experiment's SSH forwards without running remote commands."""

import argparse
import os
import subprocess
import time
from pathlib import Path

from .storage import atomic_json, process_lock

ROOT = Path(__file__).resolve().parents[1]


def supervise(target, key, directory):
    directory = Path(directory).resolve()
    key = Path(key).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stop_path = directory / "TUNNEL_STOP"
    status_path = directory / "tunnel-status.json"
    with process_lock(directory / "experiment-tunnel.lock"):
        child = None
        failures = 0

        def publish(status, **details):
            atomic_json(status_path, dict(status=status, updated_at=time.time(),
                        supervisor_pid=os.getpid(), ssh_pid=child.pid if child else None,
                        target=target, reconnects=failures, **details))

        def stop_child():
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()

        try:
            publish("starting")
            if not key.is_file():
                raise FileNotFoundError(f"SSH key does not exist: {key}")
            command = ["ssh", "-N", "-T", "-i", str(key),
                       "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
                       "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                       "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1",
                       "-R", "127.0.0.1:8767:127.0.0.1:8767",
                       "-L", "127.0.0.1:8768:127.0.0.1:8768", target]
            delay = 1
            with (directory / "tunnel-ssh.log").open("ab", buffering=0) as log:
                while not stop_path.exists():
                    started = time.monotonic()
                    try:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        # A live SSH PID alone does not prove the remote services are ready.
                        publish("ssh_running", services_verified=False)
                        while child.poll() is None and not stop_path.exists():
                            time.sleep(1)
                        if stop_path.exists():
                            break
                        exit_code = child.returncode
                        child = None
                        failures += 1
                        if time.monotonic() - started >= 60:
                            delay = 1
                        publish("reconnecting", exit_code=exit_code, retry_in_seconds=delay,
                                log_path=str((directory / "tunnel-ssh.log").resolve()))
                    except OSError as exc:
                        failures += 1
                        publish("reconnecting", error=str(exc), retry_in_seconds=delay)
                    for _ in range(delay):
                        if stop_path.exists():
                            break
                        time.sleep(1)
                    delay = min(delay * 2, 30)
            stop_child()
            child = None
            publish("stopped")
        except BaseException as exc:
            stop_child()
            child = None
            publish("failed", error=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="root@217.18.55.48")
    parser.add_argument("--key", default=str(ROOT / "data/session-access-exp002/id_ed25519"))
    parser.add_argument("--data-dir", default=str(ROOT / "data/local/exp002"))
    args = parser.parse_args()
    supervise(args.target, args.key, args.data_dir)


if __name__ == "__main__":
    main()
