"""Watch the local game service and export a terminal session without model access."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from .recording import export_session
from .server import identifier
from .storage import atomic_json, process_lock


def watch(data_dir, session, output, server_url="http://127.0.0.1:8765"):
    session = identifier(session)
    endpoint = urlsplit(server_url)
    if (endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost", "::1")
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("The finalizer requires an unauthenticated loopback HTTP game service")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "conversion-status.json"
    with process_lock(output / "finish.lock"):
        status = {"session": session, "status": "watching", "started_at": time.time(),
                  "output_directory": str(output.resolve())}

        def publish(**values):
            status.update(values, updated_at=time.time())
            atomic_json(status_path, status)

        try:
            publish()
            failures = 0
            while True:
                try:
                    with urlopen(server_url.rstrip("/") + "/sessions", timeout=15) as response:
                        sessions = json.load(response)
                    if not isinstance(sessions, list):
                        raise ValueError("Unexpected game service response")
                    state = next((item for item in sessions if item.get("session") == session), None)
                    failures = 0
                    if state is not None:
                        publish(total_steps=state.get("total_steps", state.get("steps", 0)),
                                episode=state["episode"], success=bool(state.get("success")),
                                budget_exhausted=bool(state.get("budget_exhausted")), error=None)
                        # Death alone is followed by another episode within the same budget.
                        if state.get("success") or state.get("budget_exhausted"):
                            break
                except (URLError, TimeoutError, ConnectionError) as exc:
                    failures += 1
                    publish(error=str(exc), consecutive_connection_failures=failures)
                    if failures >= 60:
                        raise RuntimeError("Game service unavailable for 60 consecutive checks") from exc
                time.sleep(5)
            publish(status="converting", terminal_reason="diamond" if state.get("success") else "step_budget")
            manifest = export_session(data_dir, session, output)
            publish(status="complete", completed_at=time.time(), manifest=manifest,
                    manifest_path=str((output / (session + "-manifest.json")).resolve()),
                    total_steps=manifest["total_steps"], frames=manifest["frames"],
                    episodes=len(manifest["episodes"]),
                    full_session_video=manifest["full_session_video"], error=None)
            print(json.dumps(status), flush=True)
            return manifest
        except BaseException as exc:
            publish(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/server")
    parser.add_argument("--session", default="diamonds-astra-1")
    parser.add_argument("--output", default="data/results")
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    watch(args.data_dir, args.session, args.output, args.server_url)


if __name__ == "__main__":
    main()
