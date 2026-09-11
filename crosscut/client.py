"""Local controller; credentials and model calls stay on this computer."""

import argparse
import base64
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .server import identifier
from .storage import atomic_json, process_lock

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.local.json"
    if not path.exists():
        path = ROOT / "config.example.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["data_dir"] = str((ROOT / config["data_dir"]).resolve())
    identifier(config["campaign"])
    return config


class RemoteError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class Client:
    def __init__(self, config):
        self.config = config
        self.base = config["server_url"].rstrip("/")
        self.root = Path(config["data_dir"]) / config["campaign"]

    def request(self, path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        request = Request(self.base + path, data=payload, headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")
                raise RemoteError(exc.code, text) from exc
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def health(self):
        health = self.request("/health")
        if health != {"service": "crosscut-crafter", "version": 1}:
            raise RuntimeError("The endpoint is not a compatible CrossCut game server")
        return health

    def save(self, state):
        state = dict(state)
        session = identifier(state["session"])
        revision = state["revision"]
        frame = state.pop("image_png_base64", None)
        if frame:
            path = self.root / "frames" / session / f"{revision:012d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                fd, temporary = tempfile.mkstemp(dir=path.parent)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(base64.b64decode(frame, validate=True))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            state["image_path"] = str(path.resolve())
        # One immutable event per committed revision; repeated observations are deduplicated.
        path = self.root / "events" / session / f"{revision:012d}.json"
        if not path.exists():
            state["observed_at"] = time.time()
            atomic_json(path, state)
        with process_lock(self.root / "publish" / (session + ".lock")):
            latest = self.root / "state" / (session + ".json")
            previous = json.loads(latest.read_text()) if latest.exists() else {}
            if previous.get("revision", 0) <= revision:
                atomic_json(latest, state)
        return state

    def observe(self, session):
        return self.save(self.request("/sessions/" + identifier(session)))

    def _mutate(self, session, operation, body):
        session = identifier(session)
        path = self.root / "pending" / (session + ".json")
        with process_lock(self.root / "locks" / (session + ".lock")):
            if path.exists():
                raise RuntimeError(f"An unacknowledged request exists. Run: python -m crosscut.client recover {session}")
            body = dict(body, request_id=uuid.uuid4().hex)
            request = {"path": f"/sessions/{session}/{operation}", "body": body}
            atomic_json(path, request)
            try:
                result = self.save(self.request(request["path"], body))
            except RemoteError as exc:
                # A server-side rejection did not execute the operation. Transport errors keep it pending.
                if exc.status in (400, 404, 409):
                    path.unlink()
                raise
            path.unlink()
            return result

    def recover(self, session):
        session = identifier(session)
        path = self.root / "pending" / (session + ".json")
        with process_lock(self.root / "locks" / (session + ".lock")):
            if not path.exists():
                return self.observe(session)
            request = json.loads(path.read_text(encoding="utf-8"))
            try:
                result = self.save(self.request(request["path"], request["body"]))
            except RemoteError as exc:
                if exc.status in (400, 404, 409):
                    atomic_json(self.root / "rejected" / (request["body"]["request_id"] + ".json"),
                                dict(request, status=exc.status, error=str(exc)))
                    path.unlink()
                raise
            path.unlink()
            return result

    def reset(self, session, seed, revision=0, length=10000):
        return self._mutate(session, "reset", {"seed": seed, "expected_revision": revision, "length": length})

    def step(self, session, actions, revision):
        return self._mutate(session, "step", {"actions": actions, "expected_revision": revision})

    def sessions(self):
        return self.request("/sessions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    sub.add_parser("sessions")
    for name in ("observe", "recover"):
        sub.add_parser(name).add_argument("session")
    reset = sub.add_parser("reset")
    reset.add_argument("session")
    reset.add_argument("--seed", type=int, required=True)
    reset.add_argument("--revision", type=int, default=0)
    reset.add_argument("--length", type=int, default=10000)
    step = sub.add_parser("step")
    step.add_argument("session")
    step.add_argument("--revision", type=int, required=True)
    step.add_argument("actions", nargs="+")
    args = parser.parse_args()
    client = Client(load_config(args.config))
    if args.command == "health":
        result = client.health()
    elif args.command == "sessions":
        result = client.sessions()
    elif args.command == "reset":
        result = client.reset(args.session, args.seed, args.revision, args.length)
    elif args.command == "step":
        result = client.step(args.session, args.actions, args.revision)
    else:
        result = getattr(client, args.command)(args.session)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
