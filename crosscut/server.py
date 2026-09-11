"""Loopback-only game service. No model SDK, credentials, or W&B on Jarvis."""

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .storage import atomic_json, process_lock

MAX_BATCH_STEPS = 32
LIFETIME_STEP_BUDGET = 1000000


class Conflict(ValueError):
    pass


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError("Identifier must contain 1-80 letters, digits, underscores or hyphens")
    return value


class SessionStore:
    def __init__(self, root, factory=None, step_budget=LIFETIME_STEP_BUDGET):
        if type(step_budget) is not int or not 1 <= step_budget <= LIFETIME_STEP_BUDGET:
            raise ValueError("step_budget must be an integer in [1, 1000000]")
        self.step_budget = step_budget
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if factory is None:
            from .game import CrafterGame
            factory = CrafterGame
        self.factory = factory
        self.lock = threading.RLock()
        self.games = {}

    def _path(self, session):
        return self.root / (identifier(session) + ".json")

    def _load(self, session):
        path = self._path(session)
        if not path.exists():
            raise FileNotFoundError("Session does not exist; reset it first")
        record = json.loads(path.read_text(encoding="utf-8"))
        # Fail closed for old/unaccounted journals rather than replenish a run.
        if record.get("step_budget") != self.step_budget:
            raise Conflict("Stored step budget differs from configured budget")
        total_steps = record.get("total_steps")
        if type(total_steps) is not int or not len(record["actions"]) <= total_steps <= self.step_budget:
            raise Conflict("Session lifetime step accounting is invalid")
        if session not in self.games:
            game = self.factory(record["seed"], record["length"])
            for action in record["actions"]:
                game.step(action)
            self.games[session] = game
        return record, self.games[session]

    def _state(self, session, record, game, include_image=True):
        state = game.snapshot(include_image=include_image)
        state.update(session=session, revision=record["revision"], episode=record["episode"])
        state.update(total_steps=record["total_steps"], step_budget=record["step_budget"],
                     budget_exhausted=record["total_steps"] >= record["step_budget"])
        return state

    def state(self, session):
        with self.lock:
            record, game = self._load(session)
            return self._state(session, record, game)

    def all_states(self):
        with self.lock:
            result = []
            for path in sorted(self.root.glob("*.json")):
                record, game = self._load(path.stem)
                result.append(self._state(path.stem, record, game, False))
            return result

    def mutate(self, session, operation, body):
        identifier(session)
        request_id = identifier(body.get("request_id"))
        fingerprint = hashlib.sha256(json.dumps([operation, body], sort_keys=True).encode()).hexdigest()
        expected = body.get("expected_revision")
        if type(expected) is not int or expected < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        with self.lock:
            path = self._path(session)
            record, game = self._load(session) if path.exists() else (None, None)
            # A retry after a lost response returns the committed state, without replaying actions.
            if record and record["request_id"] == request_id:
                if record["fingerprint"] != fingerprint:
                    raise Conflict("request_id was reused with different content")
                return self._state(session, record, game)
            revision = record["revision"] if record else 0
            if expected != revision:
                raise Conflict("Stale revision; observe again before choosing an action")
            if record and record["total_steps"] >= record["step_budget"]:
                raise Conflict("Lifetime step budget exhausted; steps and resets are disabled")
            if operation == "reset":
                if game and game.snapshot(False)["success"]:
                    raise Conflict("Diamond already obtained; winning episode is preserved")
                seed, length = body.get("seed"), body.get("length", 10000)
                if type(seed) is not int or not 0 <= seed < 2**31:
                    raise ValueError("seed must be an integer in [0, 2**31)")
                if type(length) is not int or not 1 <= length <= 100000:
                    raise ValueError("length must be an integer in [1, 100000]")
                new_game = self.factory(seed, length)
                if record:
                    atomic_json(self.root / "episodes" / session / (str(record["episode"]) + ".json"), record)
                record = {"seed": seed, "length": length, "actions": [],
                          "total_steps": record["total_steps"] if record else 0,
                          "step_budget": record["step_budget"] if record else self.step_budget,
                          "episode": record["episode"] + 1 if record else 1}
                game = new_game
            elif operation == "step":
                if record is None:
                    raise FileNotFoundError("Session does not exist; reset it first")
                actions = body.get("actions")
                if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_BATCH_STEPS:
                    raise ValueError("actions must be a nonempty list of at most 32 action names")
                if any(not isinstance(a, str) or a not in game.action_names for a in actions):
                    raise ValueError("Unknown action")
                if game.snapshot(False)["done"] or game.snapshot(False)["success"]:
                    raise Conflict("Episode ended; observe before resetting")
                try:
                    for action in actions:
                        if record["total_steps"] >= record["step_budget"]:
                            break
                        game.step(action)
                        record["actions"].append(action)
                        record["total_steps"] += 1
                        state = game.snapshot(False)
                        if state["done"] or state["success"]:
                            break
                except Exception:
                    self.games.pop(session, None)  # Restore last committed state on next read.
                    raise
            else:
                raise ValueError("Unknown operation")
            record.update(revision=revision + 1, request_id=request_id, fingerprint=fingerprint)
            try:
                atomic_json(path, record)
            except Exception:
                self.games.pop(session, None)
                raise
            self.games[session] = game
            return self._state(session, record, game)


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return  # State and action journals are the audit log.

        def send_json(self, code, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def route(self):
            parts = self.path.strip("/").split("/")
            if self.command == "GET" and parts == ["health"]:
                return {"service": "crosscut-crafter", "version": 1}
            if self.command == "GET" and parts == ["sessions"]:
                return store.all_states()
            if len(parts) == 2 and parts[0] == "sessions" and self.command == "GET":
                return store.state(parts[1])
            if len(parts) == 3 and parts[0] == "sessions" and self.command == "POST":
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Request must be a JSON object")
                return store.mutate(parts[1], parts[2], body)
            raise FileNotFoundError("Unknown endpoint")

        def handle_request(self):
            try:
                result = self.route()
            except Conflict as exc:
                self.send_json(409, {"error": str(exc)})
            except FileNotFoundError as exc:
                self.send_json(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.send_json(500, {"error": type(exc).__name__})
            else:
                self.send_json(200, result)

        do_GET = handle_request
        do_POST = handle_request

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default="data/server")
    args = parser.parse_args()
    with process_lock(Path(args.data_dir) / "server.lock"):
        store = SessionStore(args.data_dir)
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store))
        print(f"Crafter listening on 127.0.0.1:{args.port}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


if __name__ == "__main__":
    main()
