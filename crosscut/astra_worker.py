"""Local, serialized Astra visual teacher using the existing Codex CLI login."""

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from .storage import atomic_json, process_lock
from .autoplay import stop_process

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-6-astra"
DEFAULT_DATA = ROOT / "data/local/exp002/teacher"
ACTIONS = ["noop", "move_left", "move_right", "move_up", "move_down", "do", "sleep",
           "place_stone", "place_table", "place_furnace", "place_plant", "make_wood_pickaxe",
           "make_stone_pickaxe", "make_iron_pickaxe", "make_wood_sword", "make_stone_sword", "make_iron_sword"]
EPSILON = 0.05
SCHEMA = {"type": "object", "properties": {"action": {"type": "string", "enum": ACTIONS}},
          "required": ["action"], "additionalProperties": False}


class Conflict(ValueError):
    pass


class Stopped(RuntimeError):
    pass


def validate_request(body):
    if not isinstance(body, dict) or set(body) != {"request_id", "model", "frames"}:
        raise ValueError("Expected only request_id, model, and frames")
    request_id = body["request_id"]
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
        raise ValueError("Invalid request_id")
    if body["model"] != MODEL:
        raise ValueError("Only gpt-6-astra is permitted; no fallback")
    frames = body["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 4:
        raise ValueError("Supply between one and four PNG frames")
    decoded = []
    for frame in frames:
        if not isinstance(frame, str) or len(frame) > 2800000:
            raise ValueError("PNG exceeds size limit")
        try:
            png = base64.b64decode(frame, validate=True)
            with Image.open(io.BytesIO(png)) as image:
                if image.format != "PNG" or not all(16 <= n <= 1024 for n in image.size):
                    raise ValueError("Expected PNG dimensions from 16 to 1024")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated PNG is not supported")
                image.verify()
        except Exception as exc:
            raise ValueError("Invalid PNG payload or dimensions") from exc
        decoded.append(png)
    fingerprint = hashlib.sha256(json.dumps({"model": MODEL,
        "frames": [hashlib.sha256(png).hexdigest() for png in decoded]},
        sort_keys=True).encode()).hexdigest()
    return request_id, decoded, fingerprint


def normalize(result):
    """Deterministic label smoothing, not model-provided probabilities."""
    if not isinstance(result, dict) or set(result) != {"action"}:
        raise ValueError("Teacher output must contain only action")
    action = result["action"]
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError("Teacher returned an unknown Crafter action")
    values = [EPSILON / len(ACTIONS)] * len(ACTIONS)
    values[ACTIONS.index(action)] += 1 - EPSILON
    return values

def cli_infer(directory, paths, prompt, stopped, timeout=600):
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex CLI must be installed and signed in locally")
    schema = directory / "schema.json"
    output = directory / "answer.json"
    workspace = directory / "workspace"
    workspace.mkdir(exist_ok=True)
    atomic_json(schema, SCHEMA)
    command = [executable, "exec", "--model", MODEL, "--ephemeral", "--sandbox", "read-only",
               "--skip-git-repo-check", "--cd", str(workspace), "--json", "--color", "never",
               "-c", 'model_reasoning_effort="high"',
               "--output-schema", str(schema), "--output-last-message", str(output)]
    for path in paths:
        command.extend(["--image", str(path)])
    command.append("-")
    if stopped():
        raise Stopped("STOP prevents new model calls")
    usage = {}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8", creationflags=flags,
        start_new_session=(os.name != "nt"))

    def consume():
        # Discard all trace/message events. Persist only numeric usage totals.
        for line in process.stdout:
            try:
                event = json.loads(line)
                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                    usage.update({k: v for k, v in event["usage"].items()
                                  if type(v) in (int, float) and math.isfinite(v)})
            except (ValueError, TypeError):
                pass

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    try:
        process.stdin.write(prompt)
        process.stdin.close()
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if stopped():
                raise Stopped("STOP interrupted model call")
            if time.monotonic() >= deadline:
                raise TimeoutError("Astra call timed out")
            time.sleep(0.2)
        if process.returncode:
            raise RuntimeError(f"Astra exited with status {process.returncode}")
        result = json.loads(output.read_text(encoding="utf-8"))
    finally:
        stop_process(process)
        reader.join(timeout=5)
        atomic_json(directory / "usage.json", usage)
    return result, usage


class Teacher:
    def __init__(self, data=DEFAULT_DATA, infer=cli_infer, prompt=None):
        self.data = Path(data).resolve()
        self.data.mkdir(parents=True, exist_ok=True)
        self.infer = infer
        self.prompt = prompt if prompt is not None else (ROOT / "prompts/student-teacher.md").read_text(encoding="utf-8")
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        self.lock = threading.Lock()
        self.active = None

    def stopped(self):
        return (self.data / "STOP").exists()

    def status(self):
        return dict(model=MODEL, active_request_id=self.active, stopped=self.stopped(),
                    target_kind="smoothed_teacher_action", prompt_sha256=self.prompt_sha256)

    def predict(self, body):
        request_id, frames, fingerprint = validate_request(body)
        with self.lock:
            directory = self.data / "requests" / request_id
            record_path = directory / "record.json"
            attempt = 1
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record["input_sha256"] != fingerprint or record["prompt_sha256"] != self.prompt_sha256:
                    raise Conflict("Request ID reused with changed input or prompt")
                if record["status"] == "complete":
                    return dict(record["response"], cached=True)
                if record["status"] != "failed":
                    raise Conflict("Prior call is incomplete; cannot safely duplicate an ambiguous call")
                attempt = record.get("attempt", 1) + 1
            if self.stopped():
                raise Stopped("STOP prevents new model calls")
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "request.json", body)
            paths = []
            for index, png in enumerate(frames):
                path = directory / f"frame-{index}.png"
                path.write_bytes(png)
                paths.append(path)
            record = dict(request_id=request_id, input_sha256=fingerprint,
                          prompt_sha256=self.prompt_sha256, status="inflight", attempt=attempt)
            # Durable reservation precedes invocation, including across crashes.
            atomic_json(record_path, record)
            attempt_directory = directory / "attempts" / f"{attempt:06d}"
            attempt_directory.mkdir(parents=True, exist_ok=True)
            atomic_json(attempt_directory / "record.json", record)
            self.active = request_id
            try:
                result, usage = self.infer(attempt_directory, paths, self.prompt, self.stopped)
                response = dict(request_id=request_id, model=MODEL,
                    probabilities=normalize(result), target_kind="smoothed_teacher_action",
                    chosen_action=result["action"], epsilon=EPSILON,
                    input_sha256=fingerprint, prompt_sha256=self.prompt_sha256, cached=False)
                atomic_json(attempt_directory / "usage.json", usage)
                atomic_json(attempt_directory / "response.json", response)
                atomic_json(directory / "response.json", response)
                record.update(status="complete", response=response)
                atomic_json(attempt_directory / "record.json", record)
                atomic_json(record_path, record)
                return response
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                atomic_json(attempt_directory / "record.json", record)
                atomic_json(record_path, record)
                raise
            finally:
                self.active = None


def handler_for(teacher):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/health", "/status"):
                self.respond(200, teacher.status())
            else:
                self.respond(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/predict":
                return self.respond(404, {"error": "Unknown endpoint"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 11210000:
                    raise ValueError("Invalid request size")
                result = teacher.predict(json.loads(self.rfile.read(size)))
            except Conflict as exc:
                self.respond(409, {"error": str(exc)})
            except Stopped as exc:
                self.respond(503, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(502, {"error": type(exc).__name__})
            else:
                self.respond(200, result)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    with process_lock(DEFAULT_DATA / "worker.lock"):
        teacher = Teacher()
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(teacher))
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
