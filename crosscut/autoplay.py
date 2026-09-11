"""Local Codex CLI agents play remote sessions until one collects a diamond."""

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from .client import Client, RemoteError, ROOT, load_config
from .storage import atomic_json, process_lock


def decision_schema(action_names):
    return {"type": "object", "properties": {
        "actions": {"type": "array", "minItems": 1, "maxItems": 32,
                    "items": {"type": "string", "enum": list(action_names)}},
        "memory": {"type": "string"}}, "required": ["actions", "memory"], "additionalProperties": False}


def codex_command(config, state, schema_path, output_path):
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex CLI must be installed and signed in on this computer")
    if config["model"] != "gpt-6-astra":
        raise ValueError("This project requires gpt-6-astra; model fallback is disabled")
    return [executable, "exec", "--model", "gpt-6-astra", "--sandbox", "read-only",
            "--ephemeral", "--cd", str(ROOT), "--json", "--color", "never",
            "-c", f'model_reasoning_effort="{config["reasoning_effort"]}"',
            "--output-schema", str(schema_path), "--output-last-message", str(output_path),
            "--image", state["image_path"], "-"]


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def choose(config, client, session, state, memory, stop):
    directory = client.root / "decisions" / session
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{state['revision']:012d}-{time.time_ns()}"
    schema_path = directory / "schema.json"
    output_path = directory / (prefix + ".json")
    atomic_json(schema_path, decision_schema(state["action_names"]))
    prompt = (ROOT / "prompts" / "player.md").read_text(encoding="utf-8")
    prompt += "\n\nCurrent state:\n" + json.dumps(state) + "\n\nMemory:\n" + memory
    command = codex_command(config, state, schema_path, output_path)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    # Prompt stays on stdin rather than being interpreted as shell arguments.
    with (directory / (prefix + ".events.jsonl")).open("w", encoding="utf-8") as stdout, (directory / (prefix + ".stderr.log")).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                   text=True, encoding="utf-8", creationflags=flags, start_new_session=(os.name != "nt"))
        try:
            process.stdin.write(prompt)
            process.stdin.close()
            deadline = time.monotonic() + config.get("decision_timeout_seconds", 600)
            while process.poll() is None:
                if (client.root / "STOP").exists():
                    stop.set()
                if stop.wait(1) or time.monotonic() >= deadline:
                    stop_process(process)
                    if stop.is_set():
                        return None
                    raise TimeoutError(f"Astra decision timed out; inspect {stderr.name}")
            if process.returncode:
                raise RuntimeError(f"Astra exited {process.returncode}; inspect {stderr.name}")
        finally:
            stop_process(process)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    actions = result.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 32 or any(a not in state["action_names"] for a in actions):
        raise ValueError("Astra returned invalid game actions")
    if not isinstance(result.get("memory"), str):
        raise ValueError("Astra returned invalid memory")
    return result


def worker(config, index, stop):
    client = Client(config)
    session = f"{config['campaign']}-astra-{index + 1}"
    with process_lock(client.root / "players" / (session + ".lock")):
        memory_path = client.root / "memory" / (session + ".json")
        memory = json.loads(memory_path.read_text()).get("memory", "") if memory_path.exists() else ""
        try:
            state = client.recover(session)
        except RemoteError as exc:
            if exc.status != 404:
                raise
            state = client.reset(session, config["seed"] + index)
        decisions = 0
        while not stop.is_set():
            if state["success"]:
                atomic_json(client.root / "SUCCESS.json", state)
                stop.set()
                return f"{session}: diamond obtained"
            if state.get("budget_exhausted"):
                atomic_json(client.root / "BUDGET_EXHAUSTED.json", state)
                stop.set()
                return f"{session}: 1,000,000-step budget exhausted without diamond"
            if (client.root / "STOP").exists():
                stop.set()
                return f"{session}: stop requested"
            if state["done"]:
                memory += "\nPrevious episode ended. Apply its lessons to a fresh world."
                seed = (config["seed"] + index + state["episode"] * config["agents"]) % (2**31)
                state = client.reset(session, seed, state["revision"])
            maximum = config.get("max_decisions_per_agent", 0)
            if maximum and decisions >= maximum:
                return f"{session}: decision limit reached; game preserved"
            result = choose(config, client, session, state, memory, stop)
            if result is None or stop.is_set():
                return f"{session}: stopped"
            memory = result["memory"]
            atomic_json(memory_path, {"memory": memory, "based_on_revision": state["revision"]})
            state = client.step(session, result["actions"], state["revision"])
            decisions += 1
            print(f"{session}: episode={state['episode']} step={state['steps']} diamond={state['success']}", flush=True)
    return f"{session}: stopped"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true", help="Validate local settings without a server or model call")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model"] != "gpt-6-astra" or type(config["agents"]) is not int or config["agents"] != 1:
        raise ValueError("Use exactly one gpt-6-astra player")
    if config["reasoning_effort"] not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("Invalid reasoning effort")
    if args.dry_run:
        print(json.dumps({"model": config["model"], "agents": config["agents"], "server_url": config["server_url"],
                          "codex_cli": shutil.which("codex"), "model_runs_on": "this computer",
                          "objective": "collect_diamond", "started": False}, indent=2))
        return
    client = Client(config)
    client.health()
    for state in client.sessions():
        if state["session"] == f"{config['campaign']}-astra-1" and state["success"]:
            atomic_json(client.root / "SUCCESS.json", state)
            print("Diamond already verified; no model calls started.")
            return
    if (client.root / "STOP").exists():
        print("STOP file exists; no model calls started.")
        return
    stop = threading.Event()
    with process_lock(client.root / "autoplay.lock"):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=config["agents"])
        futures = [pool.submit(worker, config, index, stop) for index in range(config["agents"])]
        try:
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)
        except BaseException:
            stop.set()
            raise
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
