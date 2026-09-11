"""Session-only Jarvis API access. The token is never written to disk."""

import getpass
import json
import sys
import time
from pathlib import Path

from jarvislabs import Client


def main():
    client = Client(api_key=getpass.getpass("Jarvis API key (session only): "))
    created = set()
    target_id = 504029
    print("READY", flush=True)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request["action"]
                if action == "instances":
                    result = [{"machine_id": item.machine_id, "name": item.name,
                               "status": item.status, "ssh_command": item.ssh_command}
                              for item in client.instances.list()]
                elif action == "ssh_list":
                    result = [item.model_dump() for item in client.ssh_keys.list()]
                elif action == "ssh_add":
                    before = {item.key_id for item in client.ssh_keys.list()}
                    added = client.ssh_keys.add(Path(request["public_key_path"]).read_text().strip(), request["name"])
                    after = client.ssh_keys.list()
                    created.update(item.key_id for item in after if item.key_id not in before)
                    result = {"added": added, "temporary_key_ids": list(created)}
                elif action == "target":
                    item = client.instances.get(target_id)
                    result = {k: getattr(item, k) for k in ("machine_id", "name", "status", "template", "gpu_type", "ssh_command")}
                elif action == "refresh_ssh":
                    item = client.instances.get(target_id)
                    if item.name != "astra-crafter-mining":
                        raise ValueError("Target identity changed")
                    if item.status == "Running":
                        client.instances.pause(target_id)
                    deadline = time.monotonic() + 600
                    while client.instances.get(target_id).status != "Paused":
                        if time.monotonic() > deadline:
                            raise TimeoutError("Instance pause timed out")
                        time.sleep(5)
                    item = client.instances.resume(target_id)
                    target_id = item.machine_id
                    result = {k: getattr(item, k) for k in ("machine_id", "name", "status", "template", "gpu_type", "ssh_command")}
                elif action == "close":
                    break
                else:
                    raise ValueError("Unknown operation")
                print(json.dumps({"ok": True, "result": result}), flush=True)
            except Exception as exc:
                # Do not dump HTTP request headers, response bodies, or credentials.
                print(json.dumps({"ok": False, "error_type": type(exc).__name__}), flush=True)
    finally:
        for key_id in created:
            try:
                removed = client.ssh_keys.remove(key_id)
                print(json.dumps({"temporary_key_removed": key_id, "ok": removed}), flush=True)
            except Exception as exc:
                print(json.dumps({"cleanup_failed": key_id, "error_type": type(exc).__name__}), flush=True)
        client.close()
        print("API session closed; no credential file was created.", flush=True)


if __name__ == "__main__":
    main()
