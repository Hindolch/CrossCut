"""Verified local artifact collection and retryable W&B/Git completion delivery."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

from .storage import atomic_json, process_lock

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "exp002-astra-seed0"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(root, name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Artifact path must be a relative POSIX path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Artifact path escapes the output directory")
    # Windows aliases and alternate streams must not map different manifest entries
    # onto the same local file, even when the source server is Linux.
    for part in path.parts:
        if part.endswith((".", " ")) or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part):
            raise ValueError("Artifact path is not portable")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes through a symbolic link")
    return target


def download(base, entry, root):
    name, size, expected = entry["path"], entry["size"], entry["sha256"]
    if type(size) is not int or size < 0 or not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError("Invalid artifact size or SHA-256")
    target = safe_target(root, name)
    if target.exists():
        if target.is_file() and target.stat().st_size == size and file_hash(target) == expected:
            return target
        raise FileExistsError(f"Existing artifact differs from manifest: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + expected + ".partial")
    digest = hashlib.sha256()
    received = 0
    with urlopen(base.rstrip("/") + "/" + quote(name, safe="/"), timeout=60) as response:
        with temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                received += len(chunk)
                if received > size:
                    raise ValueError(f"Artifact exceeds declared size: {name}")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if received != size or digest.hexdigest() != expected:
        raise ValueError(f"Artifact verification failed: {name}")
    # Exclusive publication avoids overwriting a file created during a download.
    os.link(temporary, target)
    temporary.unlink()
    return target


def deliver(run, output_dir=ROOT / "data/local/exp002", server_url="http://127.0.0.1:8768"):
    """Return delivery metadata only after upload acknowledgement and Git push."""
    if run.id != RUN_ID:
        raise ValueError("Delivery run ID must match experiment 2")
    run_path = f"{run.entity}/{run.project}/{run.id}"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with process_lock(output / "delivery-artifacts.lock"):
        status_path = output / "artifact-delivery-status.json"
        checkpoint_path = output / "artifact-delivery-checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
        phase = "manifest"
        try:
            atomic_json(status_path, {"status": "working", "phase": phase})
            with urlopen(server_url.rstrip("/") + "/artifact-manifest.json", timeout=30) as response:
                manifest = json.load(response)
            entries = manifest.get("files", manifest.get("entries")) if isinstance(manifest, dict) else manifest
            if not isinstance(entries, list) or not entries:
                raise ValueError("Artifact manifest must list files")
            if isinstance(manifest, dict) and manifest.get("run_id", RUN_ID) != RUN_ID:
                raise ValueError("Artifact manifest belongs to another experiment")
            encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            manifest_hash = hashlib.sha256(encoded).hexdigest()
            root = output / "artifacts"
            root.mkdir(parents=True, exist_ok=True)
            if any(item["path"].casefold() == "_delivery/manifest.json" for item in entries):
                raise ValueError("Manifest uses the reserved delivery manifest path")
            names = [str(safe_target(root, item["path"])).casefold() for item in entries]
            if len(names) != len(set(names)):
                raise ValueError("Manifest contains duplicate artifact destinations")
            phase = "downloading"
            atomic_json(status_path, {"status": "working", "phase": phase, "files": len(entries)})
            paths = [download(server_url, entry, root) for entry in entries]
            manifest_path = output / "verified-artifact-manifest.json"
            atomic_json(manifest_path, manifest)
            phase = "uploading"
            artifact_name = checkpoint.get("artifact") if checkpoint.get("manifest_sha256") == manifest_hash and checkpoint.get("run_path") == run_path else None
            if not artifact_name:
                import wandb
                artifact = wandb.Artifact(RUN_ID + "-results", type="experiment-results",
                                          metadata={"manifest_sha256": manifest_hash, "run_id": RUN_ID})
                for entry, path in zip(entries, paths):
                    artifact.add_file(str(path), name=entry["path"])
                artifact.add_file(str(manifest_path), name="_delivery/manifest.json")
                atomic_json(status_path, {"status": "working", "phase": phase, "files": len(entries)})
                logged = run.log_artifact(artifact)
                logged.wait()
                artifact_name = logged.qualified_name
                checkpoint = {"manifest_sha256": manifest_hash, "artifact": artifact_name, "run_path": run_path}
                atomic_json(checkpoint_path, checkpoint)
            summary = dict(run_id=RUN_ID, run_path=f"{run.entity}/{run.project}/{run.id}",
                           run_url=run.url, artifact=artifact_name, manifest_sha256=manifest_hash,
                           files=len(entries), bytes=sum(item["size"] for item in entries),
                           hash_algorithm="sha256", manifest_in_artifact="_delivery/manifest.json")
            destination = ROOT / "results" / RUN_ID / "delivery.json"
            atomic_json(destination, summary)
            phase = "git_push"
            atomic_json(status_path, {"status": "working", "phase": phase, **summary})
            relative = destination.relative_to(ROOT).as_posix()
            git = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
            subprocess.run(git + ["add", "--", relative], check=True, capture_output=True)
            changed = subprocess.run(git + ["diff", "--cached", "--quiet", "--", relative], capture_output=True)
            if changed.returncode == 1:
                subprocess.run(git + ["commit", "--only", "-m", "Record verified experiment 2 artifact delivery",
                                      "--", relative], check=True, capture_output=True)
            elif changed.returncode:
                raise RuntimeError("Could not inspect result staging")
            subprocess.run(git + ["push"], check=True, capture_output=True)
            atomic_json(status_path, {"status": "complete", **summary})
            return summary
        except BaseException as exc:
            atomic_json(status_path, {"status": "failed", "phase": phase,
                                     "error": f"{type(exc).__name__}: {exc}"})
            raise
