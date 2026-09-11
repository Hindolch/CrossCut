"""Import only the requested LLM4Teach exp1/exp2 code, or verify its provenance.

No clone, submodules, weights, recordings, archived experiments, or credentials.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "experiments" / "LLM4Teach"
REPOSITORY = "nileshsarkar-ai/Game-Playing-Agents-Crafter"
COMMIT = "08961111faa8caac1e721b7e27df347c4e73d2eb"
TREE = "e53ecc612103814d180a6b7195cafc3044aa2c48"
PREFIX = "experiments/LLM4Teach"
PATHS = (
    ".gitignore", "README.md", "requirements.txt",
    "student.py", "teacher.py", "train.py", "ppo.py", "common.py",
    "evaluate.py", "schedules.py", "tracking.py", "render_training.py",
    "plot_learning.py", "aggregate.py", "run_jarvis.sh",
    "configs/gemma12b.json", "configs/ppo.json",
    "exp1/README.md", "exp1/config.json", "exp1/manifest.json",
    "exp2/README.md", "exp2/config.json", "exp2/manifest.json",
    "docs/EXPERIMENT_SPEC.md", "docs/UPSTREAM.md",
)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "CrossCut-selective-import"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(500_001)
    if len(data) > 500_000:
        raise ValueError("Selected text file exceeds the import limit")
    return data


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def import_source():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("Incomplete source tree")
    entries = {item["path"]: item for item in tree["tree"]}

    def read(path):
        entry = entries[path]
        if entry["type"] != "blob" or entry["mode"] not in ("100644", "100755", "120000"):
            raise ValueError(f"Unexpected source entry: {path}")
        data = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{PREFIX}/{path}")
        if blob_sha(data) != entry["sha"]:
            raise ValueError(f"Source blob hash mismatch: {path}")
        return data

    def one(path):
        data = read(path)
        original_sha = entries[path]["sha"]
        transformation = None
        if entries[path]["mode"] == "120000":
            # This is the only selected symlink. Materialize its selected JSON
            # target for Windows without following arbitrary filesystem links.
            if path != "exp1/config.json" or data.decode().strip() != "../configs/gemma12b.json":
                raise ValueError(f"Unexpected symbolic link: {path}")
            data = read("configs/gemma12b.json")
            transformation = "Materialized ../configs/gemma12b.json instead of a filesystem symlink"
        local = "README.upstream.md" if path == "README.md" else path
        target = DESTINATION.joinpath(*PurePosixPath(local).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite modified import: {target}")
        target.write_bytes(data)
        return {"source": f"{PREFIX}/{path}", "local": local, "source_git_blob": original_sha,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                "transformation": transformation}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(one, PATHS))
    manifest = {
        "repository": f"https://github.com/{REPOSITORY}", "commit": COMMIT,
        "source_directory": PREFIX, "scope": "Exp1/exp2 configurations and shared training/evaluation code",
        "excluded": ["upstream/", "other experiment folders", "model weights", "recordings",
                     "architecture media/PDFs", "cloud automation", "environment submodules"],
        "files": files, "file_count": len(files), "source_bytes": sum(item["bytes"] for item in files),
    }
    (DESTINATION / "IMPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify()


def verify():
    manifest = json.loads((DESTINATION / "IMPORT_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = DESTINATION / entry["local"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Imported source changed: {entry['local']}")
        if path.suffix == ".py":
            ast.parse(data, filename=str(path))
        elif path.suffix == ".json":
            json.loads(data)
    return {"verified_files": manifest["file_count"], "bytes": manifest["source_bytes"],
            "source_commit": manifest["commit"], "models_or_games_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download the explicit text-file allowlist")
    arguments = parser.parse_args()
    print(json.dumps(import_source() if arguments.fetch else verify(), indent=2))
