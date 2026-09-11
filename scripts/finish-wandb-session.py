"""Publish completed gameplay using a process-only W&B credential."""

import getpass
import json
import os
from pathlib import Path

import wandb


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "diamonds"
    verified = json.loads((results / "verification.json").read_text())
    if not verified["diamond_obtained"] or verified["verified_video_frames"] != verified["frames"]:
        raise RuntimeError("Completed, verified recording required")
    if os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Use a dedicated process without a pre-existing W&B key")
    os.environ["WANDB_API_KEY"] = getpass.getpass("W&B key (this experiment only): ")
    try:
        with wandb.init(entity="nileshsarkar-ai", project="crosscut-diamonds",
                        id="diamonds", resume="must", mode="online", name="diamonds",
                        dir=str(root / "data" / "local" / "diamonds")) as run:
            run.summary.update(verified)
            run.config.update({"model": "gpt-6-astra", "agents": 1,
                               "step_budget": 1_000_000, "seed": 42,
                               "training": False, "host_gpu": "NVIDIA L4",
                               "game_runtime": "CPU simulation on GPU host",
                               "controller_location": "local Codex task"}, allow_val_change=True)
            run.log({"complete_gameplay": wandb.Video(str(results / "gameplay.mp4"), format="mp4"),
                     "winning_frame": wandb.Image(str(results / "winning-frame.png"))})
            artifact = wandb.Artifact("diamonds-astra-1-complete-gameplay", type="gameplay",
                                      description="All 138 frames and 137 actions through the first diamond.",
                                      metadata=verified)
            artifact.add_dir(str(results))
            published = run.log_artifact(artifact, aliases=["latest", "diamond-success"])
            published.wait()
            print(json.dumps({"run_url": run.url, "artifact": published.qualified_name,
                              "artifact_id": published.id, "artifact_state": published.state}), flush=True)
    finally:
        os.environ.pop("WANDB_API_KEY", None)
        print("W&B publishing session closed; no credential file was created.", flush=True)


if __name__ == "__main__":
    main()
