"""Preserve cached game frames and export every committed frame at 10 FPS.

Video time is presentation time, not the wall-clock duration of agent decisions.
The PNG files remain the original, lossless recording.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

from .storage import atomic_json


def episode_dir(root, session, episode):
    from .server import identifier
    return Path(root) / "recordings" / identifier(session) / f"episode-{episode:06d}"


def record_frame(root, session, record, game, action):
    state = game.snapshot(include_image=True)
    png = base64.b64decode(state.pop("image_png_base64"), validate=True)
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Game did not supply a PNG frame")
    step = len(record["actions"])
    if state["steps"] != step:
        raise ValueError("Recording step disagrees with game")
    directory = episode_dir(root, session, record["episode"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{step:07d}.png"
    temporary = target.with_suffix(".png.tmp")
    with temporary.open("wb") as stream:
        stream.write(png)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    metadata = dict(session=session, episode=record["episode"], step=step,
                    total_steps=record["total_steps"], action=action,
                    png_sha256=hashlib.sha256(png).hexdigest(), state=state)
    atomic_json(target.with_suffix(".json"), metadata)


def committed_episodes(root, session):
    from .server import identifier
    root = Path(root)
    current = json.loads((root / (identifier(session) + ".json")).read_text())
    records = []
    for number in range(1, current["episode"]):
        records.append(json.loads((root / "episodes" / session / f"{number}.json").read_text()))
    records.append(current)
    return records


def export_session(root, session, output, videos=True):
    """Export journal-selected frames only; ignore any uncommitted crash tail."""
    root, output = Path(root), Path(output)
    records = committed_episodes(root, session)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(session=session, fps=10, timing="presentation; not wall-clock",
                    total_steps=records[-1]["total_steps"], frames=0, episodes=[])
    full_video = output / "full-session.mp4"
    full_tmp = output / "full-session.partial.mp4"
    full_writer = None
    try:
        if videos:
            import imageio.v2 as imageio
            full_writer = imageio.get_writer(str(full_tmp), format="FFMPEG", fps=10,
                codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                ffmpeg_params=["-crf", "0", "-preset", "veryfast"])
        for record in records:
            number = record["episode"]
            directory = episode_dir(root, session, number)
            name = f"{session}-episode-{number:06d}"
            log = output / (name + ".jsonl")
            log_tmp = log.with_suffix(".jsonl.tmp")
            video = output / (name + ".mp4")
            video_tmp = video.with_suffix(".partial.mp4")
            writer = None
            frame_count = len(record["actions"]) + 1
            try:
                if videos:
                    writer = imageio.get_writer(str(video_tmp), format="FFMPEG", fps=10,
                        codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                        ffmpeg_params=["-crf", "0", "-preset", "veryfast"])
                with log_tmp.open("w", encoding="utf-8") as stream:
                    for step in range(frame_count):
                        png_path = directory / f"{step:07d}.png"
                        png = png_path.read_bytes()
                        data = json.loads(png_path.with_suffix(".json").read_text())
                        expected_action = None if step == 0 else record["actions"][step - 1]
                        expected_total = record["total_steps"] - len(record["actions"]) + step
                        if (data["step"] != step or data["episode"] != number
                                or data["action"] != expected_action
                                or data["total_steps"] != expected_total
                                or data["png_sha256"] != hashlib.sha256(png).hexdigest()):
                            raise ValueError("Recording does not match committed journal")
                        data["png_path"] = str(png_path.resolve())
                        stream.write(json.dumps(data, sort_keys=True) + "\n")
                        if writer is not None:
                            frame = imageio.imread(png)
                            writer.append_data(frame)
                            full_writer.append_data(frame)
                    stream.flush()
                    os.fsync(stream.fileno())
                if writer is not None:
                    writer.close()
                    writer = None
                    os.replace(video_tmp, video)
                os.replace(log_tmp, log)
            finally:
                if writer is not None:
                    writer.close()
            manifest["frames"] += frame_count
            manifest["episodes"].append(dict(episode=number, seed=record["seed"],
                actions=len(record["actions"]), frames=frame_count,
                original_png_directory=str(directory.resolve()),
                log=str(log.resolve()), video=str(video.resolve()) if videos else None))
        if full_writer is not None:
            full_writer.close()
            full_writer = None
            os.replace(full_tmp, full_video)
    finally:
        if full_writer is not None:
            full_writer.close()
    manifest["full_session_video"] = str(full_video.resolve()) if videos else None
    manifest["full_session_frames"] = manifest["frames"]
    manifest["duration_seconds"] = manifest["frames"] / manifest["fps"]
    atomic_json(output / (session + "-manifest.json"), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/server")
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", default="data/results")
    args = parser.parse_args()
    print(json.dumps(export_session(args.data_dir, args.session, args.output), indent=2))


if __name__ == "__main__":
    main()
