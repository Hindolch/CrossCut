import base64
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from crosscut.recording import episode_dir, export_session
from crosscut.server import SessionStore


class FrameGame:
    action_names = ["noop", "diamond"]

    def __init__(self, seed, length):
        self.seed, self.length = seed, length
        self.steps = 0
        self.success = False

    def step(self, action):
        self.steps += 1
        self.success = action == "diamond"

    def snapshot(self, include_image=True):
        state = dict(seed=self.seed, steps=self.steps, done=False, success=self.success)
        if include_image:
            buffer = io.BytesIO()
            Image.new("RGB", (16, 16), (self.steps, self.seed % 256, 0)).save(buffer, "PNG")
            state["image_png_base64"] = base64.b64encode(buffer.getvalue()).decode()
        return state


class RecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = SessionStore(self.root, FrameGame, step_budget=3)
        self.store.mutate("player", "reset", dict(request_id="reset", expected_revision=0, seed=42))

    def step(self, actions):
        return self.store.mutate("player", "step", dict(request_id="step", expected_revision=1, actions=actions))

    def test_every_primitive_and_initial_frame_saved_and_restart_does_not_rewrite(self):
        self.step(["noop"] * 10)
        directory = episode_dir(self.root, "player", 1)
        self.assertEqual(len(list(directory.glob("*.png"))), 4)
        self.assertEqual(len(list(directory.glob("*.json"))), 4)
        entries = [json.loads((directory / f"{step:07d}.json").read_text()) for step in range(4)]
        self.assertEqual([e["action"] for e in entries], [None, "noop", "noop", "noop"])
        self.assertEqual([e["total_steps"] for e in entries], [0, 1, 2, 3])
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}
        restarted = SessionStore(self.root, FrameGame, step_budget=3)
        restarted.state("player")
        after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.iterdir()}
        self.assertEqual(before, after)

    def test_terminal_success_frame_is_recorded(self):
        self.step(["noop", "diamond", "noop"])
        directory = episode_dir(self.root, "player", 1)
        self.assertEqual(len(list(directory.glob("*.png"))), 3)
        data = json.loads((directory / "0000002.json").read_text())
        self.assertEqual(data["action"], "diamond")
        self.assertTrue(data["state"]["success"])

    def test_export_includes_all_episodes_but_excludes_uncommitted_frames(self):
        self.step(["noop"])
        self.store.mutate("player", "reset", dict(request_id="reset2", expected_revision=2, seed=43))
        # The journal commit fails AFTER frame+metadata persistence. Export must
        # select committed frames and ignore this crash tail.
        with patch("crosscut.server.atomic_json", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.store.mutate("player", "step", dict(request_id="fail", expected_revision=3, actions=["noop"]))
        self.assertTrue((episode_dir(self.root, "player", 2) / "0000001.png").exists())
        manifest = export_session(self.root, "player", self.root / "results", videos=False)
        self.assertEqual(manifest["total_steps"], 1)
        self.assertEqual(manifest["frames"], 3)
        self.assertEqual([e["frames"] for e in manifest["episodes"]], [2, 1])
        self.assertEqual(len(Path(manifest["episodes"][1]["log"]).read_text().splitlines()), 1)

    def test_export_rejects_corrupt_committed_frame(self):
        self.step(["noop"])
        (episode_dir(self.root, "player", 1) / "0000001.png").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            export_session(self.root, "player", self.root / "results", videos=False)

    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"), "imageio-ffmpeg unavailable")
    def test_video_contains_every_original_frame_in_order(self):
        import imageio.v2 as imageio
        self.step(["noop"] * 3)
        manifest = export_session(self.root, "player", self.root / "results")
        with imageio.get_reader(manifest["episodes"][0]["video"], format="FFMPEG") as reader:
            frames = [frame for frame in reader]
        self.assertEqual(len(frames), 4)
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [0, 1, 2, 3])
        self.assertEqual(manifest["fps"], 10)


if __name__ == "__main__":
    unittest.main()
