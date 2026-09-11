import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crosscut.client import Client, RemoteError
from crosscut.storage import atomic_json


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.client = Client(dict(server_url="http://127.0.0.1:8765", campaign="test", data_dir=self.directory.name))

    def test_failed_recovery_clears_terminal_rejection_but_keeps_audit(self):
        pending = self.client.root / "pending/player.json"
        atomic_json(pending, dict(path="/sessions/player/step", body=dict(request_id="abc", expected_revision=1, actions=["noop"])))
        with patch.object(self.client, "request", side_effect=RemoteError(409, "stale")):
            with self.assertRaises(RemoteError):
                self.client.recover("player")
        self.assertFalse(pending.exists())
        self.assertTrue((self.client.root / "rejected/abc.json").exists())

    def test_uncertain_transport_preserves_request_id_for_recovery(self):
        with patch.object(self.client, "request", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.client.step("player", ["noop"], 1)
        pending = self.client.root / "pending/player.json"
        saved = json.loads(pending.read_text())
        with patch.object(self.client, "request", return_value=dict(session="player", revision=2)) as request:
            self.client.recover("player")
        self.assertEqual(request.call_args.args[1], saved["body"])
        self.assertFalse(pending.exists())

    def test_late_observation_cannot_roll_back_latest_state(self):
        self.client.save(dict(session="player", revision=3))
        self.client.save(dict(session="player", revision=2))
        latest = json.loads((self.client.root / "state/player.json").read_text())
        self.assertEqual(latest["revision"], 3)

    def test_repeated_observation_does_not_rewrite_frame(self):
        state = dict(session="player", revision=1, image_png_base64="aGVsbG8=")
        first = self.client.save(state)
        frame = Path(first["image_path"])
        timestamp = frame.stat().st_mtime_ns
        self.client.save(state)
        self.assertEqual(frame.stat().st_mtime_ns, timestamp)


if __name__ == "__main__":
    unittest.main()
