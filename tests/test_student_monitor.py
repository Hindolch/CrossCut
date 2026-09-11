import argparse
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from crosscut import student_monitor as monitor


class StudentMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(data_dir=str(self.root / "local"), run_id="exp002-astra-seed0",
            entity="test", project="test", mode="offline", server_url="http://localhost:8768",
            teacher_url="http://localhost:8767", push=True, interval=1)
        self.metric_raw = '{ "env_step": 32, "loss": 0.5 }\n'

    def transport(self, url):
        if url.endswith("suite-status.json"):
            return json.dumps(dict(run_id=self.args.run_id, phase="complete"))
        if url.endswith("train/status.json"):
            return json.dumps(dict(env_step=32, updates=1, episode=1, phase="complete"))
        if url.endswith("metrics.jsonl"):
            return self.metric_raw
        if url.endswith("episodes.jsonl"):
            return ""
        return '{"requests": 2}'

    def fake_wandb(self, fail=False):
        run = types.SimpleNamespace(summary={}, logs=[])
        run.define_metric = lambda *a, **kw: None
        run.log = run.logs.append
        def finish():
            if fail:
                raise RuntimeError("finish failed")
        run.finish = finish
        return types.SimpleNamespace(init=lambda **kw: run), run

    def test_exact_spool_and_actual_step_finish_acknowledgment(self):
        fake, tracking = self.fake_wandb()
        with patch.object(monitor, "progress"):
            monitor.run(self.args, self.transport, fake)
        spooled = [json.loads(p.read_text()) for p in (self.root / "local/spool").glob("*.json")]
        self.assertIn(self.metric_raw, [item["text"] for item in spooled])
        self.assertEqual(tracking.logs[0]["env_step"], 32)
        self.assertEqual(tracking.logs[0]["train/metrics/loss"], 0.5)
        status = json.loads((self.root / "local/monitor-status.json").read_text())
        self.assertEqual(status["status"], "sync_complete")
        self.assertTrue(json.loads((self.root / "local/delivery.json").read_text())["wandb"])

    def test_partial_jsonl_append_waits_for_next_poll(self):
        self.metric_raw += '{"env_step":'
        values = monitor.poll(self.args.server_url, self.root, self.transport)
        self.assertEqual(values["/train/metrics.jsonl"], [{"env_step": 32, "loss": 0.5}])

    def test_finish_failure_does_not_acknowledge_wandb(self):
        fake, tracking = self.fake_wandb(fail=True)
        with patch.object(monitor, "progress"):
            with self.assertRaisesRegex(RuntimeError, "finish failed"):
                monitor.run(self.args, self.transport, fake)
        cursor = json.loads((self.root / "local/delivery.json").read_text())
        self.assertFalse(cursor["wandb"])
        self.assertEqual(json.loads((self.root / "local/monitor-status.json").read_text())["status"], "failed")

    def test_push_failure_retries_immutable_snapshot(self):
        snapshot = dict(marker="training-update-1", train={"env_step": 32})
        counts = dict(commit=0, push=0)
        def git(command, **kwargs):
            if "diff" in command:
                return types.SimpleNamespace(returncode=int(counts["commit"] == 0))
            if "commit" in command:
                self.assertIn("--only", command)
                counts["commit"] += 1
            if "push" in command:
                counts["push"] += 1
                if counts["push"] == 1:
                    raise subprocess.CalledProcessError(1, command)
            return types.SimpleNamespace(returncode=0)
        with patch.object(monitor, "ROOT", self.root), patch.object(monitor.subprocess, "run", side_effect=git):
            with self.assertRaises(subprocess.CalledProcessError):
                monitor.progress(snapshot, self.args.run_id)
            monitor.progress(snapshot, self.args.run_id)
        self.assertEqual(counts, dict(commit=1, push=2))

    def test_real_git_preserves_unrelated_staging(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git("init")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "Test")
        (self.root / "initial").write_text("initial")
        git("add", "initial")
        git("commit", "-m", "initial")
        (self.root / "unrelated").write_text("staged")
        git("add", "unrelated")
        with patch.object(monitor, "ROOT", self.root):
            monitor.progress(dict(marker="training-update-1"), self.args.run_id, push=False)
        self.assertEqual(git("diff", "--cached", "--name-only"), "unrelated")
        self.assertTrue(git("show", "--pretty=format:", "--name-only", "HEAD").startswith("progress/exp002-astra-seed0/"))


if __name__ == "__main__":
    unittest.main()
