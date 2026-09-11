import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from crosscut import monitor
from crosscut.storage import atomic_json


def state(revision=1, achievements=None):
    return dict(session="demo-astra-1", revision=revision, episode=1, steps=revision,
                reward=0, total_reward=0, success=False, done=False, inventory={},
                achievements=achievements or {"collect_wood": 0})


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(data_dir=str(self.root / "data"), campaign="demo",
                           model="gpt-6-astra", agents=1, autocommit=False, server_url="http://127.0.0.1:1",
                           autopush=False, wandb=dict(mode="offline", project="test"))
        self.patch = patch.object(monitor, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.spool = self.root / "data" / "demo"

    def event(self, value):
        atomic_json(self.spool / "events" / value["session"] /
                    f"{value['revision']:012d}.json", value)

    def fake_wandb(self):
        runs = []

        def init(**kwargs):
            run = types.SimpleNamespace(summary={}, logs=[], definitions=[], options=kwargs)
            run.define_metric = lambda *args, **kw: run.definitions.append((args, kw))
            run.log = lambda data: run.logs.append(data)
            run.finish = lambda: None
            runs.append(run)
            return run

        return types.SimpleNamespace(init=init, Image=lambda path: path), runs

    def test_local_then_offline_then_online_deliver_independently(self):
        self.event(state())
        monitor.run_monitor(self.config, once=True, no_wandb=True)
        fake, runs = self.fake_wandb()
        with patch.dict(sys.modules, {"wandb": fake}):
            monitor.run_monitor(self.config, once=True)
            self.config["wandb"]["mode"] = "online"
            monitor.run_monitor(self.config, once=True)
            monitor.run_monitor(self.config, once=True)
        self.assertEqual([len(run.logs) for run in runs], [1, 1, 0])
        self.assertEqual(runs[0].definitions, [
            (("demo-astra-1/total_steps",), {}),
            (("demo-astra-1/*",), {"step_metric": "demo-astra-1/total_steps"})])
        self.assertEqual(runs[0].logs[0]["demo-astra-1/total_steps"], 1)
        self.assertEqual(runs[0].logs[0]["demo-astra-1/step_budget"], 10000)
        self.assertFalse(runs[0].logs[0]["demo-astra-1/budget_exhausted"])

    def test_failed_log_keeps_telemetry_pending_and_status_visible(self):
        self.event(state())
        fake, runs = self.fake_wandb()
        original_init = fake.init

        def bad_init(**kwargs):
            run = original_init(**kwargs)
            def fail(data):
                raise RuntimeError("telemetry unavailable")
            run.log = fail
            return run

        fake.init = bad_init
        with patch.dict(sys.modules, {"wandb": fake}):
            with self.assertRaisesRegex(RuntimeError, "telemetry unavailable"):
                monitor.run_monitor(self.config, once=True)
            status = json.loads((self.spool / "monitor-status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "wandb_log")
            fake.init = original_init
            monitor.run_monitor(self.config, once=True)
        self.assertEqual(len(runs[-1].logs), 1)

    def test_commit_failure_retries_identical_immutable_snapshot(self):
        self.config["autocommit"] = True
        self.event(state())
        commits = []

        def git(command, **kwargs):
            if "commit" in command:
                commits.append(command)
                if len(commits) == 1:
                    raise subprocess.CalledProcessError(1, command, stderr=b"identity missing")
            return types.SimpleNamespace(returncode=1 if "diff" in command else 0)

        with patch.object(monitor.subprocess, "run", side_effect=git):
            with self.assertRaises(subprocess.CalledProcessError):
                monitor.run_monitor(self.config, once=True, no_wandb=True)
            snapshot = self.root / "progress/demo/demo-astra-1/000000000001.json"
            original = snapshot.read_bytes()
            monitor.run_monitor(self.config, once=True, no_wandb=True)
            self.assertEqual(snapshot.read_bytes(), original)
        self.assertEqual(len(commits), 2)
        self.assertIn("--only", commits[-1])

    def test_failed_push_retries_even_after_commit_succeeded(self):
        self.config.update(autocommit=True, autopush=True)
        self.event(state())
        counts = {"commit": 0, "push": 0}

        def git(command, **kwargs):
            if "diff" in command:
                return types.SimpleNamespace(returncode=int(counts["commit"] == 0))
            if "commit" in command:
                counts["commit"] += 1
            if "push" in command:
                counts["push"] += 1
                if counts["push"] == 1:
                    raise subprocess.CalledProcessError(1, command, stderr=b"network unavailable")
            return types.SimpleNamespace(returncode=0)

        with patch.object(monitor.subprocess, "run", side_effect=git):
            with self.assertRaises(subprocess.CalledProcessError):
                monitor.run_monitor(self.config, once=True, no_wandb=True)
            status = json.loads((self.spool / "monitor-status.json").read_text())
            self.assertIn("network unavailable", status["error"])
            monitor.run_monitor(self.config, once=True, no_wandb=True)
        self.assertEqual(counts, {"commit": 1, "push": 2})

    def test_milestones_remain_immutable_and_skip_ordinary_steps(self):
        monitor.progress_commit(self.config, state())
        monitor.progress_commit(self.config, state(2))
        monitor.progress_commit(self.config, state(3, {"collect_wood": 1}))
        files = sorted((self.root / "progress/demo/demo-astra-1").glob("*.json"))
        self.assertEqual([p.stem for p in files], ["000000000001", "000000000003"])
        self.assertEqual(json.loads(files[0].read_text())["achievements"], {"collect_wood": 0})


    def test_enabling_git_replays_local_only_delivery(self):
        self.event(state())
        monitor.run_monitor(self.config, once=True, no_wandb=True)
        self.config.update(autocommit=True, autopush=True)
        calls = []
        def git(command, **kwargs):
            calls.append(command)
            return types.SimpleNamespace(returncode=1 if "diff" in command else 0)
        with patch.object(monitor.subprocess, "run", side_effect=git):
            monitor.run_monitor(self.config, once=True, no_wandb=True)
        self.assertTrue(any("commit" in command for command in calls))
        self.assertTrue(any("push" in command for command in calls))

    def test_finish_failure_replays_telemetry(self):
        self.event(state())
        fake, runs = self.fake_wandb()
        original_init = fake.init
        def bad_init(**kwargs):
            run = original_init(**kwargs)
            def fail():
                raise RuntimeError("flush failed")
            run.finish = fail
            return run
        fake.init = bad_init
        with patch.dict(sys.modules, {"wandb": fake}):
            with self.assertRaisesRegex(RuntimeError, "flush failed"):
                monitor.run_monitor(self.config, once=True)
            fake.init = original_init
            monitor.run_monitor(self.config, once=True)
        self.assertEqual([len(run.logs) for run in runs], [1, 1])

    def test_real_git_preserves_unrelated_staged_file(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git("init")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "Monitor Test")
        (self.root / "initial.txt").write_text("initial")
        git("add", "initial.txt")
        git("commit", "-m", "initial")
        (self.root / "unrelated.txt").write_text("preserve staging")
        git("add", "unrelated.txt")
        self.config["autocommit"] = True
        monitor.progress_commit(self.config, state())
        self.assertEqual(git("diff", "--cached", "--name-only"), "unrelated.txt")
        self.assertEqual(git("show", "--pretty=format:", "--name-only", "HEAD"),
                         "progress/demo/demo-astra-1/000000000001.json")


if __name__ == "__main__":
    unittest.main()
