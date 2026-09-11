import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from PIL import Image

from crosscut.astra_worker import (Conflict, MODEL, Stopped, Teacher, cli_infer,
                                  handler_for, normalize, validate_request)


def request(name="one", color="green"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, "PNG")
    return dict(request_id=name, model=MODEL,
                frames=[base64.b64encode(buffer.getvalue()).decode()])


class TeacherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.calls = 0

        def infer(*_):
            self.calls += 1
            return {"action": "move_left"}, {"input_tokens": 123}
        self.infer = infer
        self.teacher = Teacher(self.root, infer=infer, prompt="Objective and actions only")

    def test_retry_and_restart_cache_without_second_call(self):
        first = self.teacher.predict(request())
        self.assertFalse(first["cached"])
        self.assertAlmostEqual(sum(first["probabilities"]), 1)
        self.assertEqual(first["target_kind"], "smoothed_teacher_action")
        self.assertEqual(first["chosen_action"], "move_left")
        self.assertEqual(first["epsilon"], 0.05)
        self.assertAlmostEqual(first["probabilities"][1], 0.95 + 0.05 / 17)
        self.assertAlmostEqual(first["probabilities"][0], 0.05 / 17)
        restarted = Teacher(self.root, infer=self.infer, prompt=self.teacher.prompt)
        second = restarted.predict(request())
        self.assertTrue(second["cached"])
        self.assertEqual(self.calls, 1)
        self.assertEqual(dict(first, cached=True), second)
        with self.assertRaises(Conflict):
            restarted.predict(request(color="red"))

    def test_failed_attempt_allows_audited_serialized_retry(self):
        def fail(*_):
            self.calls += 1
            raise TimeoutError("ambiguous")
        self.teacher.infer = fail
        with self.assertRaises(TimeoutError):
            self.teacher.predict(request())
        restarted = Teacher(self.root, infer=self.infer, prompt=self.teacher.prompt)
        response = restarted.predict(request())
        self.assertFalse(response["cached"])
        self.assertEqual(self.calls, 2)
        attempts = self.root / "requests/one/attempts"
        self.assertEqual(json.loads((attempts / "000001/record.json").read_text())["status"], "failed")
        self.assertEqual(json.loads((attempts / "000002/record.json").read_text())["status"], "complete")

    def test_concurrent_duplicate_requests_are_serialized_and_cached(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.teacher.predict(request()), range(4)))
        self.assertEqual(self.calls, 1)
        self.assertEqual(sum(not result["cached"] for result in results), 1)

    def test_stop_prevents_inference(self):
        (self.root / "STOP").touch()
        with self.assertRaises(Stopped):
            self.teacher.predict(request())
        self.assertEqual(self.calls, 0)

    def test_strict_validation_and_normalization(self):
        for body in (dict(request(), model="other"), dict(request(), frames=[]),
                     dict(request(), frames=["invalid"]), dict(request(), hidden_state={}),
                     dict(request(), request_id="../escape")):
            with self.assertRaises(ValueError):
                validate_request(body)
        for result in ({"action": "bad"}, {"action": True}, {"action": []},
                       {"probabilities": [1] * 17}, {"action": "noop", "extra": 1}):
            with self.assertRaises(ValueError):
                normalize(result)
        values = normalize({"action": "noop"})
        self.assertAlmostEqual(sum(values), 1)
        self.assertAlmostEqual(values[0], 0.95 + 0.05 / 17)
        self.assertTrue(all(v == 0.05 / 17 for v in values[1:]))

    def test_http_contract_with_mock_inference(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.teacher))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}"
            with urlopen(endpoint + "/health") as response:
                self.assertEqual(json.load(response)["model"], MODEL)
            payload = json.dumps(request()).encode()
            with urlopen(Request(endpoint + "/predict", data=payload,
                                 headers={"Content-Type": "application/json"})) as response:
                self.assertEqual(len(json.load(response)["probabilities"]), 17)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cli_retains_usage_but_not_event_traces(self):
        captured = []

        class Process:
            returncode = 0
            stdin = io.StringIO()
            stdout = io.StringIO('{"type":"item.completed","text":"PRIVATE_TRACE"}\n'
                                 '{"type":"turn.completed","usage":{"input_tokens":8}}\n')

            def poll(self):
                return 0

        def popen(command, **kwargs):
            captured.extend(command)
            Path(command[command.index("--output-last-message") + 1]).write_text(
                json.dumps({"action": "noop"}))
            return Process()

        with patch("crosscut.astra_worker.shutil.which", return_value="codex"), \
                patch("crosscut.astra_worker.subprocess.Popen", side_effect=popen):
            result, usage = cli_infer(self.root, [self.root / "frame.png"], "prompt", lambda: False)
        self.assertEqual(captured[captured.index("--model") + 1], MODEL)
        self.assertIn("--ephemeral", captured)
        self.assertIn('model_reasoning_effort="high"', captured)
        self.assertEqual(captured[captured.index("--sandbox") + 1], "read-only")
        self.assertIn("--output-schema", captured)
        self.assertEqual(usage, {"input_tokens": 8})
        self.assertFalse(any("PRIVATE_TRACE" in path.read_text() for path in self.root.glob("*.json")))


    def test_stop_interrupts_running_cli_process(self):
        class Process:
            stdin = io.StringIO()
            stdout = io.StringIO("")

            def poll(self):
                return None

        process = Process()
        checks = iter([False, True])
        with patch("crosscut.astra_worker.shutil.which", return_value="codex"), patch("crosscut.astra_worker.subprocess.Popen", return_value=process), patch("crosscut.astra_worker.stop_process") as terminate:
            with self.assertRaises(Stopped):
                cli_infer(self.root, [self.root / "frame.png"], "prompt", lambda: next(checks))
            terminate.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
