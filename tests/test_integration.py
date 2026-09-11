import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from crosscut.client import Client
from crosscut.server import SessionStore, handler_for


class IntegrationTests(unittest.TestCase):
    def test_real_crafter_through_http_and_replay(self):
        with tempfile.TemporaryDirectory() as server_dir, tempfile.TemporaryDirectory() as local_dir:
            store = SessionStore(server_dir)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                client = Client(dict(server_url=f"http://127.0.0.1:{httpd.server_port}", campaign="test", data_dir=local_dir))
                self.assertEqual(client.health()["service"], "crosscut-crafter")
                initial = client.reset("player", 42)
                self.assertEqual(initial["steps"], 0)
                state = client.step("player", ["move_left", "do", "move_up"], initial["revision"])
                self.assertEqual(state["steps"], 3)
                self.assertEqual(state["total_steps"], 3)
                restored = SessionStore(server_dir).state("player")
                expected = store.state("player")
                self.assertEqual(restored, expected)
                self.assertTrue(state["image_path"].endswith(".png"))
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
