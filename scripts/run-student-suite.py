"""Run the real GPU experiment detached; serve read-only loopback telemetry."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crosscut.storage import atomic_json, process_lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'experiments/LLM4Teach/exp2/astra.json'
    config = json.loads(config_path.read_text())
    def status(phase, **extra):
        atomic_json(out / 'suite-status.json', dict(run_id=config['run_id'], phase=phase,
            updated_at=time.time(), config=config, **extra))
    with process_lock(out / 'suite.lock'):
        if (out / 'train').exists():
            raise FileExistsError('Existing training data; refuse to reset the budget or overwrite evidence')
        (out / 'eval').mkdir(exist_ok=True)
        (out / 'eval/episodes.jsonl').touch()
        status('training')
        server = ThreadingHTTPServer(('127.0.0.1', 8768), partial(SimpleHTTPRequestHandler, directory=str(out)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        command = [sys.executable, '-u', str(ROOT / 'experiments/LLM4Teach/astra_runtime.py'),
                   '--config', str(config_path), '--output', str(out / 'train')]
        try:
            with (out / 'training.log').open('a', buffering=1) as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(out / 'process.json', {'pid': child.pid, 'command': command})
                code = child.wait()
            status('complete' if code == 0 else 'failed', exit_code=code,
                evaluation='Not launched: 1,000,000 total environment steps are reserved for training')
        except BaseException as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise
        # Keep final results reachable until the controller has collected them.
        while not (out / 'HTTP_STOP').exists():
            time.sleep(5)
        server.shutdown()


if __name__ == '__main__':
    main()
