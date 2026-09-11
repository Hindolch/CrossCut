"""Restart local tracking after failures without persisting its API credential."""

import time
from .client import Client, load_config
from .monitor import run_monitor


def main():
    config = load_config()
    root = Client(config).root
    while not (root / "MONITOR_STOP").exists():
        try:
            run_monitor(config)
            return
        except Exception as exc:
            print(f"Tracking failure: {type(exc).__name__}; retrying in 10 seconds", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
