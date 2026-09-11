"""Keep the local game tunnel connected until TUNNEL_STOP is requested."""

import argparse
import os
import subprocess
import time

from .client import Client, load_config
from .storage import process_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    root = Client(load_config()).root
    command = ["ssh", "-N", "-T", "-i", args.identity, "-o", "IdentitiesOnly=yes",
               "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=3", "-L", "127.0.0.1:8765:127.0.0.1:8765", args.host]
    with process_lock(root / "tunnel.lock"):
        while not (root / "TUNNEL_STOP").exists():
            print("Connecting game tunnel", flush=True)
            process = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                while process.poll() is None:
                    if (root / "TUNNEL_STOP").exists():
                        process.terminate()
                        break
                    time.sleep(1)
                process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            if not (root / "TUNNEL_STOP").exists():
                print("Tunnel disconnected; reconnecting in 2 seconds", flush=True)
                time.sleep(2)


if __name__ == "__main__":
    main()
