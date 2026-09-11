"""Keep this Windows controller awake while its foreground thread is alive."""

import ctypes
import os
import time
from contextlib import contextmanager


@contextmanager
def keep_awake():
    if os.name == "nt":
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        if not function(0x80000001):
            raise OSError("Windows rejected the keep-awake request")
    try:
        yield
    finally:
        if os.name == "nt":
            function(0x80000000)


if __name__ == "__main__":
    with keep_awake():
        print(f"Keep-awake active, PID {os.getpid()}", flush=True)
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            pass
