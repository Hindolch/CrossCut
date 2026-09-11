#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${CROSSCUT_PYTHON:-python3}"
port="${CROSSCUT_PORT:-8765}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo 'CROSSCUT_PORT must be between 1024 and 65535.' >&2
  exit 1
fi
if [[ ! -x .venv-server/bin/python ]]; then
  "$python_bin" -m venv .venv-server
fi
.venv-server/bin/python -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), "Use Python 3.11-3.13"'
.venv-server/bin/python -m pip install -r requirements-server.txt
mkdir -p data
nohup .venv-server/bin/python -m crosscut.server --port "$port" --data-dir data/server > data/server.log 2>&1 < /dev/null &
server_pid=$!
sleep 1
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo 'Server exited; inspect data/server.log.' >&2
  exit 1
fi
printf '%s\n' "$server_pid" > data/server.pid
printf 'Detached Crafter PID %s, loopback port %s. Log: data/server.log\n' "$server_pid" "$port"

nohup .venv-server/bin/python -m crosscut.finish --data-dir data/server --session "${CROSSCUT_SESSION:-diamonds-astra-1}" --output data/results --server-url "http://127.0.0.1:$port" > data/finish.log 2>&1 < /dev/null &
printf '%s\n' "$!" > data/finish.pid
