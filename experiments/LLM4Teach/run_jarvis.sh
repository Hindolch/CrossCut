#!/usr/bin/env bash
# Run inside an existing JarvisLabs GPU instance, from any directory.
set -euo pipefail
cd "$(dirname "$0")"
condition="${1:?Usage: bash run_jarvis.sh ppo|gemma12b|gemma31b|exp2 SEED RUN_NAME}"
seed="${2:?Specify training seed}"
run_name="${3:?Specify a unique run name}"
case "$condition" in ppo|gemma12b|gemma31b|exp2) ;; *) exit 2 ;; esac
config="configs/${condition}.json"
if [[ "$condition" == "exp2" ]]; then config="exp2/config.json"; fi
configured_steps="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_steps"])' "$config")"
steps="${4:-$configured_steps}"
episodes="${5:-100}"
python train.py --config "$config" --seed "$seed" --steps "$steps" --output "outputs/${run_name}"
python evaluate.py --checkpoint "outputs/${run_name}/checkpoint-${steps}.pt" \
  --output "outputs/${run_name}-eval" --seed 10000 --episodes "$episodes"
