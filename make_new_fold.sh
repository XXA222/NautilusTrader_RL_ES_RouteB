set -euo pipefail
cd "/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB"
export PYTHONPATH="packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src:packages/world_model/src"
packages/world_model/environment/.venv-wsl/bin/python /mnt/c/Users/QX/Documents/ChatGPT/RouteB/make_new_fold.py