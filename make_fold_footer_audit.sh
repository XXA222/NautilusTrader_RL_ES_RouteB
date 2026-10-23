set -euo pipefail
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'
export PYTHONPATH='packages/data/src:packages/contracts/src:packages/world_model/src:packages/orchestrator/src'
packages/world_model/environment/.venv-wsl/bin/python /mnt/c/Users/QX/Documents/ChatGPT/RouteB/make_fold_footer_audit.py