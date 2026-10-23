set -euo pipefail
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'
export PYTHONPATH='packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src'
.tmp/routeb-host-linux-314/bin/python /mnt/c/Users/QX/Documents/ChatGPT/RouteB/audit_run.py