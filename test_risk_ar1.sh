set -euo pipefail
cd "/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB"
export PYTHONPATH="packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
packages/world_model/environment/.venv-wsl/bin/python -m pytest -q tests/world_model/test_routeb_quality_v2.py