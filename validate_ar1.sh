set -euo pipefail
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'
export PYTHONPATH='packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src'
packages/world_model/environment/.venv-wsl/bin/python -m pytest -q tests/world_model/test_routeb_quality_v2.py
packages/world_model/environment/.venv-wsl/bin/python -m ruff check packages/world_model/src/rles_world_model/routeb/jax_model.py tests/world_model/test_routeb_quality_v2.py qa/world_model/routeb_prior_gradient_smoke.py
packages/world_model/environment/.venv-wsl/bin/python -m compileall -q packages/world_model/src/rles_world_model/routeb/jax_model.py tests/world_model/test_routeb_quality_v2.py qa/world_model/routeb_prior_gradient_smoke.py