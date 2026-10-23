set -euo pipefail
cd "/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB"
export PYTHONPATH="packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
root="/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021"
mkdir -p "$root/ensemble"
packages/world_model/environment/.venv-wsl/bin/python qa/world_model/run_registered_fold_ensemble.py --registered-fold-config configs/routeb/fresh_trxusdt_20250801_stochastic_risk_path_aligned_60_15_15_fold.json --e1-configuration configs/routeb/fresh_trxusdt_20250801_e1_stochastic_risk_path_aligned_ensemble.json --e2-configuration configs/routeb/fresh_trxusdt_20250801_e2_stochastic_risk_path_aligned_ensemble.json --worker-python packages/world_model/environment/.venv-wsl/bin/python --upstream-checkout .tmp/dreamerv3-route-b --output-dir "$root/ensemble/run-01" --model-seeds 31,102,20261018 --paths-per-seed 32 --forecast-seed 20261018 --horizons 1,5,15,30,60 --anchor-interval-minutes 120 --context-lookback 128 --volatility-window 15 --block-length 12 --bootstrap-replicates 2000 --seed 0 --timeout-seconds 1800