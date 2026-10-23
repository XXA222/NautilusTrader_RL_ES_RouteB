set -euo pipefail
cd "/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB"
export PYTHONPATH="packages/data/src:packages/contracts/src"
worker="packages/world_model/environment/.venv-wsl/bin/python"
root="/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021"
"$worker" qa/world_model/prepare_routeb_binance_fold_source.py --raw-dir "$root/raw" --output-dir "$root/source/curated" --symbol TRXUSDT --instrument TRXUSDT-PERP.BINANCE --start-open 2025-08-01T00:00:00Z --end-open 2025-11-03T00:00:00Z --availability-delay-ms 1000 --row-group-boundary-available-at-ns 1760745660000000000