set -euo pipefail
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'
export PYTHONPATH='packages/data/src:packages/contracts/src:packages/world_model/src'
packages/world_model/environment/.venv-wsl/bin/python qa/world_model/prepare_routeb_binance_fold_source.py \
  --raw-dir '.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/source/raw' \
  --output-dir '.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/source/curated' \
  --symbol TRXUSDT --instrument TRXUSDT-PERP.BINANCE \
  --start-open '2026-02-11T00:00:00Z' --end-open '2026-05-16T00:00:00Z' \
  --availability-delay-ms 1000 \
  --row-group-boundary-available-at-ns 1777507260000000000 \
  > '.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/source/preparation.stdout.json'
cat '.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/source/preparation.stdout.json'