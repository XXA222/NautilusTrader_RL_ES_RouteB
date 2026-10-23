set -euo pipefail
cd "/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB"
export PYTHONPATH="packages/data/src:packages/contracts/src"
root=".artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021"
packages/world_model/environment/.venv-wsl/bin/python qa/world_model/prepare_routeb_binance_fold_source.py \
  --raw-dir "$root/source/raw" \
  --output-dir "$root/source/curated" \
  --symbol TRXUSDT --instrument TRXUSDT-PERP.BINANCE \
  --start-open 2026-05-19T00:00:00Z \
  --end-open 2026-08-21T00:00:00Z \
  --availability-delay-ms 1000 \
  --row-group-boundary-available-at-ns 1785888060000000000
