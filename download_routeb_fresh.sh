set -euo pipefail
root="/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021"
mkdir -p "$root/raw"
cd "$root/raw"
for month in 2025-08 2025-09 2025-10 2025-11; do
  name="TRXUSDT-1m-${month}.zip"
  base="https://data.binance.vision/data/futures/um/monthly/klines/TRXUSDT/1m/${name}"
  curl -fLsS --retry 3 "$base" -o "$name"
  curl -fLsS --retry 3 "${base}.CHECKSUM" -o "${name}.CHECKSUM"
done
ls -lh
