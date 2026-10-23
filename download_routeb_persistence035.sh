set -euo pipefail
root="/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021"
mkdir -p "$root/source/raw"
cd "$root/source/raw"
for month in 2026-05 2026-06 2026-07 2026-08; do
  name="TRXUSDT-1m-${month}.zip"
  base="https://data.binance.vision/data/futures/um/monthly/klines/TRXUSDT/1m/${name}"
  curl -fLsS --retry 5 --retry-all-errors --retry-delay 2 "$base" -o "$name"
  curl -fLsS --retry 5 --retry-all-errors --retry-delay 2 "${base}.CHECKSUM" -o "${name}.CHECKSUM"
done
sha256sum ./*.zip ./*.CHECKSUM


