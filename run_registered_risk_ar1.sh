set -euo pipefail
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'
export PYTHONPATH='packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src'
out='.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/ensemble'
mkdir -p "$out"
.tmp/routeb-host-linux-314/bin/python qa/world_model/run_registered_fold_ensemble.py \
  --registered-fold-config configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json \
  --e1-configuration configs/routeb/fresh_trxusdt_20251106_e1_persistent_risk_innovation_ensemble.json \
  --e2-configuration configs/routeb/fresh_trxusdt_20251106_e2_persistent_risk_innovation_ensemble.json \
  --worker-python packages/world_model/environment/.venv-wsl/bin/python \
  --upstream-checkout .tmp/dreamerv3-route-b \
  --output-dir "$out/run-01" \
  --model-seeds 31,102,20261018 --paths-per-seed 32 \
  --anchor-interval-minutes 120 --block-length 12 --bootstrap-replicates 2000 \
  > "$out/run-01.stdout.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/run-01.exit"
cat "$out/run-01.stdout.log"
exit "$status"