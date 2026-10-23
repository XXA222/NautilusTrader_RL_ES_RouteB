set +e
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB' || exit 20
out='.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021'
mkdir -p "$out"
export ROUTEB_DREAMERV3_CHECKOUT="$PWD/.tmp/dreamerv3-route-b"
export PYTHONPATH="$PWD/packages/world_model/src:$PWD/packages/data/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$PWD/packages/world_model/environment/.venv-wsl/bin/python" qa/world_model/routeb_prior_gradient_smoke.py --output "$out/synthetic_prior_gradient_smoke.json" > "$out/synthetic_prior_gradient_smoke.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/synthetic_prior_gradient_smoke.exit"
exit "$status"