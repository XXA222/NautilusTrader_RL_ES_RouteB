set -euo pipefail
cd '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'
export PYTHONPATH='packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src'
for p in .tmp/routeb-host-linux-314/bin/python packages/world_model/environment/.venv-wsl/bin/python; do
  "$p" -c 'import sys; print(sys.executable,sys.version.split()[0]); import pyarrow, rles_world_model; print("PYARROW",pyarrow.__version__)'
done