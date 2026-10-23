import json,sys
from pathlib import Path
root=Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
sys.path[:0]=[str(root/'packages/world_model/src'),str(root/'packages/data/src'),str(root/'packages/contracts/src'),str(root/'packages/orchestrator/src'),str(root/'packages/research/src')]
from rles_world_model.routeb.worker_client import WorkerClient,WorkerConfig
worker=WorkerClient(WorkerConfig((root/'packages/world_model/environment/.venv-wsl/bin/python'),(root/'.tmp/dreamerv3-route-b').resolve(),timeout_seconds=1800))
result=worker.preflight()
print(json.dumps({'worker_preflight':'PASS','result':result},indent=2,default=str))

