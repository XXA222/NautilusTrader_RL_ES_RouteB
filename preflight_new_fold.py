import hashlib, json, sys
from pathlib import Path
root=Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
sys.path[:0]=[str(root/'packages/world_model/src'),str(root/'packages/data/src'),str(root/'packages/contracts/src'),str(root/'packages/orchestrator/src'),str(root/'packages/research/src')]
from rles_world_model.cli import _fold, _routeb_code_identity, _validate_registered_b0_fold
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.jax_train import TrainingConfig
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_data.io import curated_source_identity, curated_event_time_bounds
p=root/'configs/routeb/fresh_trxusdt_20250801_stochastic_risk_path_aligned_60_15_15_fold.json'
registered=json.loads(p.read_text())
fold=_fold(registered['fold']); _validate_registered_b0_fold(fold)
source=curated_source_identity(registered['config']['environment']['market_path'])
assert source['marker_sha256']==registered['marker_sha256'] and source['snapshot_hash']==fold.dataset_hash
bounds=curated_event_time_bounds(registered['config']['environment']['market_path'],expected_marker_sha256=registered['marker_sha256'])
assert bounds['min_ts_event_ns']<=fold.train.warmup_start_ns and bounds['max_ts_event_ns']>=fold.test.score_end_ns-60_000_000_000
for path, cls in [(root/'configs/routeb/fresh_trxusdt_20250801_e1_stochastic_risk_path_aligned_ensemble.json',GRUConfig),(root/'configs/routeb/fresh_trxusdt_20250801_e2_stochastic_risk_path_aligned_ensemble.json',RSSMConfig)]:
 d=json.loads(path.read_text());
 if cls: cls(**d['model'])
 TrainingConfig(**d['training'])
identity=_routeb_code_identity(root)
print(json.dumps({'preflight':'PASS_NO_RESERVATION','repo':identity,'fold_sha256':fold.hash,'source':source,'footer':bounds,'config_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,root/'configs/routeb/fresh_trxusdt_20250801_e1_stochastic_risk_path_aligned_ensemble.json',root/'configs/routeb/fresh_trxusdt_20250801_e2_stochastic_risk_path_aligned_ensemble.json']},'source_preparation_sha256':hashlib.sha256((Path(registered['config']['environment']['market_path']).parent/'source_preparation.json').read_bytes()).hexdigest()},indent=2))

