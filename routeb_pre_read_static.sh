set -euo pipefail
cd "/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB"
export PYTHONPATH="packages/world_model/src:packages/data/src:packages/contracts/src:packages/orchestrator/src:packages/research/src"
packages/world_model/environment/.venv-wsl/bin/python - <<'PY'
import json, sqlite3
from pathlib import Path
import pyarrow.parquet as pq
from rles_world_model.cli import _fold, _read_json, _validate_registered_b0_fold
from rles_data.io import curated_source_identity
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.jax_train import TrainingConfig
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig

root=Path.cwd()
fold_path=root/'configs/routeb/fresh_trxusdt_20260211_risk_conditioning_strength025_60_15_15_fold.json'
raw=_read_json(str(fold_path))
fold=_fold(raw['fold'])
_validate_registered_b0_fold(fold)
e1=_read_json(str(root/'configs/routeb/fresh_trxusdt_20260211_e1_persistent_risk_strength025_ensemble.json'))
e2=_read_json(str(root/'configs/routeb/fresh_trxusdt_20260211_e2_persistent_risk_strength025_ensemble.json'))
t1=TrainingConfig(**e1['training']); t2=TrainingConfig(**e2['training'])
assert t1==t2
m1=GRUConfig(**e1['model']); m2=RSSMConfig(**e2['model'])
assert m2.rollout_risk_noise_persistence==0.65
assert m2.rollout_risk_conditioning_strength==0.25
source=Path(raw['config']['environment']['market_path'])
identity=curated_source_identity(str(source))
assert identity['snapshot_hash']==fold.dataset_hash
assert identity['marker_sha256']==raw['marker_sha256']
pf=pq.ParquetFile(source)
meta=pf.metadata
schema=pf.schema_arrow
col=schema.get_field_index('available_at_ns')
groups=[]
for i in range(meta.num_row_groups):
    rg=meta.row_group(i)
    idx=next(j for j in range(rg.num_columns) if rg.column(j).path_in_schema=='available_at_ns')
    st=rg.column(idx).statistics
    groups.append({'row_group':i,'rows':rg.num_rows,'min':st.min,'max':st.max})
scope_start=fold.train.warmup_start_ns
scope_end=fold.validation.score_end_ns
cross=[g['row_group'] for g in groups if g['min'] < scope_end and g['max'] >= scope_end]
selected=[g['row_group'] for g in groups if g['max'] >= scope_start and g['min'] < scope_end]
assert not cross, cross
assert selected==list(range(78)), selected
assert groups[78]['min']>=scope_end
db=root/'.artifacts/research-access.sqlite'
conn=sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True)
access=conn.execute("SELECT count(*) FROM access WHERE market=? AND family=? AND start_ns<? AND end_ns>?",('TRXUSDT','perpetual-market-v1',scope_end,scope_start)).fetchone()[0]
intent=conn.execute("SELECT count(*) FROM query_intent WHERE market=? AND family=? AND start_ns<? AND end_ns>?",('TRXUSDT','perpetual-market-v1',scope_end,scope_start)).fetchone()[0]
conn.close()
worker=WorkerClient(WorkerConfig(root/'packages/world_model/environment/.venv-wsl/bin/python',root/'.tmp/dreamerv3-route-b',timeout_seconds=1800))
env=worker.preflight()
out={'fold_hash':fold.hash,'fold_scope':{'start_ns':scope_start,'end_ns':scope_end},'fold_dataset_hash':fold.dataset_hash,'source_identity':identity,'row_groups_total':len(groups),'row_groups_reader_scope':selected,'row_groups_crossing_boundary':cross,'first_group_after_scope':groups[78],'e1_model':m1.__dict__,'e2_model':m2.__dict__,'training_config':t1.__dict__,'worker_environment':env,'ledger_access_overlap':access,'ledger_query_intent_overlap':intent}
print(json.dumps(out,sort_keys=True,indent=2,default=str))
assert access==0 and intent==0
PY
