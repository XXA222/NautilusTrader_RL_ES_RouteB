import hashlib,json,pathlib,sqlite3
import pyarrow.parquet as pq
from rles_world_model.cli import _fold,_read_json,_routeb_code_identity,_validate_registered_b0_fold
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_world_model.routeb.jax_model import RSSMConfig
from rles_world_model.routeb.jax_train import TrainingConfig
from rles_world_model.routeb.worker_client import WorkerClient,WorkerConfig
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
fold_path=root/'configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json'
e1_path=root/'configs/routeb/fresh_trxusdt_20251106_e1_persistent_risk_innovation_ensemble.json'
e2_path=root/'configs/routeb/fresh_trxusdt_20251106_e2_persistent_risk_innovation_ensemble.json'
registered=_read_json(str(fold_path)); fold=_fold(registered['fold']); _validate_registered_b0_fold(fold)
e1=_read_json(str(e1_path)); e2=_read_json(str(e2_path))
model1=GRUConfig(**e1['model']); model2=RSSMConfig(**e2['model']); training1=TrainingConfig(**e1['training']); training2=TrainingConfig(**e2['training'])
if training1!=training2 or model2.rollout_risk_noise_persistence!=0.65: raise RuntimeError('candidate configuration mismatch')
market=pathlib.Path(registered['config']['environment']['market_path'])
metadata=pq.ParquetFile(market).metadata
column=metadata.schema.names.index('available_at_ns')
lo=fold.train.warmup_start_ns; hi=fold.validation.score_end_ns
scoped=[]; crossings=[]
for i in range(metadata.num_row_groups):
    stats=metadata.row_group(i).column(column).statistics
    mn=int(stats.min); mx=int(stats.max)
    if mn>=lo and mx<hi: scoped.append(i)
    elif mn<hi and mx>=lo: crossings.append(i)
if crossings or scoped!=list(range(78)): raise RuntimeError(f'footer scope mismatch: {scoped=} {crossings=}')
identity=_routeb_code_identity(root)
manifest=json.loads((root/'qa/reports/source_manifest_current.json').read_text())
worker=WorkerClient(WorkerConfig(root/'packages/world_model/environment/.venv-wsl/bin/python',root/'.tmp/dreamerv3-route-b')).preflight()
artifact_paths=[
'configs/routeb/fresh_trxusdt_20251106_e1_persistent_risk_innovation_ensemble.json',
'configs/routeb/fresh_trxusdt_20251106_e2_persistent_risk_innovation_ensemble.json',
'configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/raw/download_receipt.json',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/market.parquet',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/market.parquet.COMPLETED.json',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/source_preparation.json',
'qa/world_model/run_registered_fold_ensemble.py','packages/world_model/environment/pyproject.toml','packages/world_model/environment/uv.lock']
sha={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in artifact_paths}
db=root/'.artifacts/research-access.sqlite'
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
    access=con.execute('SELECT COUNT(*) FROM access WHERE market=? AND family=? AND start_ns < ? AND end_ns > ?',('TRXUSDT','perpetual-market-v1',hi,lo)).fetchone()[0]
    intent=con.execute('SELECT COUNT(*) FROM query_intent WHERE market=? AND family=? AND start_ns < ? AND end_ns > ?',('TRXUSDT','perpetual-market-v1',hi,lo)).fetchone()[0]
if access or intent: raise RuntimeError(f'ledger overlaps found: {access=} {intent=}')
report={'schema':'routeb.preregistered-fold-pre-read-identity.v1','repository_commit':identity['repository_commit'],'routeb_source_code_sha256':identity['source_code_sha256'],'source_manifest_identity_sha256':manifest['source_identity_sha256'],'fold_hash':fold.hash,'fold_scope_ns':[lo,hi],'scoped_row_groups':scoped,'crossing_row_groups':crossings,'candidate_and_source_sha256':sha,'worker_preflight':worker,'ledger_preflight_read_only':{'database':'.artifacts/research-access.sqlite','market':'TRXUSDT','family':'perpetual-market-v1','interval_ns':[lo,hi],'access_overlap':access,'query_intent_overlap':intent},'test_payload_decoded':False,'created_before_registered_reader':True}
out=root/'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/pre_read_identity.json'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
print(json.dumps(report,indent=2,sort_keys=True))