import hashlib,json,pathlib,sqlite3
from rles_world_model.cli import _fold,_read_json,_validate_registered_b0_fold
from rles_world_model.routeb.worker_client import WorkerClient,WorkerConfig
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
fold_path=root/'configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json'
fold_json=_read_json(str(fold_path)); fold=_fold(fold_json['fold']); _validate_registered_b0_fold(fold)
worker=WorkerClient(WorkerConfig(root/'packages/world_model/environment/.venv-wsl/bin/python',root/'.tmp/dreamerv3-route-b')).preflight()
db=root/'.artifacts/research-access.sqlite'; uri=db.as_uri()+'?mode=ro'
with sqlite3.connect(uri,uri=True) as con:
    access=con.execute("SELECT COUNT(*) FROM access WHERE market=? AND family=? AND start_ns < ? AND end_ns > ?",('TRXUSDT','perpetual-market-v1',1769126460000000000,1762387260000000000)).fetchone()[0]
    intent=con.execute("SELECT COUNT(*) FROM query_intent WHERE market=? AND family=? AND start_ns < ? AND end_ns > ?",('TRXUSDT','perpetual-market-v1',1769126460000000000,1762387260000000000)).fetchone()[0]
manifest=json.loads((root/'qa/reports/source_manifest_current.json').read_text())
artifacts=[
'configs/routeb/fresh_trxusdt_20251106_e1_persistent_risk_innovation_ensemble.json',
'configs/routeb/fresh_trxusdt_20251106_e2_persistent_risk_innovation_ensemble.json',
'configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/market.parquet',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/market.parquet.COMPLETED.json',
'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/source_preparation.json',
'qa/world_model/run_registered_fold_ensemble.py',
'packages/world_model/environment/pyproject.toml','packages/world_model/environment/uv.lock']
sha={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in artifacts}
print(json.dumps({'fold_hash':fold.hash,'fold_scope':[fold.train.warmup_start_ns,fold.validation.score_end_ns],'source_manifest_identity':manifest['source_identity_sha256'],'sha256':sha,'worker_preflight':worker,'ledger_read_only_counts':{'access_overlap':access,'query_intent_overlap':intent}},indent=2))