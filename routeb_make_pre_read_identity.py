from __future__ import annotations
import hashlib, json, sqlite3, os
from pathlib import Path
os.chdir('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
root=Path.cwd()
import pyarrow.parquet as pq
from rles_world_model.cli import _fold, _read_json, _routeb_code_identity, _validate_registered_b0_fold
from rles_data.io import curated_source_identity
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
rel_fold='configs/routeb/fresh_trxusdt_20260211_risk_conditioning_strength025_60_15_15_fold.json'
rel_e1='configs/routeb/fresh_trxusdt_20260211_e1_persistent_risk_strength025_ensemble.json'
rel_e2='configs/routeb/fresh_trxusdt_20260211_e2_persistent_risk_strength025_ensemble.json'
raw=_read_json(rel_fold); fold=_fold(raw['fold']); _validate_registered_b0_fold(fold)
start,end=fold.train.warmup_start_ns,fold.validation.score_end_ns
source=Path(raw['config']['environment']['market_path'])
srcid=curated_source_identity(str(source))
assert srcid['snapshot_hash']==fold.dataset_hash and srcid['marker_sha256']==raw['marker_sha256']
pf=pq.ParquetFile(source); groups=[]
for i in range(pf.metadata.num_row_groups):
    rg=pf.metadata.row_group(i)
    idx=next(j for j in range(rg.num_columns) if rg.column(j).path_in_schema=='available_at_ns')
    st=rg.column(idx).statistics
    groups.append({'row_group':i,'rows':rg.num_rows,'min_available_at_ns':st.min,'max_available_at_ns':st.max})
scope_groups=[g['row_group'] for g in groups if g['min_available_at_ns']<end and g['max_available_at_ns']>=start]
cross=[g['row_group'] for g in groups if g['min_available_at_ns']<end<=g['max_available_at_ns']]
assert scope_groups==list(range(78)) and not cross and groups[78]['min_available_at_ns']>=end
db=root/'.artifacts/research-access.sqlite'
conn=sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True)
access=conn.execute("SELECT count(*) FROM access WHERE market=? AND family=? AND start_ns<? AND end_ns>?",('TRXUSDT','perpetual-market-v1',end,start)).fetchone()[0]
intent=conn.execute("SELECT count(*) FROM query_intent WHERE market=? AND family=? AND start_ns<? AND end_ns>?",('TRXUSDT','perpetual-market-v1',end,start)).fetchone()[0]
conn.close()
assert access==0 and intent==0,(access,intent)
worker=WorkerClient(WorkerConfig(root/'packages/world_model/environment/.venv-wsl/bin/python',root/'.tmp/dreamerv3-route-b',timeout_seconds=1800))
worker_env=worker.preflight()
manifest=json.loads((root/'qa/reports/source_manifest_current.json').read_text())
source_root=root/'.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/source'
files={
 'registered_fold_config':root/rel_fold,
 'e1_config':root/rel_e1,'e2_config':root/rel_e2,
 'download_receipt':source_root/'raw/download_receipt.json',
 'curated_parquet':source,
 'completion_marker':source.with_name(source.name+'.COMPLETED.json'),
 'source_preparation':source.parent/'source_preparation.json',
}
out={
 'schema':'routeb.pre_read_identity.v1',
 'repository_commit':_routeb_code_identity()['repository_commit'],
 'routeb_code_identity':_routeb_code_identity(),
 'source_manifest_identity_sha256':manifest['source_identity_sha256'],
 'source_manifest_file_count':len(manifest['files']),
 'fold_hash':fold.hash,
 'fold_scope':{'start_ns':start,'end_ns':end,'includes_test':False},
 'source_identity':srcid,
 'file_sha256':{k:sha(p) for k,p in files.items()},
 'footer_audit':{'row_groups_total':len(groups),'reader_scope_groups':scope_groups,'crossing_groups':cross,'first_group_after_scope':groups[78]},
 'ledger_read_only_preflight':{'access_overlap':access,'query_intent_overlap':intent},
 'worker_environment':worker_env,
 'payload_rows_decoded':False,
}
outpath=root/'.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/pre_read_identity.json'
if outpath.exists(): raise SystemExit(f'refusing to overwrite {outpath}')
outpath.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(out,sort_keys=True,indent=2,default=str))

