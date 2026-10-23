import hashlib,json,sqlite3,os
from pathlib import Path
os.chdir('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
root=Path.cwd()
import pyarrow.parquet as pq
from rles_world_model.cli import _fold,_read_json,_routeb_code_identity,_validate_registered_b0_fold
from rles_data.io import curated_source_identity
from rles_world_model.routeb.worker_client import WorkerClient,WorkerConfig
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
fold_rel='configs/routeb/fresh_trxusdt_20260519_risk_persistence035_60_15_15_fold.json'
e1_rel='configs/routeb/fresh_trxusdt_20260519_e1_risk_persistence035_strength025_ensemble.json'
e2_rel='configs/routeb/fresh_trxusdt_20260519_e2_risk_persistence035_strength025_ensemble.json'
reg=_read_json(fold_rel); fold=_fold(reg['fold']); _validate_registered_b0_fold(fold)
start,end=fold.train.warmup_start_ns,fold.validation.score_end_ns
src=Path(reg['config']['environment']['market_path']); source_id=curated_source_identity(str(src))
assert source_id['snapshot_hash']==fold.dataset_hash and source_id['marker_sha256']==reg['marker_sha256']
pf=pq.ParquetFile(src); groups=[]
for i in range(pf.metadata.num_row_groups):
 rg=pf.metadata.row_group(i); ix=next(j for j in range(rg.num_columns) if rg.column(j).path_in_schema=='available_at_ns'); st=rg.column(ix).statistics
 groups.append({'row_group':i,'rows':rg.num_rows,'min_available_at_ns':st.min,'max_available_at_ns':st.max})
scope=[g['row_group'] for g in groups if g['min_available_at_ns']<end and g['max_available_at_ns']>=start]
cross=[g['row_group'] for g in groups if g['min_available_at_ns']<end<=g['max_available_at_ns']]
assert scope==list(range(78)) and not cross and groups[78]['min_available_at_ns']>=end
db=root/'.artifacts/research-access.sqlite'; c=sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True)
a=c.execute('select count(*) from access where market=? and family=? and start_ns<? and end_ns>?',('TRXUSDT','perpetual-market-v1',end,start)).fetchone()[0]
q=c.execute('select count(*) from query_intent where market=? and family=? and start_ns<? and end_ns>?',('TRXUSDT','perpetual-market-v1',end,start)).fetchone()[0]; c.close(); assert a==0 and q==0
worker=WorkerClient(WorkerConfig(root/'packages/world_model/environment/.venv-wsl/bin/python',root/'.tmp/dreamerv3-route-b',timeout_seconds=1800)).preflight()
manifest=json.loads((root/'qa/reports/source_manifest_current.json').read_text())
sr=root/'.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021/source'
files={'fold_config':root/fold_rel,'e1_config':root/e1_rel,'e2_config':root/e2_rel,'download_may':sr/'raw/TRXUSDT-1m-2026-05.zip','download_june':sr/'raw/TRXUSDT-1m-2026-06.zip','download_july':sr/'raw/TRXUSDT-1m-2026-07.zip','download_august':sr/'raw/TRXUSDT-1m-2026-08.zip','curated_parquet':src,'completion_marker':src.with_name(src.name+'.COMPLETED.json'),'source_preparation':src.parent/'source_preparation.json'}
code=_routeb_code_identity()
out={'schema':'routeb.pre_read_identity.v1','repository_commit':code['repository_commit'],'routeb_code_identity':code,'source_manifest_identity_sha256':manifest['source_identity_sha256'],'source_manifest_file_count':len(manifest['files']),'fold_hash':fold.hash,'fold_scope':{'start_ns':start,'end_ns':end,'includes_test':False},'source_identity':source_id,'file_sha256':{k:sha(v) for k,v in files.items()},'footer_audit':{'row_groups_total':len(groups),'reader_scope_groups':scope,'crossing_groups':cross,'first_group_after_scope':groups[78]},'ledger_read_only_preflight':{'access_overlap':a,'query_intent_overlap':q},'worker_environment':worker,'payload_rows_decoded':False}
outp=root/'.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021/pre_read_identity.json'
if outp.exists(): raise SystemExit('refusing to overwrite '+str(outp))
outp.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(out,sort_keys=True,indent=2))

