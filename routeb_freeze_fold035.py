from pathlib import Path
import json,sqlite3
import pyarrow.parquet as pq
from rles_world_model.cli import _fold,_read_json,_validate_registered_b0_fold
from rles_data.io import curated_source_identity
from rles_world_model.routeb.worker_client import WorkerClient,WorkerConfig
root=Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
base=_read_json(str(root/'configs/routeb/fresh_trxusdt_20260211_risk_conditioning_strength025_60_15_15_fold.json'))
raw=base['fold']
delta=97*86400*1_000_000_000
for name,win in raw.items():
 if isinstance(win,dict):
  for key,val in win.items():
   if key.endswith('_ns') and val is not None: win[key]=val+delta
raw['dataset_hash']='7819bff058090b9e18adc70773c268713d4dd4556f2512084ac6702285bc358d'
outroot=root/'.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021'
market=outroot/'source/curated/market.parquet'
rawconfig={
 'schema_version':'rles.world_model.registered_fold_baseline.v0',
 'label':'routeb-fold-trxusdt-binance-20260519-risk-persistence035-v1',
 'instrument':'TRXUSDT-PERP.BINANCE',
 'config':{'environment':{'instrument':'TRXUSDT-PERP.BINANCE','market_path':str(market)}},
 'fold':raw,
 'marker_sha256':'f1ff58a3344ac44b41bf031c2ec4e7e7a5615ad6cb66e0170cbb7578b3fe5523',
 'allow_legacy_isolated_snapshot':False,
}
path=root/'configs/routeb/fresh_trxusdt_20260519_risk_persistence035_60_15_15_fold.json'
path.write_text(json.dumps(rawconfig,sort_keys=True,indent=2)+'\n',encoding='utf-8')
fold=_fold(raw); _validate_registered_b0_fold(fold)
identity=curated_source_identity(str(market))
assert identity['snapshot_hash']==fold.dataset_hash and identity['marker_sha256']==rawconfig['marker_sha256']
pf=pq.ParquetFile(market); groups=[]
for i in range(pf.metadata.num_row_groups):
 rg=pf.metadata.row_group(i); idx=next(j for j in range(rg.num_columns) if rg.column(j).path_in_schema=='available_at_ns'); st=rg.column(idx).statistics
 groups.append({'row_group':i,'rows':rg.num_rows,'min':st.min,'max':st.max})
start,end=fold.train.warmup_start_ns,fold.validation.score_end_ns
selected=[g['row_group'] for g in groups if g['min']<end and g['max']>=start]
cross=[g for g in groups if g['min']<end<=g['max']]
assert selected==list(range(78)) and not cross and groups[78]['min']>=end
conn=sqlite3.connect('file:'+str(root/'.artifacts/research-access.sqlite')+'?mode=ro',uri=True)
access=conn.execute("select count(*) from access where market=? and family=? and start_ns<? and end_ns>?",('TRXUSDT','perpetual-market-v1',end,start)).fetchone()[0]
intent=conn.execute("select count(*) from query_intent where market=? and family=? and start_ns<? and end_ns>?",('TRXUSDT','perpetual-market-v1',end,start)).fetchone()[0]
conn.close(); assert access==0 and intent==0
w=WorkerClient(WorkerConfig(root/'packages/world_model/environment/.venv-wsl/bin/python',root/'.tmp/dreamerv3-route-b',timeout_seconds=1800)).preflight()
print(json.dumps({'fold_config':str(path),'fold_config_hash':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),'fold_hash':fold.hash,'scope':{'start_ns':start,'end_ns':end},'identity':identity,'row_groups':len(groups),'selected_groups':selected,'crossings':cross,'first_after':groups[78],'ledger_access_overlap':access,'ledger_query_intent_overlap':intent,'worker':w},sort_keys=True,indent=2,default=str))

