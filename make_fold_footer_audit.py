import hashlib,json,pathlib
from datetime import timedelta
import pyarrow.parquet as pq
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
base=json.loads((root/'configs/routeb/fresh_trxusdt_20250801_stochastic_risk_path_aligned_60_15_15_fold.json').read_text())
shift=97*86400*1_000_000_000
for section in ('train','validation','test'):
    for field,value in base['fold'][section].items():
        if field.endswith('_ns'):
            base['fold'][section][field]=value+shift
base['fold']['dataset_hash']='d9c202e17488d5e4059347074aa23a8533e5d5a1aed1e1d0891b0dca146ab956'
base['marker_sha256']='57bca13d146f70d44b0b3e322a4a99323ff1b0bd1757868dd9f998cd7405209b'
base['config']['environment']['market_path']='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/source/curated/market.parquet'
base['label']='routeb-fold-trxusdt-binance-20251106-persistent-risk-innovation-v1'
out=root/'configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json'
out.write_text(json.dumps(base,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('fold_config_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
print('fold_config',json.dumps(base,indent=2))
pqfile=pq.ParquetFile(base['config']['environment']['market_path'])
metadata=pqfile.metadata
names=metadata.schema.names
col=names.index('available_at_ns')
lo=base['fold']['train']['warmup_start_ns']; hi=base['fold']['validation']['score_end_ns']
inside=[]; crossing=[]
for i in range(metadata.num_row_groups):
    stats=metadata.row_group(i).column(col).statistics
    mn=int(stats.min); mx=int(stats.max)
    if mn>=lo and mx<hi: inside.append(i)
    elif mn<hi and mx>=lo: crossing.append((i,mn,mx))
print('FOOTER_AUDIT',json.dumps({'row_groups':metadata.num_row_groups,'rows':metadata.num_rows,'scoped_groups':inside,'crossing_groups':crossing,'first_test_group_stats':{'min':int(metadata.row_group(len(inside)).column(col).statistics.min),'max':int(metadata.row_group(len(inside)).column(col).statistics.max)} if len(inside)<metadata.num_row_groups else None,'scope':[lo,hi]}))