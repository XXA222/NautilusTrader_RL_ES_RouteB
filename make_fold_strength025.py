import hashlib,json,pathlib
import pyarrow.parquet as pq
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
base=json.loads((root/'configs/routeb/fresh_trxusdt_20251106_persistent_risk_innovation_60_15_15_fold.json').read_text())
shift=97*86400*1_000_000_000
for section in ('train','validation','test'):
    for field,value in base['fold'][section].items():
        if field.endswith('_ns'): base['fold'][section][field]=value+shift
base['fold']['dataset_hash']='528bdc786707272888248e088cc364198a570db24b7b5fc6b2a997db2947a378'
base['marker_sha256']='fcbeaa361e912d926b53d3cebcd83db151f948cc45980428dd429fe7d8f6c5d7'
base['config']['environment']['market_path']='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/source/curated/market.parquet'
base['label']='routeb-fold-trxusdt-binance-20260211-risk-conditioning-strength025-v1'
out=root/'configs/routeb/fresh_trxusdt_20260211_risk_conditioning_strength025_60_15_15_fold.json'
out.write_text(json.dumps(base,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print('fold_config_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
pqf=pq.ParquetFile(base['config']['environment']['market_path']); m=pqf.metadata; ci=m.schema.names.index('available_at_ns')
lo=base['fold']['train']['warmup_start_ns']; hi=base['fold']['validation']['score_end_ns']; inside=[]; crossings=[]
for i in range(m.num_row_groups):
 s=m.row_group(i).column(ci).statistics; mn=int(s.min); mx=int(s.max)
 if mn>=lo and mx<hi: inside.append(i)
 elif mn<hi and mx>=lo: crossings.append(i)
print('footer_audit',json.dumps({'rows':m.num_rows,'row_groups':m.num_row_groups,'scope':[lo,hi],'scoped_groups':inside,'crossing_groups':crossings,'first_test_group_available_min':int(m.row_group(len(inside)).column(ci).statistics.min)}))
print('fold_config',json.dumps(base,indent=2))