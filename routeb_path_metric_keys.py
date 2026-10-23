import json
p='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json'
x=json.load(open(p))['predictive_comparison']
for k in ['path_log_realized_volatility_comparison','path_risk_scores']:
 v=x.get(k)
 print(k,type(v), list(v) if isinstance(v,dict) else len(v) if isinstance(v,list) else v)
 if isinstance(v,dict):
  for kk,vv in v.items():
   print(' ',kk,type(vv), list(vv)[:20] if isinstance(vv,dict) else len(vv) if isinstance(vv,list) else vv)
 if isinstance(v,list):
  for row in v[:20]: print(json.dumps(row,sort_keys=True))
print('paired E2 15m')
for row in x['paired_crps_differences']:
 if row.get('candidate')=='E2' and row.get('horizon')==15:
  print(row)

