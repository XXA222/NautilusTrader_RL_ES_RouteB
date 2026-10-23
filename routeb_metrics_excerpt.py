import json
p='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json'
d=json.load(open(p)); x=d['predictive_comparison']
for k,v in x.items():
 if k in ['path_log_realized_volatility_comparison','path_risk_scores','paired_crps_differences','overall','protocol']:
  print('\n###',k)
  if k=='overall':
   for model,rows in v.items():
    for row in rows:
     if row.get('horizon') in [15,60] and ('path' not in model.lower()):
      print(model,{a:row.get(a) for a in ['horizon','crps','coverage_90','coverage_95']})
  else:
   s=json.dumps(v,indent=2,sort_keys=True)
   print(s[:12000])

