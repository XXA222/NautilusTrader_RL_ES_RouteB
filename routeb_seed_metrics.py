import json,pathlib
base=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/ensemble/run-01')
for seed in [31,102,20261018]:
 d=json.load(open(base/f'seed-{seed}/fold_experiment_report.json'))
 print('\nSEED',seed,'keys',list(d.keys()))
 pc=d.get('predictive_comparison',{})
 print('pc keys',list(pc.keys()))
 for key in ['path_log_realized_volatility_comparison','paired_crps_differences','overall','generation_diagnostics']:
  if key in pc: print(key,json.dumps(pc[key],sort_keys=True)[:4500])

