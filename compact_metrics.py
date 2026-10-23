import json,pathlib
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/ensemble/run-01')
r=json.loads((root/'multi_seed_fold_experiment_report.json').read_text()); p=r['predictive_comparison']
base=p['path_log_realized_volatility_comparison']
print('POOLED_15M_RV',json.dumps(base))
for name in ('E1','E2'):
 print('GEN',name,json.dumps({k:v for k,v in p['generation_diagnostics'][name].items() if k not in ('touch_reliability',)}))
for seed in r['training_seeds']:
 sr=json.loads((root/f'seed-{seed}'/'fold_experiment_report.json').read_text())['predictive_comparison']
 print('SEED',seed,'rv',json.dumps(sr['path_log_realized_volatility_comparison']))
 for row in sr['paired_crps_differences']:
  if row['candidate']=='E2' and row['baseline']=='E0_S0_ZERO_DRIFT_FIXED_VOL' and row['horizon'] in (15,):print('SEED_RETURN15',json.dumps(row))
for row in p['paired_crps_differences']:
 if row['candidate']=='E2' and row['baseline']=='E0_S0_ZERO_DRIFT_FIXED_VOL' and row['horizon']==15:print('POOL_RETURN15',json.dumps(row))
print('MODEL_SELECTION',p['model_selection'])