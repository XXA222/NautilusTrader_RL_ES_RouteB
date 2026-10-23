import json
from pathlib import Path
root=Path('.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021/ensemble/run-01')
for seed in ('31','102','20261018'):
 d=json.loads((root/f'seed-{seed}'/'fold_experiment_report.json').read_text())
 q=d['predictive_comparison']
 p=q['path_log_realized_volatility_comparison']
 selected=p['baseline_selected_by_validation_crps']
 print('\nSEED',seed,'status',d['status'],'selected',selected)
 print('PATHLOG horizon',p['horizon'],'means',p['mean_crps_by_candidate'])
 print('PATH_DIFF',p['paired_block_bootstrap'])
 print('RETURN15 E2')
 for x in q['paired_crps_differences']:
  if x['candidate']=='E2' and x['horizon']==15: print(x)
 diag=q.get('generation_diagnostics',{})
 print('DIAGNOSTIC_KEYS',list(diag))
