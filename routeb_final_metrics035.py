import json
p='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json'
d=json.load(open(p)); x=d['predictive_comparison']
print('PATH',json.dumps(x['path_log_realized_volatility_comparison'],sort_keys=True))
base=x['path_log_realized_volatility_comparison']['baseline_selected_by_validation_crps']
print('RETURN15',json.dumps([r for r in x['paired_crps_differences'] if r['candidate']=='E2' and r['horizon']==15 and r['baseline']==base],sort_keys=True))
print('DIAG',json.dumps({m:{k:v[k] for k in ['generated_abs_return_lag1_autocorrelation','realized_abs_return_lag1_autocorrelation','generated_rolling_rv_mean','realized_rolling_rv_mean','generated_rolling_rv_p95','realized_rolling_rv_p95','generated_path_valid_rate']} for m,v in x['generation_diagnostics'].items()},sort_keys=True))

