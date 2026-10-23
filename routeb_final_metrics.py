import json
p='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json'
x=json.load(open(p))['predictive_comparison']
print('PATH',json.dumps(x['path_log_realized_volatility_comparison'],sort_keys=True))
print('DIAG',json.dumps(x['generation_diagnostics'],sort_keys=True))
print('RETURN15',json.dumps([r for r in x['paired_crps_differences'] if r['candidate']=='E2' and r['horizon']==15 and r['baseline']==x['path_log_realized_volatility_comparison']['baseline_selected_by_validation_crps']],sort_keys=True))
print('RISK',json.dumps(x['path_risk_scores'],sort_keys=True)[:2000])

