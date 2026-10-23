import json,pathlib
base=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021/ensemble/run-01')
for seed in [31,102,20261018]:
 d=json.load(open(base/f'seed-{seed}/fold_experiment_report.json'))
 x=d['predictive_comparison']; p=x['path_log_realized_volatility_comparison']; b=p['baseline_selected_by_validation_crps']
 diff=p['paired_block_bootstrap']; g=x['generation_diagnostics']
 ret=[r for r in x['paired_crps_differences'] if r['candidate']=='E2' and r['horizon']==15 and r['baseline']==b]
 print(json.dumps({'seed':seed,'baseline':b,'path_logrv_mean_crps':p['mean_crps_by_candidate'],'path_logrv_E2_vs_baseline':diff,'return_crps_E2_vs_baseline_15m':ret,'generation':{m:{k:v[k] for k in ['generated_abs_return_lag1_autocorrelation','realized_abs_return_lag1_autocorrelation','generated_rolling_rv_mean','realized_rolling_rv_mean','generated_rolling_rv_p95','realized_rolling_rv_p95','generated_path_valid_rate']} for m,v in g.items()}},sort_keys=True))


