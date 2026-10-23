import json
p='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20260211-risk-conditioning-strength025-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json'
d=json.load(open(p))
for k in ['predictive_comparison','native_risk_comparison_by_seed','prior_risk_forecast_artifact_by_seed','candidate_training_runs_by_seed']:
 v=d.get(k)
 print('\n###',k)
 s=json.dumps(v,indent=2,sort_keys=True)
 print(s[:18000])

