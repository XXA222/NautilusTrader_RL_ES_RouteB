import json,pathlib,hashlib,sqlite3
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
out=root/'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/ensemble/run-01'
r=json.loads((out/'multi_seed_fold_experiment_report.json').read_text()); p=r['predictive_comparison']
for k in ['model_selection','path_log_realized_volatility_comparison','paired_crps_differences','path_risk_scores']:
 print(k, json.dumps(p.get(k),indent=2,sort_keys=True)[:12000])
print('data_audit',json.dumps(r['data_audit'],indent=2,sort_keys=True))
print('generation_diagnostics',json.dumps(p['generation_diagnostics'],indent=2,sort_keys=True))
for key,item in r['pooled_forecast_artifacts'].items():
 path=out/item['artifact']; actual=hashlib.sha256(path.read_bytes()).hexdigest(); print('artifact_sha',key,actual,item['sha256'],actual==item['sha256'])
with sqlite3.connect((root/'.artifacts/research-access.sqlite').as_uri()+'?mode=ro',uri=True) as con:
 print('access_columns',con.execute('pragma table_info(access)').fetchall())
 print('state_counts',con.execute('select state,count(*) from access group by state').fetchall())