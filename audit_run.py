import json,pathlib,sqlite3,hashlib
from rles_world_model.routeb.fold_experiment import verify_fold_experiment_report_digest
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
out=root/'.artifacts/world_model/fresh-trxusdt-20251106-persistent-risk-innovation-20261021/ensemble/run-01'
report=json.loads((out/'multi_seed_fold_experiment_report.json').read_text())
receipt=json.loads((out/'research_completion_receipt.json').read_text())
print('report_keys',sorted(report))
print('predictive_keys',sorted(report['predictive_comparison']))
print('predictive',json.dumps(report['predictive_comparison'],indent=2)[:15000])
print('pooled_artifacts',json.dumps(report['pooled_forecast_artifacts'],indent=2))
print('receipt',json.dumps(receipt,indent=2))
print('digest_valid',verify_fold_experiment_report_digest(report))
print('report_sha256',report['report_sha256'])
print('scope',report['read_scope'],'test_payload_decoded',report['test_payload_decoded'])
db=root/'.artifacts/research-access.sqlite'
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
    con.row_factory=sqlite3.Row
    key=receipt['access_key']
    access=dict(con.execute('select access_key,state,report_sha,start_ns,end_ns from access where access_key=?',(key,)).fetchone())
    development=dict(con.execute('select access_key,status,report_sha from development_result where access_key=?',(key,)).fetchone())
    label='routeb-fold-trxusdt-binance-20251106-persistent-risk-innovation-v1'
    attempts=[dict(x) for x in con.execute('select attempt_key,candidate_id,seed,candidate_sha from trial_attempt where protocol_id=? order by seed,candidate_id',(f'routeb-r2:{label}',))]
    event_counts=[dict(x) for x in con.execute('select a.candidate_id,a.seed,e.status,count(*) as n from trial_attempt a join trial_attempt_event e using(attempt_key) where a.protocol_id=? group by a.candidate_id,a.seed,e.status order by a.seed,a.candidate_id,e.status',(f'routeb-r2:{label}',))]
print('ledger_access',json.dumps(access,indent=2)); print('ledger_development',json.dumps(development,indent=2)); print('attempts',json.dumps(attempts,indent=2)); print('events',json.dumps(event_counts,indent=2))