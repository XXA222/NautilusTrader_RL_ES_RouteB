from pathlib import Path
import hashlib,json,sqlite3,sys
root=Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
run=root/'.artifacts/world_model/fresh-trxusdt-20260519-risk-persistence035-20261021/ensemble/run-01'
report=json.loads((run/'multi_seed_fold_experiment_report.json').read_text())
receipt=json.loads((run/'research_completion_receipt.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
payload=dict(report); stored=payload.pop('report_sha256')
actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert stored==actual
assert report['status']=='COMPLETED'
assert report['training_seeds']==[31,102,20261018]
assert report['forecast_context']['anchors']==180 and report['total_forecast_paths_per_anchor']==96
assert report['test_payload_decoded'] is False and report['forecast_context']['test_payload_decoded'] is False
assert report['read_scope']['includes_test'] is False
for name,item in report['pooled_forecast_artifacts'].items():
 p=run/item['artifact']; assert sha(p)==item['sha256']
assert receipt=={'schema':'routeb.development-result-receipt.v1','access_key':report['research_access']['access_key'],'status':'completed','report_sha256':stored}
db=root/'.artifacts/research-access.sqlite'
c=sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True)
protocol='routeb-r2:routeb-fold-trxusdt-binance-20260519-risk-persistence035-v1'
rows=c.execute('select attempt_key,candidate_id,seed from trial_attempt where protocol_id=? order by seed,candidate_id',(protocol,)).fetchall()
ev={}
for key,candidate,seed in rows:
 ev[(seed,candidate)]=c.execute('select event_key,status,detail_json from trial_attempt_event where attempt_key=? order by recorded',(key,)).fetchall()
completion=c.execute('select status,report_sha,detail_json from development_result where access_key=?',(receipt['access_key'],)).fetchone()
c.close()
assert len(rows)==6, rows
assert all([(e[0],e[1]) for e in ev[k]]==[('registered','registered'),('started','started'),('completed','completed')] for k in ev),ev
assert completion and completion[0]=='completed' and completion[1]==stored,completion
print(json.dumps({
 'status':'PASS','report_sha256':stored,'digest_verified':True,
 'completion_receipt':'completed','ledger_development_result':completion[0],
 'attempts':[{'seed':s,'candidate':c,'events':[(e[0],e[1]) for e in ev[(s,c)]]} for _,c,s in rows],
 'pooled_artifacts':{name:{'sha256':item['sha256'],'verified':True,'shape':item['shape']} for name,item in report['pooled_forecast_artifacts'].items()},
 'test_payload_decoded':report['test_payload_decoded'],
 'evidence_scope':report['evidence_scope'],
 'decision_scope':report['decision_scope'],
 'predictive_comparison_keys':list(report['predictive_comparison'].keys()),
 'data_audit':report['data_audit'],
},sort_keys=True,indent=2,default=str))



