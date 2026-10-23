import hashlib,json,sqlite3
from pathlib import Path
root=Path('.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021/ensemble/run-01')
p=root/'multi_seed_fold_experiment_report.json'; d=json.loads(p.read_text())
claimed=d['report_sha256']; body=dict(d); body.pop('report_sha256')
canonical=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
actual=hashlib.sha256(canonical).hexdigest()
receipt=json.loads((root/'research_completion_receipt.json').read_text())
c=sqlite3.connect('.artifacts/research-access.sqlite'); c.row_factory=sqlite3.Row
key=d['research_access']['access_key']
result=c.execute('SELECT * FROM development_result WHERE access_key=?',(key,)).fetchone()
attempts=c.execute("SELECT a.candidate_id,a.seed,e.status,e.event_key FROM trial_attempt a JOIN trial_attempt_event e USING(attempt_key) WHERE a.protocol_id=? ORDER BY a.seed,a.candidate_id,e.recorded",('routeb-r2:'+d['reservation_label'],)).fetchall()
artifacts=[]
for candidate,info in d['pooled_forecast_artifacts'].items():
 f=root/info['artifact']; artifacts.append((candidate,hashlib.sha256(f.read_bytes()).hexdigest()==info['sha256']))
print(json.dumps({'report_hash_claimed':claimed,'report_hash_recomputed':actual,'report_hash_valid':claimed==actual,'completion_receipt_matches':receipt['report_sha256']==claimed,'ledger_completion':dict(result) if result else None,'test_payload_decoded':d['test_payload_decoded'],'read_scope':d['read_scope'],'pooled_artifacts_valid':artifacts,'attempt_events':[dict(a) for a in attempts]},indent=2))
