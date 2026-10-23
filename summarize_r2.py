import hashlib,json
from pathlib import Path
p=Path('.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021/ensemble/run-01/multi_seed_fold_experiment_report.json')
d=json.loads(p.read_text())
print('top_keys',list(d))
for k,v in d.items():
 if isinstance(v,dict): print('DICT',k,list(v)[:40])
 elif isinstance(v,list): print('LIST',k,len(v))
 else: print('SCALAR',k,repr(v)[:180])
print('report_file_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
print('receipt',Path(p.parent/'research_completion_receipt.json').read_text())
for name in ('seed-31','seed-102','seed-20261018'):
 q=p.parent/name/'fold_experiment_report.json'
 x=json.loads(q.read_text())
 print('SEED',name,'keys',list(x))
 for k,v in x.items():
  if any(t in k.lower() for t in ('metric','score','gate','crps','risk','skill')): print(' ',k,repr(v)[:1000])
