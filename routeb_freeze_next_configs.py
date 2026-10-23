import json
from pathlib import Path
root=Path(r'D:\Program Files\NautilusTrader_RL_ES_RouteB')
e1=json.loads((root/'configs/routeb/fresh_trxusdt_20260211_e1_persistent_risk_strength025_ensemble.json').read_text(encoding='utf-8'))
e2=json.loads((root/'configs/routeb/fresh_trxusdt_20260211_e2_persistent_risk_strength025_ensemble.json').read_text(encoding='utf-8'))
e2['model']['rollout_risk_noise_persistence']=0.35
e2['model']['rollout_risk_conditioning_strength']=0.25
for name,value in [
 ('fresh_trxusdt_20260519_e1_risk_persistence035_strength025_ensemble.json',e1),
 ('fresh_trxusdt_20260519_e2_risk_persistence035_strength025_ensemble.json',e2),
]:
 (root/'configs/routeb'/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')

