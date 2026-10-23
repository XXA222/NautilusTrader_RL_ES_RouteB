import sqlite3
p=r'D:\Program Files\NautilusTrader_RL_ES_RouteB\.artifacts\research-access.sqlite'
c=sqlite3.connect(p); c.row_factory=sqlite3.Row
r=c.execute("SELECT * FROM access WHERE identity LIKE '%38521aee%'").fetchone()
print('ACCESS',dict(r) if r else None)
if r: print('RESULTS',[dict(x) for x in c.execute('SELECT * FROM development_result WHERE access_key=?',(r['access_key'],))])
print('INTENTS',[dict(x) for x in c.execute("SELECT * FROM query_intent WHERE identity LIKE '%38521aee%'")])
print('ATTEMPTS',[dict(x) for x in c.execute("SELECT attempt_key,candidate_id,seed FROM trial_attempt WHERE protocol_id LIKE '%20250426%'")])
