import sqlite3,datetime
p=r'D:\Program Files\NautilusTrader_RL_ES_RouteB\.artifacts\research-access.sqlite'
c=sqlite3.connect(p)
for row in c.execute("SELECT market,family,start_ns,end_ns,state,report_sha FROM access WHERE market='TRXUSDT' ORDER BY start_ns"):
 print(row[0],datetime.datetime.fromtimestamp(row[2]/1e9,datetime.timezone.utc).isoformat(),datetime.datetime.fromtimestamp(row[3]/1e9,datetime.timezone.utc).isoformat(),row[4],row[5])
