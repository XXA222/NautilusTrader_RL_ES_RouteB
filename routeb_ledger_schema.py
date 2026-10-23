import sqlite3
p='/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/research-access.sqlite'
c=sqlite3.connect('file:'+p+'?mode=ro',uri=True)
for name,sql in c.execute("select name,sql from sqlite_master where name like '%trial_attempt%' or name like '%development_result%'"):
 print(name,sql)
c.close()
