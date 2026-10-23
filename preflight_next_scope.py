import datetime,sqlite3,pathlib,json
UTC=datetime.timezone.utc
start=datetime.datetime(2026,2,11,0,1,tzinfo=UTC); end=datetime.datetime(2026,4,30,0,1,tzinfo=UTC)
ns=lambda d:int(d.timestamp())*1_000_000_000
root=pathlib.Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB'); db=root/'.artifacts/research-access.sqlite'
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
 a=con.execute('SELECT COUNT(*) FROM access WHERE market=? AND family=? AND start_ns < ? AND end_ns > ?',('TRXUSDT','perpetual-market-v1',ns(end),ns(start))).fetchone()[0]
 q=con.execute('SELECT COUNT(*) FROM query_intent WHERE market=? AND family=? AND start_ns < ? AND end_ns > ?',('TRXUSDT','perpetual-market-v1',ns(end),ns(start))).fetchone()[0]
print(json.dumps({'source_open':['2026-02-11T00:00:00Z','2026-05-16T00:00:00Z'],'train_validation_ns':[ns(start),ns(end)],'train_validation_utc':[start.isoformat(),end.isoformat()],'test_score_utc':['2026-05-01T00:01:00Z','2026-05-16T00:01:00Z'],'access_overlap':a,'query_intent_overlap':q},indent=2))