import json
import sqlite3
import sys


connection = sqlite3.connect(sys.argv[1])
if len(sys.argv) == 4:
    start_ns = int(sys.argv[2])
    end_ns = int(sys.argv[3])
    for table in ("access", "query_intent"):
        count = connection.execute(
            f"SELECT COUNT(*) FROM {table} "
            "WHERE market = ? AND family = ? AND start_ns < ? AND end_ns > ?",
            ("TRXUSDT", "perpetual-market-v1", end_ns, start_ns),
        ).fetchone()[0]
        print(f"{table}_overlap={count}")
    raise SystemExit
tables = [
    row[0]
    for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    )
]
print("TABLES", json.dumps(tables, indent=2))
for table in tables:
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    print(table, json.dumps(columns))
    if "market" in columns:
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE market LIKE ? ORDER BY rowid",
            ("%TRX%",),
        ).fetchall()
        print("TRX_ROWS", json.dumps(rows, default=str))
