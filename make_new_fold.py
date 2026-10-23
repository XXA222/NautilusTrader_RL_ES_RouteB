import hashlib
import json
from pathlib import Path
import sys

root = Path('/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB')
sys.path.insert(0, str(root / 'packages/world_model/src'))
sys.path.insert(0, str(root / 'packages/data/src'))
from rles_world_model.cli import _fold
from rles_data.io import curated_event_time_bounds
old = json.loads(Path('/mnt/c/Users/QX/Documents/ChatGPT/RouteB/scratch-final/old_fold.json').read_text(encoding='utf-8'))
shift = 97 * 24 * 60 * 60 * 1_000_000_000
fold = old['fold']
for name in ('train', 'validation', 'test'):
    for key in ('warmup_start_ns', 'score_start_ns', 'score_end_ns'):
        fold[name][key] += shift
fold['dataset_hash'] = '2cbafd76127a661054a3b2fbe92539ab6f07249ccf128812784e077ddf4562d5'
label = 'routeb-fold-trxusdt-binance-20250801-stochastic-risk-path-aligned-v1'
market = '/mnt/d/Program Files/NautilusTrader_RL_ES_RouteB/.artifacts/world_model/fresh-trxusdt-20250801-stochastic-risk-path-aligned-20261021/source/curated/market.parquet'
source_dir = Path(market).parent
marker_path = source_dir / 'market.parquet.COMPLETED.json'
marker = json.loads(marker_path.read_text(encoding='utf-8'))
marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
payload = {
  'allow_legacy_isolated_snapshot': False,
  'config': {'environment': {'instrument': 'TRXUSDT-PERP.BINANCE', 'market_path': market}},
  'fold': fold,
  'instrument': 'TRXUSDT-PERP.BINANCE',
  'label': label,
  'marker_sha256': marker_sha,
  'schema_version': 'rles.world_model.registered_fold_baseline.v0',
}
out = root / 'configs/routeb/fresh_trxusdt_20250801_stochastic_risk_path_aligned_60_15_15_fold.json'
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
fold_hash = _fold(fold).hash
parquet_path = Path(market)
import pyarrow.parquet as pq
parquet = pq.ParquetFile(parquet_path)
start, end = fold['train']['warmup_start_ns'], fold['validation']['score_end_ns']
scoped = []
crossing = []
columns = {name: i for i, name in enumerate(parquet.schema_arrow.names)}
for i in range(parquet.metadata.num_row_groups):
    rg = parquet.metadata.row_group(i)
    event = rg.column(columns['ts_event_ns']).statistics
    available = rg.column(columns['available_at_ns']).statistics
    if int(event.max) >= start and int(event.min) < end or int(available.max) >= start and int(available.min) < end:
        if int(event.min) >= start and int(event.max) < end and int(available.min) >= start and int(available.max) < end:
            scoped.append(i)
        else:
            crossing.append(i)
print(json.dumps({
 'fold_config':str(out), 'fold_sha256':fold_hash,
 'fold_config_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'source_preparation_sha256':hashlib.sha256((source_dir/'source_preparation.json').read_bytes()).hexdigest(),
 'parquet_sha256':hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
 'marker_sha256':marker_sha, 'dataset_hash':marker['snapshot_hash'],
 'footer_bounds':curated_event_time_bounds(parquet_path,expected_marker_sha256=marker_sha),
 'row_group_count':parquet.metadata.num_row_groups,
 'registered_scope_groups':len(scoped), 'registered_scope_crossings':crossing,
 'registered_scope_group_first_last':[scoped[0],scoped[-1]],
},indent=2))
