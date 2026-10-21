from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

from rles_data.binance_archive import archive_events
from rles_data.core import CuratedStore, Event, MINUTE_NS
from rles_data.io import curated_event_time_bounds, curated_source_identity, write_curated, write_fact_json


def month_range(start: date, end_exclusive: date):
    current = date(start.year, start.month, 1)
    final = date(end_exclusive.year, end_exclusive.month, 1)
    while current <= final:
        yield current
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)


def ns_utc(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--instrument', default='BTCUSDT-PERP.BINANCE')
    parser.add_argument('--start-open', default='2025-09-20T00:00:00Z')
    parser.add_argument('--end-open', default='2026-10-18T00:00:00Z', help='exclusive UTC open-time boundary')
    parser.add_argument('--availability-delay-ms', type=int, default=1000)
    parser.add_argument('--row-group-boundary-available-at-ns', action='append', default=[],
                        type=int, help='split Parquet row groups before the first event at this availability timestamp')
    args = parser.parse_args()
    if args.availability_delay_ms <= 0:
        raise SystemExit('availability delay must be positive')
    if not re.fullmatch(r'[A-Z0-9]{3,24}', args.symbol):
        raise SystemExit('invalid exchange symbol')
    start = datetime.fromisoformat(args.start_open.replace('Z', '+00:00')).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end_open.replace('Z', '+00:00')).astimezone(timezone.utc)
    if start.minute or start.second or end.minute or end.second or end <= start:
        raise SystemExit('source interval must be ordered and minute aligned')
    start_day, last_day = start.date(), (end - timedelta(minutes=1)).date()
    sources = []
    for month in month_range(start_day, last_day + timedelta(days=1)):
        key = month.strftime('%Y-%m')
        monthly_name = f'{args.symbol}-1m-{key}.zip'
        monthly_path = args.raw_dir / monthly_name
        if monthly_path.is_file():
            sources.append((monthly_path, f'https://data.binance.vision/data/futures/um/monthly/klines/{args.symbol}/1m/{monthly_name}'))
            continue
        cursor = max(start_day, month)
        month_end = (date(month.year + (month.month == 12), month.month % 12 + 1, 1) - timedelta(days=1))
        final_day = min(last_day, month_end)
        while cursor <= final_day:
            day_name = f'{args.symbol}-1m-{cursor:%Y-%m-%d}.zip'
            daily_path = args.raw_dir / day_name
            if not daily_path.is_file():
                raise SystemExit(f'missing required archive: {daily_path}')
            sources.append((daily_path, f'https://data.binance.vision/data/futures/um/daily/klines/{args.symbol}/1m/{day_name}'))
            cursor += timedelta(days=1)
    events: list[Event] = []
    source_receipts = []
    delay_ns = args.availability_delay_ms * 1_000_000
    start_event_ns = ns_utc(start) + MINUTE_NS
    end_event_ns = ns_utc(end)
    for archive_path, source_url in sources:
        sidecar = archive_path.with_suffix(archive_path.suffix + '.CHECKSUM')
        if not sidecar.is_file():
            raise SystemExit(f'missing Binance checksum: {sidecar}')
        match = re.search(r'([0-9a-fA-F]{64})\s+\*?' + re.escape(archive_path.name) + r'\s*$', sidecar.read_text(encoding='ascii').strip())
        actual_hash = sha256(archive_path.read_bytes()).hexdigest()
        if not match or actual_hash != match.group(1).lower():
            raise SystemExit(f'Binance checksum mismatch: {archive_path.name}')
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise SystemExit(f'corrupt archive member: {archive_path.name}')
        archive_hash = actual_hash
        converted = archive_events(
            archive_path,
            stream='klines',
            instrument=args.instrument,
            source_url=source_url,
            expected_sha256=archive_hash,
            delay_ns=delay_ns,
        )
        events.extend(event for event in converted if start_event_ns <= event.ts_event_ns <= end_event_ns)
        source_receipts.append({'file': archive_path.name, 'url': source_url, 'sha256': archive_hash,
                                'bytes': archive_path.stat().st_size})
    events.sort(key=lambda event: (event.available_at_ns, event.ts_event_ns, event.event_id))
    expected_count = int((end_event_ns - start_event_ns) // MINUTE_NS) + 1
    if len(events) != expected_count:
        raise SystemExit(f'bar count mismatch: expected {expected_count}, received {len(events)}')
    if any(right.ts_event_ns - left.ts_event_ns != MINUTE_NS for left, right in zip(events, events[1:])):
        raise SystemExit('source contains a missing or duplicate one-minute close')
    if any(event.payload['instrument'] != args.instrument for event in events):
        raise SystemExit('instrument identity mismatch')
    target = args.output_dir.resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f'output directory must be empty: {target}')
    target.mkdir(parents=True, exist_ok=True)
    store = CuratedStore(events)
    parquet_path = target / 'market.parquet'
    available_times = [event.available_at_ns for event in events]
    if any(value is None for value in available_times):
        raise SystemExit('source contains an event without availability time')
    row_group_boundaries = tuple(sorted({
        bisect_left(available_times, boundary)
        for boundary in args.row_group_boundary_available_at_ns
    }))
    if any(offset <= 0 or offset >= len(events) for offset in row_group_boundaries):
        raise SystemExit('row-group boundary must fall inside the prepared source interval')
    dataset_hash = write_curated(
        parquet_path, store, row_group_size=1440,
        row_group_boundaries=row_group_boundaries,
    )
    identity = curated_source_identity(parquet_path)
    bounds = curated_event_time_bounds(parquet_path, expected_marker_sha256=identity['marker_sha256'])
    if identity['schema_version'] != '0.4.1' or identity['snapshot_hash'] != dataset_hash:
        raise SystemExit('curated identity validation failed')
    report = {
        'schema': 'routeb.binance-fold-source-preparation.v1',
        'source': f'Binance Public Data, USD-M Futures, {args.symbol} 1m klines',
        'instrument': args.instrument,
        'source_open_interval_utc': {'start_inclusive': args.start_open, 'end_exclusive': args.end_open},
        'event_interval_ns': {'start_inclusive': start_event_ns, 'end_inclusive': end_event_ns},
        'availability': 'derived_after_closed_bar',
        'availability_delay_ns': delay_ns,
        'bar_count': len(events),
        'row_group_size': 1440,
        'row_group_boundaries': [
            {'event_offset': offset, 'available_at_ns': available_times[offset]}
            for offset in row_group_boundaries
        ],
        'dataset_hash': dataset_hash,
        'marker_sha256': identity['marker_sha256'],
        'source_file_sha256': identity['file_sha256'],
        'footer_event_bounds_ns': bounds,
        'source_archives': source_receipts,
        'economics_evidence': 'not supplied; no fees, funding or instrument rules inferred',
    }
    write_fact_json(target / 'source_preparation.json', report)
    print(json.dumps({'status': 'PREPARED', 'curated_path': str(parquet_path), **report}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
