from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from rles_data.core import DAY_NS
from rles_data.io import curated_event_time_bounds, curated_source_identity
from rles_orchestrator.research_access import (
    complete_routeb_fresh_fold,
    record_routeb_candidate_event,
    register_routeb_candidate_attempt,
    routeb_fresh_fold_access_key,
)
from rles_world_model.baselines import score_baselines
from rles_world_model.cli import (
    _fold, _read_json, _registered_fold_baseline, _routeb_code_identity,
    _validate_registered_b0_fold,
)
from rles_world_model.data_audit import audit_replay
from rles_world_model.routeb.fold_evaluation import build_fold_predictive_report
from rles_world_model.routeb.fold_experiment import (
    run_fold_experiment, verify_fold_experiment_report_digest,
)
from rles_world_model.routeb.jax_gru import GRUConfig
from rles_world_model.routeb.jax_model import PINNED_UPSTREAM, RSSMConfig
from rles_world_model.routeb.jax_train import SNAPSHOT_SHA, TrainingConfig
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig


def _fresh_json(path: Path, value: dict) -> None:
    if path.exists():
        raise ValueError(f'refusing to overwrite frozen artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == 'multi_seed_fold_experiment_report.json':
        _seal_report_digest(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def _seal_report_digest(report: dict) -> str:
    payload = dict(report)
    payload.pop('report_sha256', None)
    digest = sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')).hexdigest()
    report['report_sha256'] = digest
    verified = dict(report)
    recorded = verified.pop('report_sha256')
    actual = sha256(json.dumps(
        verified, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')).hexdigest()
    if recorded != actual:
        raise ValueError('aggregate report digest failed its post-seal verification')
    return digest


def _read_forecasts(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    report = json.loads((directory / 'fold_experiment_report.json').read_text(encoding='utf-8'))
    request = json.loads((directory / 'validation_forecast_request.json').read_text(encoding='utf-8'))
    anchors = np.asarray(request['anchor_timestamps_ns'], dtype=np.int64)
    path = directory / f'{name.lower()}_validation_forecasts.npz'
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {'bars'}:
            raise ValueError(f'{name} forecast archive has unexpected members')
        bars = np.asarray(payload['bars'], dtype=np.float64)
    return anchors, bars, report


def _sha_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Run E1/E2 with multiple seeds under one fresh RouteB Fold reservation.')
    parser.add_argument('--registered-fold-config', required=True, type=Path)
    parser.add_argument('--e1-configuration', required=True, type=Path)
    parser.add_argument('--e2-configuration', required=True, type=Path)
    parser.add_argument('--worker-python', required=True, type=Path)
    parser.add_argument('--upstream-checkout', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model-seeds', default='31,102,20261018')
    parser.add_argument('--paths-per-seed', type=int, default=32)
    parser.add_argument('--forecast-seed', type=int, default=20261018)
    parser.add_argument('--horizons', default='1,5,15,30,60')
    parser.add_argument('--anchor-interval-minutes', type=int, default=120)
    parser.add_argument('--context-lookback', type=int, default=128)
    parser.add_argument('--volatility-window', type=int, default=15)
    parser.add_argument('--block-length', type=int, default=12, help='12 two-hour anchors form a 24-hour bootstrap block')
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()

    seeds = tuple(int(piece.strip()) for piece in args.model_seeds.split(','))
    horizons = tuple(int(piece.strip()) for piece in args.horizons.split(','))
    if (len(seeds) < 3 or tuple(sorted(set(seeds))) != seeds or any(seed < 0 for seed in seeds)
            or not horizons or tuple(sorted(set(horizons))) != horizons
            or horizons[-1] > 60 or args.paths_per_seed < 2
            or args.anchor_interval_minutes < 1 or args.block_length < 1
            or args.bootstrap_replicates < 1 or args.timeout_seconds < 1):
        raise ValueError('invalid preregistered seed, horizon, forecast, or bootstrap protocol')
    output = args.output_dir.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('output directory must be new and its parent must already exist')
    if not args.worker_python.is_file() or not args.upstream_checkout.is_dir():
        raise ValueError('pinned worker Python and upstream checkout are required')
    upstream_commit = __import__('subprocess').run(
        ['git', '-C', str(args.upstream_checkout.resolve()), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if upstream_commit != PINNED_UPSTREAM:
        raise ValueError('upstream checkout differs from the pinned DreamerV3 commit')

    registered = _read_json(str(args.registered_fold_config))
    e1_raw, e2_raw = _read_json(str(args.e1_configuration)), _read_json(str(args.e2_configuration))
    expected = {'schema_version', 'config', 'fold', 'marker_sha256', 'label', 'instrument',
                'allow_legacy_isolated_snapshot'}
    if set(registered) != expected:
        raise ValueError('registered Fold config schema mismatch')
    if set(e1_raw) != {'model', 'training'} or set(e2_raw) != {'model', 'training'}:
        raise ValueError('candidate configs must contain exactly model and training objects')
    e1_model = asdict(GRUConfig(**e1_raw['model']))
    e2_model = asdict(RSSMConfig(**e2_raw['model']))
    e1_training = TrainingConfig(**e1_raw['training'])
    e2_training = TrainingConfig(**e2_raw['training'])
    if asdict(e1_training) != asdict(e2_training):
        raise ValueError('E1/E2 must share the same per-seed optimizer budget')

    fold = _fold(registered['fold'])
    _validate_registered_b0_fold(fold)
    train_days = (fold.train.score_end_ns - fold.train.score_start_ns) // DAY_NS
    if fold.test.score_end_ns - fold.train.score_start_ns != (train_days + 32) * DAY_NS:
        raise ValueError('the frozen 60/15/15 or 360/15/15 fold must reserve both one-day gaps')
    environment = registered['config'].get('environment', {})
    market_path = environment.get('market_path')
    if (registered['allow_legacy_isolated_snapshot'] is not False
            or registered['instrument'] != environment.get('instrument')
            or not isinstance(registered['instrument'], str)
            or not registered['instrument'].endswith('-PERP.BINANCE')
            or 'USDT-PERP.BINANCE' not in registered['instrument']
            or not isinstance(market_path, str)):
        raise ValueError('fresh RouteB Fold must use a row-group USD-M perpetual source')
    source_identity = curated_source_identity(market_path)
    if (source_identity['schema_version'] != '0.4.1'
            or source_identity['marker_sha256'] != registered['marker_sha256']
            or source_identity['snapshot_hash'] != fold.dataset_hash):
        raise ValueError('frozen source identity differs from the registered Fold')
    footer_bounds = curated_event_time_bounds(
        market_path, expected_marker_sha256=registered['marker_sha256'])
    if (footer_bounds['min_ts_event_ns'] > fold.train.warmup_start_ns
            or footer_bounds['max_ts_event_ns'] < fold.test.score_end_ns - 60_000_000_000):
        raise ValueError('0.4.1 footer does not cover the complete Fold')

    code_identity = _routeb_code_identity()
    output.mkdir(parents=True, exist_ok=False)
    worker = WorkerClient(WorkerConfig(
        args.worker_python.absolute(), args.upstream_checkout.resolve(),
        timeout_seconds=args.timeout_seconds,
    ))
    worker_environment = worker.preflight()
    configurations_by_seed: dict[int, dict[str, dict]] = {}
    attempt_keys: dict[tuple[int, str], str] = {}
    for model_seed in seeds:
        training = asdict(e1_training)
        training['seed'] = model_seed
        configurations = {
            'E1': {'model': e1_model, 'training': training},
            'E2': {'model': e2_model, 'training': training},
        }
        configurations_by_seed[model_seed] = configurations
        for candidate, configuration in configurations.items():
            attempt_keys[(model_seed, candidate)] = register_routeb_candidate_attempt(
                label=registered['label'], candidate_id=candidate,
                configuration=configuration, fold_sha256=fold.hash,
                feature_sha256=fold.feature_hash, seed=model_seed,
            )
    seed_reports: dict[str, dict] = {}
    e1_paths: list[np.ndarray] = []
    e2_paths: list[np.ndarray] = []
    common_anchors: np.ndarray | None = None
    per_seed_provenance: dict[str, dict[str, dict]] = {}
    receipt, fold, replay = _registered_fold_baseline(str(args.registered_fold_config), code_identity)
    access_window = {
        'start_ns': fold.train.warmup_start_ns,
        'end_ns': fold.validation.score_end_ns,
    }
    access_key = routeb_fresh_fold_access_key(
        registered['config'], access_window, label=receipt['label'],
        fold_sha256=fold.hash, repository_commit=code_identity['repository_commit'],
        source_code_sha256=code_identity['source_code_sha256'],
        source_identity=source_identity,
    )
    for (model_seed, candidate), attempt_key in attempt_keys.items():
        record_routeb_candidate_event(
            attempt_key, event_key='started', status='started',
            detail={
                'access_key': access_key,
                'seed': model_seed,
                'candidate': candidate,
                'test_payload_decoded': False,
            },
        )
    receipt_payload = {
        'schema': 'routeb.registered-fold-receipt.v1',
        'reservation_label': receipt['label'], 'marker_sha256': receipt['marker_sha256'],
        'dataset_hash': fold.dataset_hash, 'fold_hash': fold.hash,
        'access_key': access_key, **code_identity,
        'read_scope': {'start_ns': fold.train.warmup_start_ns,
                       'end_ns': fold.validation.score_end_ns, 'includes_test': False},
    }
    _fresh_json(output / 'registration_receipt.json', receipt_payload)

    for model_seed in seeds:
        configurations = configurations_by_seed[model_seed]
        seed_dir = output / f'seed-{model_seed}'
        seed_dir.mkdir()
        _fresh_json(seed_dir / 'registration_receipt.json', receipt_payload)
        try:
            run_report = run_fold_experiment(
                replay=replay, fold=fold, train_scope_id=receipt['label'],
                candidate_configurations=configurations, worker=worker, output_dir=seed_dir,
                repository_commit=code_identity['repository_commit'],
                source_code_sha256=code_identity['source_code_sha256'],
                marker_sha256=receipt['marker_sha256'], reservation_label=receipt['label'],
                forecast_seed=args.forecast_seed + model_seed,
                forecast_paths=args.paths_per_seed, horizons=horizons,
                anchor_interval_minutes=args.anchor_interval_minutes,
                context_lookback=args.context_lookback,
                volatility_window=args.volatility_window, block_length=args.block_length,
                bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
            )
            if not verify_fold_experiment_report_digest(run_report):
                raise ValueError(f'seed {model_seed} report digest verification failed')
        except Exception as exc:
            for candidate in configurations:
                record_routeb_candidate_event(
                    attempt_keys[(model_seed, candidate)],
                    event_key='failed', status='failed',
                    detail={
                        'error': type(exc).__name__,
                        'detail': str(exc),
                        'test_payload_decoded': False,
                    },
                )
            raise
        seed_reports[str(model_seed)] = run_report
        for candidate in configurations:
            candidate_run = run_report['candidate_training_runs'][candidate]
            record_routeb_candidate_event(
                attempt_keys[(model_seed, candidate)],
                event_key='completed', status='completed',
                detail={
                    'seed_report_sha256': run_report['report_sha256'],
                    'checkpoint_sha256': candidate_run['checkpoint_sha256'],
                    'optimizer_updates': candidate_run['optimizer_updates'],
                    'test_payload_decoded': False,
                },
            )
        e1_anchors, one_e1, _ = _read_forecasts(seed_dir, 'E1')
        e2_anchors, one_e2, _ = _read_forecasts(seed_dir, 'E2')
        if not np.array_equal(e1_anchors, e2_anchors):
            raise ValueError(f'seed {model_seed} E1/E2 anchors differ')
        if common_anchors is None:
            common_anchors = e1_anchors
        elif not np.array_equal(common_anchors, e1_anchors):
            raise ValueError(f'seed {model_seed} forecast anchors differ across seeds')
        if one_e1.shape[1] != args.paths_per_seed or one_e2.shape[1] != args.paths_per_seed:
            raise ValueError(f'seed {model_seed} path count mismatch')
        e1_paths.append(one_e1)
        e2_paths.append(one_e2)
        per_seed_provenance[str(model_seed)] = run_report['candidate_training_runs']

    if common_anchors is None:
        raise ValueError('no trained seed forecasts were produced')
    pooled = {
        'E1': np.concatenate(e1_paths, axis=1),
        'E2': np.concatenate(e2_paths, axis=1),
    }
    total_paths = len(seeds) * args.paths_per_seed
    pooled_paths: dict[str, dict] = {}
    for name, values in pooled.items():
        path = output / f'pooled_{name.lower()}_validation_forecasts.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, bars=values)
        pooled_paths[name] = {'artifact': path.name, 'sha256': _sha_file(path),
                              'shape': list(values.shape), 'paths_per_seed': args.paths_per_seed,
                              'training_seeds': list(seeds)}

    representative = seed_reports[str(seeds[0])]['candidate_training_runs']
    predictive = build_fold_predictive_report(
        replay, fold, pooled, representative, {'E1': common_anchors, 'E2': common_anchors},
        train_scope_id=receipt['label'], snapshot_sha256=SNAPSHOT_SHA,
        forecast_seed=args.forecast_seed, horizons=horizons,
        anchor_interval_minutes=args.anchor_interval_minutes,
        context_lookback=args.context_lookback,
        volatility_window=args.volatility_window, block_length=args.block_length,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
    )
    predictive['authorization_verified'] = True
    predictive['authorization_note'] = "Verified against this report's one-shot RouteB registration receipt."
    provenance = predictive.get('model_provenance', {})
    for name in ('E1', 'E2'):
        if name in provenance:
            provenance[name]['training_seeds'] = list(seeds)
            provenance[name]['paths_per_seed'] = args.paths_per_seed
            provenance[name]['member_checkpoint_sha256'] = [
                per_seed_provenance[str(s)][name]['checkpoint_sha256'] for s in seeds]
            provenance[name]['member_configuration_sha256'] = [
                per_seed_provenance[str(s)][name]['configuration_sha256'] for s in seeds]
            provenance[name]['optimizer_updates_per_member'] = [
                per_seed_provenance[str(s)][name]['optimizer_updates'] for s in seeds]
    predictive.pop('report_sha256', None)
    predictive['report_sha256'] = sha256(json.dumps(
        predictive, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    audit = audit_replay(replay)
    baselines = [asdict(score) for score in score_baselines(
        replay, fold, seed=args.seed, ensemble_size=total_paths,
        volatility_lookback=args.volatility_window, bootstrap_block=args.block_length,
    )]
    aggregate = {
        'schema': 'routeb.multi-seed-fold-experiment.v1', 'status': 'COMPLETED',
        'evidence_scope': 'registered_fresh_fold_train_validation',
        'dataset_hash': fold.dataset_hash, 'fold_hash': fold.hash,
        'fold_manifest': {
            'schema_version': fold.schema_version, 'dataset_hash': fold.dataset_hash,
            'feature_hash': fold.feature_hash, 'train': asdict(fold.train),
            'validation': asdict(fold.validation), 'test': asdict(fold.test),
            'purge_ns': fold.purge_ns, 'embargo_ns': fold.embargo_ns,
            'scaler_hash': fold.scaler_hash,
        },
        'repository_commit': code_identity['repository_commit'],
        'source_code_sha256': code_identity['source_code_sha256'],
        'runner_sha256': _sha_file(Path(__file__).resolve()),
        'marker_sha256': receipt['marker_sha256'], 'reservation_label': receipt['label'],
        'source_identity': source_identity, 'footer_time_bounds': footer_bounds,
        'research_access': {
            'access_key': access_key,
            'completion_protocol': 'routeb.development-result-receipt.v1',
        },
        'read_scope': {'start_ns': fold.train.warmup_start_ns,
                       'end_ns': fold.validation.score_end_ns, 'includes_test': False},
        'data_audit': asdict(audit),
        'training_seeds': list(seeds), 'paths_per_seed': args.paths_per_seed,
        'total_forecast_paths_per_anchor': total_paths,
        'forecast_anchor_interval_minutes': args.anchor_interval_minutes,
        'bootstrap_block_anchors': args.block_length,
        'bootstrap_block_hours': args.block_length * args.anchor_interval_minutes / 60,
        'candidate_training_runs_by_seed': per_seed_provenance,
        'native_risk_comparison_by_seed': {
            seed: seed_reports[seed]['predictive_comparison'].get('native_risk_comparison')
            for seed in seed_reports
        },
        'prior_risk_forecast_artifact_by_seed': {
            seed: seed_reports[seed].get('prior_risk_forecast_artifact')
            for seed in seed_reports
        },
        'worker_environment': worker_environment,
        'per_seed_report_sha256': {s: seed_reports[s]['report_sha256'] for s in seed_reports},
        'forecast_context': {'anchors': len(common_anchors), 'paths_per_seed': args.paths_per_seed,
                             'pooled_paths': total_paths, 'horizon': horizons[-1],
                             'test_payload_decoded': False},
        'pooled_forecast_artifacts': pooled_paths,
        'baselines': baselines, 'predictive_comparison': predictive,
        'test_payload_decoded': False, 'economic_certification': False,
        'decision_scope': 'R2 predictive evidence only; this run does not certify trading economics or authorize R3-R6.',
    }
    _seal_report_digest(aggregate)
    _fresh_json(output / 'multi_seed_fold_experiment_report.json', aggregate)
    completion_receipt = complete_routeb_fresh_fold(
        registered['config'], access_window, label=receipt['label'],
        fold_sha256=fold.hash, repository_commit=code_identity['repository_commit'],
        source_code_sha256=code_identity['source_code_sha256'],
        source_identity=source_identity,
        report_sha256=aggregate['report_sha256'],
        detail={
            'report_schema': aggregate['schema'],
            'test_payload_decoded': False,
            'economic_certification': False,
        },
    )
    _fresh_json(output / 'research_completion_receipt.json', completion_receipt)
    print(json.dumps({'status': aggregate['status'], 'report': str(output / 'multi_seed_fold_experiment_report.json'),
                      'training_seeds': list(seeds), 'anchors': len(common_anchors),
                      'paths_per_seed': args.paths_per_seed, 'pooled_paths': total_paths,
                      'dataset_hash': fold.dataset_hash, 'fold_hash': fold.hash,
                      'report_sha256': aggregate['report_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure = {
            'schema': 'routeb.multi-seed-runner-failure.v1',
            'status': 'BLOCKED', 'error': type(exc).__name__,
            'detail': str(exc), 'traceback': traceback.format_exc(),
        }
        try:
            output_arg = sys.argv.index('--output-dir') + 1
            output_dir = Path(sys.argv[output_arg])
            failure_path = output_dir / 'runner_failure.json'
            if output_dir.is_dir() and not failure_path.exists():
                failure_path.write_text(json.dumps(failure, sort_keys=True, indent=2,
                                                   ensure_ascii=False) + '\n',
                                        encoding='utf-8')
        except (ValueError, IndexError, OSError):
            pass
        print(json.dumps({'status': 'BLOCKED', 'error': type(exc).__name__, 'detail': str(exc)},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        raise
