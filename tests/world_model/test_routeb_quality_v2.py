"""Adversarial regression suite for the v2 Route B additive source overlay."""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rles_world_model.routeb.market import Candle, MarketSequence, MINUTE_NS
from rles_world_model.routeb.codec import OHLCVCodec, TrainOnlyStandardizer
from rles_world_model.routeb.evaluation import multi_horizon_scores, return_paths
from rles_world_model.routeb.sequences import (
    SequenceBatch, make_windows, export_train_sequences, load_worker_batch,
)
from rles_world_model.routeb.integration import (
    prepare_forecast_context, prepare_fold_train_sequences,
)
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_data.core import FoldManifest, Window
from rles_world_model.routeb.imagined_account import ImaginedAccount
from rles_world_model.routeb.checkpoint import optimizer_recipe
from rles_world_model.routeb.jax_train import TrainingConfig
from rles_world_model.routeb.jax_model import RSSMConfig

UPSTREAM = 'e01491fad6434b2245a3b8ca201dd7faedcc458c'
SNAPSHOT = '80194a587444bf4a11f06f6e273723e5754f2483'


def candles(count=128, *, start=1):
    price = 100.0
    result = []
    for i in range(start, start + count):
        op = price * (1 + 0.0001)
        close = op * (1 - 0.00002 * math.cos(i / 3))
        result.append(Candle(i * MINUTE_NS, i * MINUTE_NS, 'ETH-PERP',
                             op, max(op, close) * 1.001,
                             min(op, close) * 0.999, close, 3.0))
        price = close
    return tuple(result)


def prepared():
    bars = candles()
    seq = MarketSequence(bars, 'dataset-20261018', 'train')
    feature = OHLCVCodec.encode(bars[1:], previous_close=bars[0].close)
    scaler = TrainOnlyStandardizer.fit(feature, dataset_id=seq.dataset_id, scope='train')
    return bars, seq, scaler


def test_masks_include_last_target_and_end_at_train_boundary():
    _, seq, scaler = prepared()
    b = make_windows(seq, scaler=scaler, length=32, burn_in=8, stride=19)
    assert np.all(~b.masks[:, :8])
    assert np.all(b.masks[:, 8:])
    assert np.all(b.reset[:, 0]) and not np.any(b.reset[:, 1:])
    assert np.max(b.timestamps) <= seq.candles[-1].ts_close_ns


def test_loss_interval_masks_only_train_score_targets():
    _, seq, scaler = prepared()
    score_start = 40 * MINUTE_NS
    score_end = 110 * MINUTE_NS
    batch = make_windows(
        seq, scaler=scaler, length=24, burn_in=5, stride=8,
        loss_interval=(score_start, score_end),
    )
    assert np.all(batch.timestamps[batch.masks] >= score_start)
    assert np.all(batch.timestamps[batch.masks] < score_end)
    assert np.all(batch.masks[:, -1])
    assert np.all(np.diff(batch.masks.astype(np.int8), axis=1) >= 0)


def test_scoring_includes_first_post_burn_in_target():
    # nll next step uses target mask[1:], with reset at time zero only.
    _, seq, scaler = prepared()
    b = make_windows(seq, scaler=scaler, length=16, burn_in=5)
    valid = b.masks[:, 1:] * (~b.reset[:, 1:])
    assert valid.sum(axis=1).tolist() == [11] * len(b.features)


def test_fold_train_artifact_fits_scaler_and_targets_on_train_score_only(tmp_path):
    dataset_hash = "a" * 64
    fold = FoldManifest(
        "0.4.0", dataset_hash, "b" * 64,
        Window(1 * MINUTE_NS, 20 * MINUTE_NS, 90 * MINUTE_NS),
        Window(90 * MINUTE_NS, 100 * MINUTE_NS, 140 * MINUTE_NS),
        Window(140 * MINUTE_NS, 150 * MINUTE_NS, 160 * MINUTE_NS),
        0, 0, None,
    )
    frames = []
    for index in range(1, 140):
        # Keep Train warmup and Validation visibly different so accidental
        # inclusion in scaler fitting is immediately detectable.
        base = 2.0 if index < 20 else 100.0 if index < 90 else 10_000.0
        open_price = base + index * 0.01
        close = open_price + 0.02
        frames.append(MarketFrame(
            index * MINUTE_NS, index * MINUTE_NS, "ETH-PERP",
            open_price, close + 0.1, open_price - 0.1, close, float(index),
        ))
    replay = MarketReplay(tuple(frames), dataset_hash)
    target = tmp_path / "fold-train.npz"
    _, scaler, summary = prepare_fold_train_sequences(
        fold_replay=replay, fold=fold, train_scope_id="train-fold-001",
        target_npz=target, length=16, burn_in=4, stride=8,
    )

    # The scaler reference uses only coordinates whose candle timestamps fall
    # in [Train.score_start, Train.score_end).
    train_candles = tuple(
        Candle(frame.ts_close_ns, frame.available_at_ns, frame.instrument,
               frame.open, frame.high, frame.low, frame.close, frame.volume)
        for frame in frames if frame.ts_close_ns < fold.train.score_end_ns
    )
    coordinates = OHLCVCodec.encode(train_candles[1:], previous_close=train_candles[0].close)
    timestamps = np.asarray([bar.ts_close_ns for bar in train_candles[1:]])
    expected = TrainOnlyStandardizer.fit(
        coordinates[(timestamps >= fold.train.score_start_ns)
                    & (timestamps < fold.train.score_end_ns)],
        dataset_id=dataset_hash, scope="train",
    )
    np.testing.assert_array_equal(scaler.mean, expected.mean)
    np.testing.assert_array_equal(scaler.std, expected.std)
    assert summary["validation_payload_in_worker"] is False
    assert summary["test_payload_in_worker"] is False
    batch, metadata = load_worker_batch(
        target, expected_scope="train-fold-001", snapshot_sha=SNAPSHOT,
        upstream_sha=UPSTREAM,
    )
    assert metadata["train_scope_id"] == "train-fold-001"
    assert metadata["fold_sha256"] == fold.hash
    assert np.all(batch.timestamps[batch.masks] >= fold.train.score_start_ns)
    assert np.all(batch.timestamps[batch.masks] < fold.train.score_end_ns)
    assert set(np.load(target, allow_pickle=False).files) == {
        "features", "reset", "masks", "timestamps", "risk_targets", "risk_masks",
        "risk_horizons", "risk_barrier_fractions", "scaler_mean", "scaler_std",
    }
    assert batch.risk_targets.shape == (*batch.features.shape[:2], 5, 6)
    assert batch.risk_masks.shape == (*batch.features.shape[:2], 5)
    assert np.all(batch.risk_masks[..., 4] <= batch.risk_masks[..., 3])
    assert np.all(batch.risk_masks[..., 3] <= batch.risk_masks[..., 2])
    assert np.all(batch.risk_masks[..., 2] <= batch.risk_masks[..., 1])
    assert np.all(batch.risk_masks[..., 1] <= batch.risk_masks[..., 0])
    assert batch.risk_masks.any()
    from rles_world_model.routeb.risk_targets import estimate_train_barrier_fractions
    score_frames=[frame for frame in frames
                  if fold.train.score_start_ns<=frame.ts_close_ns<fold.train.score_end_ns]
    expected_barriers=estimate_train_barrier_fractions(
        np.asarray([frame.close for frame in score_frames]),
        train_timestamps_ns=np.asarray([frame.ts_close_ns for frame in score_frames]),
        horizons=batch.risk_horizons)
    np.testing.assert_array_equal(batch.risk_barrier_fractions,expected_barriers)
    np.testing.assert_array_equal(
        np.asarray(metadata["risk_barrier_fractions"]),expected_barriers)
    for horizon_index, horizon in enumerate(batch.risk_horizons):
        valid = batch.risk_masks[..., horizon_index]
        assert np.all(batch.timestamps[valid] >= fold.train.score_start_ns)
        assert np.all(batch.timestamps[valid] + horizon * MINUTE_NS
                      < fold.train.score_end_ns)


def test_fold_train_artifact_requires_exact_authorized_replay_coverage(tmp_path):
    dataset_hash = "c" * 64
    fold = FoldManifest(
        "0.4.0", dataset_hash, "d" * 64,
        Window(MINUTE_NS, 20 * MINUTE_NS, 90 * MINUTE_NS),
        Window(90 * MINUTE_NS, 100 * MINUTE_NS, 140 * MINUTE_NS),
        Window(140 * MINUTE_NS, 150 * MINUTE_NS, 160 * MINUTE_NS),
        0, 0, None,
    )
    frames = []
    for index in range(1, 139):
        price = 100.0 + index * 0.01
        frames.append(MarketFrame(index * MINUTE_NS, index * MINUTE_NS,
                                  "ETH-PERP", price, price + 0.1,
                                  price - 0.1, price, 1.0))
    replay = MarketReplay(tuple(frames), dataset_hash)
    with pytest.raises(ValueError, match="cover Train warmup through Validation end"):
        prepare_fold_train_sequences(
            fold_replay=replay, fold=fold, train_scope_id="train-fold-002",
            target_npz=tmp_path / "incomplete.npz", length=16,
            burn_in=4, stride=8,
        )

    complete_frames = tuple(frames) + (
        MarketFrame(139 * MINUTE_NS, 141 * MINUTE_NS, "ETH-PERP",
                    101.39, 101.49, 101.29, 101.39, 1.0),
    )
    late_replay = MarketReplay(complete_frames, dataset_hash)
    with pytest.raises(ValueError, match="cover Train warmup through Validation end"):
        prepare_fold_train_sequences(
            fold_replay=late_replay, fold=fold, train_scope_id="train-fold-003",
            target_npz=tmp_path / "late-availability.npz", length=16,
            burn_in=4, stride=8,
        )


def test_scaler_owns_readonly_arrays_and_binding():
    bars, _, scaler = prepared()
    ref = scaler.fingerprint
    with pytest.raises(ValueError):
        scaler.mean[0] = 999.
    assert scaler.fingerprint == ref
    alt = TrainOnlyStandardizer(scaler.mean + 1., scaler.std, scaler.dataset_id, 'train')
    assert alt.fingerprint != ref
    with pytest.raises(ValueError):
        TrainOnlyStandardizer(np.zeros(5), np.ones(5), 'x', 'validation')


def test_train_npz_rejects_tamper_and_wrong_scaler(tmp_path):
    _, seq, scaler = prepared()
    b = make_windows(seq, scaler=scaler, length=32, burn_in=8)
    path = tmp_path / 'train.npz'
    export_train_sequences(path, b, scaler, snapshot_sha=SNAPSHOT,
                           upstream_sha=UPSTREAM, train_scope_id='train-reserved')
    restored, meta = load_worker_batch(path, expected_scope='train-reserved',
                                       snapshot_sha=SNAPSHOT, upstream_sha=UPSTREAM)
    assert np.array_equal(restored.masks, b.masks)
    assert meta['scaler_sha256'] == scaler.fingerprint
    data = json.loads(path.with_suffix('.json').read_text())
    data['scaler_sha256'] = '0' * 64
    path.with_suffix('.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='scaler'):
        load_worker_batch(path, expected_scope='train-reserved', snapshot_sha=SNAPSHOT,
                          upstream_sha=UPSTREAM)


def test_total_path_returns_count_gaps_with_real_anchor():
    # Two bars: open jumps 100 -> 110, closes 121, next opens 132, closes 145.2.
    p = np.array([[[[110, 122, 109, 121, 10],
                    [132, 146, 131, 145.2, 11]]]], dtype=float)
    r = return_paths(p, previous_close=100)
    assert r.shape == (1, 1, 2)
    assert r[0,0,0] == pytest.approx(math.log(121/100))
    assert r[0,0,1] == pytest.approx(math.log(145.2/121))
    assert r.sum() == pytest.approx(math.log(145.2/100))


def test_return_paths_rejects_invalid_prices_or_anchor():
    p = np.ones((2, 4, 3, 5))
    with pytest.raises(ValueError):
        return_paths(p, previous_close=np.array([100, -1]))
    p[0,0,1,3] = np.nan
    with pytest.raises(ValueError):
        return_paths(p,previous_close=100.)


def test_multi_horizon_requires_nonempty_anchors():
    with pytest.raises(ValueError):
        multi_horizon_scores(np.empty((0, 3, 5)), np.empty((0,5)))


def test_codec_requires_aligned_start_time():
    with pytest.raises(ValueError, match='aligned'):
        OHLCVCodec.decode(np.zeros((3, 5)), previous_close=100,
                          start_close_ns=MINUTE_NS + 1, instrument='ETH')


def test_forecast_context_binds_scaler_and_timestamps(tmp_path):
    bars, _, scaler = prepared()
    request = prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-registered', train_npz_sha256='a'*64,
        output_context_npz=tmp_path/'history.npz',
        output_request_json=tmp_path/'request.json', horizon=5, n_paths=16, seed=31)
    assert request['schema']=='routeb.forecast-request.v2'
    assert request['scaler_sha256']==scaler.fingerprint
    with np.load(tmp_path/'history.npz', allow_pickle=False) as arrays:
        assert arrays['close_timestamps'][-1] == bars[-1].ts_close_ns
        assert np.all(arrays['available_timestamps']<=bars[-1].ts_close_ns)
    with pytest.raises(ValueError):
        prepare_forecast_context(
            visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
            train_scope_id='train-registered', train_npz_sha256='a'*64,
            output_context_npz=tmp_path/'bad.npz', output_request_json=tmp_path/'bad.json',
            horizon=61, n_paths=16, seed=31)


def test_optimization_recipe_can_extend_steps_not_change_seed():
    model = RSSMConfig()
    old = TrainingConfig(steps=2, report_interval=1)
    new = TrainingConfig(steps=100, report_interval=17)
    assert optimizer_recipe(model, old) == optimizer_recipe(model, new)
    assert optimizer_recipe(model, old) != optimizer_recipe(model, replace(new, seed=7))


def test_account_rejects_repeated_bars_and_backdated_order():
    acc = ImaginedAccount(initial_equity=1000, fee_maker=.0002, fee_taker=.0004)
    bb = candles(3, start=10)
    acc.on_bar(bb[0])
    with pytest.raises(ValueError, match='increasing'):
        acc.on_bar(bb[0])
    with pytest.raises(ValueError, match='historical'):
        acc.submit(target_qty=1, decision_ns=bb[0].ts_close_ns-MINUTE_NS)


def test_limit_marketable_open_is_charged_taker_fee():
    acc = ImaginedAccount(initial_equity=1000, fee_maker=.0001, fee_taker=.001)
    bb = candles(4)
    acc.on_bar(bb[0])
    # Limit buy above next opening price: marketable at open, not maker.
    acc.submit(target_qty=1, kind='limit', limit_price=bb[1].open*1.1,
               decision_ns=bb[0].ts_close_ns)
    fill = acc.on_bar(bb[1])
    assert fill is not None and not fill.maker
    assert fill.price == pytest.approx(bb[1].open)
    assert fill.fee == pytest.approx(fill.price*.001)


def test_limit_touch_after_open_is_maker_scenario():
    acc = ImaginedAccount(initial_equity=1000, fee_maker=.0001, fee_taker=.001)
    bb = candles(4)
    acc.on_bar(bb[0])
    # Price must fall through limit after open for a nominal maker fill.
    limit = (bb[1].open + bb[1].low) / 2
    acc.submit(target_qty=1,kind='limit',limit_price=limit,
               decision_ns=bb[0].ts_close_ns)
    fill=acc.on_bar(bb[1])
    assert fill is not None and fill.maker
    assert fill.price == pytest.approx(limit)
    assert fill.fee == pytest.approx(limit*.0001)


def test_malformed_sequence_reset_masks_rejected():
    _, seq, scaler = prepared()
    b=make_windows(seq,scaler=scaler,length=32,burn_in=8)
    bad_mask=b.masks.copy()
    bad_mask[:,-1] = False
    with pytest.raises(ValueError, match='last target'):
        SequenceBatch(b.features, b.reset, bad_mask, b.timestamps)


def test_full_hierarchy_train_tensors_have_causal_indices():
    from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
    # 2 full UTC days + 1 extra hour, synthetic Train only.
    raw = candles(2940)
    seq = MarketSequence(raw, 'hierarchy-train', 'train')
    sample, fitted = build_train_hierarchy(seq, burn_in=64)
    assert sample.minute_x.shape == (1, 2939, 5)
    assert sample.hour_x.shape == (1, 49, 5)
    assert sample.day_x.shape == (1, 2, 5)
    assert sample.day_index[0,1437] == -1
    assert sample.day_index[0,1438] == 0
    assert sample.day_index[0,2878] == 1
    assert sample.minute_mask[0,63] == 0
    assert sample.minute_mask[0,64] == 1
    assert sample.dataset_id == seq.dataset_id
    assert sample.scaler_fingerprints == (
        fitted.minute.fingerprint, fitted.hour.fingerprint, fitted.day.fingerprint)
    for indices, availability in ((sample.hour_index[0],sample.hour_available_at),
                                  (sample.day_index[0],sample.day_available_at)):
        active=indices>=0
        assert np.all(availability[indices[active]]<=sample.minute_available_at[active])
    same, other = build_train_hierarchy(seq, burn_in=64, frozen_scalers=fitted)
    assert np.array_equal(same.hour_x,sample.hour_x)
    assert same.scaler_fingerprints == sample.scaler_fingerprints
    assert other.day.fingerprint == fitted.day.fingerprint


def test_full_hierarchy_rejects_insufficient_daily_context():
    from rles_world_model.routeb.hierarchy_dataset import build_train_hierarchy
    seq = MarketSequence(candles(1400), 'short', 'train')
    with pytest.raises(ValueError, match='completed'):
        build_train_hierarchy(seq, burn_in=32)


def test_f2_refuses_incomplete_day_event_without_missing_history_silence():
    from rles_world_model.routeb.synthetic_feedback import SyntheticFeedback
    # Observed prefix does not contain enough of current UTC day.
    pre = candles(70,start=1370)
    sim = SyntheticFeedback(pre, anchor_ns=1439*MINUTE_NS)
    before = len(sim.simulated)
    with pytest.raises(ValueError, match='1440m bucket'):
        sim.advance(candles(1,start=1440)[0])
    assert len(sim.simulated)==before


def test_checkpoint_detects_corruption_and_tree_permutation(tmp_path):
    jax = pytest.importorskip('jax')
    from rles_world_model.routeb.checkpoint import save_checkpoint, load_checkpoint
    params = {'a':jax.numpy.array([1.0],dtype=jax.numpy.float32),
              'b':jax.numpy.array([2.0],dtype=jax.numpy.float32)}
    opt = (jax.numpy.array(3,dtype=jax.numpy.int32),)
    recipe = {'model':'small'}
    identity = {'train_scope_id':'train-mechanism','dataset_id':'ds'}
    dest = tmp_path/'ckpt.npz'
    save_checkpoint(dest,params=params,opt_state=opt,step=3,seed=11,
                    identity=identity,config=recipe)
    invalid_template = {'a':jax.numpy.array([1.0],dtype=jax.numpy.float32),
                        'c':jax.numpy.array([2.0],dtype=jax.numpy.float32)}
    with pytest.raises(ValueError,match='pytree'):
        load_checkpoint(dest,params_template=invalid_template,opt_template=opt,
                        expected_identity=identity,expected_config=recipe)
    dest.write_bytes(dest.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='corruption'):
        load_checkpoint(dest,params_template=params,opt_template=opt,
                        expected_identity=identity,expected_config=recipe)


def test_worker_rejects_tampered_context_before_forward(tmp_path, monkeypatch, capsys):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    bars, _, scaler = prepared()
    _request = prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-reserved', train_npz_sha256='b'*64,
        output_context_npz=tmp_path/'context.npz',
        output_request_json=tmp_path/'request.json', horizon=2, n_paths=2, seed=1)
    config = {'model':asdict(RSSMConfig()),'training':asdict(TrainingConfig())}
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'context.npz').write_bytes((tmp_path/'context.npz').read_bytes()+b'tamper')
    monkeypatch.setattr(cli,'forecast_standardized',lambda **kwargs: (_ for _ in ()).throw(
        AssertionError('must reject before computing a forecast')))
    status=cli.main(['forecast','--request',str(tmp_path/'request.json'),
                     '--checkout',str(tmp_path/'unused'),
                     '--checkpoint',str(tmp_path/'unused.npz'),
                     '--output',str(tmp_path/'out.npz'),
                     '--configuration',str(tmp_path/'config.json')])
    assert status==2
    err=json.loads(capsys.readouterr().err)
    assert err['status']=='BLOCKED' and 'checksum' in err['detail']
    assert not (tmp_path/'out.npz').exists()


@pytest.mark.parametrize(('field','value','message'),[
    ('instrument',' ','instrument'),
    ('instrument','ETHUSDT-PERP.SIM ','instrument'),
    ('anchor_close',float('nan'),'anchor close'),
    ('anchor_close',False,'anchor close'),
    ('anchor_close',0.0,'anchor close'),
])
def test_worker_rejects_invalid_anchor_or_instrument_before_forward(
        tmp_path, monkeypatch, capsys, field, value, message):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict

    bars, _, scaler = prepared()
    prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-boundary', train_npz_sha256='f'*64,
        output_context_npz=tmp_path/'context.npz',
        output_request_json=tmp_path/'request.json', horizon=2, n_paths=2, seed=17)
    request_path=tmp_path/'request.json'
    request=json.loads(request_path.read_text())
    request[field]=value
    request_path.write_text(json.dumps(request))
    (tmp_path/'config.json').write_text(json.dumps({
        'model':asdict(RSSMConfig()),'training':asdict(TrainingConfig())}))
    monkeypatch.setattr(cli,'forecast_standardized',lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError('invalid request must fail before model inference')))

    status=cli.main(['forecast','--request',str(request_path),
                     '--checkout',str(tmp_path/'unused'),
                     '--checkpoint',str(tmp_path/'unused.npz'),
                     '--output',str(tmp_path/'out.npz'),
                     '--configuration',str(tmp_path/'config.json')])
    assert status==2
    error=json.loads(capsys.readouterr().err)
    assert error['status']=='BLOCKED' and message in error['detail']
    assert not (tmp_path/'out.npz').exists()


@pytest.mark.parametrize(('source','duplicate'),[
    ('configuration','{"model":{},"model":{},"training":{}}'),
    ('request','{"schema":"bad","schema":"routeb.forecast-request.v2"}'),
])
def test_worker_cli_rejects_duplicate_json_keys_before_inference(
        tmp_path, monkeypatch, capsys, source, duplicate):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    bars, _, scaler = prepared()
    request_path=tmp_path/'request.json'
    if source=='request':
        prepare_forecast_context(
            visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
            train_scope_id='train-duplicate-json', train_npz_sha256='e'*64,
            output_context_npz=tmp_path/'context.npz', output_request_json=request_path,
            horizon=2, n_paths=2, seed=13)
        request_path.write_text(duplicate,encoding='utf-8')
        config={'model':asdict(RSSMConfig()),'training':asdict(TrainingConfig())}
        (tmp_path/'config.json').write_text(json.dumps(config),encoding='utf-8')
    else:
        request_path.write_text('{}',encoding='utf-8')
        (tmp_path/'config.json').write_text(duplicate,encoding='utf-8')
    monkeypatch.setattr(cli,'forecast_standardized',lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError('ambiguous JSON must fail before model inference')))
    status=cli.main(['forecast','--request',str(request_path),
        '--checkout',str(tmp_path/'unused'),'--checkpoint',str(tmp_path/'unused.npz'),
        '--output',str(tmp_path/'out.npz'),'--configuration',str(tmp_path/'config.json')])
    assert status==2
    error=json.loads(capsys.readouterr().err)
    assert error['status']=='BLOCKED' and 'duplicate JSON object key' in error['detail']
    assert not (tmp_path/'out.npz').exists()


def test_worker_cli_mocked_end_to_end_schema_and_geometry(tmp_path, monkeypatch, capsys):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    bars, _, scaler = prepared()
    _request = prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-allowed', train_npz_sha256='a'*64,
        output_context_npz=tmp_path/'context.npz',
        output_request_json=tmp_path/'request.json', horizon=3, n_paths=2, seed=31)
    (tmp_path/'config.json').write_text(json.dumps({
        'model':asdict(RSSMConfig()),'training':asdict(TrainingConfig())}))
    monkeypatch.setattr(cli,'forecast_standardized',lambda **kwargs: np.zeros((2,3,5),dtype=np.float32))
    assert cli.main(['forecast','--request',str(tmp_path/'request.json'),
                     '--checkout',str(tmp_path/'unused'),
                     '--checkpoint',str(tmp_path/'unused.npz'),
                     '--output',str(tmp_path/'out.npz'),
                     '--configuration',str(tmp_path/'config.json')])==0
    response=json.loads(capsys.readouterr().out)
    assert response['status']=='COMPLETED' and response['paths']==2
    with np.load(tmp_path/'out.npz',allow_pickle=False) as data:
        assert data['bars'].shape==(2,3,5)
        assert np.all(data['bars'][...,0:4]>0)


def test_e1_worker_cli_emits_common_f1_ohlcv_schema(tmp_path, monkeypatch, capsys):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    from rles_world_model.routeb.jax_gru import GRUConfig
    bars, _, scaler = prepared()
    prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-e1-allowed', train_npz_sha256='d'*64,
        output_context_npz=tmp_path/'e1-context.npz',
        output_request_json=tmp_path/'e1-request.json', horizon=3, n_paths=2, seed=43)
    (tmp_path/'e1-config.json').write_text(json.dumps({
        'model':asdict(GRUConfig(hidden=8)),
        'training':asdict(TrainingConfig(steps=2,batch_size=2))}))
    monkeypatch.setattr(cli,'forecast_gru',lambda **kwargs: np.zeros((2,3,5),dtype=np.float32))
    assert cli.main(['e1-forecast','--request',str(tmp_path/'e1-request.json'),
                     '--checkout',str(tmp_path/'unused'),
                     '--checkpoint',str(tmp_path/'e1-checkpoint.npz'),
                     '--output',str(tmp_path/'e1-out.npz'),
                     '--configuration',str(tmp_path/'e1-config.json')])==0
    response=json.loads(capsys.readouterr().out)
    assert response['status']=='COMPLETED' and response['kind']=='E1_PRIOR_PATHS'
    with np.load(tmp_path/'e1-out.npz',allow_pickle=False) as data:
        assert data['bars'].shape==(2,3,5)
        assert np.isfinite(data['bars']).all()
        assert np.all(data['bars'][...,1]>=np.maximum(data['bars'][...,0],data['bars'][...,3]))
        assert np.all(data['bars'][...,2]<=np.minimum(data['bars'][...,0],data['bars'][...,3]))


@pytest.mark.parametrize(('command','candidate','kind'),[
    ('forecast-batch','e2','F1_BATCH_PRIOR_PATHS'),
    ('e1-forecast-batch','e1','E1_BATCH_PRIOR_PATHS'),
])
def test_worker_cli_batched_forecasts_preserve_anchor_axis(
        tmp_path, monkeypatch, capsys, command, candidate, kind):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    from rles_world_model.routeb.jax_gru import GRUConfig
    from rles_world_model.routeb.market import MINUTE_NS

    bars, _, scaler = prepared()
    single_request = prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-batch-allowed', train_npz_sha256='b'*64,
        output_context_npz=tmp_path/'single.npz',
        output_request_json=tmp_path/'single.json', horizon=3, n_paths=2, seed=41)
    with np.load(tmp_path/'single.npz',allow_pickle=False) as source:
        history=source['history']
    anchors=bars[-1].ts_close_ns+np.arange(6,dtype=np.int64)*MINUTE_NS
    closes=anchors[:,None]-np.arange(len(history)-1,-1,-1,dtype=np.int64)[None,:]*MINUTE_NS
    decisions=anchors+1_000_000_000
    request={**single_request,'schema':'routeb.forecast-batch-request.v1',
             'anchor_timestamps_ns':anchors.tolist(),
             'decision_timestamps_ns':decisions.tolist()}
    request.pop('anchor_ts_ns')
    request.pop('anchor_close')
    request.pop('first_future_close_ns')
    context=tmp_path/'batch.npz'
    np.savez_compressed(context,histories=np.stack([history]*len(anchors)),
        scaler_mean=scaler.mean,scaler_std=scaler.std,anchor_timestamps_ns=anchors,
        decision_timestamps_ns=decisions,close_timestamps=closes,
        available_timestamps=closes+1_000_000_000,
        anchor_closes=np.asarray([bars[-1].close]*len(anchors)))
    request['history_npz']=str(context.resolve())
    request['context_sha256']=hashlib.sha256(context.read_bytes()).hexdigest()
    (tmp_path/'request-batch.json').write_text(json.dumps(request))
    model=GRUConfig(hidden=8) if candidate=='e1' else RSSMConfig()
    (tmp_path/'config-batch.json').write_text(json.dumps({
        'model':asdict(model),'training':asdict(TrainingConfig(steps=2,batch_size=2))}))
    forecast=cli.forecast_gru if candidate=='e1' else cli.forecast_standardized
    chunk_seeds=[]
    def fake_forecast(**kwargs):
        chunk_seeds.append(kwargs['forecast_seed'])
        return np.zeros((len(kwargs['history']),3,5),dtype=np.float32)
    monkeypatch.setattr(cli,forecast.__name__,fake_forecast)
    assert cli.main([command,'--request',str(tmp_path/'request-batch.json'),
        '--checkout',str(tmp_path/'unused'),'--checkpoint',str(tmp_path/'checkpoint.npz'),
        '--output',str(tmp_path/'batch-out.npz'),
        '--configuration',str(tmp_path/'config-batch.json')])==0
    response=json.loads(capsys.readouterr().out)
    assert response['status']=='COMPLETED' and response['kind']==kind
    assert response['sampled_paths_per_anchor']==8 and response['rejected_paths']==0
    assert response['rejected_paths_by_anchor']==[0]*6
    assert response['anchors']==6 and response['paths']==2
    assert response['anchor_chunk_size']==4
    assert response['chunk_seed_rule']=='request_seed_plus_chunk_start_index'
    assert chunk_seeds==[41,45]
    with np.load(tmp_path/'batch-out.npz',allow_pickle=False) as output:
        assert output['bars'].shape==(6,2,3,5)
        assert np.isfinite(output['bars']).all()
        assert np.all(output['bars'][...,1]>=np.maximum(output['bars'][...,0],output['bars'][...,3]))
        assert np.all(output['bars'][...,2]<=np.minimum(output['bars'][...,0],output['bars'][...,3]))


def test_worker_cli_returns_provenance_bound_next_distribution(tmp_path, monkeypatch, capsys):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    bars, _, scaler = prepared()
    prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-allowed', train_npz_sha256='c'*64,
        output_context_npz=tmp_path/'context.npz',
        output_request_json=tmp_path/'request.json', horizon=1, n_paths=2, seed=37)
    (tmp_path/'config.json').write_text(json.dumps({
        'model':asdict(RSSMConfig(forecast_distribution='student_t')),'training':asdict(TrainingConfig())}))
    monkeypatch.setattr(cli, 'forecast_distribution_standardized',
                        lambda **kwargs: (np.zeros((1,5),dtype=np.float32),
                                          np.ones((1,5),dtype=np.float32)))
    assert cli.main(['next-distribution','--request',str(tmp_path/'request.json'),
                     '--checkout',str(tmp_path/'unused'),
                     '--checkpoint',str(tmp_path/'unused.npz'),
                     '--output',str(tmp_path/'distribution.npz'),
                     '--configuration',str(tmp_path/'config.json')])==0
    response=json.loads(capsys.readouterr().out)
    assert response['kind']=='E2_NEXT_DISTRIBUTION'
    assert response['distribution_family']=='student_t'
    assert response['student_t_df']==5.0
    assert response['scale_semantics']=='marginal_standard_deviation'
    with np.load(tmp_path/'distribution.npz',allow_pickle=False) as output:
        assert set(output.files)=={'standardized_mean','standardized_scale'}
        assert output['standardized_mean'].shape==(1,5)
        assert np.all(output['standardized_scale']>0)



def test_worker_cli_returns_mixture_parameters_with_marginal_moments(tmp_path, monkeypatch, capsys):
    import rles_world_model.routeb.worker_cli as cli
    from dataclasses import asdict
    bars, _, scaler = prepared()
    prepare_forecast_context(
        visible_history=bars, frozen_scaler=scaler, anchor_ns=bars[-1].ts_close_ns,
        train_scope_id='train-allowed', train_npz_sha256='c'*64,
        output_context_npz=tmp_path/'context.npz',
        output_request_json=tmp_path/'request.json', horizon=1, n_paths=2, seed=37)
    (tmp_path/'config.json').write_text(json.dumps({
        'model':asdict(RSSMConfig(forecast_distribution='mixture_gaussian')),
        'training':asdict(TrainingConfig())}))
    means=np.asarray([[[0.0]*5,[2.0]*5]],dtype=np.float32)
    scales=np.ones((1,2,5),dtype=np.float32)
    weights=np.asarray([[0.5,0.5]],dtype=np.float32)
    monkeypatch.setattr(cli,'forecast_distribution_standardized',lambda **kwargs:{
        'mean':np.ones((1,5),dtype=np.float32),
        'scale':np.full((1,5),np.sqrt(2.0),dtype=np.float32),
        'mixture_means':means,'mixture_scales':scales,'mixture_weights':weights})
    assert cli.main(['next-distribution','--request',str(tmp_path/'request.json'),
                     '--checkout',str(tmp_path/'unused'),
                     '--checkpoint',str(tmp_path/'unused.npz'),
                     '--output',str(tmp_path/'mixture.npz'),
                     '--configuration',str(tmp_path/'config.json')])==0
    response=json.loads(capsys.readouterr().out)
    assert response['distribution_family']=='mixture_gaussian'
    assert response['mixture_components']==2
    with np.load(tmp_path/'mixture.npz',allow_pickle=False) as artifact:
        assert set(artifact.files)=={'standardized_mean','standardized_scale',
            'mixture_means','mixture_scales','mixture_weights'}


def test_worker_client_requires_genuine_artifact(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import rles_world_model.routeb.worker_client as client
    settings=client.WorkerConfig(python_executable=Path(sys.executable),
                                 upstream_checkout=tmp_path)
    c=client.WorkerClient(settings)
    mock = {'status':'COMPLETED','kind':'F1_PRIOR_PATHS',
            'output':str(tmp_path/'unwritten.npz'), 'paths':2,'horizon':3,
            'sha256':'0'*64}
    monkeypatch.setattr(client.subprocess,'run',lambda *args,**kwargs:
                        SimpleNamespace(returncode=0,stdout=json.dumps(mock),stderr=''))
    with pytest.raises(client.WorkerFailure,match='failed to produce'):
        c.forecast(request=tmp_path/'request.json',checkpoint=tmp_path/'checkpoint.npz',
                   output=tmp_path/'unwritten.npz',configuration=tmp_path/'config.json')


def test_worker_client_rejects_duplicate_keys_in_worker_response(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import rles_world_model.routeb.worker_client as client
    worker=client.WorkerClient(client.WorkerConfig(
        python_executable=Path(sys.executable),upstream_checkout=tmp_path))
    stdout='{"status":"COMPLETED","kind":"F1_PRIOR_PATHS","kind":"E1_PRIOR_PATHS"}'
    monkeypatch.setattr(client.subprocess,'run',lambda *args,**kwargs:
        SimpleNamespace(returncode=0,stdout=stdout,stderr=''))
    with pytest.raises(client.WorkerFailure,match='duplicate JSON object key: kind'):
        worker.forecast(request=tmp_path/'request.json',checkpoint=tmp_path/'checkpoint.npz',
                        output=tmp_path/'out.npz',configuration=tmp_path/'config.json')


def test_worker_client_accepts_only_valid_three_dimensional_forecast(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import rles_world_model.routeb.worker_client as client
    settings=client.WorkerConfig(python_executable=Path(sys.executable),
                                 upstream_checkout=tmp_path)
    worker=client.WorkerClient(settings)
    bars=np.zeros((2,3,5),dtype=np.float64)
    bars[...,0]=100.0
    bars[...,1]=102.0
    bars[...,2]=99.0
    bars[...,3]=101.0
    bars[...,4]=2.0
    artifact=bars

    def fake_run(command, **kwargs):
        output=Path(command[command.index('--output')+1])
        np.savez_compressed(output,bars=artifact)
        result={'status':'COMPLETED','kind':'F1_PRIOR_PATHS','output':str(output),
                'paths':2,'horizon':3,
                'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')

    monkeypatch.setattr(client.subprocess,'run',fake_run)
    result=worker.forecast(request=tmp_path/'request.json',
        checkpoint=tmp_path/'checkpoint.npz',output=tmp_path/'forecast.npz',
        configuration=tmp_path/'configuration.json')
    assert result['kind']=='F1_PRIOR_PATHS'

    artifact=bars[:,None,:,:]
    with pytest.raises(client.WorkerFailure,match='shape/geometry'):
        worker.forecast(request=tmp_path/'request.json',
            checkpoint=tmp_path/'checkpoint.npz',output=tmp_path/'invalid-forecast.npz',
            configuration=tmp_path/'configuration.json')


def test_worker_client_validates_batched_forecast_anchor_and_path_axes(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import rles_world_model.routeb.worker_client as client
    worker=client.WorkerClient(client.WorkerConfig(
        python_executable=Path(sys.executable),upstream_checkout=tmp_path))
    bars=np.zeros((2,3,4,5),dtype=np.float64)
    bars[...,0]=100.0
    bars[...,1]=102.0
    bars[...,2]=99.0
    bars[...,3]=101.0
    bars[...,4]=2.0
    def fake_run(command, **kwargs):
        output=Path(command[command.index('--output')+1])
        np.savez_compressed(output,bars=bars)
        result={'status':'COMPLETED','kind':'F1_BATCH_PRIOR_PATHS','output':str(output),
                'anchors':2,'paths':3,'horizon':4,'sampled_paths_per_anchor':12,
                'anchor_chunk_size':4,
                'chunk_seed_rule':'request_seed_plus_chunk_start_index',
                'rejected_paths':0,'rejected_paths_by_anchor':[0,0],
                'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')
    monkeypatch.setattr(client.subprocess,'run',fake_run)
    result=worker.forecast_batch(request=tmp_path/'request.json',
        checkpoint=tmp_path/'checkpoint.npz',output=tmp_path/'forecast-batch.npz',
        configuration=tmp_path/'configuration.json',candidate='e2')
    assert result['kind']=='F1_BATCH_PRIOR_PATHS'

    bars=bars[:,None,:,:,:]
    with pytest.raises(client.WorkerFailure,match='shape/geometry'):
        worker.forecast_batch(request=tmp_path/'request.json',
            checkpoint=tmp_path/'checkpoint.npz',output=tmp_path/'invalid-batch.npz',
            configuration=tmp_path/'configuration.json',candidate='E2')


def test_worker_client_validates_next_distribution_artifact(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import rles_world_model.routeb.worker_client as client
    settings=client.WorkerConfig(python_executable=Path(sys.executable),upstream_checkout=tmp_path)
    worker=client.WorkerClient(settings)
    def fake_run(command, **kwargs):
        output=Path(command[command.index('--output')+1])
        np.savez_compressed(output,standardized_mean=np.zeros((1,5)),
                            standardized_scale=np.ones((1,5)))
        result={'status':'COMPLETED','kind':'E2_NEXT_DISTRIBUTION','output':str(output),
                'paths':1,'horizon':1,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                'distribution_family':'gaussian','student_t_df':None,'mixture_components':None,
                'scale_semantics':'marginal_standard_deviation'}
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')
    monkeypatch.setattr(client.subprocess,'run',fake_run)
    result=worker.next_distribution(request=tmp_path/'request.json',
        checkpoint=tmp_path/'checkpoint.npz',output=tmp_path/'distribution.npz',
        configuration=tmp_path/'config.json')
    assert result['kind']=='E2_NEXT_DISTRIBUTION'



def test_worker_client_validates_mixture_marginal_moments(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import rles_world_model.routeb.worker_client as client
    worker=client.WorkerClient(client.WorkerConfig(
        python_executable=Path(sys.executable),upstream_checkout=tmp_path))
    def fake_run(command, **kwargs):
        output=Path(command[command.index('--output')+1])
        means=np.asarray([[[0.0]*5,[2.0]*5]],dtype=np.float32)
        scales=np.ones((1,2,5),dtype=np.float32)
        weights=np.asarray([[0.5,0.5]],dtype=np.float32)
        np.savez_compressed(output,standardized_mean=np.ones((1,5),dtype=np.float32),
            standardized_scale=np.full((1,5),np.sqrt(2.0),dtype=np.float32),
            mixture_means=means,mixture_scales=scales,mixture_weights=weights)
        result={'status':'COMPLETED','kind':'E2_NEXT_DISTRIBUTION','output':str(output),
            'paths':1,'horizon':1,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'distribution_family':'mixture_gaussian','student_t_df':None,
            'mixture_components':2,'scale_semantics':'marginal_standard_deviation'}
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')
    monkeypatch.setattr(client.subprocess,'run',fake_run)
    result=worker.next_distribution(request=tmp_path/'request.json',
        checkpoint=tmp_path/'checkpoint.npz',output=tmp_path/'mixture.npz',
        configuration=tmp_path/'config.json')
    assert result['distribution_family']=='mixture_gaussian'


def test_rssm_config_rejects_nans_and_malformed_architecture():
    with pytest.raises(ValueError):
        RSSMConfig(min_scale=float('nan'))
    with pytest.raises(ValueError):
        RSSMConfig(deter=31,blocks=8)
    with pytest.raises(ValueError):
        RSSMConfig(classes=True)
    assert RSSMConfig(risk_horizon_weights=[0.5, 0.5, 1.0, 1.5, 1.5]).risk_horizon_weights == (
        0.5, 0.5, 1.0, 1.5, 1.5,
    )
    with pytest.raises(ValueError):
        RSSMConfig(risk_horizon_weights=(0.5, 0.5))
    with pytest.raises(ValueError):
        RSSMConfig(risk_horizon_weights=(1.0, 1.0, 1.0, 1.0, float('inf')))
    assert RSSMConfig(forecast_distribution="student_t", forecast_student_t_df=5.0).forecast_distribution == "student_t"
    assert RSSMConfig(forecast_distribution="mixture_gaussian").forecast_distribution == "mixture_gaussian"
    assert RSSMConfig(rollout_risk_conditioning=True).rollout_risk_conditioning is True
    assert RSSMConfig(rollout_risk_conditioning=True,
                      rollout_risk_sampling=True).rollout_risk_sampling is True
    assert RSSMConfig(
        rollout_risk_conditioning=True, rollout_risk_sampling=True,
        rollout_risk_noise_persistence=0.65,
    ).rollout_risk_noise_persistence == 0.65
    assert RSSMConfig(rollout_risk_conditioning_strength=0.5).rollout_risk_conditioning_strength == 0.5
    for invalid_strength in (True, -0.1, 1.1, float("inf"), "0.5"):
        with pytest.raises(ValueError):
            RSSMConfig(rollout_risk_conditioning_strength=invalid_strength)
    for invalid_persistence in (True, -0.01, 1.0, float("nan"), "0.65"):
        with pytest.raises(ValueError):
            RSSMConfig(rollout_risk_noise_persistence=invalid_persistence)
    with pytest.raises(ValueError):
        RSSMConfig(rollout_risk_conditioning=1)
    with pytest.raises(ValueError):
        RSSMConfig(rollout_risk_sampling=True)
    with pytest.raises(ValueError):
        RSSMConfig(rollout_risk_noise_persistence=0.65)
    with pytest.raises(ValueError):
        RSSMConfig(rollout_risk_conditioning=True,
                   rollout_risk_noise_persistence=0.65)
    with pytest.raises(ValueError):
        RSSMConfig(rollout_risk_conditioning=True, rollout_risk_sampling=1)
    assert RSSMConfig(forecast_distribution="student_t", rollout_risk_conditioning=True,
                      rollout_risk_conditioning_strength=0.5).rollout_risk_conditioning
    with pytest.raises(ValueError):
        RSSMConfig(forecast_distribution="mixture_gaussian", rollout_risk_conditioning=True)
    assert RSSMConfig(forecast_objective="crps").forecast_objective == "crps"
    assert RSSMConfig(forecast_objective="open_loop_crps",
                      open_loop_crps_horizon=15).open_loop_crps_horizon == 15
    path_config=RSSMConfig(forecast_objective="path_log_rv_crps",
                           path_log_rv_crps_horizon=15,
                           path_log_rv_crps_paths=8)
    assert path_config.path_log_rv_crps_horizon==15
    assert path_config.path_log_rv_crps_paths==8
    assert path_config.path_log_rv_crps_scale==1.0
    multi_path_config=RSSMConfig(
        forecast_objective="path_log_rv_crps",
        path_log_rv_crps_horizon=15,
        path_log_rv_crps_horizons=(15,30,60),
        path_log_rv_crps_paths=4)
    assert multi_path_config.path_log_rv_crps_horizons==(15,30,60)
    high_sample_config=RSSMConfig(
        forecast_objective="path_log_rv_crps",
        path_log_rv_crps_horizon=15,
        path_log_rv_crps_horizons=(15,30,60),
        path_log_rv_crps_paths=32)
    assert high_sample_config.path_log_rv_crps_paths==32
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="path_log_rv_crps",
                   path_log_rv_crps_horizons=(15,30,60),
                   path_log_rv_crps_paths=33)
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="path_log_rv_crps",
                   path_log_rv_crps_horizons=(15,60,30))
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="path_log_rv_crps",
                   path_log_rv_crps_horizons=(15,45))
    with pytest.raises(ValueError):
        RSSMConfig(forecast_distribution="student_t", forecast_objective="crps")
    with pytest.raises(ValueError):
        RSSMConfig(forecast_distribution="mixture_gaussian",
                   forecast_objective="open_loop_crps")
    with pytest.raises(ValueError):
        RSSMConfig(forecast_distribution="student_t",
                   forecast_objective="path_log_rv_crps")
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="path_log_rv_crps",
                   path_log_rv_crps_horizon=3)
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="path_log_rv_crps",
                   path_log_rv_crps_paths=1)
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="open_loop_crps", open_loop_crps_horizon=61)
    with pytest.raises(ValueError):
        RSSMConfig(forecast_objective="unknown")
    with pytest.raises(ValueError):
        RSSMConfig(forecast_distribution="unknown")
    with pytest.raises(ValueError):
        RSSMConfig(forecast_distribution="student_t", forecast_student_t_df=2.0)


def test_risk_conditioned_return_scale_matches_native_log_rv_in_raw_units():
    jnp = pytest.importorskip("jax.numpy")
    from rles_world_model.routeb.jax_model import _risk_conditioned_return_scale

    scale = jnp.asarray([[2.0, 3.0, 4.0, 5.0, 6.0]], dtype=jnp.float32)
    risk_mean = jnp.zeros((1, 5, 4), dtype=jnp.float32)
    risk_mean = risk_mean.at[:, 0, 1].set(jnp.log(0.004))
    risk_scale = jnp.zeros_like(risk_mean)
    feature_std = jnp.asarray([0.001, 0.002, 0.3, 0.4, 1.0], dtype=jnp.float32)

    adjusted = _risk_conditioned_return_scale(
        scale, risk_mean, risk_scale, feature_std,
    )
    tempered = _risk_conditioned_return_scale(
        scale, risk_mean, risk_scale, feature_std, strength=0.5,
    )
    unconditioned = _risk_conditioned_return_scale(
        scale, risk_mean, risk_scale, feature_std, strength=0.0,
    )
    student_t_adjusted = _risk_conditioned_return_scale(
        scale, risk_mean, risk_scale, feature_std, student_t_df=5.0,
    )
    sampled_risk_scale = risk_scale.at[:, 0, 1].set(0.5)
    sampled_low = _risk_conditioned_return_scale(
        scale, risk_mean, sampled_risk_scale, feature_std,
        risk_noise=jnp.asarray([-1.0], dtype=jnp.float32),
    )
    sampled_median = _risk_conditioned_return_scale(
        scale, risk_mean, sampled_risk_scale, feature_std,
        risk_noise=jnp.zeros((1,), dtype=jnp.float32),
    )
    sampled_high = _risk_conditioned_return_scale(
        scale, risk_mean, sampled_risk_scale, feature_std,
        risk_noise=jnp.asarray([1.0], dtype=jnp.float32),
    )
    sampled_disabled = _risk_conditioned_return_scale(
        scale, risk_mean, sampled_risk_scale, feature_std, strength=0.0,
        risk_noise=jnp.asarray([1.0], dtype=jnp.float32),
    )
    generated_return_std = jnp.sqrt(
        jnp.square(adjusted[0, 0] * feature_std[0])
        + jnp.square(adjusted[0, 1] * feature_std[1]),
    )
    implied_expected_rv = generated_return_std * jnp.sqrt(2.0 / jnp.pi)
    np.testing.assert_allclose(implied_expected_rv, 0.004, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(tempered[0, :2], jnp.sqrt(adjusted[0, :2] * scale[0, :2]),
                               rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(unconditioned, scale)
    assert sampled_low[0, 0] < sampled_median[0, 0] < sampled_high[0, 0]
    assert sampled_low[0, 1] < sampled_median[0, 1] < sampled_high[0, 1]
    np.testing.assert_array_equal(sampled_disabled, scale)
    np.testing.assert_array_equal(adjusted[0, 2:], scale[0, 2:])
    t_abs_ratio = (3.0**0.5 * math.gamma(2.0)
                   / (math.pi**0.5 * math.gamma(2.5)))
    t_return_std = jnp.sqrt(
        jnp.square(student_t_adjusted[0, 0] * feature_std[0])
        + jnp.square(student_t_adjusted[0, 1] * feature_std[1]),
    )
    np.testing.assert_allclose(t_return_std * t_abs_ratio, 0.004, rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match="risk noise"):
        _risk_conditioned_return_scale(
            scale, risk_mean, sampled_risk_scale, feature_std,
            risk_noise=jnp.zeros((2,), dtype=jnp.float32),
        )


def test_risk_noise_ar1_transform_preserves_stationary_variance_and_shape():
    jnp = pytest.importorskip("jax.numpy")
    from rles_world_model.routeb.jax_model import _correlate_risk_noise

    previous = jnp.asarray([1.0, -1.0, 0.5], dtype=jnp.float32)
    innovation = jnp.asarray([0.0, 1.0, -1.0], dtype=jnp.float32)
    rho = 0.65
    expected = rho * previous + math.sqrt(1.0 - rho**2) * innovation
    actual = _correlate_risk_noise(previous, innovation, rho)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(_correlate_risk_noise(None, innovation, rho), innovation)
    np.testing.assert_array_equal(_correlate_risk_noise(previous, innovation, 0.0), innovation)
    rng = np.random.default_rng(650)
    stationary_previous = rng.normal(size=65_536).astype(np.float32)
    independent_innovation = rng.normal(size=65_536).astype(np.float32)
    stationary_actual = np.asarray(_correlate_risk_noise(
        jnp.asarray(stationary_previous), jnp.asarray(independent_innovation), rho,
    ))
    assert abs(float(stationary_actual.var()) - 1.0) < 0.02
    assert abs(float(np.corrcoef(stationary_previous, stationary_actual)[0, 1]) - rho) < 0.02
    with pytest.raises(ValueError, match="shapes"):
        _correlate_risk_noise(previous, jnp.zeros((2,), dtype=jnp.float32), rho)


def test_optimizer_recipe_normalizes_tuple_fields_for_json_checkpoint_roundtrip():
    model = RSSMConfig(
        risk_horizon_weights=(0.5, 0.5, 1.0, 1.5, 1.5),
        path_log_rv_crps_horizons=(15, 30, 60),
        rollout_risk_conditioning=True, rollout_risk_sampling=True,
        rollout_risk_noise_persistence=0.65,
        forecast_distribution="student_t", forecast_student_t_df=5.0,
    )
    recipe = optimizer_recipe(model, TrainingConfig())

    assert recipe['model']['risk_horizon_weights'] == [0.5, 0.5, 1.0, 1.5, 1.5]
    assert recipe['model']['path_log_rv_crps_horizons'] == [15, 30, 60]
    assert recipe['model']['forecast_distribution'] == 'student_t'
    assert recipe['model']['forecast_student_t_df'] == 5.0
    assert recipe['model']['rollout_risk_noise_persistence'] == 0.65
    assert json.loads(json.dumps(recipe)) == recipe


def test_invalid_ohlc_path_rejected_even_with_positive_closes():
    x=np.ones((1,2,3,5))
    x[...,1]=0.5 # high is below open and close
    with pytest.raises(ValueError):
        return_paths(x,previous_close=100.)
