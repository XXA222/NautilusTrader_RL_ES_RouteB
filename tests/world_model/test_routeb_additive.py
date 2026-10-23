"""Local deterministic tests; no NautilusTrader, DreamerV3 or JAX required."""
import json
import math
import numpy as np
import pytest
from rles_world_model.routeb.market import Candle,MarketSequence,MINUTE_NS
from rles_world_model.routeb.clock import MultiScaleClock
from rles_world_model.routeb.codec import OHLCVCodec,TrainOnlyStandardizer
from rles_world_model.routeb.sequences import (make_windows,export_train_sequences,
                                                load_worker_batch)
from rles_world_model.routeb.hierarchy import build_plan,causal_state_lookup
from rles_world_model.routeb.evaluation import ensemble_crps,multi_horizon_scores
from rles_world_model.routeb.synthetic_feedback import SyntheticFeedback
from rles_world_model.routeb.imagined_account import ImaginedAccount

SHA='8'*40
UP='e01491fad6434b2245a3b8ca201dd7faedcc458c'


def bars(n,start=1,*,vol=3.,delay=0):
    out=[];prev=100.
    for i in range(start,start+n):
        op=prev*(1+0.0001)
        cl=op*(1+math.sin(i/37)*0.001)
        out.append(Candle(i*MINUTE_NS,i*MINUTE_NS+delay,"ETHUSDT-PERP.SIM",
                          op,max(op,cl)*1.001,min(op,cl)*0.999,cl,vol))
        prev=cl
    return tuple(out)


def test_codec_roundtrip_and_geometry():
    x=bars(50)
    encoded=OHLCVCodec.encode(x,previous_close=100.)
    rebuilt=OHLCVCodec.decode(encoded,previous_close=100.,start_close_ns=MINUTE_NS,
                               instrument="ETHUSDT-PERP.SIM")
    for a,b in zip(x,rebuilt):
        assert b.high>=max(b.open,b.close)>=b.low
        assert b.low<=min(b.open,b.close)
        assert b.close==pytest.approx(a.close,rel=1e-7)
        assert b.high==pytest.approx(a.high,rel=1e-7)
        assert b.low==pytest.approx(a.low,rel=1e-7)
    with pytest.raises(ValueError):
        OHLCVCodec.decode(np.array([[30,0,0,0,0]]),previous_close=100.,
                          start_close_ns=MINUTE_NS,instrument="ETH")


def test_train_only_scaling_and_boundary():
    records=bars(300)
    series=MarketSequence(records,"dataset-demo","train")
    train=OHLCVCodec.encode(records[1:],previous_close=records[0].close)
    with pytest.raises(ValueError):
        TrainOnlyStandardizer.fit(train,dataset_id="dataset-demo",scope="validation")
    scaler=TrainOnlyStandardizer.fit(train,dataset_id="dataset-demo",scope="train")
    windows=make_windows(series,scaler=scaler,length=64,burn_in=8,stride=16)
    assert windows.features.shape[1:]==(64,5)
    assert not windows.masks[:,0].any()
    assert windows.masks[:,-1].all() # final target must contribute to NLL
    assert np.all(windows.reset[:,0])
    with pytest.raises(ValueError):
        make_windows(MarketSequence(records,"dataset-demo","validation"),
                     scaler=scaler,length=64)


def test_npz_authorized_scope_roundtrip(tmp_path):
    records=bars(200)
    seq=MarketSequence(records,"ds","train")
    scaler=TrainOnlyStandardizer.fit(OHLCVCodec.encode(records[1:],previous_close=records[0].close),
                                    dataset_id="ds",scope="train")
    windows=make_windows(seq,scaler=scaler,length=32,burn_in=8)
    artifact=tmp_path/'train.npz'
    export_train_sequences(artifact,windows,scaler,snapshot_sha=SHA,
                           upstream_sha=UP,train_scope_id='reserved-train-1')
    restored,meta=load_worker_batch(artifact,expected_scope='reserved-train-1',
                                    snapshot_sha=SHA,upstream_sha=UP)
    assert np.array_equal(windows.features,restored.features)
    assert meta['purpose']=='train'
    with pytest.raises(ValueError):
        load_worker_batch(artifact,expected_scope='validation',
                          snapshot_sha=SHA,upstream_sha=UP)


def test_hour_day_close_availability_and_future_mutation():
    history=bars(1500)
    clock=MultiScaleClock(history)
    assert len(clock.buckets(60,as_of_ns=59*MINUTE_NS))==0
    assert len(clock.buckets(60,as_of_ns=60*MINUTE_NS))==1
    assert len(clock.buckets(1440,as_of_ns=1439*MINUTE_NS))==0
    assert len(clock.buckets(1440,as_of_ns=1440*MINUTE_NS))==1
    plan=build_plan(history)
    assert plan.hour_last_visible[58]==-1
    assert plan.hour_last_visible[59]==0
    assert plan.day_last_visible[1438]==-1
    assert plan.day_last_visible[1439]==0
    future=list(history)
    b=future[-1]
    future[-1]=Candle(b.ts_close_ns,b.available_at_ns,b.instrument,b.open,
                       max(b.high,100000.),b.low,b.close,b.volume)
    assert MultiScaleClock(future).latest(60,1440*MINUTE_NS)==clock.latest(60,1440*MINUTE_NS)
    delayed=list(history)
    last=delayed[59]
    delayed[59]=Candle(last.ts_close_ns,last.ts_close_ns+MINUTE_NS,
                       last.instrument,last.open,last.high,last.low,last.close,last.volume)
    assert MultiScaleClock(delayed).latest(60,60*MINUTE_NS) is None


def test_slow_latent_lookup_prohibits_future():
    states=np.arange(15,dtype=np.float32).reshape(5,3)
    out=causal_state_lookup(states,np.array([-1,0,0,2,4]))
    assert np.all(out[0]==0)
    assert np.all(out[3]==states[2])
    with pytest.raises(ValueError):
        causal_state_lookup(states,np.array([5]))


def test_synthetic_cross_hour_uses_only_simulated_future():
    history=bars(59)
    synth=SyntheticFeedback(history,anchor_ns=59*MINUTE_NS)
    new=bars(1,start=60)[0]
    completed=synth.advance(new)
    assert len(completed)==1
    assert completed[0].timeframe_minutes==60
    assert completed[0].close_ns==60*MINUTE_NS
    assert completed[0].high==max(x.high for x in history+(new,))
    with pytest.raises(ValueError):
        synth.advance(new)


def test_crps_and_forecast_coverage():
    vals=np.array([[0,0],[0,2]],dtype=np.float32)
    assert ensemble_crps(vals,np.array([2,1]))==pytest.approx(1.25)
    p=np.zeros((3,64,10),dtype=float)
    y=np.zeros((3,10))
    report=multi_horizon_scores(p,y,horizons=(1,5,10),tail_threshold=-0.02)
    assert len(report)==3 and all(s.crps==0 and s.coverage_80==1 for s in report)


def test_imagined_limit_market_cancellation_and_equity():
    account=ImaginedAccount(initial_equity=1000,fee_maker=0,fee_taker=.001)
    history=bars(5)
    account.submit(target_qty=1,decision_ns=1*MINUTE_NS)
    assert account.on_bar(history[0]) is None
    fill=account.on_bar(history[1])
    assert fill is not None and not fill.maker
    assert account.qty==1
    assert account.equity==pytest.approx(1000-fill.fee + (history[1].close-fill.price))
    account.submit(target_qty=0,decision_ns=2*MINUTE_NS,kind="limit",
                   limit_price=history[2].high*1.1)
    assert account.on_bar(history[2]) is None
    assert account.cancel()
    assert account.pending is None
    assert account.economics_eligible is False


def test_checkpoint_roundtrip_full_optimizer_tree(tmp_path):
    jax=pytest.importorskip("jax")
    from rles_world_model.routeb.checkpoint import save_checkpoint,load_checkpoint
    p={"linear": {"w":jax.numpy.array([[1.,2.],[3.,4.]],dtype=jax.numpy.float32)},
       "bias":jax.numpy.array([.2,.7],dtype=jax.numpy.float32)}
    o={"count":jax.numpy.array(13,dtype=jax.numpy.int32),
       "moment":jax.tree.map(lambda x:jax.numpy.zeros_like(x),p)}
    identity={"train_scope_id":"real-registered-train","dataset_id":"ds","snapshot_sha":SHA}
    cfg={"width":8,"lr":.01}
    path=tmp_path/'state.npz'
    save_checkpoint(path,params=p,opt_state=o,step=47,seed=31,identity=identity,config=cfg)
    weights,optim,step,seed=load_checkpoint(path,params_template=p,opt_template=o,
                                            expected_identity=identity,expected_config=cfg)
    assert step==47 and seed==31
    assert np.array_equal(weights['linear']['w'],p['linear']['w'])
    assert int(optim['count'])==13
    with pytest.raises(ValueError,match='identity'):
        load_checkpoint(path,params_template=p,opt_template=o,
                        expected_identity={**identity,'train_scope_id':'not-authorized'},
                        expected_config=cfg)
    path.with_suffix('.json').write_text(
        '{"schema":"routeb.jax-checkpoint.v2","schema":"other"}',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='duplicate JSON key'):
        load_checkpoint(path, params_template=p, opt_template=o,
                        expected_identity=identity, expected_config=cfg)


def test_bundle_restore_rejects_ambiguous_manifest(tmp_path):
    from rles_data.core import DataError
    from rles_world_model.bundle import load_bundle

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text(
        '{"schema_version":"rles.world_model.bundle.v1",'
        '"schema_version":"other"}', encoding='utf-8',
    )
    with pytest.raises(DataError, match='bundle manifest or state archive is unreadable'):
        load_bundle(bundle, None, expected_feature_spec_hash='0' * 64,
                    expected_source_code_sha256='1' * 64)


def test_delayed_hourly_bucket_visibility_no_future():
    xs=list(bars(120))
    x=xs[59]
    # Data can arrive one minute later but must remain monotonic.
    xs[59]=Candle(x.ts_close_ns,x.ts_close_ns+MINUTE_NS,x.instrument,
                  x.open,x.high,x.low,x.close,x.volume)
    clock=MultiScaleClock(xs)
    assert clock.latest(60,60*MINUTE_NS) is None
    assert clock.latest(60,61*MINUTE_NS).close_ns==60*MINUTE_NS
    schedule=build_plan(tuple(xs))
    assert schedule.hour_last_visible[58]==-1
    assert schedule.hour_last_visible[59]==0  # decision at its delayed availability


def test_synthetic_day_close_event():
    prefix=bars(1439)
    sim=SyntheticFeedback(prefix,anchor_ns=1439*MINUTE_NS)
    events=sim.advance(bars(1,start=1440)[0])
    assert {x.timeframe_minutes for x in events}=={60,1440}


def test_forecast_request_from_visible_only(tmp_path):
    from rles_world_model.routeb.integration import prepare_forecast_context
    history=bars(12)
    x=OHLCVCodec.encode(history[1:],previous_close=history[0].close)
    scaler=TrainOnlyStandardizer.fit(x,dataset_id='ds',scope='train')
    req=prepare_forecast_context(visible_history=history,frozen_scaler=scaler,
        anchor_ns=history[-1].ts_close_ns,train_scope_id='train-authorized-demo',train_npz_sha256='f'*64,
        output_context_npz=tmp_path/'context.npz',output_request_json=tmp_path/'request.json',
        horizon=5,n_paths=16,seed=31)
    assert req['first_future_close_ns']==history[-1].ts_close_ns+MINUTE_NS
    assert json.loads((tmp_path/'request.json').read_text())['anchor_ts_ns']==history[-1].ts_close_ns
    with pytest.raises(ValueError,match='unavailable'):
        late=list(history)
        b=late[-1]
        late[-1]=Candle(b.ts_close_ns,b.ts_close_ns+MINUTE_NS,
                        b.instrument,b.open,b.high,b.low,b.close,b.volume)
        prepare_forecast_context(visible_history=tuple(late),frozen_scaler=scaler,
            anchor_ns=b.ts_close_ns,train_scope_id='train-demo',train_npz_sha256='f'*64,
            output_context_npz=tmp_path/'bad.npz',output_request_json=tmp_path/'bad.json',
            horizon=5,n_paths=16,seed=31)
