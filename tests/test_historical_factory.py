Warning: truncated output (original token count: 24647)
Total output lines: 1946

"""Synthetic contract tests for production assembly; never real-data claims."""
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS
from rles_data.features import make_feature_row
from rles_data.io import write_curated
from rles_data.contracts import to_market_event
from rles_gym_bridge.historical import (HistoricalEnvironmentFactory, _money, _visible_batches_by_decision,
                                        _CompactReportSnapshot, _CompactReportSnapshotSeries,
                                        _SpooledReportSnapshotSequence,
                                        historical_feature_spec, partition_decision_lanes,
                                        scheduled_funding_time_features)
from rles_gym_bridge.episode import BarrierFrame, EpisodeReplay
from rles_nautilus_adapter.session import VisibleBatch
from rles_contracts.dto import DataQualityGrade
from rles_risk import SoftRiskParams

pytestmark = pytest.mark.micro_golden


@pytest.mark.parametrize("value", ["1,2 USDT", "1,234.50 USDT", "12 USDC", "NaN", "1 USDT extra"])
def test_native_account_money_rejects_ambiguous_or_wrong_currency(value):
    with pytest.raises(DataError):
        _money(value, "USDT")


@pytest.mark.parametrize(("value", "expected"), [
    ("1234.50000000 USDT", Decimal("1234.50000000")),
    ("1234.50000000", Decimal("1234.50000000")),
])
def test_native_account_money_accepts_strict_decimal_wire_values(value, expected):
    assert _money(value, "USDT") == expected


def test_checked_in_ethusdt_real_config_has_valid_soft_risk_domain():
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/ethusdt_20240101_real.json").read_text(encoding="utf-8"))
    SoftRiskParams.from_search_dict(config["environment"]["soft_risk"])


def test_actor_account_scale_is_causal_and_legacy_checkpoint_defaults_stay_raw(tmp_path):
    from rles_orchestrator.application import _training_actor_spec
    from rles_orchestrator.workflow import Route

    factory = HistoricalEnvironmentFactory(**recipe(tmp_path))
    config = {"output": str(tmp_path / "run")}
    before = _training_actor_spec(factory, (2,), config, Route.A1)
    future = Event("future-mark", "mark", 25 * MINUTE_NS, 25 * MINUTE_NS, 25 * MINUTE_NS,
                   Availability.NATIVE, "b" * 64, {"instrument": factory.instrument, "price": "1000000"},
                   "synthetic:future-perturbation")
    factory.store = CuratedStore((*factory.store.events, future))
    after = _training_actor_spec(factory, (2,), config, Route.A1)
    assert after.signature == before.signature
    manifest = Path(config["output"]) / "run.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema": "rles.historical-run.v1", "run_id": "legacy", "attempt_id": "legacy",
                                    "recipe_hash": "a" * 64, "run_fingerprint": "b" * 64}))
    legacy = _training_actor_spec(factory, (2,), {**config, "resume_checkpoint": "legacy-boundary"}, Route.A1)
    assert legacy.version == 1 and legacy.input_scale is None


def test_historical_frame_batch_partition_is_ordered_complete_and_linear_windowed():
    batches = ((10, "pre-window"), (20, "first"), (30, "boundary"), (40, "later"))
    frames = _visible_batches_by_decision(batches, (25, 30, 40))

    assert frames == (("pre-window", "first"), ("boundary",), ("later",))
    assert tuple(batch for frame in frames for batch in frame) == tuple(batch for _, batch in batches)


def test_factory_lazy_episode_frames_match_windowed_visible_event_grouping(tmp_path, monkeypatch):
    import rles_gym_bridge.historical as historical_module

    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=30))
    # CuratedStore owns the validated events; keeping a second Fold-wide tuple
    # of the same OHLCV Event references serves no consumer.
    assert not hasattr(factory, "_bars")
    warmup_minutes = max(
        field.lookback_bars * field.timeframe_minutes
        for field in factory.features.fields if field.source_kind == "ohlcv_1m"
    ) + 2
    original_features = historical_module.visible_market_features
    feature_calls = []

    def counted_features(*args, **kwargs):
        feature_calls.append(kwargs["decision_ts_ns"])
        return original_features(*args, **kwargs)

    monkeypatch.setattr(historical_module, "visible_market_features", counted_features)
    for start_offset in (0, 7):
        feature_calls.clear()
        episode_rows = factory.rows[start_offset:start_offset + 8]
        episode_feed_start = max(
            0, episode_rows[0].decision_ts_ns - warmup_minutes * MINUTE_NS,
        )
        frames = factory._episode_barrier_frames(
            start_offset=start_offset, episode_rows=episode_rows,
            feed_start_ns=episode_feed_start,
        )
        assert not isinstance(frames, tuple)
        assert len(feature_calls) == 0
        for index, row in enumerate(episode_rows):
            start_available_ns = (episode_feed_start if index == 0
                                  else episode_rows[index - 1].decision_ts_ns + 1)
            expected_groups = {}
            for event in factory.store.available_event_window(
                    start_available_ns, row.decision_ts_ns):
                if (event.kind in {"instrument_rule", "funding_update"}
                        or event.payload.get("instrument") != factory.instrument
                        or event.available_at_ns is None):
                    continue
                expected_groups.setdefault(event.available_at_ns, []).append(
                    to_market_event(event, quality_grade=DataQualityGrade(
                        factory.quality.grade.value)))
            expected_batches = tuple(
                VisibleBatch(timestamp, tuple(events), True)
                for timestamp, events in sorted(expected_groups.items())
            )
            frame = frames[index]
            assert frame.batches == expected_batches
            assert frame.market == original_features(
                row, factory.features, factory.schema, instrument=factory.instrument,
                decision_ts_ns=row.decision_ts_ns,
            )
            assert frame.time_features == {}
            assert feature_calls[-1] == row.decision_ts_ns
        assert frames[0] == frames[0]  # repeated reads are deterministic, not retained in a Fold cache
        assert feature_calls[-1] == episode_rows[0].decision_ts_ns
        assert not hasattr(factory, "_barrier_frames_by_row")


def test_cached_episode_frames_preserve_native_replay_boundary_and_ledger(tmp_path, monkeypatch):
    from collections import defaultdict

    from rles_research.native_metrics import native_report
    from rles_gym_bridge.l2_bridge import visible_market_features

    config = recipe(tmp_path, bar_count=24)
    cached_factory = HistoricalEnvironmentFactory(**config)
    reference_factory = HistoricalEnvironmentFactory(**config)

    def uncached_frames(*, start_offset, episode_rows, feed_start_ns):
        groups = defaultdict(list)
        for event in reference_factory.store.available_event_window(
                feed_start_ns, episode_rows[-1].decision_ts_ns):
            if event.kind in {"instrument_rule", "funding_update"}:
                continue
            if event.payload.get("instrument") != reference_factory.instrument:
                continue
            if event.available_at_ns is None:
                continue
            groups[event.available_at_ns].append(to_market_event(
                event, quality_grade=DataQualityGrade(reference_factory.quality.grade.value),
            ))
        batches = tuple((timestamp, VisibleBatch(timestamp, tuple(events), True))
                        for timestamp, events in sorted(groups.items()))
        batch_groups = _visible_batches_by_decision(
            batches, tuple(row.decision_ts_ns for row in episode_rows),
        )
        frames = []
        for row, visible_batches in zip(episode_rows, batch_groups, strict=True):
            frames.append(BarrierFrame(
                row.decision_ts_ns, visible_batches,
                visible_market_features(row, reference_factory.features,
                                        reference_factory.schema,
                                        instrument=reference_factory.instrument,
                                        decision_ts_ns=row.decision_ts_ns),
                {},
            ))
        return tuple(frames)

    monkeypatch.setattr(reference_factory, "_episode_barrier_frames", uncached_frames)
    reference_env = reference_factory.make_env()
    cached_resume_env = None
    try:
        reference_env.reset(seed=317)
        prefix = []
        for action in (0.5, -0.5):
            prefix.append(reference_env.step(np.asarray([action], dtype=np.float32)))
        replay = EpisodeReplay.from_dict(reference_env.rles_export_replay())
        continuation = [reference_env.step(np.asarray([action], dtype=np.float32))
                        for action in (0.25, 0.0, 0.5)]
        reference_report = native_report(
            reference_factory.snapshots,
            fill_references=reference_factory.fill_references,
            sample_interval_ns=MINUTE_NS,
        )
        reference_env.close()

        cached_resume_env = cached_factory.make_env(replay=replay)
        resumed_observation, _info = cached_resume_env.reset(seed=317)
        assert np.array_equal(resumed_observation, prefix[-1][0])
        assert cached_resume_env.rles_boundary_token == replay.boundary_token
        for action, expected in zip((0.25, 0.0, 0.5), continuation, strict=True):
            actual = cached_resume_env.step(np.asarray([action], dtype=np.float32))
            assert np.array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
        cached_report = native_report(
            cached_factory.snapshots,
            fill_references=cached_factory.fill_references,
            sample_interval_ns=MINUTE_NS,
        )
        assert cached_report["metrics"] == reference_report["metrics"]
        for field in ("total_fee", "total_slippage_cost", "total_funding",
                      "net_return", "max_drawdown", "turnover_ratio"):
            assert cached_report["metrics"][field] == reference_report["metrics"][field]
        economic_fill_fields = (
            "instrument_id", "order_side", "last_qty", "last_px", "commission", "ts_event",
        )
        assert [tuple(fill.get(key) for key in economic_fill_fields)
                for fill in cached_report["fills"]] == [
                    tuple(fill.get(key) for key in economic_fill_fields)
                    for fill in reference_report["fills"]
                ]
        assert cached_factory.snapshots[-1].equity == reference_factory.snapshots[-1].equity
        assert cached_factory.snapshots[-1].positions == reference_factory.snapshots[-1].positions
    finally:
        reference_env.close()
        if cached_resume_env is not None:
            cached_resume_env.close()


def test_training_environment_can_skip_diagnostic_snapshot_retention_without_changing_rollout(tmp_path):
    config = recipe(tmp_path, bar_count=24)
    reference_factory = HistoricalEnvironmentFactory(**config)
    lean_factory = HistoricalEnvironmentFactory(**config)
    actions = (0.5, -0.25, 0.0, 0.75)

    reference = reference_factory.make_env()
    try:
        reference_observation, reference_info = reference.reset(seed=71)
        reference_steps = [reference.step(np.asarray([action], dtype=np.float32))
                           for action in actions]
    finally:
        reference.close()
    assert reference_factory.snapshots

    lean = lean_factory.make_env(retain_report_snapshots=False)
    try:
        lean_observation, lean_info = lean.reset(seed=71)
        lean_steps = [lean.step(np.asarray([action], dtype=np.float32)) for action in actions]
    finally:
        lean.close()

    assert np.array_equal(reference_observation, lean_observation)
    assert reference_info == lean_info
    for expected, actual in zip(reference_steps, lean_steps, strict=True):
        assert np.array_equal(expected[0], actual[0])
        assert expected[1:] == actual[1:]
    assert lean_factory.snapshots == []
    assert lean_factory.fill_references == {}


def test_stress_execution_identity_changes_only_the_mode_from_frozen_base(tmp_path):
    environment = recipe(tmp_path)
    base = HistoricalEnvironmentFactory(**environment)
    stress = HistoricalEnvironmentFactory(**{
        **environment, "fit_scaler": False, "scaler": base.scaler, "mode": "stress",
    })

    assert stress.mode == "stress"
    assert stress.config_hash != base.config_hash
    assert stress.base_config_hash == base.config_hash
    assert stress.execution.fingerprint() == base.execution.fingerprint()
    assert stress.scaler.hash == base.scaler.hash


def test_nondefault_reward_scale_is_bound_to_historical_training_identity(tmp_path):
    environment = recipe(tmp_path)
    raw = HistoricalEnvironmentFactory(**environment)
    legacy_default = HistoricalEnvironmentFactory(**environment, reward_scale=1.0)
    scaled = HistoricalEnvironmentFactory(**environment, reward_scale=10_000.0)

    assert raw.config_hash == legacy_default.config_hash
    assert raw.transition_hash == legacy_default.transition_hash
    assert scaled.config_hash != raw.config_hash
    assert scaled.transition_hash != raw.transition_hash
    assert scaled.metadata()["reward_config"]["reward_scale"] == 10_000.0
    assert scaled.make_env()._reward.config.reward_scale == 10_000.0


def test_nondefault_drawdown_reward_shaping_is_bound_to_historical_identity(tmp_path):
    environment = recipe(tmp_path)
    raw = HistoricalEnvironmentFactory(**environment)
    default_explicit = HistoricalEnvironmentFactory(**environment, reward_lambda_dd=0.0)
    shaped = HistoricalEnvironmentFactory(**environment, reward_lambda_dd=0.1)

    assert raw.config_hash == default_explicit.config_hash
    assert raw.transition_hash == default_explicit.transition_hash
    assert shaped.config_hash != raw.config_hash
    assert shaped.transition_hash != raw.transition_hash
    assert shaped.metadata()["reward_config"]["lambda_dd"] == 0.1
    assert shaped.metadata()["reward_config"]["lambda_guard"] == 0.0
    env = shaped.make_env()
    try:
        assert env._reward.config.lambda_dd == 0.1
        assert env._reward.config.lambda_guard == 0.0
    finally:
        env.close()


def test_rank_partitioned_training_episodes_start_in_distinct_time_lanes(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=30))
    full_steps = len(factory.rows) - 3
    episode_steps = 4
    first = factory.make_env(episode_rank=0, episode_count=2, episode_steps=episode_steps)
    try:
        _first_obs, first_info = first.reset(seed=11)
    finally:
        first.close()
    last = factory.make_env(episode_rank=1, episode_count=2, episode_steps=episode_steps)
    try:
        _last_obs, last_info = last.reset(seed=22)
        assert first_info["episode_window"]["decision_steps"] == episode_steps
        assert last_info["episode_window"]["decision_steps"] == episode_steps
        assert first_info["episode_window"]["start_ns"] < last_info["episode_window"]["start_ns"]
        assert last_info["episode_window"]["start_ns"] == factory.rows[
            full_steps // 2
        ].decision_ts_ns
        assert first_info["episode_window"]["episode_index"] == 0
        assert last_info["episode_window"]["episode_index"] == 0
    finally:
        last.close()


def test_rank_decision_lanes_partition_every_step_with_balanced_remainders():
    lanes = partition_decision_lanes(129_597, 90)
    sizes = [end - start for start, end in lanes]

    assert lanes[0][0] == 0
    assert lanes[-1][1] == 129_597
    assert all(previous[1] == following[0]
               for previous, following in zip(lanes, lanes[1:]))
    assert sum(sizes) == 129_597
    assert sizes.count(1_440) == 87
    assert sizes.count(1_439) == 3


def test_rank_partition_rejects_duplicate_or_oversized_segments(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=12))
    full_steps = len(factory.rows) - 3
    with pytest.raises(DataError, match="rank-exclusive time lane"):
        factory.make_env(episode_rank=0, episode_count=2, episode_steps=full_steps)
    with pytest.raises(DataError, match="invalid rotating historical episode partition"):
        factory.make_env(episode_steps=full_steps + 1)


def test_historical_episode_windows_rotate_reproducibly_within_rank_lane(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=30))
    episode_steps = 4
    env = factory.make_env(episode_rank=0, episode_count=2, episode_steps=episode_steps)
    try:
        _, first = env.reset(seed=11, options={"rles_episode_index": 0})
        _, second = env.reset(seed=12, options={"rles_episode_index": 1})
    finally:
        env.close()
    duplicate = factory.make_env(episode_rank=0, episode_count=2, episode_steps=episode_steps)
    try:
        _, repeated = duplicate.reset(seed=99, options={"rles_episode_index": 1})
    finally:
        duplicate.close()
    assert first["episode_window"]["start_offset"] != second["episode_window"]["start_offset"]
    assert second["episode_window"] == repeated["episode_window"]
    assert 0 <= second["episode_window"]["start_offset"] < (len(factory.rows) - 3) // 2


def test_historical_factory_accepts_v2_profile_and_binds_its_schema(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=80, start_minute=62),
                                           feature_profile="causal_mtf_v2")

    assert factory.feature_profile == "causal_mtf_v2"
    assert "market.trend_efficiency_15m" in factory.schema.names
    assert "market.trend_efficiency_60m" in factory.schema.names


def test_historical_factory_v3_binds_schedule_time_inputs_separately(tmp_path):
    environment = recipe(tmp_path, bar_count=80, start_minute=62)
    symbol = environment["instrument"]
    environment["coverage_policy"]["known_schedule_times_ns"] = {symbol: (90 * MINUTE_NS,)}
    factory = HistoricalEnvironmentFactory(**environment, feature_profile="causal_mtf_v3")
    v2 = HistoricalEnvironmentFactory(**environment, feature_profile="causal_mtf_v2")

    assert "market.trend_efficiency_60m" in factory.schema.names
    assert "time.funding_settlement_within_24h" in factory.schema.names
    assert "time.funding_settlement_time_to_event_24h" in factory.schema.names
    assert factory.schema.hash != v2.schema.hash
    assert "time.funding_settlement_within_24h" not in v2.schema.names


def test_funding_schedule_countdown_is_bounded_and_uses_strict_future_schedule():
    schedule = (10 * MINUTE_NS, 40 * MINUTE_NS)

    assert scheduled_funding_time_features(9 * MINUTE_NS, schedule) == {
        "funding_settlement_within_24h": 1.0,
        "funding_settlement_time_to_event_24h": pytest.approx(1 / 1440),
    }
    assert scheduled_funding_time_features(10 * MINUTE_NS, schedule)[
        "funding_settlement_time_to_event_24h"
    ] == pytest.approx(30 / 1440)
    assert scheduled_funding_time_features(20 * MINUTE_NS, schedule)[
        "funding_settlement_time_to_event_24h"
    ] == pytest.approx(20 / 1440)
    assert scheduled_funding_time_features(40 * MINUTE_NS, schedule) == {
        "funding_settlement_within_24h": 0.0,
        "funding_settlement_time_to_event_24h": 1.0,
    }


def recipe(tmp_path, *, price_jump=False, bar_count=20, start_minute=6,
           base_currency="ETH", quote_currency="USDT", contract_multiplier="1"):
    symbol = f"{base_currency}{quote_currency}-PERP.SIM"
    quantity_unit = base_currency
    rule = {"instrument": symbol, "quantity_unit": quantity_unit,
            "base_currency": base_currency, "quote_currency": quote_currency,
            "contract_multiplier": contract_multiplier,
            "tick_size": "0.01", "step_size": "0.001", "min_qty": "0.001",
            "min_notional": "1", "maker_fee": "0.001", "taker_fee": "0.001", "initial_margin_rate": "1",
            "maintenance_margin_rate": "0.35", "effective_ns": MINUTE_NS,
            "valid_until_ns": (bar_count + 1) * MINUTE_NS,
            "rule_version": "SYNTHETIC-TEST-ONLY"}
    source = tmp_path / "synthetic-rule.json"
    source.write_text(json.dumps(rule), encoding="utf-8")
    fingerprint = sha256(source.read_bytes()).hexdigest()
    rule_event = Event("rule", "instrument_rule", MINUTE_NS, MINUTE_NS, MINUTE_NS, Availability.NATIVE,
                       fingerprint, rule, "synthetic:factory-test-rule")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"sources": [{"path": str(source), "sha256": fingerprint, "provenance": "synthetic:test"}],
                                  "events": [asdict(rule_event)]}), encoding="utf-8")
    events = []
    for i in range(1, bar_count + 1):
        timestamp = i * MINUTE_NS
        price = "1040" if price_jump and i >= 8 else "1000"
        for kind, payload in {
            "ohlcv_1m": {"open": price, "high": price, "low": price, "close": price, "volume": "1000"},
            "mark": {"price": price}, "index": {"price": price},
        }.items():
            events.append(Event(f"{kind}-{i}", kind, timestamp, timestamp, timestamp, Availability.NATIVE,
                                "a" * 64, {"instrument": symbol, **payload}, f"synthetic:{kind}"))
    for minute in range(1, bar_count + 1, 20):
        timestamp = minute * MINUTE_NS
        events.append(Event(f"rate-{minute}", "funding_update", timestamp, timestamp, timestamp,
                            Availability.NATIVE, "a" * 64,
                            {"instrument": symbol, "rate": "0.001"}, "synthetic:funding_update"))
    market = tmp_path / "market.parquet"
    write_curated(market, CuratedStore(events))
    latency = {"model_id": "base", "decision_ns": 1, "submit_ns": 1, "venue_ns": 1}
    execution = {"schema_version": "0.4.0", "deadband": "0.01", "pending_order_policy": "cancel_replace",
                 "order_type": "MARKET", "tif": "GTC", "terminal_policy": "force_flatten_with_cost",
                 "base_latency": latency, "stress_latency": {**latency, "model_id": "stress", "venue_ns": 2},
                 "base_fill": {"model_id": "base", "slippage_bps": "1", "data_granularity": "bar"},
                 "stress_fill": {"model_id": "stress", "slippage_bps": "2", "data_granularity": "bar"}}
    return dict(market_path=market, rules_path=rules, instrument=symbol,
                start_ns=start_minute * MINUTE_NS, end_ns=(bar_count - 1) * MINUTE_NS,
                coverage_policy={"maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS, "funding_update": 30 * MINUTE_NS},
                                 "settlement_times_ns": {symbol: ()}, "provenance": "synthetic:factory-test"},
                execution=execution, starting_balance="10000", risk_budget_notional="1000",
                hard_risk={"max_notional": "1000", "max_leverage": "1", "margin_buffer_fraction": "0.1",
                           "min_equity": "1", "incomplete_data_max_leverage": "1", "require_mark": True, "require_funding": True},
                soft_risk={"target_scale": "1", "max_turnover_fraction": "1", "drawdown_throttle": "0", "funding_sensitivity": "0"},
                fit_scaler=True)


def test_historical_planner_uses_frozen_quantity_unit_and_contract_multiplier(tmp_path, monkeypatch):
    import rles_gym_bridge.historical as historical

    planned = []
    original = historical.to_contract_plan

    def capture(decision, factory, *args, latency_model_id, quantity_unit):
        planned.append((decision.target_qty, quantity_unit))
        return original(decision, factory, *args, latency_model_id=latency_model_id,
                        quantity_unit=quantity_unit)

    monkeypatch.setattr(historical, "to_contract_plan", capture)
    factory = HistoricalEnvironmentFactory(**recipe(
        tmp_path, bar_count=14, base_currency="BTC", contract_multiplier="2",
    ))
    units = dict(zip(factory.schema.names, factory.schema.units, strict=True))
    assert units["market.volume_5m"] == "BTC"
    assert units["account.equity"] == "USDT"
    assert units["account.position_quantity"] == "BTC"
    env = factory.make_env()
    try:
        env.reset(seed=17)
        env.step(np.asarray([0.5], dtype=np.float32))
    finally:
        env.close()

    assert planned
    assert all(unit == "BTC" for _quantity, unit in planned)
    assert any(quantity == Decimal("0.250") for quantity, _unit in planned)


def test_historical_observation_and_risk_exposures_apply_contract_multiplier(tmp_path):
    from types import SimpleNamespace

    factory = HistoricalEnvironmentFactory(**recipe(
        tmp_path, bar_count=14, base_currency="BTC", contract_multiplier="2",
    ))
    snapshot = SimpleNamespace(
        mark=Decimal("1000"), equity=Decimal("10000"), currency="USDT",
        positions=({"quantity": "0.5", "side": "LONG"},),
        account_events=({"balances": ({"currency": "USDT", "total": "10000", "free": "9000"},)},),
        margins=(), orders=({"side": "BUY", "leaves_qty": "0.25"},),
        cumulative_submitted_qty=Decimal("1"), cumulative_filled_qty=Decimal("0.5"),
    )

    state = factory.observation_state(snapshot)

    assert state["account"]["net_exposure"] == pytest.approx(1.0)
    assert state["account"]["position_quantity"] == pytest.approx(1.0)
    assert state["execution"]["pending_signed_exposure"] == pytest.approx(0.5)
    assert state["execution"]["pending_gross_exposure"] == pytest.approx(0.5)


def test_causal_mtf_profile_adds_regime_features_without_future_leakage():
    maximum_age = {"mark": 60 * MINUTE_NS, "index": 60 * MINUTE_NS,
                   "funding_update": 60 * MINUTE_NS}
    legacy = historical_feature_spec(maximum_age, quantity_unit="ETH", quote_currency="USDT")
    profile_v1 = historical_feature_spec(maximum_age, quantity_unit="ETH", quote_currency="USDT",
                                         profile="causal_mtf_v1")
    profile = historical_feature_spec(maximum_age, quantity_unit="ETH", quote_currency="USDT",
                                      profile="causal_mtf_v2")
    assert profile.fields[:len(legacy.fields)] == legacy.fields
    assert profile.fields[:len(profile_v1.fields)] == profile_v1.fields
    assert tuple(field.name for field in profile.fields[len(legacy.fields):len(profile_v1.fields)]) == (
        "return_5m", "return_15m", "return_60m", "volatility_15m", "volatility_60m",
    )
    assert tuple(field.name for field in profile.fields[len(profile_v1.fields):]) == (
        "trend_efficiency_15m", "trend_efficiency_60m",
    )
    assert profile.max_lookback_ns == 61 * MINUTE_NS

    symbol = "ETHUSDT-PERP.SIM"
    events = []
    closes = []
    for minute in range(1, 62):
        timestamp = minute * MINUTE_NS
        close = float(100 + minute)
        closes.append(close)
        for kind, payload in {
            "ohlcv_1m": {"open": str(close), "high": str(close), "low": str(close),
                         "close": str(close), "volume": "10"},
            "mark": {"price": str(close)},
            "index": {"price": str(close)},
        }.items():
            events.append(Event(f"{kind}-{minute}", kind, timestamp, timestamp, timestamp,
                                Availability.NATIVE, "a" * 64, {"instrument": symbol, **payload},
                                f"synthetic:causal-mtf:{kind}"))
    events.append(Event("funding", "funding_update", MINUTE_NS, MINUTE_NS, MINUTE_NS,
                        Availability.NATIVE, "a" * 64,
                        {"instrument": symbol, "rate": "0.001"}, "synthetic:causal-mtf:funding"))

    decision = 61 * MINUTE_NS
    store = CuratedStore(events)
    row = make_feature_row(store, profile, decision, instrument=symbol)
    values = dict(zip((field.name for field in profile.fields), row.values))
    assert values["return_60m"] == pytest.approx(closes[-1] / closes[0] - 1)
    assert values["trend_efficiency_15m"] == pytest.approx(1.0)
    assert values["trend_efficiency_60m"] == pytest.approx(1.0)
    assert row.available_at_ns <= decision

    future = Event("future-bar", "ohlcv_1m", 62 * MINUTE_NS, 62 * MINUTE_NS, 62 * MINUTE_NS,
                   Availability.NATIVE, "b" * 64,
                   {"instrument": symbol, "open": "1000000", "high": "1000000", "low": "1000000",
                    "close": "1000000", "volume": "1000000"}, "synthetic:future-perturbation")
    extended_store = CuratedStore((*events, future))
    extended_row = make_feature_row(extended_store, profile, decision, instrument=symbol)
    assert extended_row.values == row.values


def test_trend_efficiency_encodes_chop_is_bounded_and_rejects_gaps():
    from math import isfinite
    from rles_data.core import FeatureField, FeatureSpec

    symbol = "ETHUSDT-PERP.SIM"
    events = []
    for minute in range(1, 62):
        close = 100.0 if minute < 47 or minute % 2 == 0 else 101.0
        timestamp = minute * MINUTE_NS
        for kind, payload in {
            "ohlcv_1m":…9647 tokens truncated…sert resources["gpu_max_allocated_bytes"] is None
        assert resources["gpu_max_reserved_bytes"] is None
        bundle_record = json.loads((tmp_path / "run" / "bundle.json").read_text(encoding="utf-8"))
        assert bundle_record["runtime_resources"]["gpu_lease"] is None
        assert bundle_record["runtime_resources"]["cuda_allocator_fraction"] is None
        assert bundle_record["actor"]["training_diagnostics_hash"] == sha256(
            learning.read_bytes()
        ).hexdigest()
        original_learning = learning.read_bytes()
        learning.write_bytes(original_learning + b"\n")
        with pytest.raises(DataError, match="integrity mismatch"):
            backtest(exploratory, result["output"])
        learning.write_bytes(original_learning)
        bundle = tmp_path / "run" / "bundle.json"
        original = bundle.read_bytes()
        bundle.write_bytes(original + b"\n")
        with pytest.raises(DataError, match="integrity mismatch"):
            backtest(exploratory, result["output"])
        bundle.write_bytes(original)


def test_application_parallel_rl_uses_rotating_rank_exclusive_train_lanes(tmp_path):
    from rles_orchestrator.application import train

    config = recipe(tmp_path)
    config["end_ns"] = 13 * MINUTE_NS
    config["market_path"], config["rules_path"] = str(config["market_path"]), str(config["rules_path"])
    run = {
        "environment": config,
        "evaluation": {"start_ns": 13 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                       "coverage_policy": config["coverage_policy"]},
        "output": str(tmp_path / "parallel-run"), "algorithm": "PPO", "steps": 4,
        "hidden_layers": [2], "rl_num_envs": 2, "rl_episode_steps": 2,
        "rl_corpus_episode_steps": 2,
        "hyperparameters": {"n_steps": 2, "batch_size": 4, "n_epochs": 1},
    }
    result = train(run)
    coverage = result["training_coverage"]
    assert coverage["environment_count"] == 2
    assert coverage["nominal_episode_decision_steps"] == 2
    assert coverage["sampling_policy"] == "rank_exclusive_coprime_rotating_windows_v1"
    assert coverage["initial_episode_start_offsets"][0] == 0
    assert coverage["initial_episode_start_offsets"][1] > 0
    assert coverage["upper_bound_unique_train_window_steps"] == coverage[
        "train_window_decision_steps"
    ]
    assert coverage["upper_bound_train_window_coverage_fraction"] == 1.0


def test_parallel_rl_checkpoint_restores_each_partition_mid_episode(tmp_path, monkeypatch):
    from rles_orchestrator.application import train
    from rles_rl_sb3.provider import RLProvider

    environment = recipe(tmp_path)
    environment["end_ns"] = 13 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {
        "environment": environment,
        "evaluation": {"start_ns": 13 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                       "coverage_policy": environment["coverage_policy"]},
        "output": str(tmp_path / "parallel-resume"), "algorithm": "PPO", "steps": 4,
        "hidden_layers": [2], "rl_num_envs": 2, "rl_episode_steps": 2,
        "rl_corpus_episode_steps": 2,
        "hyperparameters": {"n_steps": 1, "batch_size": 2, "n_epochs": 1},
        "checkpoint_interval_steps": 2,
    }
    original_train = RLProvider.train
    calls = 0

    def stop_after_first_boundary(provider, total_timesteps, *, resumed=False, callback=None):
        nonlocal calls
        if calls:
            raise RuntimeError("interrupt parallel learner after durable boundary")
        calls += 1
        return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

    monkeypatch.setattr(RLProvider, "train", stop_after_first_boundary)
    with pytest.raises(RuntimeError, match="interrupt parallel learner"):
        train(run)
    checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000002"
    replays = json.loads((checkpoint / "episode_replays.json").read_text(encoding="utf-8"))
    assert len(replays) == 2
    assert all(len(replay["actions"]) == 1 for replay in replays)
    assert all(replay["schema"] == "rles.episode-replay.v2" for replay in replays)
    assert [replay["episode_index"] for replay in replays] == [0, 0]

    monkeypatch.setattr(RLProvider, "train", original_train)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    assert result["training_coverage"]["environment_count"] == 2
    assert (Path(run["output"]) / "COMPLETED.json").is_file()


@pytest.mark.parametrize("route", ("A1", "A4"))
def test_application_es_and_hybrid_use_installed_historical_spawn_factory(tmp_path, route, monkeypatch):
    from rles_orchestrator.application import backtest, diagnose_train, train

    config = recipe(tmp_path)
    config["end_ns"] = 12 * MINUTE_NS
    config["market_path"], config["rules_path"] = str(config["market_path"]), str(config["rules_path"])
    run = {"environment": config, "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                                                   "coverage_policy": config["coverage_policy"]},
           "output": str(tmp_path / "run"), "algorithm": "PPO", "route": route,
           "steps": 4, "hidden_layers": [2], "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "distillation": {"epochs": 120, "tolerance": 0.12},
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
    if route == "A4":
        from rles_rl_sb3.provider import RLProvider

        original_train = RLProvider.train
        calls = 0

        def interrupt_after_first_durable_chunk(provider, total_timesteps, *, resumed=False, callback=None):
            nonlocal calls
            if calls:
                raise RuntimeError("injected A4 interruption after R1 checkpoint")
            calls += 1
            return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

        monkeypatch.setattr(RLProvider, "train", interrupt_after_first_durable_chunk)
        with pytest.raises(RuntimeError, match="injected A4 interruption"):
            train(run)
        checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000002"
        assert (checkpoint / "COMPLETED").is_file()
        monkeypatch.setattr(RLProvider, "train", original_train)
        run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    restored = backtest({**run, "evaluation": {**run["evaluation"], "split": "validation"}}, result["output"])
    assert result["metrics"] == restored["metrics"]
    assert restored["terminal_flat"]
    bundle_root = Path(result["output"])
    bundle_files_before = {str(path.relative_to(bundle_root)): sha256(path.read_bytes()).hexdigest()
                           for path in bundle_root.rglob("*") if path.is_file()}
    with monkeypatch.context() as diagnostic_patch:
        from rles_data.io import curated_source_identity, read_curated

        def forbidden(*_args, **_kwargs):
            pytest.fail("Train diagnostics must not refit scaling or construct an evaluation")
        diagnostic_patch.setattr("rles_gym_bridge.historical.fit_train", forbidden)
        diagnostic_patch.setattr("rles_orchestrator.application._evaluation_factory", forbidden)
        diagnostic_patch.setattr("rles_orchestrator.application.ResearchLedger", forbidden)
        diagnostic_patch.setattr(
            "rles_orchestrator.application.read_registered_train_window",
            lambda scoped_config, _window, **_kwargs:
                read_curated(scoped_config["environment"]["market_path"]),
        )
        scoped_run = {
            **run, "evaluation": None,
            "train_data_access": {
                "start_ns": MINUTE_NS, "end_ns": 31 * MINUTE_NS,
                "marker_sha256": curated_source_identity(config["market_path"])["marker_sha256"],
                "label": "train-diagnostic-test",
            },
        }
        diagnostic = diagnose_train(scoped_run, bundle_root)
        assert diagnostic["diagnostic_scope"]["split"] == "train_inner"
        assert diagnostic["diagnostic_scope"]["evaluation_queries"] == 0
        assert diagnostic["diagnostic_scope"]["end_ns"] == 12 * MINUTE_NS
        assert diagnostic["action_diagnostics"]["count"] > 0
        assert diagnostic["execution_attribution"]["components_known"] is True
        assert diagnostic["execution_attribution"]["quantity_rounding_components_known"] is True
        assert diagnostic["quantity_rounding_diagnostics"]["components_known"] is True
        assert diagnostic["quantity_rounding_diagnostics"]["count"] == diagnostic["action_diagnostics"]["count"]
        assert diagnostic["terminal_flat"] is True
        assert "account_rows" not in diagnostic
        with pytest.raises(DataError, match="may not replace"):
            diagnose_train({**scoped_run,
                            "environment": {**config, "end_ns": 20 * MINUTE_NS}}, bundle_root)
    assert bundle_files_before == {
        str(path.relative_to(bundle_root)): sha256(path.read_bytes()).hexdigest()
        for path in bundle_root.rglob("*") if path.is_file()
    }
    from os import name as os_name

    resource_paths = list(Path(result["output"]).glob("E1-resources-*.json"))
    assert resource_paths
    resource_report = json.loads(resource_paths[0].read_text())
    assert resource_report["configured_policy"]["enforcement"] == (
        "windows_job_commit_memory" if os_name == "nt" else "quota_only")
    if os_name == "nt":
        assert resource_report["workers"][0]["active_processes"] == 0
        assert resource_report["workers"][0]["peak_process_commit_bytes"] > 0
    manifest = json.loads((Path(result["output"]) / "run.json").read_text())
    assert manifest["actor_spec"]["version"] == 2
    scales = manifest["actor_spec"]["input_scale"]
    names = manifest["actor_spec"]["observation_names"]
    assert scales[names.index("account.cash")] == pytest.approx(0.0001)
    assert scales[names.index("account.position_quantity")] == pytest.approx(1.0)
    if route == "A4":
        assert manifest["canonical_ppo_output"] is True
        transfer = json.loads((Path(result["output"]) / "R1-transfer.json").read_text())["result"]
        assert transfer["method"] == "DIRECT"
        assert transfer["parity"]["max_abs_error"] < 1e-5
    if route == "A4":
        from rles_research.ledger import ResearchLedger

        root = Path(result["output"])
        receipts = ResearchLedger(root / "research.sqlite").evidence_summary(
            json.loads((root / "run.json").read_text(encoding="utf-8"))["run_id"],
        )["stage_receipts"]
        assert {item["stage"] for item in receipts} == {"E1", "R1", "E2"}
        # E2 now includes one paired exact-neutral candidate in each generation.
        assert sum(json.loads(item["usage_json"]).get("lane_requests", 0) for item in receipts) == 9
        assert len(list(root.glob("FAILED-*.json"))) == 1
        assert (root / "R1-transfer.json").is_file()
        checkpoint_dirs = sorted((Path(result["output"]) / "R1-checkpoints").glob("step-*"))
        assert [path.name for path in checkpoint_dirs] == ["step-000000000002", "step-000000000004"]
        for checkpoint in checkpoint_dirs:
            manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["schema"] == "rles.l4.stage.v4"
            assert "episode_replays.json" in manifest["files"]
            assert (checkpoint / "COMPLETED").is_file()
            replays = json.loads((checkpoint / "episode_replays.json").read_text(encoding="utf-8"))
            assert len(replays) == 1
            assert replays[0]["actions"]
            assert len(replays[0]["actions"]) <= int(checkpoint.name.rsplit("-", 1)[1])
            assert replays[0]["boundary_token"] == manifest["boundary"]["env_tokens"][0]


@pytest.mark.parametrize("route", ("A0", "A2", "A3"))
def test_application_routes_resume_partial_r1_from_replay_checkpoint(tmp_path, monkeypatch, route):
    from rles_orchestrator.application import train
    from rles_rl_sb3.provider import RLProvider

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / f"resume-{route.lower()}-run"), "algorithm": "PPO", "route": route,
           "steps": 4, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
    if route in {"A2", "A3"}:
        run.update(population=4, generations=1,
                   distillation={"epochs": 120, "tolerance": 0.12})
    original_train = RLProvider.train
    calls = 0

    def stop_after_first_boundary(provider, total_timesteps, *, resumed=False, callback=None):
        nonlocal calls
        if calls:
            raise RuntimeError("injected interruption after durable R1 boundary")
        calls += 1
        return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

    monkeypatch.setattr(RLProvider, "train", stop_after_first_boundary)
    with pytest.raises(RuntimeError, match="injected interruption"):
        train(run)
    checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000002"
    assert (checkpoint / "COMPLETED").is_file()
    assert (Path(run["output"]) / "FAILED.json").is_file()

    monkeypatch.setattr(RLProvider, "train", original_train)
    changed_recipe = dict(run, seed=run.get("seed", 31) + 1,
                          resume_checkpoint=str(checkpoint))
    with pytest.raises(DataError, match="configuration differs"):
        train(changed_recipe)
    assert (Path(run["output"]) / "FAILED.json").is_file()
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    root = Path(result["output"])
    assert (root / "COMPLETED.json").is_file()
    assert (root / "FAILED.json").is_file() is False
    assert list(root.glob("FAILED-*.json"))
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert [item["stage"] for item in bundle["lineage"]] == {
        "A0": ["R1"], "A2": ["R1", "E2"], "A3": ["E1", "R1"],
    }[route]
    rl_entry = next(item for item in bundle["lineage"] if item["stage"] == "R1")
    assert rl_entry["usage"]["rl_steps"] == 4
    assert (root / "R1-checkpoints" / "step-000000000004" / "COMPLETED").is_file()


@pytest.mark.parametrize("route", ("A1", "A3", "A4"))
def test_application_routes_resume_e1_from_completed_generation(tmp_path, monkeypatch, route):
    from rles_es_evotorch.provider import EvoTorchProvider
    from rles_orchestrator.application import train

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / f"resume-{route.lower()}-e1-run"), "algorithm": "PPO", "route": route,
           "steps": 4, "population": 4, "generations": 2, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
    if route in {"A3", "A4"}:
        run["distillation"] = {"epochs": 120, "tolerance": 0.12}

    original_continue = EvoTorchProvider.continue_evolution
    calls = 0

    def stop_before_second_generation(provider, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected E1 interruption after generation one")
        return original_continue(provider, *args, **kwargs)

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", stop_before_second_generation)
    with pytest.raises(RuntimeError, match="injected E1 interruption"):
        train(run)

    root = Path(run["output"])
    checkpoint = root / "E1" / "generation-00000001" / json.loads(
        (root / "run.json").read_text(encoding="utf-8"),
    )["run_id"]
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == 1
    assert (root / "FAILED.json").is_file()

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", original_continue)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    assert (root / "COMPLETED.json").is_file()
    assert not (root / "FAILED.json").exists()
    assert list(root.glob("FAILED-*.json"))
    assert (root / "E1" / "generation-00000002" / manifest["attempt_id"] / "COMPLETED").is_file()
    assert result["route"] == route


@pytest.mark.parametrize("route", ("A2", "A4"))
def test_application_routes_resume_e2_and_revalidate_existing_r1_actor(tmp_path, monkeypatch, route):
    from rles_es_evotorch.provider import EvoTorchProvider
    from rles_orchestrator.application import backtest, train

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / f"resume-{route.lower()}-e2-run"), "algorithm": "PPO", "route": route,
           "steps": 4, "population": 4, "generations": 2, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
    if route == "A4":
        run["distillation"] = {"epochs": 120, "tolerance": 0.12}

    original_continue = EvoTorchProvider.continue_evolution
    e2_calls = 0

    def stop_during_second_e2_generation(provider, *args, **kwargs):
        nonlocal e2_calls
        if provider.config.algorithm == "cmaes":
            e2_calls += 1
            if e2_calls == 2:
                raise RuntimeError("injected E2 interruption after generation one")
        return original_continue(provider, *args, **kwargs)

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", stop_during_second_e2_generation)
    with pytest.raises(RuntimeError, match="injected E2 interruption"):
        train(run)

    root = Path(run["output"])
    run_id = json.loads((root / "run.json").read_text(encoding="utf-8"))["run_id"]
    checkpoint = root / "E2" / "generation-00000001" / run_id
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == 1
    actor_hash = (root / "r1" / "COMPLETED").read_text(encoding="ascii")
    assert (root / "FAILED.json").is_file()
    if route == "A4":
        # Simulate interruption after the payload is durable but before its
        # digest sidecar is restored by the E1 recovery pass.
        e1_digest_path = root / "E1-selected-actor.json.sha256"
        assert e1_digest_path.is_file()
        e1_digest_path.unlink()

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", original_continue)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    report = backtest({**run, "evaluation": {**run["evaluation"], "split": "validation"}}, result["output"])
    assert report["terminal_flat"]
    assert (root / "r1" / "COMPLETED").read_text(encoding="ascii") == actor_hash
    assert (root / "E2" / "generation-00000002" / run_id / "COMPLETED").is_file()
    assert (root / "COMPLETED.json").is_file()
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if route == "A4":
        e1_artifact = bundle["stage_actor_artifacts"]["E1"]
        e1_path = root / e1_artifact["path"]
        assert sha256(e1_path.read_bytes()).hexdigest() == e1_artifact["sha256"]
        assert e1_path.with_suffix(e1_path.suffix + ".sha256").read_text(encoding="ascii") == e1_artifact["sha256"]
        e1_payload = json.loads(e1_path.read_text(encoding="utf-8"))
        e1_lineage = next(row for row in bundle["lineage"] if row["stage"] == "E1")
        assert e1_payload["actor_hash"] == e1_lineage["actor_hash"] == e1_artifact["actor_hash"]
        assert e1_payload["lane_evidence_hash"] == e1_artifact["lane_evidence_hash"] == (
            e1_lineage["metadata"]["lane_evidence_hash"]
        )
        from rles_es_evotorch.actor import ActorGenomeSpec, genome_hash

        e1_spec_values = dict(e1_payload["actor_spec"])
        e1_spec_values["observation_names"] = tuple(e1_spec_values["observation_names"])
        e1_spec_values["hidden_sizes"] = tuple(e1_spec_values["hidden_sizes"])
        e1_spec_values["input_scale"] = (
            None if e1_spec_values.get("input_scale") is None else tuple(e1_spec_values["input_scale"])
        )
        assert genome_hash(ActorGenomeSpec(**e1_spec_values),
                           np.asarray(e1_payload["genome"], dtype=np.float32)) == e1_payload["actor_hash"]
    else:
        assert bundle["stage_actor_artifacts"] == {}


def test_application_finalizes_full_budget_r1_checkpoint_without_retraining(tmp_path, monkeypatch):
    from rles_orchestrator.application import train
    from rles_rl_sb3.provider import RLProvider

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / "finalize-run"), "algorithm": "PPO", "route": "A0",
           "steps": 4, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
    original_export = RLProvider.export_inference

    def interrupt_before_inference_export(_provider, _directory, _fixed_observations):
        raise RuntimeError("injected interruption after full-budget checkpoint")

    monkeypatch.setattr(RLProvider, "export_inference", interrupt_before_inference_export)
    with pytest.raises(RuntimeError, match="full-budget checkpoint"):
        train(run)
    checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000004"
    assert (checkpoint / "COMPLETED").is_file()

    monkeypatch.setattr(RLProvider, "export_inference", original_export)

    def reject_retraining(*_args, **_kwargs):
        raise AssertionError("finalize-only recovery must not train or charge new steps")

    monkeypatch.setattr(RLProvider, "train", reject_retraining)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    root = Path(result["output"])
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["lineage"][0]["usage"]["rl_steps"] == 4
    assert (root / "COMPLETED.json").is_file()
    assert list(root.glob("FAILED-*.json"))

def test_compact_report_positions_are_immutable_facts(tmp_path):
    config = recipe(tmp_path, bar_count=32)
    config["compact_report_snapshots"] = True
    factory = HistoricalEnvironmentFactory(**config)
    env = factory.make_env()
    try:
        env.reset(seed=83)
        for _ in range(12):
            observation, reward, terminated, truncated, info = env.step(
                np.asarray([0.5], dtype=np.float32),
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    positions = [snapshot.positions[0] for snapshot in factory.snapshots
                 if snapshot.positions]
    assert len(positions) >= 3
    compact_accounts = [snapshot.account for snapshot in factory.snapshots]
    assert compact_accounts[0] is compact_accounts[-1]
    with pytest.raises(TypeError):
        compact_accounts[0]["balance"] = "0"
    assert factory.snapshots[0].equity == next(iter(factory.snapshots)).equity
    assert factory.snapshots[-1].decision_ts_ns == tuple(factory.snapshots)[-1].decision_ts_ns
    assert factory.snapshots[1:3] == tuple(factory.snapshots)[1:3]
    with pytest.raises(TypeError):
        positions[0]["quantity"] = "0"


def test_compact_snapshot_numeric_columns_preserve_decimal_and_timestamp_values():
    series = _CompactReportSnapshotSeries()
    equities = (Decimal("-0.00100"), Decimal("1.2300"),
                Decimal("123456789012345678901234567890.12300"), Decimal("NaN"))
    marks = (Decimal("-0"), Decimal("0.0001"), Decimal("1E+33"), None)
    timestamps = (1, -7, (1 << 63) + 1, True)
    for equity, mark, timestamp in zip(equities, marks, timestamps, strict=True):
        series.append(_CompactReportSnapshot(
            equity=equity, currency="USDT", positions=(), orders=(), account={},
            account_events=(), fills=(), funding=(), all_orders=(),
            liquidation_orders=(), margins=(), mark=mark,
            liquidation_capability=True, decision_ts_ns=timestamp,
            cumulative_submitted_qty=Decimal("0.000"),
            cumulative_filled_qty=Decimal("-0.000"),
        ))

    for index, (equity, mark, timestamp) in enumerate(zip(equities, marks, timestamps, strict=True)):
        restored = series[index]
        assert restored.equity.as_tuple() == equity.as_tuple()
        if isinstance(mark, Decimal):
            assert restored.mark.as_tuple() == mark.as_tuple()
        else:
            assert restored.mark is mark
        assert restored.decision_ts_ns is timestamp or restored.decision_ts_ns == timestamp
        assert restored.cumulative_submitted_qty.as_tuple() == Decimal("0.000").as_tuple()
        assert restored.cumulative_filled_qty.as_tuple() == Decimal("-0.000").as_tuple()


def test_compact_snapshot_run_columns_retain_only_fact_changes():
    series = _CompactReportSnapshotSeries()
    stable_position = (MappingProxyType({"position_id": "P1", "quantity": "1"}),)
    changed_position = (MappingProxyType({"position_id": "P1", "quantity": "2"}),)
    fill = ({"event_id": "F1", "last_qty": "1"},)
    for index in range(6):
        series.append(_CompactReportSnapshot(
            equity=Decimal(100 + index), currency="USDT",
            positions=stable_position if index < 4 else changed_position,
            orders=(), account={}, account_events=(),
            fills=fill if index == 2 else (), funding=(), all_orders=(),
            liquidation_orders=(), margins=(), mark=Decimal(10),
            liquidation_capability=True, decision_ts_ns=index,
            cumulative_submitted_qty=Decimal(1 if index >= 3 else 0),
            cumulative_filled_qty=Decimal(1 if index >= 3 else 0),
        ))

    assert len(series) == 6
    assert len(series._positions._values) == 2
    assert len(series._orders._values) == 1
    assert len(series._fills._values) == 3
    assert len(series._funding._values) == 1
    assert len(series._all_orders._values) == 1
    assert len(series._liquidation_orders._values) == 1
    assert len(series._cumulative_submitted_qty._values) == 2
    assert series[0].positions is stable_position
    assert series[4].positions is changed_position
    assert series[1].fills == ()
    assert series[2].fills == fill
    assert series[5].cumulative_submitted_qty == Decimal(1)


def test_spooled_summary_snapshots_round_trip_exact_report_facts():
    spool = _SpooledReportSnapshotSequence()
    snapshot = _CompactReportSnapshot(
        equity=Decimal("-0.00100"), currency="USDT",
        positions=(MappingProxyType({"quantity": Decimal("1.2300")}),),
        orders=(), account={}, account_events=({"value": Decimal("-sNaN007")},),
        fills=({"event_id": "F1", "last_qty": Decimal("-0.000")},),
        funding=(), all_orders=(), liquidation_orders=(), margins=(Decimal("-Infinity"),),
        mark=Decimal("1E+33"), liquidation_capability=True,
        decision_ts_ns=1_700_000_000_000_000_000,
        cumulative_submitted_qty=Decimal("2.00"),
        cumulative_filled_qty=Decimal("1.000"),
    )
    try:
        spool.append(snapshot)
        restored = spool[0]
        assert len(spool) == 1
        assert restored.equity.as_tuple() == snapshot.equity.as_tuple()
        assert restored.positions[0]["quantity"].as_tuple() == Decimal("1.2300").as_tuple()
        assert restored.fills[0]["last_qty"].as_tuple() == Decimal("-0.000").as_tuple()
        assert restored.account_events[0]["value"].as_tuple() == Decimal("-sNaN007").as_tuple()
        assert restored.margins[0].as_tuple() == Decimal("-Infinity").as_tuple()
        assert restored.mark.as_tuple() == Decimal("1E+33").as_tuple()
        iterated = next(iter(spool))
        assert iterated.equity.as_tuple() == snapshot.equity.as_tuple()
    finally:
        spool.close()

