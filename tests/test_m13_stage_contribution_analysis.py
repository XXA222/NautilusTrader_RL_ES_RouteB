from __future__ import annotations

from pathlib import Path

import pytest

from qa.analyze_m13_stage_contribution import (
    _load_json, _verify_equity_reconciliation, summarize_mode,
)


def _arm(*, equity: tuple[float, float], fee: float, fills: int, submits: int) -> dict:
    metrics = {
        "net_return": equity[-1] / 10_000 - 1,
        "max_drawdown": 0.01,
        "sharpe": -0.5,
        "sortino": -0.8,
        "turnover_ratio": 0.2,
        "total_fee": fee,
        "total_funding": 0.1,
        "total_slippage_cost": 0.05,
        "gross_return_before_reported_costs": 0.002,
    }
    return {
        "account_rows": [
            {"ts_ns": 10, "equity": equity[0]},
            {"ts_ns": 20, "equity": equity[1]},
        ],
        "costs_already_in_native_equity": True,
        "equity_reconciliation": {
            "rows": [{"ts_ns": 20, "residual": "0", "tolerance": "0.0000001"}],
            "max_abs_residual": "0",
        },
        "terminal_flat": True,
        "metrics": metrics,
        "fills": [{} for _ in range(fills)],
        "execution_decision_diagnostics": {
            "decision_count": 3,
            "submit_decision_count": submits,
            "native_order_count": submits,
        },
        "risk_hard_interventions": 1,
    }


def _mode_payload(*, e2_timestamps: tuple[int, int] = (10, 20)) -> dict:
    r1 = _arm(equity=(10_000, 9_990), fee=2, fills=4, submits=3)
    e2 = _arm(equity=(10_000, 9_995), fee=1, fills=2, submits=2)
    e2["account_rows"][0]["ts_ns"], e2["account_rows"][1]["ts_ns"] = e2_timestamps
    return {
        "r1_neutral": r1,
        "e2_selected": e2,
        "summary": {
            "same_native_timestamps": True,
            "final_equity_r1": 9_990,
            "final_equity_e2": 9_995,
            "native_equity_delta_e2_minus_r1": 5,
        },
    }


def test_summarize_mode_recomputes_paired_economic_and_execution_deltas() -> None:
    result = summarize_mode("base", _mode_payload())

    assert result["paired_account_rows"] == 2
    assert result["policy_decisions_per_arm"] == 3
    assert result["native_equity_delta_e2_minus_r1"] == 5
    assert result["metric_deltas_e2_minus_r1"]["total_fee"] == -1
    assert result["metric_deltas_e2_minus_r1"]["net_return"] == pytest.approx(0.0005)
    assert result["execution_deltas_e2_minus_r1"] == {
        "fills": -2,
        "submit_decisions": -1,
        "native_orders": -1,
        "hard_risk_interventions": 0,
    }


def test_summarize_mode_rejects_unpaired_timestamps() -> None:
    with pytest.raises(ValueError, match="timestamps are not strictly paired"):
        summarize_mode("base", _mode_payload(e2_timestamps=(10, 21)))


@pytest.mark.parametrize("payload", [
    b'{"schema":"first","schema":"second"}',
    b'{"outer":{"identity":1,"identity":2}}',
])
def test_revision2_json_reader_rejects_duplicate_keys(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _load_json(path)


def test_revision2_json_reader_normalizes_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b"{\"schema\":\"\xff\"}")
    with pytest.raises(ValueError, match="cannot read JSON object"):
        _load_json(path)


@pytest.mark.parametrize("residual,tolerance", [(float("nan"), 1.0), (0.0, True)])
def test_revision2_equity_reconciliation_rejects_non_finite_and_bool_numbers(
    residual: float, tolerance: float,
) -> None:
    report = {
        "costs_already_in_native_equity": True,
        "terminal_flat": True,
        "equity_reconciliation": {
            "rows": [{"ts_ns": 1, "residual": residual, "tolerance": tolerance}],
            "max_abs_residual": 0.0,
        },
    }
    with pytest.raises(ValueError):
        _verify_equity_reconciliation(report, "test")


def test_revision2_summarizer_rejects_boolean_metric_and_boolean_count() -> None:
    payload = _mode_payload()
    payload["r1_neutral"]["metrics"]["total_fee"] = True
    with pytest.raises(ValueError, match="finite number"):
        summarize_mode("base", payload)

    payload = _mode_payload()
    payload["r1_neutral"]["execution_decision_diagnostics"]["decision_count"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        summarize_mode("base", payload)
