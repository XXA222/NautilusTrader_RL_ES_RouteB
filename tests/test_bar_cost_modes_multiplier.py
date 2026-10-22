from decimal import Decimal
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qa"))
from qa.compare_bar_cost_modes_20261017 import _order_metrics
from rles_data.core import DataError


def test_bar_cost_order_quote_notional_uses_contract_multiplier() -> None:
    orders = [{"client_order_id": "order-1", "quantity": "1"}]
    fills = [{"client_order_id": "order-1", "last_qty": "1", "last_px": "100"}]

    metrics = _order_metrics(orders, fills, contract_multiplier=Decimal("2"))

    assert metrics["filled_quote_notional"] == "200"
    assert metrics["contract_multiplier"] == "2"


@pytest.mark.parametrize("multiplier", [Decimal("0"), Decimal("NaN")])
def test_bar_cost_order_metrics_reject_invalid_contract_multiplier(multiplier: Decimal) -> None:
    with pytest.raises(DataError, match="finite positive Decimal"):
        _order_metrics([], [], contract_multiplier=multiplier)
