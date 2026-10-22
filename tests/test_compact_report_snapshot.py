"""The per-barrier native report container stays compact at long horizons."""
import sys
from types import SimpleNamespace

from rles_gym_bridge.historical import _CompactReportSnapshot


def test_compact_report_snapshot_has_no_per_instance_dict_and_uses_less_memory() -> None:
    fields = {
        "equity": "1000 USDT",
        "currency": "USDT",
        "positions": (),
        "orders": (),
        "account": {},
        "account_events": (),
        "fills": (),
        "funding": (),
        "all_orders": (),
        "liquidation_orders": (),
        "margins": (),
        "mark": "1.0",
        "liquidation_capability": True,
        "decision_ts_ns": 1,
        "cumulative_submitted_qty": 0,
        "cumulative_filled_qty": 0,
    }
    compact = _CompactReportSnapshot(**fields)
    legacy = SimpleNamespace(**fields)

    assert not hasattr(compact, "__dict__")
    assert sys.getsizeof(compact) < sys.getsizeof(legacy) + sys.getsizeof(legacy.__dict__)
    assert all(getattr(compact, key) == value for key, value in fields.items())
