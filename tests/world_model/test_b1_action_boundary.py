from __future__ import annotations

import numpy as np

from rles_world_model.upstream import rssm_market_observe


def test_trade_action_cannot_change_market_transition_input() -> None:
    class FakeRSSM:
        def observe(self, carry, tokens, action, reset, training, *, single):
            return action

    outputs = [rssm_market_observe(
        FakeRSSM(), None, None, None, False,
        trading_action=np.full((2, 1), policy_action, dtype=np.float32), single=True,
    )['market_dummy'] for policy_action in (-1.0, 0.0, 1.0)]
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])
    np.testing.assert_array_equal(outputs[0], np.zeros((2, 1), dtype=np.float32))
