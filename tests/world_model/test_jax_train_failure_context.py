from __future__ import annotations

import hashlib

import numpy as np

from rles_world_model.routeb.jax_train import _update_failure_context


def test_update_failure_context_reports_only_selected_train_batch_summaries() -> None:
    features = np.asarray([
        [[1.0, -2.0, 0.5, 0.0, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5]],
        [[-3.0, 4.0, 2.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
    ], dtype=np.float32)
    risk_targets = np.zeros((2, 2, 2, 6), dtype=np.float32)
    risk_targets[1, 0, 1, 3] = -17.5
    risk_targets[1, 1, 0, 0] = 999.0  # masked label must not affect the summary
    risk_masks = np.zeros((2, 2, 2), dtype=np.bool_)
    risk_masks[1, 0, 1] = True
    selection = np.asarray([1], dtype=np.int64)

    summary = _update_failure_context(
        step=401, sampler_seed=4321, selection=selection, features=features,
        risk_targets=risk_targets, risk_masks=risk_masks,
        cause=RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS\nextra detail"))

    selection_sha = hashlib.sha256(np.asarray(selection, dtype="<i8").tobytes()).hexdigest()
    assert "step=401" in summary
    assert "sampler_seed=4321" in summary
    assert f"selection_sha256={selection_sha}" in summary
    assert "feature_abs_max=4" in summary
    assert "valid_risk_target_abs_max=17.5" in summary
    assert "cause_type=RuntimeError" in summary
    assert "cause=CUDA_ERROR_ILLEGAL_ADDRESS" in summary
    assert "extra detail" not in summary
    assert "999" not in summary
