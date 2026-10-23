import numpy as np
import pytest

from rles_world_model.routeb.evaluation import sample_e0_paths


def test_e0_paths_are_finite_reproducible_and_anchor_causal():
    train = np.asarray([0.001, -0.002, 0.0005, 0.003, -0.0015, 0.0002])
    visible = np.asarray([[0.0001, -0.0003], [0.001, -0.0005]])
    paths = sample_e0_paths(train, visible, paths=32, horizon=15, seed=31)
    repeated = sample_e0_paths(train, visible, paths=32, horizon=15, seed=31)
    assert set(paths) == {
        "S0_ZERO_DRIFT_FIXED_VOL",
        "S1_EWMA_VOL_TRAIN_INNOVATIONS",
        "S2_STATIONARY_BLOCK_BOOTSTRAP",
    }
    assert all(path.shape == (2, 32, 15) for path in paths.values())
    assert all(np.isfinite(path).all() for path in paths.values())
    assert all(np.array_equal(paths[name], repeated[name]) for name in paths)

    changed_later_anchor = visible.copy()
    changed_later_anchor[1] *= 100
    changed = sample_e0_paths(train, changed_later_anchor, paths=32, horizon=15, seed=31)
    assert np.array_equal(paths["S1_EWMA_VOL_TRAIN_INNOVATIONS"][0],
                          changed["S1_EWMA_VOL_TRAIN_INNOVATIONS"][0])
    assert np.array_equal(paths["S0_ZERO_DRIFT_FIXED_VOL"][0],
                          changed["S0_ZERO_DRIFT_FIXED_VOL"][0])


@pytest.mark.parametrize("kwargs", [
    {"paths": 1, "horizon": 1, "seed": 0},
    {"paths": 2, "horizon": 0, "seed": 0},
    {"paths": 2, "horizon": 1, "seed": -1},
    {"paths": 2, "horizon": 1, "seed": 0, "ewma_decay": 1.0},
])
def test_e0_rejects_invalid_sampling_configuration(kwargs):
    with pytest.raises(ValueError):
        sample_e0_paths(np.asarray([0.1, -0.1]), np.zeros((1, 0)), **kwargs)
