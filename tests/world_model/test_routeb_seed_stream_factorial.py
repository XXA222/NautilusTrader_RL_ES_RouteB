from __future__ import annotations

import pytest

from qa.world_model.routeb_seed_stream_factorial import _factorial_contrasts


def test_factorial_contrasts_recover_known_binary_effects() -> None:
    roots = {
        "tied_31": (0, 0, 0),
        "initialization_only_102": (1, 0, 0),
        "sampling_only_102": (0, 1, 0),
        "model_noise_only_102": (0, 0, 1),
        "initialization_sampling_102": (1, 1, 0),
        "initialization_model_noise_102": (1, 0, 1),
        "sampling_model_noise_102": (0, 1, 1),
        "tied_102": (1, 1, 1),
    }
    results = {}
    for label, (initialization, sampling, model_noise) in roots.items():
        value = (3 + 5 * initialization + 2 * sampling - model_noise
                 + 4 * initialization * sampling
                 + 7 * initialization * model_noise
                 + 6 * sampling * model_noise
                 + 8 * initialization * sampling * model_noise)
        results[label] = {"metrics": {"loss": float(value)}}

    contrast = _factorial_contrasts(results, ("loss",))["loss"]

    assert contrast["main_effects_mean_difference"] == {
        "initialization": 12.5,
        "sampling": 9.0,
        "model_noise": 7.5,
    }
    assert contrast["pairwise_difference_in_differences"] == {
        "initialization:sampling": 8.0,
        "initialization:model_noise": 11.0,
        "sampling:model_noise": 10.0,
    }
    assert contrast["three_way_difference"] == 8.0


def test_factorial_contrasts_require_all_eight_cases() -> None:
    with pytest.raises(RuntimeError, match="all eight"):
        _factorial_contrasts({"tied_31": {"metrics": {"loss": 1.0}}}, ("loss",))
