import inspect
import numpy as np
import pytest
from autoencoder.evaluate import choose_threshold, evaluate_scenario


def test_threshold_is_validation_percentile():
    val = np.arange(1, 101, dtype=float)
    assert choose_threshold(val, 95) == pytest.approx(np.percentile(val, 95))


def test_threshold_function_accepts_only_validation_errors():
    assert list(inspect.signature(choose_threshold).parameters) == ["val_errors", "percentile"]


def test_changing_test_data_does_not_change_threshold():
    rng = np.random.default_rng(0)
    val = rng.random(1000)
    test_a, labels_a = rng.random(500), rng.integers(0, 2, 500)
    test_b, labels_b = rng.random(500) * 10, rng.integers(0, 2, 500)
    thr_a = evaluate_scenario(val, test_a, labels_a)[0]["limiar"]
    thr_b = evaluate_scenario(val, test_b, labels_b)[0]["limiar"]
    assert thr_a == thr_b


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        choose_threshold([], 95)
    with pytest.raises(ValueError):
        choose_threshold([1.0, 2.0], 120)