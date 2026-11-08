"""Integração com cache oficial: nunca baixa dados durante o pytest."""

from pathlib import Path

import numpy as np
import pytest

from autoencoder.data import FILES, load_raw, make_dataloaders, make_splits


@pytest.fixture(scope="module")
def official_raw():
    directory = Path(__file__).resolve().parents[1] / "data"
    if not all((directory / name).exists() for name in FILES.values()):
        pytest.skip("Execute load_raw() ou o notebook para baixar o Fashion-MNIST")
    return load_raw(directory, download=False)


@pytest.mark.integration
@pytest.mark.parametrize("anomaly", [8, 5])
def test_official_data_acceptance(official_raw, anomaly):
    splits = make_splits(official_raw, anomaly_class=anomaly)
    repeated = make_splits(official_raw, anomaly_class=anomaly)
    assert (len(splits.train.labels), len(splits.val.labels), len(splits.test.labels)) == (45900, 8100, 10000)
    assert np.intersect1d(splits.train.indices, splits.val.indices).size == 0
    normal_indices = np.flatnonzero(official_raw["train_labels"] != anomaly)
    np.testing.assert_array_equal(np.sort(np.r_[splits.train.indices, splits.val.indices]), normal_indices)
    for name in ("train", "val", "test"):
        split = getattr(splits, name)
        assert split.images.dtype == np.float32
        assert np.isfinite(split.images).all()
        assert 0 <= split.images.min() <= split.images.max() <= 1
        np.testing.assert_array_equal(split.indices, getattr(repeated, name).indices)
        np.testing.assert_array_equal(split.images, getattr(repeated, name).images)
        counts = np.bincount(split.labels, minlength=10)
        expected = np.full(10, {"train": 5100, "val": 900, "test": 1000}[name])
        if name != "test":
            expected[anomaly] = 0
        np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(splits.test.labels, official_raw["test_labels"])
    np.testing.assert_array_equal(splits.test.images, official_raw["test_images"].astype(np.float32) / 255)
    loaders = make_dataloaders(splits)
    for name in ("train", "val", "test"):
        assert sum(len(images) for images, labels in loaders[name]) == len(getattr(splits, name).labels)
