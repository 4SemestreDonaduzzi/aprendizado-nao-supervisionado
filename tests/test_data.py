"""Testes offline dos riscos que invalidariam o experimento."""

import gzip
import hashlib
import io
import struct

import numpy as np
import pytest
import torch

from autoencoder import data


@pytest.fixture
def raw():
    rng = np.random.default_rng(123)
    return {
        "train_images": rng.integers(0, 256, (400, 28, 28), dtype=np.uint8),
        "train_labels": np.repeat(np.arange(10, dtype=np.uint8), 40),
        "test_images": rng.integers(0, 256, (30, 28, 28), dtype=np.uint8),
        "test_labels": np.tile(np.arange(10, dtype=np.uint8), 3),
    }


@pytest.mark.parametrize("anomaly", [8, 5])
def test_no_anomalies_and_exact_normal_coverage(raw, anomaly):
    splits = data.make_splits(raw, anomaly_class=anomaly)
    assert anomaly not in splits.train.labels
    assert anomaly not in splits.val.labels
    assert np.intersect1d(splits.train.indices, splits.val.indices).size == 0
    used = np.concatenate([splits.train.indices, splits.val.indices])
    assert np.unique(used).size == len(used)
    np.testing.assert_array_equal(np.sort(used), np.flatnonzero(raw["train_labels"] != anomaly))
    assert splits.train.source == splits.val.source == "official_train"
    assert splits.test.source == "official_test"
    # Todas as nove classes normais, inclusive as posteriores à classe excluída.
    for cls in set(range(10)) - {anomaly}:
        assert np.sum(splits.train.labels == cls) == 34
        assert np.sum(splits.val.labels == cls) == 6


@pytest.mark.parametrize("anomaly", [8, 5])
def test_normalization_alignment_and_untouched_official_test(raw, anomaly):
    original = {key: value.copy() for key, value in raw.items()}
    splits = data.make_splits(raw, anomaly_class=anomaly)
    for name in ("train", "val", "test"):
        split = getattr(splits, name)
        source = "test" if name == "test" else "train"
        assert split.images.dtype == np.float32
        assert split.labels.dtype == np.int64
        assert np.isfinite(split.images).all()
        assert 0 <= split.images.min() <= split.images.max() <= 1
        np.testing.assert_array_equal(split.labels, raw[f"{source}_labels"][split.indices])
        np.testing.assert_allclose(
            split.images, raw[f"{source}_images"][split.indices].astype(np.float32) / 255,
        )
    np.testing.assert_array_equal(splits.test.indices, np.arange(30))
    np.testing.assert_array_equal(splits.test.labels, raw["test_labels"])
    assert (splits.test.labels == anomaly).sum() == 3
    for key in raw:
        np.testing.assert_array_equal(raw[key], original[key])


def test_same_seed_repeats_split_and_different_seed_changes_it(raw):
    first = data.make_splits(raw, seed=42)
    np.random.seed(999)  # o split não deve depender do gerador global
    second = data.make_splits(raw, seed=42)
    third = data.make_splits(raw, seed=43)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(first, name).indices, getattr(second, name).indices)
        np.testing.assert_array_equal(getattr(first, name).images, getattr(second, name).images)
    assert not np.array_equal(first.train.indices, third.train.indices)
    np.testing.assert_array_equal(first.test.images, third.test.images)


@pytest.mark.parametrize("options", [
    {"anomaly_class": -1}, {"anomaly_class": 10}, {"anomaly_class": 5.0},
    {"anomaly_class": True}, {"validation_fraction": 0},
    {"validation_fraction": 1}, {"validation_fraction": float("nan")}, {"seed": -1},
])
def test_invalid_split_options_fail(raw, options):
    with pytest.raises(ValueError):
        data.make_splits(raw, **options)


def test_rejects_already_normalized_or_misaligned_raw_arrays(raw):
    with pytest.raises(ValueError, match="uint8"):
        data.make_splits({**raw, "train_images": raw["train_images"].astype(np.float32)})
    with pytest.raises(ValueError, match="incompatível"):
        data.make_splits({**raw, "train_labels": raw["train_labels"][:-1]})
    bad_labels = raw["train_labels"].copy()
    bad_labels[0] = 10
    with pytest.raises(ValueError, match="Rótulos"):
        data.make_splits({**raw, "train_labels": bad_labels})


@pytest.mark.parametrize("shape", [(5,), (3, 28, 28)])
def test_idx_round_trip(tmp_path, shape):
    expected = np.arange(np.prod(shape), dtype=np.int64).astype(np.uint8).reshape(shape)
    path = tmp_path / "sample.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(bytes([0, 0, 8, len(shape)]))
        handle.write(struct.pack(">" + "I" * len(shape), *shape))
        handle.write(expected.tobytes())
    np.testing.assert_array_equal(data.read_idx(path), expected)


@pytest.mark.parametrize("payload", [
    b"bad", b"\0\0\x08\x01\0", b"\0\0\x08\x01" + struct.pack(">I", 2) + b"x",
    b"\0\0\x08\x01" + struct.pack(">I", 1) + b"xy",
    b"\0\0\x08\x01" + struct.pack(">I", 0),
])
def test_idx_rejects_invalid_or_truncated_payload(tmp_path, payload):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(payload)
    with pytest.raises(ValueError):
        data.read_idx(path)


def test_downloader_reuses_valid_cache_and_checks_downloads(tmp_path, monkeypatch):
    contents = {filename: filename.encode() for filename in data.FILES.values()}
    monkeypatch.setattr(data, "MD5", {name: hashlib.md5(blob).hexdigest() for name, blob in contents.items()})
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        return io.BytesIO(contents[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    data.download_data(tmp_path)
    assert len(requested) == 4
    data.download_data(tmp_path)
    assert len(requested) == 4  # sem nova requisição
    assert not list(tmp_path.glob("*.part"))
    for name, content in contents.items():
        assert (tmp_path / name).read_bytes() == content


def test_failed_download_never_becomes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupted"))
    with pytest.raises(ValueError, match="Checksum"):
        data.download_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_interrupted_download_removes_temporary_file(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(data, "urlopen", interrupted)
    with pytest.raises(OSError):
        data.download_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_corrupt_existing_cache_fails_without_overwriting(tmp_path, monkeypatch):
    filename = next(iter(data.FILES.values()))
    (tmp_path / filename).write_bytes(b"corrupted")

    def network_forbidden(*args, **kwargs):
        pytest.fail("Arquivo existente não deve ser baixado novamente")

    monkeypatch.setattr(data, "urlopen", network_forbidden)
    with pytest.raises(ValueError, match="Checksum"):
        data.download_data(tmp_path)
    assert (tmp_path / filename).read_bytes() == b"corrupted"


def test_loader_shapes_reproducibility_and_no_dropped_samples(raw):
    splits = data.make_splits(raw)
    first = data.make_dataloaders(splits, batch_size=64)
    torch.manual_seed(9876)
    second = data.make_dataloaders(splits, batch_size=64)
    for name in ("train", "val", "test"):
        batches = list(first[name])
        other_batches = list(second[name])
        assert sum(len(x) for x, y in batches) == len(getattr(splits, name).labels)
        for (images, labels), (other_images, other_labels) in zip(batches, other_batches):
            assert images.shape[1:] == (1, 28, 28)
            assert images.dtype == torch.float32 and labels.dtype == torch.int64
            assert torch.equal(images, other_images) and torch.equal(labels, other_labels)
        combined = torch.cat([images for images, _ in batches])[:, 0].numpy()
        if name == "train":
            assert not np.array_equal(combined, splits.train.images)
        else:
            np.testing.assert_array_equal(combined, getattr(splits, name).images)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"num_workers": -1}])
def test_invalid_loader_options(raw, options):
    with pytest.raises(ValueError):
        data.make_dataloaders(data.make_splits(raw), **options)


def test_yaml_config_resolves_project_path_and_selected_scenario(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "default.yaml"
    path.write_text("seed: 123\ndata:\n  directory: data\n  anomaly_class: 5\n", encoding="utf-8")
    monkeypatch.chdir(config_dir)
    config = data.load_config(path)
    assert config.directory == tmp_path / "data"
    assert config.anomaly_class == 5 and config.seed == 123
    assert config.validation_fraction == 0.15
