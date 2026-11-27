"""Testes offline da extração de latentes a partir de um checkpoint."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder.inference import encode_dataset, load_trained_model
from autoencoder.model import Autoencoder


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    model = Autoencoder(latent_dim=4)
    checkpoint = tmp_path / "ae.pt"
    torch.save(model.state_dict(), checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  latent_dim: 4\n", encoding="utf-8")
    images = torch.rand(37, 1, 28, 28)  # 37 não é múltiplo do lote: testa o último lote
    labels = torch.arange(37) % 10
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)
    return model, checkpoint, config, loader, images, labels


def test_load_trained_model_restores_weights_in_eval_mode(setup):
    model, checkpoint, config, *_ = setup
    loaded = load_trained_model(config, checkpoint)
    assert not loaded.training
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(expected, actual)


def test_load_trained_model_explains_missing_checkpoint(setup, tmp_path):
    _, _, config, *_ = setup
    with pytest.raises(FileNotFoundError, match="train_cenarios"):
        load_trained_model(config, tmp_path / "inexistente.pt")


def test_encode_dataset_keeps_order_and_matches_encode(setup):
    model, _, _, loader, images, labels = setup
    result = encode_dataset(model, loader)
    assert result["latent"].shape == (37, 4)
    np.testing.assert_array_equal(result["labels"], labels.numpy())
    with torch.no_grad():
        np.testing.assert_allclose(result["latent"], model.encode(images).numpy(), rtol=1e-5, atol=1e-6)

