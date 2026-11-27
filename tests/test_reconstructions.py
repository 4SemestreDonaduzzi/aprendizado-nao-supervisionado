"""Testes offline das reconstruções, da seleção de falhas e das grades."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from autoencoder import reconstructions  # noqa: E402
from autoencoder.evaluate import reconstruction_errors  # noqa: E402
from autoencoder.model import Autoencoder  # noqa: E402


def test_reconstruct_dataset_keeps_order_and_matches_evaluation_scores():
    torch.manual_seed(0)
    model = Autoencoder(latent_dim=4)
    images = torch.rand(37, 1, 28, 28)  # 37 não é múltiplo do lote: testa o último lote
    labels = torch.arange(37) % 10
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)

    result = reconstructions.reconstruct_dataset(model, loader)
    assert result["images"].shape == result["reconstructions"].shape == (37, 28, 28)
    assert result["latent"].shape == (37, 4)
    np.testing.assert_array_equal(result["images"], images.squeeze(1).numpy())
    np.testing.assert_array_equal(result["labels"], labels.numpy())
    with torch.no_grad():
        np.testing.assert_allclose(result["latent"], model.encode(images).numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result["reconstructions"], model(images).squeeze(1).numpy(), rtol=1e-5, atol=1e-6)
    # O escore precisa ser idêntico ao da avaliação para FP/FN baterem com o limiar.
    np.testing.assert_array_equal(result["mse"], reconstruction_errors(model, loader)["mse"])
    assert not model.training
