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


def test_extreme_indices_orders_and_respects_mask():
    errors = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
    np.testing.assert_array_equal(reconstructions.extreme_indices(errors, 2, largest=True), [2, 4])
    np.testing.assert_array_equal(reconstructions.extreme_indices(errors, 2, largest=False), [1, 3])
    mask = np.array([True, False, False, True, True])
    np.testing.assert_array_equal(reconstructions.extreme_indices(errors, 5, largest=True, mask=mask), [4, 0, 3])
    with pytest.raises(ValueError):
        reconstructions.extreme_indices(errors, 0, largest=True)
    with pytest.raises(ValueError):
        reconstructions.extreme_indices(errors, 2, largest=True, mask=mask[:3])


def test_failure_indices_follow_threshold_rule():
    # Mesma regra de threshold_metrics: anomalia se erro > limiar (igualdade = normal).
    errors = np.array([0.1, 0.6, 0.9, 0.5, 0.2, 0.8, 0.5])
    is_anomaly = np.array([0, 0, 0, 0, 1, 1, 1])
    result = reconstructions.failure_indices(errors, is_anomaly, threshold=0.5, k=10)
    np.testing.assert_array_equal(result["fp"], [2, 1])  # normais acima, maior erro primeiro
    np.testing.assert_array_equal(result["fn"], [4, 6])  # anomalias no limiar ou abaixo, menor primeiro
    pred = errors > 0.5
    assert len(result["fp"]) == np.sum(pred & (is_anomaly == 0))
    assert len(result["fn"]) == np.sum(~pred & (is_anomaly == 1))


def test_reconstruction_grid_layout_and_error_scale(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.random((4, 28, 28)).astype(np.float32)
    recon = np.clip(images + rng.normal(0, 0.1, images.shape), 0, 1).astype(np.float32)
    np.testing.assert_allclose(reconstructions.error_maps(images, recon), np.abs(images - recon))
    fig = reconstructions.plot_reconstruction_grid(images, recon, [f"#{i}" for i in range(4)],
                                                   title="grade", error_vmax=0.5)
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 12  # 3 linhas x 4 exemplos (a barra de cores é à parte)
    error_panels = [ax for ax in panels if ax.images[0].get_cmap().name == "inferno"]
    assert len(error_panels) == 4
    assert all(ax.images[0].get_clim() == (0, 0.5) for ax in error_panels)
    path = tmp_path / "grid.png"
    fig.savefig(path)
    plt.close(fig)
    assert path.stat().st_size > 0
    with pytest.raises(ValueError):
        reconstructions.plot_reconstruction_grid(images, recon, captions=["só um"])
    with pytest.raises(ValueError):
        reconstructions.error_maps(images, recon[:2])
