"""Testes offline das projeções e gráficos, com dados sintéticos."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from autoencoder import visualize  # noqa: E402


@pytest.fixture
def blobs():
    """Três grupos bem separados em 20 dimensões; rótulos 0, 1 e 8."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 10, (3, 20))
    labels = np.repeat([0, 1, 8], 40)
    features = centers[np.repeat(np.arange(3), 40)] + rng.normal(0, 0.5, (120, 20))
    return features.astype(np.float32), labels


def test_as_matrix_flattens_images_and_rejects_bad_input():
    assert visualize.as_matrix(np.zeros((5, 28, 28))).shape == (5, 784)
    assert visualize.as_matrix(np.zeros((5, 32))).shape == (5, 32)
    with pytest.raises(ValueError):
        visualize.as_matrix(np.zeros(5))
    with pytest.raises(ValueError):
        visualize.as_matrix(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        visualize.as_matrix(np.array([[1.0, np.nan]]))


def test_stratified_sample_keeps_proportions_and_is_reproducible():
    labels = np.repeat(np.arange(10), 1000)  # mesmo balanço do teste oficial
    sample = visualize.stratified_sample(labels, 3000, seed=1)
    assert len(sample) == 3000 and np.unique(sample).size == 3000
    np.testing.assert_array_equal(np.bincount(labels[sample]), np.full(10, 300))
    np.testing.assert_array_equal(sample, visualize.stratified_sample(labels, 3000, seed=1))
    assert not np.array_equal(sample, visualize.stratified_sample(labels, 3000, seed=2))


def test_stratified_sample_exact_size_with_unbalanced_classes():
    labels = np.array([0] * 7 + [1] * 2 + [8] * 1)
    for n in range(1, 10):
        sample = visualize.stratified_sample(labels, n)
        assert len(sample) == n and np.unique(sample).size == n
    np.testing.assert_array_equal(visualize.stratified_sample(labels, 50), np.arange(10))


def test_pca_shape_and_variance(blobs):
    features, _ = blobs
    coords, variance = visualize.project_pca(features)
    assert coords.shape == (120, 2)
    assert variance.shape == (2,) and 0 < variance.sum() <= 1
    np.testing.assert_allclose(coords, visualize.project_pca(features)[0])


def test_tsne_separates_clear_groups(blobs):
    features, labels = blobs
    coords = visualize.project_tsne(features, perplexity=10, pca_components=5)
    assert coords.shape == (120, 2) and np.all(np.isfinite(coords))
    # Cada ponto deve estar mais perto do centro do seu grupo que dos outros.
    centers = np.stack([coords[labels == c].mean(0) for c in (0, 1, 8)])
    nearest = np.argmin(((coords[:, None] - centers[None]) ** 2).sum(-1), axis=1)
    np.testing.assert_array_equal(np.array([0, 1, 8])[nearest], labels)


def test_tsne_rejects_too_few_points():
    with pytest.raises(ValueError):
        visualize.project_tsne(np.zeros((10, 3)), perplexity=30)


def test_plots_and_save(blobs, tmp_path):
    features, labels = blobs
    coords, _ = visualize.project_pca(features)

    fig = visualize.plot_by_class(coords, labels, "classe")
    assert len(fig.axes[0].collections) == 3
    plt.close(fig)

    fig = visualize.plot_normal_vs_anomaly(coords, labels, anomaly_class=8)
    normal, anomaly = fig.axes[0].collections
    assert len(normal.get_offsets()) == 80 and len(anomaly.get_offsets()) == 40
    plt.close(fig)

    fig = visualize.plot_projection_panel(coords, labels, 8, "PCA", "teste")
    path = visualize.save_figure(fig, tmp_path / "sub" / "panel.png")
    assert path.exists() and path.stat().st_size > 0
    assert not plt.get_fignums()


def test_plots_reject_mismatched_labels(blobs):
    features, labels = blobs
    with pytest.raises(ValueError):
        visualize.plot_by_class(features[:, :2], labels[:-1])
