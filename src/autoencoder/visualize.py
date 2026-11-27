"""Projeções 2D (PCA e t-SNE), vizinhanças por classe e gráficos.

As funções recebem matrizes, não o modelo: pixels brutos (N, 28, 28) ou
(N, 784), ou a saída de ``encode()`` com formato (N, dimensão_latente).
Para obter os latentes de um checkpoint, use ``inference``.

- PCA é linear e preserva a maior variância possível em 2D, mas descarta o
  resto. Distâncias na figura são aproximadas: pontos próximos na projeção
  podem estar distantes no espaço original.
- t-SNE é não linear e preserva vizinhanças locais. Tamanhos de grupos e
  distâncias entre grupos NÃO devem ser interpretados. Use uma amostra de
  2 a 3 mil pontos, senão a execução fica lenta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from .data import CLASS_NAMES

NORMAL_COLOR = "#9aa5b1"
# Magenta: fora da paleta tab10 usada em plot_by_class, para que a cor da
# anomalia não se confunda com nenhuma classe (o vermelho da tab10 é Dress).
ANOMALY_COLOR = "#e600e6"


def as_matrix(features) -> np.ndarray:
    """Converte (N, ...) em uma matriz float32 (N, D), achatando imagens."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim < 2 or len(matrix) == 0:
        raise ValueError("features deve ter formato (N, ...) com N > 0")
    matrix = matrix.reshape(len(matrix), -1)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features contém NaN ou infinito")
    return matrix


def _check_labels(labels, n: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.shape != (n,):
        raise ValueError(f"labels deve ter formato ({n},), recebido {labels.shape}")
    return labels


def stratified_sample(labels, n_samples: int, seed: int = 42) -> np.ndarray:
    """Índices ordenados de uma amostra estratificada por classe, sem reposição.

    Mantém a proporção de cada classe (inclusive a anômala), para que a
    amostra usada no t-SNE represente o conjunto inteiro. Se n_samples for
    maior ou igual ao total, devolve todos os índices.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels deve ser unidimensional")
    if n_samples < 1:
        raise ValueError("n_samples deve ser positivo")
    if n_samples >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    # Maiores restos recebem as vagas que sobram do arredondamento para baixo.
    quotas = counts * n_samples / len(labels)
    take = np.floor(quotas).astype(int)
    remainder = n_samples - take.sum()
    take[np.argsort(-(quotas - take), kind="stable")[:remainder]] += 1
    chosen = [
        rng.choice(np.flatnonzero(labels == cls), size=k, replace=False)
        for cls, k in zip(classes, take)
    ]
    return np.sort(np.concatenate(chosen))


def project_pca(features, n_components: int = 2, seed: int = 42):
    """Projeta com PCA. Devolve (coordenadas (N, n_components), variância explicada)."""
    matrix = as_matrix(features)
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(matrix)
    return coords.astype(np.float32), pca.explained_variance_ratio_


def project_tsne(
    features,
    perplexity: float = 30.0,
    seed: int = 42,
    pca_components: int | None = 50,
) -> np.ndarray:
    """Projeta com t-SNE em 2D. Passe uma amostra (ver ``stratified_sample``).

    Com muitas dimensões (ex.: 784 pixels), reduz antes para ``pca_components``
    com PCA, prática padrão que acelera e remove ruído. Matrizes já pequenas,
    como o espaço latente, seguem direto para o t-SNE.
    """
    matrix = as_matrix(features)
    if len(matrix) <= perplexity:
        raise ValueError("t-SNE exige mais pontos do que a perplexity")
    if pca_components is not None and matrix.shape[1] > pca_components:
        matrix = PCA(n_components=pca_components, random_state=seed).fit_transform(matrix)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed)
    return tsne.fit_transform(matrix).astype(np.float32)


def _new_axes(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    return ax.figure, ax


def _finish(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_by_class(
    coords, labels, title: str = "", ax=None, class_names=CLASS_NAMES,
    xlabel: str = "Componente 1", ylabel: str = "Componente 2",
    point_size: float = 4,
):
    """Dispersão 2D colorida pelas 10 classes originais."""
    coords = np.asarray(coords)
    labels = _check_labels(labels, len(coords))
    fig, ax = _new_axes(ax)
    cmap = plt.get_cmap("tab10")
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=point_size, alpha=0.6,
            color=cmap(int(cls) % 10), label=class_names[int(cls)], rasterized=True,
        )
    ax.legend(markerscale=4, fontsize=8, loc="best", frameon=True)
    _finish(ax, title, xlabel, ylabel)
    return fig


def plot_normal_vs_anomaly(
    coords, labels, anomaly_class: int, title: str = "", ax=None,
    class_names=CLASS_NAMES,
    xlabel: str = "Componente 1", ylabel: str = "Componente 2",
    point_size: float = 4,
):
    """Dispersão 2D: classes normais em cinza e a classe anômala em destaque.

    A anomalia é desenhada por último, para não ficar escondida sob os normais.
    """
    coords = np.asarray(coords)
    labels = _check_labels(labels, len(coords))
    fig, ax = _new_axes(ax)
    anomaly = labels == anomaly_class
    ax.scatter(
        coords[~anomaly, 0], coords[~anomaly, 1], s=point_size, alpha=0.4,
        color=NORMAL_COLOR, label="Normal", rasterized=True,
    )
    ax.scatter(
        coords[anomaly, 0], coords[anomaly, 1], s=point_size * 1.5, alpha=0.8,
        color=ANOMALY_COLOR, label=f"Anomalia ({class_names[anomaly_class]})",
        rasterized=True,
    )
    ax.legend(markerscale=4, fontsize=9, loc="best", frameon=True)
    _finish(ax, title, xlabel, ylabel)
    return fig


def plot_projection_panel(
    coords, labels, anomaly_class: int, method: str, source: str,
    class_names=CLASS_NAMES, axis_labels=None,
):
    """Figura com dois painéis lado a lado: por classe e normal vs. anomalia."""
    xlabel, ylabel = axis_labels or (f"{method} 1", f"{method} 2")
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6))
    plot_by_class(coords, labels, "Por classe", ax=left, class_names=class_names,
                  xlabel=xlabel, ylabel=ylabel)
    plot_normal_vs_anomaly(coords, labels, anomaly_class, "Normal vs. anomalia",
                           ax=right, class_names=class_names, xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(f"{method} sobre {source} (n = {len(coords)})")
    fig.tight_layout()
    return fig


def neighbor_class_matrix(features, labels, k: int = 10, classes=None):
    """Composição dos k vizinhos mais próximos, medida no espaço ORIGINAL.

    ``matrix[i, j]`` é a fração dos vizinhos de pontos da classe ``classes[i]``
    que pertencem à classe ``classes[j]``; cada linha soma 1. A diagonal é a
    pureza da vizinhança; a linha da anomalia mostra com quem ela se mistura.
    Ao contrário de uma figura 2D, esta medida não sofre distorção da projeção.
    Cada ponto é excluído da própria vizinhança.
    """
    matrix = as_matrix(features)
    labels = _check_labels(labels, len(matrix))
    if not 1 <= k < len(matrix):
        raise ValueError("k deve estar entre 1 e N - 1")
    classes = np.unique(labels) if classes is None else np.asarray(classes)
    position = {int(c): i for i, c in enumerate(classes)}
    if set(np.unique(labels).tolist()) - position.keys():
        raise ValueError("classes não contém todos os rótulos presentes")
    # kneighbors() sem argumentos exclui o próprio ponto de cada consulta.
    neighbors = NearestNeighbors(n_neighbors=k).fit(matrix).kneighbors(return_distance=False)
    rows = np.array([position[int(c)] for c in labels])
    counts = np.zeros((len(classes), len(classes)))
    np.add.at(counts, (np.repeat(rows, k), rows[neighbors].ravel()), 1)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0), classes


def plot_neighbor_matrix(matrix, classes, title: str = "", ax=None, class_names=CLASS_NAMES,
                         anomaly_class: int | None = None):
    """Mapa de calor de ``neighbor_class_matrix`` com os percentuais anotados."""
    fig, ax = (plt.subplots(figsize=(7.5, 6.5)) if ax is None else (ax.figure, ax))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    names = [class_names[int(c)] + (" *" if c == anomaly_class else "") for c in classes]
    ax.set_xticks(range(len(classes)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), names)
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = matrix[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.55 else "#1e293b")
    ax.set_xlabel("Classe do vizinho")
    ax.set_ylabel("Classe do ponto")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, format=PercentFormatter(1.0, decimals=0))
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    """Salva a figura (criando a pasta) e fecha para liberar memória."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
