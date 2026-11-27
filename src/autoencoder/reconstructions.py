"""Reconstruções por imagem, seleção de falhas e grades original | reconstruído | erro.

``reconstruct_dataset`` faz só inferência (modelo em ``eval()``, sem gradiente) e
preserva a ordem do loader: validação e teste não embaralham, então o índice
``i`` de qualquer vetor devolvido corresponde à imagem ``i`` do split.
As demais funções recebem arrays, não o modelo.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def reconstruct_dataset(model, loader, device="cpu") -> dict[str, np.ndarray]:
    """Originais, latentes, reconstruções e erro MSE por imagem, na ordem do loader.

    ``images`` e ``reconstructions`` têm formato (N, 28, 28) em [0, 1];
    ``latent`` tem formato (N, latent_dim). O MSE usa as mesmas operações de
    ``evaluate.reconstruction_errors``, portanto é o mesmo escore do limiar.
    """
    model.eval()
    originals, latents, reconstructions, mse, labels = [], [], [], [], []
    for images, y in loader:
        images = images.to(device)
        latent = model.encode(images)
        output = model.decode(latent).reshape_as(images)  # igual a model(images)
        # Mesmas operações de evaluate.reconstruction_errors: valores idênticos,
        # sem diferenças de arredondamento perto do limiar.
        mse.append(((output - images).flatten(1) ** 2).mean(dim=1).cpu())
        originals.append(images.cpu())
        latents.append(latent.cpu())
        reconstructions.append(output.cpu())
        labels.append(y)
    return {
        "images": torch.cat(originals).squeeze(1).numpy(),
        "latent": torch.cat(latents).numpy(),
        "reconstructions": torch.cat(reconstructions).squeeze(1).numpy(),
        "mse": torch.cat(mse).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def _check_mask(mask, n: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != (n,):
        raise ValueError(f"a máscara deve ter formato ({n},), recebido {mask.shape}")
    return mask.astype(bool)


def error_maps(images, reconstructions) -> np.ndarray:
    """Erro absoluto por pixel ``|x − x̂|``, no mesmo formato das imagens."""
    images, reconstructions = np.asarray(images), np.asarray(reconstructions)
    if images.shape != reconstructions.shape:
        raise ValueError("images e reconstructions devem ter o mesmo formato")
    return np.abs(images - reconstructions)


def extreme_indices(errors, k: int, largest: bool, mask=None) -> np.ndarray:
    """Índices dos ``k`` maiores (ou menores) erros, do mais extremo ao menos.

    ``mask`` restringe a busca a um subconjunto (ex.: só uma classe). Se houver
    menos de ``k`` candidatos, devolve todos. Empates seguem a ordem original.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1:
        raise ValueError("errors deve ser um vetor 1D")
    if k < 1:
        raise ValueError("k deve ser positivo")
    candidates = np.arange(len(errors)) if mask is None else np.flatnonzero(_check_mask(mask, len(errors)))
    order = np.argsort(-errors[candidates] if largest else errors[candidates], kind="stable")
    return candidates[order[:k]]


def failure_indices(errors, is_anomaly, threshold: float, k: int) -> dict[str, np.ndarray]:
    """Falsos positivos e falsos negativos mais extremos para um limiar.

    Usa a mesma regra de ``evaluate.threshold_metrics``: anomalia se
    ``erro > limiar``. ``fp`` são normais acima do limiar, do maior erro para o
    menor (alarmes falsos mais "convictos"); ``fn`` são anomalias no limiar ou
    abaixo, do menor erro para o maior (anomalias mais bem escondidas).
    """
    errors = np.asarray(errors, dtype=float)
    is_anomaly = _check_mask(is_anomaly, len(errors))
    flagged = errors > threshold
    return {
        "fp": extreme_indices(errors, k, largest=True, mask=flagged & ~is_anomaly),
        "fn": extreme_indices(errors, k, largest=False, mask=~flagged & is_anomaly),
    }


def plot_reconstruction_grid(images, reconstructions, captions=None, title: str = "",
                             error_vmax: float | None = None):
    """Grade com uma coluna por exemplo: original | reconstruído | mapa de erro.

    Originais e reconstruções usam a escala fixa [0, 1]. Os mapas ``|x − x̂|``
    compartilham ``error_vmax`` (padrão: o maior erro exibido); passe o mesmo
    valor para várias figuras quando elas precisarem ser comparáveis.
    """
    images = np.asarray(images).reshape(-1, 28, 28)
    reconstructions = np.asarray(reconstructions).reshape(-1, 28, 28)
    maps = error_maps(images, reconstructions)
    n = len(images)
    if n == 0:
        raise ValueError("Nenhum exemplo para exibir")
    if captions is not None and len(captions) != n:
        raise ValueError("captions deve ter um texto por exemplo")
    vmax = float(maps.max()) if error_vmax is None else float(error_vmax)

    fig, axes = plt.subplots(3, n, figsize=(1.45 * n + 1.2, 5.0), squeeze=False)
    rows = (("Original", images, "gray", 1.0), ("Reconstruído", reconstructions, "gray", 1.0),
            ("|x − x̂|", maps, "inferno", vmax))
    for r, (label, data, cmap, top) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            shown = ax.imshow(data[c], cmap=cmap, vmin=0, vmax=top)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0 and captions is not None:
                ax.set_title(captions[c], fontsize=7.5)
    # Eixo próprio para a barra: assim ela não encolhe a última coluna da grade.
    fig.subplots_adjust(right=0.9)
    axes[2, -1].apply_aspect()  # posição real da imagem, já com proporção 1:1
    row = axes[2, -1].get_position()
    cax = fig.add_axes([0.915, row.y0, 0.012, row.height])
    fig.colorbar(shown, cax=cax, label="erro absoluto")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig
