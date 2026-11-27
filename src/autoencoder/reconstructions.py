"""Reconstruções por imagem a partir de um Autoencoder treinado.

``reconstruct_dataset`` faz só inferência (modelo em ``eval()``, sem gradiente) e
preserva a ordem do loader: validação e teste não embaralham, então o índice
``i`` de qualquer vetor devolvido corresponde à imagem ``i`` do split.
"""

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

