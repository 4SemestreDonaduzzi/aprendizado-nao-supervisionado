"""Carrega checkpoints treinados e extrai os vetores latentes de um loader.

Estas funções só fazem inferência: o modelo fica em ``eval()`` e sem gradiente.
A ordem das saídas é a mesma do loader (validação e teste não embaralham), então
o índice ``i`` de qualquer vetor devolvido corresponde à imagem ``i`` do split.
"""

from pathlib import Path

import numpy as np
import torch

from .model import Autoencoder

# nome do cenário -> (classe anômala, checkpoint treinado SEM essa classe)
SCENARIOS = {
    "bag": (8, "models/ae_bag.pt"),
    "sandal": (5, "models/ae_sandal.pt"),
}


def load_trained_model(config_path: str | Path, checkpoint: str | Path, device="cpu") -> Autoencoder:
    """Recria a arquitetura do YAML e carrega os pesos salvos no treino."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint}\n"
            "Treine o cenário antes: python scripts/train_cenarios.py --scenario <bag|sandal>"
        )
    model = Autoencoder.from_config(config_path)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def encode_dataset(model: Autoencoder, loader, device="cpu") -> dict[str, np.ndarray]:
    """Vetores latentes ``encode(x)`` de todas as imagens, na ordem do loader.

    Devolve ``latent`` (N, latent_dim) e ``labels`` (N,) com as classes originais.
    """
    model.eval()
    latents, labels = [], []
    for images, y in loader:
        latents.append(model.encode(images.to(device)).cpu())
        labels.append(y)
    return {"latent": torch.cat(latents).numpy(), "labels": torch.cat(labels).numpy()}

