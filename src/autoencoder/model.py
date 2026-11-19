"""Autoencoder denso para imagens Fashion-MNIST achatadas."""

from pathlib import Path

import torch
from torch import nn
import yaml


class Autoencoder(nn.Module):
    """Rede densa 784 → 256 → 64 → latent → 64 → 256 → 784."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim deve ser um inteiro positivo")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    @classmethod
    def from_config(cls, path: str | Path = "configs/default.yaml") -> "Autoencoder":
        """Constrói o modelo usando `model.latent_dim` do YAML informado."""
        with Path(path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        model_config = config.get("model") if isinstance(config, dict) else None
        if not isinstance(model_config, dict) or "latent_dim" not in model_config:
            raise ValueError("A configuração deve conter model.latent_dim")
        return cls(latent_dim=model_config["latent_dim"])

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mapeia imagens (B, 784) ou (B, 1, 28, 28) ao gargalo linear."""
        return self.encoder(self._flatten(inputs))

    @staticmethod
    def _flatten(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2 and inputs.shape[1] == 784:
            return inputs
        if inputs.ndim == 4 and tuple(inputs.shape[1:]) == (1, 28, 28):
            return inputs.flatten(1)
        raise ValueError("inputs deve ter formato (B, 784) ou (B, 1, 28, 28)")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstrói entradas no intervalo [0, 1] a partir do espaço latente."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent deve ter formato (B, {self.latent_dim})")
        return self.decoder(latent)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(inputs))
        return reconstruction.reshape_as(inputs)

    def count_parameters(self) -> int:
        """Retorna o número de parâmetros treináveis."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE para intensidades contínuas normalizadas em [0, 1]."""
    return nn.functional.mse_loss(reconstruction, target)
