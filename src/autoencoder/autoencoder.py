import os
import torch
import torch.nn as nn
import yaml


class Autoencoder(nn.Module):
  """Autoencoder Denso Baseline para o Fashion-MNIST (Issue #4).

  Arquitetura:
    Encoder: 784 -> 256 -> 64 -> latent_dim (ReLU)
    Decoder: latent_dim -> 64 -> 256 -> 784 (ReLU, saída Sigmoid)
  """

  def __init__(self, latent_dim: int = 16):
    super().__init__()

    # Validação estrita da dimensão latente para os testes automatizados
    if (
        isinstance(latent_dim, bool)
        or not isinstance(latent_dim, int)
        or latent_dim <= 0
    ):
      raise ValueError(
          f"latent_dim deve ser um número inteiro positivo, recebido:"
          f" {latent_dim}"
      )

    self.latent_dim = latent_dim

    # Camadas do Encoder
    self.encoder = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, self.latent_dim),
    )

    # Camadas do Decoder
    self.decoder = nn.Sequential(
        nn.Linear(self.latent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
        nn.Sigmoid(),
    )

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Codifica a entrada de 784 dimensões para o espaço latente (latent_dim)."""
    return self.encoder(x)

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    """Decodifica a representação latente de volta para 784 dimensões."""
    return self.decoder(z)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Executa o fluxo completo de reconstrução (encode -> decode)."""
    latent = self.encode(x)
    reconstruction = self.decode(latent)
    return reconstruction

  def count_parameters(self) -> int:
    """Retorna a contagem total de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  @classmethod
  def from_config(cls, config_path: str = "configs/default.yaml"):
    """Instancia o Autoencoder lendo o 'latent_dim' do arquivo de configuração YAML."""
    if not os.path.exists(config_path):
      raise FileNotFoundError(
          f"Arquivo de configuração não encontrado em: {config_path}"
      )

    with open(config_path, "r", encoding="utf-8") as f:
      config = yaml.safe_load(f)

    latent_dim = config.get("latent_dim", 16)
    return cls(latent_dim=latent_dim)


def reconstruction_loss(
    inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
  """Calcula a perda de reconstrução por Erro Quadrático Médio (MSE Loss)

  entre as imagens reconstruídas e as originais em [7].
  """
  criterion = nn.MSELoss()
  return criterion(inputs, targets)
