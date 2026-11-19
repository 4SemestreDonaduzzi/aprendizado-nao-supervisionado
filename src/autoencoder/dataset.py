import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
  """Carrega as configurações do arquivo YAML."""
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Arquivo de config não encontrado: {config_path}")
  with open(config_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def get_dataloaders(config_path: str = "configs/default.yaml"):
  """Carrega e pré-processa o dataset Fashion-MNIST seguindo a estratégia One-Class Learning (Issue #2).

  Retorna:
      train_loader: DataLoader de treino contendo apenas a classe normal (~85% dos dados normais).
      val_loader: DataLoader de validação contendo apenas a classe normal (~15% dos dados normais).
      test_loader: DataLoader de teste contendo classes normais e anômalas (10k imagens).
  """
  config = load_config(config_path)

  # Extrai configurações do bloco 'data' e da raiz do YAML
  data_cfg = config.get("data", {})
  data_dir = data_cfg.get("directory", "data")
  anomaly_class = data_cfg.get("anomaly_class", 8)
  val_fraction = data_cfg.get("validation_fraction", 0.15)
  batch_size = data_cfg.get("batch_size", 128)
  num_workers = data_cfg.get("num_workers", 0)
  seed = config.get("seed", 42)

  # Transformação: Converte para Tensor (pixels em [0, 1]) e achata para 784 dimensões
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.view(-1)),  # Reshape (1, 28, 28) -> (784,)
  ])

  # Download e carregamento oficial do Fashion-MNIST na pasta data/
  full_train_dataset = torchvision.datasets.FashionMNIST(
      root=data_dir, train=True, download=True, transform=transform
  )
  test_dataset = torchvision.datasets.FashionMNIST(
      root=data_dir, train=False, download=True, transform=transform
  )

  # 1. Filtragem One-Class: Seleciona apenas os índices das classes normais (diferentes da anomaly_class)
  train_targets = full_train_dataset.targets
  normal_indices = (train_targets != anomaly_class).nonzero(as_tuple=True)[0]

  # 2. Divisão estratificada e reproduzível entre Treino e Validação usando a semente fixa
  generator = torch.Generator().manual_seed(seed)
  num_normal = len(normal_indices)
  num_val = int(num_normal * val_fraction)
  num_train = num_normal - num_val

  # Embaralha os índices normais de forma estritamente reproduzível
  permuted_indices = normal_indices[
      torch.randperm(num_normal, generator=generator)
  ]
  train_indices = permuted_indices[:num_train]
  val_indices = permuted_indices[num_train:]

  # Subsets de treino e validação (somente imagens normais)
  train_subset = Subset(full_train_dataset, train_indices)
  val_subset = Subset(full_train_dataset, val_indices)

  # 3. Criação dos DataLoaders PyTorch
  train_loader = DataLoader(
      train_subset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
  )
  val_loader = DataLoader(
      val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
  )

  return train_loader, val_loader, test_loader
