import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import yaml

# ========================================== 
# 1. DEFINIÇÃO DAS CLASSES E FUNÇÕES
# ==========================================

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: torch.nn.Module, checkpoint_path: str, epoch: int):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, checkpoint_path)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(val_loss, model, checkpoint_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} épocas sem melhora.")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, checkpoint_path: str):
        dir_name = os.path.dirname(checkpoint_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if self.verbose:
            print(f"Loss de validação reduziu para {val_loss:.6f}. Salvando checkpoint em '{checkpoint_path}'.")
        torch.save(model.state_dict(), checkpoint_path)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError(f"latent_dim deve ser um número inteiro positivo, recebido: {latent_dim}")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction

    @classmethod
    def from_config(cls, config_path: str = "configs/default.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado em: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        latent_dim = config.get("latent_dim", 16)
        return cls(latent_dim=latent_dim)


def reconstruction_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    criterion = nn.MSELoss()
    return criterion(inputs, targets)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de config não encontrado: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataloaders(config_path: str = "configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("directory", "data")
    anomaly_class = data_cfg.get("anomaly_class", 8)
    val_fraction = data_cfg.get("validation_fraction", 0.15)
    batch_size = data_cfg.get("batch_size", 128)
    num_workers = data_cfg.get("num_workers", 0)
    seed = config.get("seed", 42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    full_train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_targets = full_train_dataset.targets
    normal_indices = (train_targets != anomaly_class).nonzero(as_tuple=True)[0]

    generator = torch.Generator().manual_seed(seed)
    num_normal = len(normal_indices)
    num_val = int(num_normal * val_fraction)
    num_train = num_normal - num_val

    permuted_indices = normal_indices[torch.randperm(num_normal, generator=generator)]
    train_indices = permuted_indices[:num_train]
    val_indices = permuted_indices[num_train:]

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# ==========================================
# 2. ROTINA DE TREINAMENTO
# ==========================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    config_dir = 'configs'
    config_path = f'{config_dir}/default.yaml'
    os.makedirs(config_dir, exist_ok=True)
    if not os.path.exists(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("""
seed: 42
latent_dim: 16
learning_rate: 0.001
patience: 5
min_delta: 0.00001
max_epochs: 10
checkpoint_path: 'models/best_autoencoder.pt'
history_path: 'reports/history.json'
figure_path: 'reports/figures/loss_curves.png'
data:
  directory: 'data'
  anomaly_class: 8
  validation_fraction: 0.15
  batch_size: 128
  num_workers: 0
""")

    config = load_config(config_path)
    seed = config.get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Treinando no dispositivo: {device}')

    train_loader, val_loader, _ = get_dataloaders(config_path)
    model = Autoencoder.from_config(config_path).to(device)

    lr = config.get('learning_rate', 0.001)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 1e-5)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)

    checkpoint_path = config.get('checkpoint_path', 'models/best_autoencoder.pt')
    max_epochs = config.get('max_epochs', 100)

    history = {'train_loss': [], 'val_loss': []}
    print('\nIniciando o loop de treinamento...')

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_train_loss = 0.0
        for inputs in train_loader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = reconstruction_loss(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = reconstruction_loss(outputs, inputs)
                running_val_loss += loss.item() * inputs.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Época [{epoch:03d}/{max_epochs:03d}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}')
        early_stopping(val_loss, model, checkpoint_path, epoch)
        if early_stopping.early_stop:
            print(f'\nTreinamento interrompido via Early Stopping na época {epoch}.')
            break

    history_path = config.get('history_path', 'reports/history.json')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)
    print(f"Histórico de treino salvo em '{history_path}'.")

    figure_path = config.get('figure_path', 'reports/figures/loss_curves.png')
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.axvline(x=early_stopping.best_epoch, color='red', linestyle='--', label=f'Melhor Checkpoint (Época {early_stopping.best_epoch})')
    plt.title('Curvas de Aprendizado - Autoencoder (Loss de Reconstrução)')
    plt.xlabel('Épocas')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()
    print(f"Gráfico de curvas de loss salvo em '{figure_path}'.")
    print('\nTreinamento concluído com sucesso!')

if __name__ == '__main__':
    train()
