import json
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yaml

# Imports usando a estrutura de pacote oficial do repositório
from autoencoder.data import load_config, load_splits, make_dataloaders
from autoencoder.early_stopping import EarlyStopping
from autoencoder.model import Autoencoder, reconstruction_loss
from autoencoder.reproducibility import set_seed


def read_training_options(config_path: str = 'configs/default.yaml') -> dict:
  """Lê o YAML diretamente para extrair o dicionário da seção 'training:'."""
  if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
      content = yaml.safe_load(f) or {}
      return content.get('training', {})
  return {}


def train():
  config_path = 'configs/default.yaml'

  # Carrega DataConfig e os parâmetros de treino
  data_config = load_config(config_path)
  train_cfg = read_training_options(config_path)

  learning_rate = train_cfg.get('learning_rate', 0.001)
  patience = train_cfg.get('patience', 10)
  min_delta = train_cfg.get('min_delta', 1e-5)
  max_epochs = train_cfg.get('max_epochs', 100)

  # Define a semente oficial de reproduzibilidade
  set_seed(data_config.seed)

  start_time = time.time()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device_name = (
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
  )

  print(f'Treinando no dispositivo: {device} ({device_name})')

  # Carrega os DataLoaders oficiais usando o batch_size da config (128)
  splits = load_splits(data_config)
  dataloaders = make_dataloaders(splits, batch_size=data_config.batch_size)
  train_loader = dataloaders['train']
  val_loader = dataloaders['val']

  # Instancia modelo oficial da main
  model = Autoencoder.from_config(config_path).to(device)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  early_stopping = EarlyStopping(
      patience=patience, min_delta=min_delta, verbose=True
  )
  checkpoint_path = 'models/best_autoencoder.pt'

  history = {'train_loss': [], 'val_loss': []}

  print('\nIniciando o loop de treinamento...')

  for epoch in range(1, max_epochs + 1):
    # FASE DE TREINO
    model.train()
    running_train_loss = 0.0

    for inputs, _ in train_loader:
      inputs = inputs.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = reconstruction_loss(outputs, inputs)
      loss.backward()
      optimizer.step()

      running_train_loss += loss.item() * inputs.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)

    # FASE DE VALIDAÇÃO
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
      for inputs, _ in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = reconstruction_loss(outputs, inputs)
        running_val_loss += loss.item() * inputs.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(
        f'Época [{epoch:03d}/{max_epochs:03d}] - Train Loss: {train_loss:.6f} -'
        f' Val Loss: {val_loss:.6f}'
    )

    early_stopping(val_loss, model, checkpoint_path, epoch)
    if early_stopping.early_stop:
      print(f'\nEarly Stopping ativado na época {epoch}.')
      break

  elapsed_time = time.time() - start_time

  # Registro completo de histórico (com hardware e tempo de execução)
  history_data = {
      'train_loss': history['train_loss'],
      'val_loss': history['val_loss'],
      'best_epoch': early_stopping.best_epoch,
      'best_val_loss': early_stopping.best_loss,
      'training_time_seconds': round(elapsed_time, 2),
      'hardware_used': device_name,
  }

  history_path = 'reports/history.json'
  os.makedirs(os.path.dirname(history_path), exist_ok=True)
  with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history_data, f, indent=4)

  # Geração e salvamento do gráfico das curvas de perda
  figure_path = 'reports/figures/loss_curves.png'
  os.makedirs(os.path.dirname(figure_path), exist_ok=True)
  epochs_range = range(1, len(history['train_loss']) + 1)

  plt.figure(figsize=(9, 5))
  plt.plot(
      epochs_range,
      history['train_loss'],
      label='Train Loss',
      color='#1f77b4',
      linewidth=2,
  )
  plt.plot(
      epochs_range,
      history['val_loss'],
      label='Validation Loss',
      color='#ff7f0e',
      linewidth=2,
  )
  plt.axvline(
      x=early_stopping.best_epoch,
      color='red',
      linestyle='--',
      label=f'Melhor Checkpoint (Época {early_stopping.best_epoch})',
  )
  plt.title('Curvas de Aprendizado - Autoencoder Baseline')
  plt.xlabel('Épocas')
  plt.ylabel('MSE Loss')
  plt.legend()
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.tight_layout()
  plt.savefig(figure_path, dpi=300)
  plt.close()

  print(
      f'\nTreinamento concluído. Melhor época: {early_stopping.best_epoch} com'
      f' Val Loss: {early_stopping.best_loss:.6f}'
  )


if __name__ == '__main__':
  train()
