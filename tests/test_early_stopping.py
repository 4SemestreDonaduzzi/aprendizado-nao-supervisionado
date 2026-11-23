import os
import pytest
import torch
import torch.nn as nn
from src.autoencoder.early_stopping import EarlyStopping


class DummyModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(1, 1)


def test_early_stopping_resets_counter_on_improvement(tmp_path):
  checkpoint_path = str(tmp_path / 'best_model.pt')
  model = DummyModel()
  es = EarlyStopping(patience=3, min_delta=1e-4, verbose=False)

  es(0.5, model, checkpoint_path, epoch=1)
  assert es.best_loss == 0.5
  assert os.path.exists(checkpoint_path)

  # Loss piora -> contador sobe
  es(0.6, model, checkpoint_path, epoch=2)
  assert es.counter == 1
  assert not es.early_stop

  # Loss melhora -> contador zera
  es(0.4, model, checkpoint_path, epoch=3)
  assert es.counter == 0
  assert es.best_loss == 0.4


def test_early_stopping_triggers_after_patience(tmp_path):
  checkpoint_path = str(tmp_path / 'best_model.pt')
  model = DummyModel()
  es = EarlyStopping(patience=2, min_delta=1e-4, verbose=False)

  es(0.5, model, checkpoint_path, epoch=1)
  es(0.55, model, checkpoint_path, epoch=2)
  es(0.55, model, checkpoint_path, epoch=3)

  assert es.early_stop is True
