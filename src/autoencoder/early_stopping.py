import os
import torch


class EarlyStopping:
  """Monitora a loss de validação e interrompe o treinamento quando a perda para de cair,

  salvando o melhor checkpoint do modelo PyTorch.
  """

  def __init__(
      self, patience: int = 10, min_delta: float = 1e-5, verbose: bool = True
  ):
    self.patience = patience
    self.min_delta = min_delta
    self.verbose = verbose
    self.counter = 0
    self.best_loss = None
    self.early_stop = False
    self.best_epoch = 0

  def __call__(
      self,
      val_loss: float,
      model: torch.nn.Module,
      checkpoint_path: str,
      epoch: int,
  ):
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
        print(
            f"EarlyStopping: {self.counter}/{self.patience} épocas sem"
            " melhora."
        )
      if self.counter >= self.patience:
        self.early_stop = True

  def save_checkpoint(
      self, val_loss: float, model: torch.nn.Module, checkpoint_path: str
  ):
    dir_name = os.path.dirname(checkpoint_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)

    if self.verbose:
      print(
          f"Loss de validação reduziu para {val_loss:.6f}. Salvando checkpoint"
          f" em '{checkpoint_path}'."
      )
    torch.save(model.state_dict(), checkpoint_path)
