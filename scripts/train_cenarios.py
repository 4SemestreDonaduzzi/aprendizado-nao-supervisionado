"""Treina o Autoencoder de um cenário (Bag ou Sandal).

treina só com imagens normais e guarda o melhor modelo (menor loss de validação).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from autoencoder.data import load_config, load_splits, make_dataloaders  # noqa: E402
from autoencoder.early_stopping import EarlyStopping  # noqa: E402
from autoencoder.model import Autoencoder, reconstruction_loss  # noqa: E402
from autoencoder.reproducibility import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# cenário -> classe anômala (que NÃO aparece no treino nem na validação)
SCENARIOS = {"bag": 8, "sandal": 5}


def read_training_options(config_path: Path) -> dict:
    """Lê a seção 'training' do YAML, com valores padrão seguros."""
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    options = {"learning_rate": 1e-3, "patience": 5, "min_delta": 1e-5, "max_epochs": 50}
    options.update(raw.get("training", {}))
    return options


def average_loss(model, loader, device, optimizer=None) -> float:
    """Loss média por imagem. Com optimizer treina; sem ele só avalia."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, _original_labels in loader:  # o rótulo NÃO é usado no treino
            images = images.to(device)
            loss = reconstruction_loss(model(images), images)  # alvo = a própria imagem
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * images.size(0)
            count += images.size(0)
    return total / count


def save_loss_curves(history: dict, best_epoch: int, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Treino", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validação", linewidth=2)
    ax.axvline(best_epoch, color="red", linestyle="--", label=f"Melhor modelo (época {best_epoch})")
    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    anomaly_class = SCENARIOS[args.scenario]
    config = replace(load_config(args.config), anomaly_class=anomaly_class)
    options = read_training_options(args.config)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cenário: {args.scenario} (classe anômala = {anomaly_class}) | dispositivo: {device}")

    splits = load_splits(config)
    loaders = make_dataloaders(splits, config.batch_size, config.num_workers)
    print(f"Treino: {len(splits.train.labels)} | Validação: {len(splits.val.labels)} imagens normais")

    model = Autoencoder.from_config(args.config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"])
    early_stopping = EarlyStopping(patience=options["patience"], min_delta=options["min_delta"])

    checkpoint = ROOT / "models" / f"ae_{args.scenario}.pt"
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, options["max_epochs"] + 1):
        train_loss = average_loss(model, loaders["train"], device, optimizer)
        val_loss = average_loss(model, loaders["val"], device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Época {epoch:03d}/{options['max_epochs']:03d} | treino {train_loss:.6f} | validação {val_loss:.6f}")

        early_stopping(val_loss, model, str(checkpoint), epoch)
        if early_stopping.early_stop:
            print(f"Early stopping na época {epoch}. Melhor época: {early_stopping.best_epoch}.")
            break

    history_path = ROOT / "reports" / f"history_{args.scenario}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_loss_curves(
        history, early_stopping.best_epoch,
        f"Curvas de aprendizado: cenário {args.scenario}",
        ROOT / "reports" / "figures" / f"loss_curves_{args.scenario}.png",
    )
    print(f"Modelo salvo em {checkpoint}")


if __name__ == "__main__":
    main()