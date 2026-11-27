"""Lógica da demonstração interativa (issue #10), separada da interface.

O app Streamlit (``app/app.py``) só exibe o que estas funções calculam. Assim, a
regra de decisão pode ser testada sem o Streamlit e é exatamente a mesma da
avaliação (#6): limiar = percentil do erro na VALIDAÇÃO, nunca no teste.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .data import load_config, load_splits, make_dataloaders
from .evaluate import choose_threshold, reconstruction_errors
from .inference import SCENARIOS, load_trained_model

OUTCOME_LABELS = {
    "TP": "Anomalia detectada (acerto)",
    "TN": "Normal reconhecida (acerto)",
    "FP": "Alarme falso (normal marcada como anomalia)",
    "FN": "Anomalia não detectada",
}


@dataclass
class ScenarioData:
    """Tudo o que o app precisa de um cenário, calculado uma única vez."""

    name: str
    anomaly_class: int
    model: torch.nn.Module
    test_images: np.ndarray  # (N, 1, 28, 28), valores em [0, 1]
    test_labels: np.ndarray  # (N,)
    test_mse: np.ndarray  # (N,) erro de reconstrução de cada imagem do teste
    val_mse: np.ndarray  # (M,) erro na validação (só normais), base do limiar


def load_scenario(config_path: str | Path, scenario: str) -> ScenarioData:
    """Carrega o checkpoint do cenário e calcula os erros de validação e teste."""
    if scenario not in SCENARIOS:
        raise ValueError(f"cenário desconhecido: {scenario!r}; use {sorted(SCENARIOS)}")
    anomaly_class, checkpoint = SCENARIOS[scenario]
    config = replace(load_config(config_path), anomaly_class=anomaly_class)
    model = load_trained_model(config_path, checkpoint)
    splits = load_splits(config)
    loaders = make_dataloaders(splits, batch_size=config.batch_size, num_workers=0)
    val = reconstruction_errors(model, loaders["val"])
    test = reconstruction_errors(model, loaders["test"])
    return ScenarioData(
        name=scenario,
        anomaly_class=anomaly_class,
        model=model,
        test_images=splits.test.images,
        test_labels=splits.test.labels,
        test_mse=test["mse"],
        val_mse=val["mse"],
    )


@torch.no_grad()
def reconstruct_image(model: torch.nn.Module, image: np.ndarray) -> dict:
    """Reconstrói uma imagem (28×28 ou 1×28×28) e mede o erro.

    Devolve ``reconstruction`` e ``error_map`` (28×28, |x − x̂|) e ``mse``
    (média do erro quadrático, a mesma medida da avaliação).
    """
    image = np.asarray(image, dtype=np.float32)
    if image.size != 28 * 28:
        raise ValueError(f"esperava uma imagem 28×28, recebi formato {image.shape}")
    x = torch.from_numpy(image.reshape(1, 1, 28, 28))
    model.eval()
    x_hat = model(x).reshape(28, 28).cpu().numpy()
    original = image.reshape(28, 28)
    diff = x_hat - original
    return {
        "reconstruction": x_hat,
        "error_map": np.abs(diff),
        "mse": float((diff**2).mean()),
    }


def classify(mse: float, threshold: float) -> bool:
    """Regra de decisão do projeto: anomalia quando o erro passa do limiar."""
    return bool(mse > threshold)


def outcome(predicted_anomaly: bool, is_anomaly: bool) -> str:
    """TP, TN, FP ou FN, comparando a decisão com o rótulo real."""
    if predicted_anomaly:
        return "TP" if is_anomaly else "FP"
    return "FN" if is_anomaly else "TN"


def test_outcomes(data: ScenarioData, threshold: float) -> np.ndarray:
    """Resultado (TP/TN/FP/FN) de cada imagem do teste com o limiar dado."""
    predicted = data.test_mse > threshold
    actual = data.test_labels == data.anomaly_class
    return np.where(predicted, np.where(actual, "TP", "FP"), np.where(actual, "FN", "TN"))


__all__ = [
    "OUTCOME_LABELS",
    "ScenarioData",
    "choose_threshold",
    "classify",
    "load_scenario",
    "outcome",
    "reconstruct_image",
    "test_outcomes",
]
