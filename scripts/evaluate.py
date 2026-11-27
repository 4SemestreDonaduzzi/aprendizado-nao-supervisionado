"""Avalia o detector de anomalias nos dois cenários (Bag e Sandal).

Uso, na raiz do repositório:

    python scripts/evaluate.py

Para cada cenário o script:
1. carrega os dados com load_splits (mesma divisão usada no treino);
2. carrega o checkpoint treinado para aquele cenário;
3. calcula o erro de reconstrução por imagem na validação e no teste;
4. escolhe o limiar SOMENTE com os erros da validação (percentis 95 e 99);
5. mede as métricas no teste, gera o histograma e salva as tabelas em reports/.

Todos os números gravados em reports/ saem desta execução. Nada é digitado à mão.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.data import CLASS_NAMES, load_config, load_splits, make_dataloaders  # noqa: E402
from autoencoder.evaluate import (  # noqa: E402
    choose_threshold,
    error_summary_by_class,
    evaluate_scenario,
    reconstruction_errors,
)
from autoencoder.inference import SCENARIOS, load_trained_model  # noqa: E402
from autoencoder.reproducibility import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERCENTILES = (95, 99)


def plot_histogram(test_errors, is_anomaly, thresholds, anomaly_name, path: Path) -> None:
    """Erro de reconstrução no teste: normais vs. anômalos, com os limiares."""
    normal = test_errors[is_anomaly == 0]
    anomalous = test_errors[is_anomaly == 1]
    low = max(float(test_errors.min()), 1e-6)
    bins = np.logspace(np.log10(low), np.log10(float(test_errors.max())), 60)

    fig, ax = plt.subplots(figsize=(9, 5))
    # density=True porque há ~9x mais normais; senão as anômalas somem.
    ax.hist(normal, bins=bins, density=True, alpha=0.6, label="Normais (teste)", color="#9aa5b1")
    ax.hist(anomalous, bins=bins, density=True, alpha=0.6, label=f"Anômalas: {anomaly_name} (teste)", color="#e600e6")
    for percentile, threshold in thresholds.items():
        ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"Limiar P{percentile} (validação)")
    ax.set_xscale("log")
    ax.set_xlabel("Erro de reconstrução (MSE por imagem, escala log)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Erro de reconstrução: normais vs. {anomaly_name}")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    """Tabela em Markdown para colar no relatório (gerada dos mesmos números)."""
    columns = list(rows[0])

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(row[c]) for c in columns) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scenario(name: str, anomaly_class: int, checkpoint: Path, config_path: Path, figures_dir: Path):
    base_config = load_config(config_path)
    config = replace(base_config, anomaly_class=anomaly_class)
    set_seed(config.seed)

    splits = load_splits(config)
    loaders = make_dataloaders(splits, config.batch_size, config.num_workers)
    model = load_trained_model(config_path, checkpoint)

    val = reconstruction_errors(model, loaders["val"])
    test = reconstruction_errors(model, loaders["test"])

    # Proteção contra vazamento de dados: a validação só pode ter imagens normais.
    if (val["labels"] == anomaly_class).any():
        raise RuntimeError("A validação contém a classe anômala; verifique a divisão dos dados.")

    is_anomaly = (test["labels"] == anomaly_class).astype("int64")
    anomaly_name = CLASS_NAMES[anomaly_class]
    print(f"[{name}] anômala={anomaly_name} | val={len(val['mse'])} | "
          f"teste={len(test['mse'])} (anômalas={int(is_anomaly.sum())})")

    # Limiar: só erros de validação. O teste entra apenas para medir o resultado.
    detection = evaluate_scenario(val["mse"], test["mse"], is_anomaly, PERCENTILES)
    detection_rows = [
        {"cenario": name, "classe_anomala": anomaly_name, "n_val": len(val["mse"]),
         "n_teste": len(test["mse"]), **row}
        for row in detection
    ]

    class_rows = []
    for subset, errors in (("validacao", val), ("teste", test)):
        for row in error_summary_by_class(errors["mse"], errors["mae"], errors["labels"]):
            class_rows.append({"cenario": name, "conjunto": subset, **row})

    thresholds = {q: choose_threshold(val["mse"], q) for q in PERCENTILES}
    plot_histogram(test["mse"], is_anomaly, thresholds, anomaly_name,
                   figures_dir / f"hist_erro_{name}.png")
    return detection_rows, class_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    args = parser.parse_args()

    detection_rows, class_rows = [], []
    for name in args.scenarios:
        anomaly_class, checkpoint = SCENARIOS[name]
        detection, per_class = run_scenario(
            name, anomaly_class, ROOT / checkpoint, args.config, args.reports / "figures")
        detection_rows += detection
        class_rows += per_class

    write_csv(detection_rows, args.reports / "resultados_deteccao.csv")
    write_markdown(detection_rows, args.reports / "resultados_deteccao.md")
    write_csv(class_rows, args.reports / "resultados_por_classe.csv")
    print(f"Resultados salvos em {args.reports}")


if __name__ == "__main__":
    main()