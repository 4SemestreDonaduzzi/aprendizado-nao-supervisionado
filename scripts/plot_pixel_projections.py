"""Baseline da #7: PCA e t-SNE sobre os PIXELS BRUTOS do conjunto de teste.

Serve de comparação para o espaço latente do Autoencoder. Quando o modelo
existir, basta trocar ``features`` pela saída de ``encode()``. Execute a
partir da raiz do projeto:

    python scripts/plot_pixel_projections.py [--tsne-samples 3000]
"""

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

from autoencoder import visualize  # noqa: E402
from autoencoder.data import load_config, load_splits  # noqa: E402
from autoencoder.reproducibility import set_seed  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=root / "configs/default.yaml", type=Path)
    parser.add_argument("--tsne-samples", default=3000, type=int)
    parser.add_argument("--perplexity", default=30.0, type=float)
    parser.add_argument("--out", default=root / "reports/figures", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    test = load_splits(config).test
    anomaly = config.anomaly_class
    features = test.images  # (10000, 28, 28): trocar por encode(...) depois

    # PCA usa o teste inteiro: é rápido e determinístico.
    started = time.monotonic()
    pca_coords, variance = visualize.project_pca(features, seed=config.seed)
    axis = (f"PC1 ({variance[0]:.1%} da variância)", f"PC2 ({variance[1]:.1%} da variância)")
    fig = visualize.plot_projection_panel(
        pca_coords, test.labels, anomaly, "PCA", "pixels brutos", axis_labels=axis,
    )
    pca_path = visualize.save_figure(fig, args.out / f"pixels_pca_anomaly{anomaly}.png")
    print(f"PCA: {time.monotonic() - started:.1f}s, variância 2D = {variance.sum():.1%}")

    # t-SNE usa uma amostra estratificada, senão fica lento.
    started = time.monotonic()
    sample = visualize.stratified_sample(test.labels, args.tsne_samples, seed=config.seed)
    tsne_coords = visualize.project_tsne(
        features[sample], perplexity=args.perplexity, seed=config.seed,
    )
    fig = visualize.plot_projection_panel(
        tsne_coords, test.labels[sample], anomaly, "t-SNE", "pixels brutos",
    )
    tsne_path = visualize.save_figure(fig, args.out / f"pixels_tsne_anomaly{anomaly}.png")
    print(f"t-SNE: {time.monotonic() - started:.1f}s com {len(sample)} pontos")

    summary = {
        "source": "pixels brutos (teste oficial)",
        "anomaly_class": anomaly,
        "seed": config.seed,
        "pca_points": len(pca_coords),
        "pca_explained_variance": [round(float(v), 4) for v in variance],
        "tsne_points": len(sample),
        "tsne_perplexity": args.perplexity,
        "figures": [p.relative_to(root).as_posix() for p in (pca_path, tsne_path)],
    }
    summary_path = root / f"reports/pixel_projections_anomaly{anomaly}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Figuras: {', '.join(summary['figures'])}")


if __name__ == "__main__":
    main()
