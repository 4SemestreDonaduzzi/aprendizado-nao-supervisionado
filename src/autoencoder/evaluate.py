import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from .data import CLASS_NAMES

 # Erro por imagem (MSE e MAE)
@torch.no_grad()
def reconstruction_errors(model, loader, device="cpu"):
    """Devolve MSE, MAE e rótulo originais de CADA imagem."""
    model.eval()
    mse, mae, labels = [], [], []
    for images, y in loader:
        images = images.to(device)
        diff = (model(images) - images).flatten(1)   # (B, 784)
        mse.append((diff ** 2).mean(dim=1).cpu())
        mae.append(diff.abs().mean(dim=1).cpu())
        labels.append(y)
    return {"mse": torch.cat(mse).numpy(),
            "mae": torch.cat(mae).numpy(),
            "labels": torch.cat(labels).numpy()}

# Métricas de reconstrução, globais e por classe

def error_summary_by_class(mse, mae, labels):
    groups = [("global", np.ones(len(labels), dtype=bool))]
    groups += [(CLASS_NAMES[c], labels == c) for c in np.unique(labels)]
    rows = []
    for name, mask in groups:
        m = float(mse[mask].mean())
        rows.append({"grupo": name, "n": int(mask.sum()), "MSE": m,
                     "MAE": float(mae[mask].mean()), "RMSE": m ** 0.5})
    return rows


#limiar

def choose_threshold(val_errors, percentile):
    """Limiar = percentil do erro na validação. NÃO recebe teste nem rótulos."""
    val_errors = np.asarray(val_errors, dtype=float)
    if val_errors.ndim != 1 or val_errors.size == 0:
        raise ValueError("val_errors deve ser um vetor 1D não vazio")
    if not 0 < percentile < 100:
        raise ValueError("percentile deve estar entre 0 e 100")
    return float(np.percentile(val_errors, percentile))

    #Métricas independentes de limiar

def ranking_metrics(scores, is_anomaly):
    return {"roc_auc": float(roc_auc_score(is_anomaly, scores)),
            "pr_auc": float(average_precision_score(is_anomaly, scores))}

# Métricas dependentes do limiar

def threshold_metrics(scores, is_anomaly, threshold):
    pred = (scores > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        is_anomaly, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(is_anomaly, pred, labels=[0, 1]).ravel()
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

#função que junta tudo para um cenário

def evaluate_scenario(val_errors, test_errors, test_is_anomaly, percentiles=(95, 99)):
    ranking = ranking_metrics(test_errors, test_is_anomaly)
    rows = []
    for q in percentiles:
        thr = choose_threshold(val_errors, q)        # só validação!
        rows.append({"percentil": q, "limiar": thr, **ranking,
                     **threshold_metrics(test_errors, test_is_anomaly, thr)})
    return rows