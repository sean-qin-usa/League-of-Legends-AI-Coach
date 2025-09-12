import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

def print_metrics(y_true, p_pred, label: str | None = None):
    auc = roc_auc_score(y_true, p_pred) if len(set(y_true))>1 else float("nan")
    acc = accuracy_score(y_true, (p_pred>=0.5).astype(int))
    brier = brier_score_loss(y_true, p_pred)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}AUC: {auc:.4f}")
    print(f"{prefix}ACC: {acc:.4f}")
    print(f"{prefix}BRIER: {brier:.4f}")
    return {"auc":auc,"acc":acc,"brier":brier}
