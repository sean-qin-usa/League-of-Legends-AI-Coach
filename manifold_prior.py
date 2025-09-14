"""Manifold prior: approximate cluster label via nearest neighbor to sampled training points.

Loads t-SNE/DBSCAN results along with the sampled original features to
assign a cluster label to a new state at runtime, then maps to a small bias.
"""

from typing import Dict
import os, joblib
import numpy as np
import pandas as pd

_PACK = None

def _load(models_dir: str = "models"):
    global _PACK
    path = os.path.join(models_dir, "manifold_tsne_dbscan.joblib")
    try:
        if os.path.exists(path):
            _PACK = joblib.load(path)
        else:
            _PACK = None
    except Exception:
        _PACK = None

def manifold_label(state: Dict[str, float], models_dir: str = "models") -> int:
    if _PACK is None:
        _load(models_dir)
    if not _PACK:
        return -1
    X_cols = _PACK.get("X_cols") or []
    Xs = _PACK.get("X_sample")
    labels = _PACK.get("labels")
    if Xs is None or labels is None or not X_cols:
        return -1
    # Build row vector in same feature space
    row = pd.DataFrame([state])
    for c in X_cols:
        if c not in row.columns:
            row[c] = 0.0
    v = row[X_cols].values.astype(float)
    # Nearest neighbor (euclidean)
    try:
        dif = Xs - v
        d2 = np.einsum('ij,ij->i', dif, dif)
        idx = int(np.argmin(d2))
        return int(labels[idx])
    except Exception:
        return -1

def manifold_bias(action: str, label: int) -> float:
    if label < 0:
        return 0.0
    a = (action or "").upper()
    # Basic mapping per cluster id; tune as needed
    if label % 5 == 0:
        if a.startswith("TAKE_") or a.startswith("SETUP_"):
            return 0.01
    elif label % 5 == 1:
        if a.startswith("JOIN_FIGHT") or a.startswith("GANK_") or a.startswith("LOOK_FOR_PICK"):
            return 0.01
    elif label % 5 == 2:
        if a.startswith("FARM_") or a in ("CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS","DEEP_VISION_SWEEP"):
            return 0.01
    elif label % 5 == 3:
        if a.startswith("INVADE_"):
            return 0.01
    elif label % 5 == 4:
        if a.startswith("PRESS_TOWER") or a.startswith("PRESS_INHIB"):
            return 0.01
    return 0.0

