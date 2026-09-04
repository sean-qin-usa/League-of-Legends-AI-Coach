"""Archetype utilities: load PCA+KMeans and compute action biases."""

from typing import Dict, Tuple
import os, joblib

_PCA = None
_KM = None

def _load_models(models_dir: str = "models"):
    global _PCA, _KM
    try:
        p_pca = os.path.join(models_dir, "archetypes_pca.joblib")
        p_km = os.path.join(models_dir, "archetypes_kmeans.joblib")
        if os.path.exists(p_pca) and os.path.exists(p_km):
            _PCA = joblib.load(p_pca)
            _KM = joblib.load(p_km)
    except Exception:
        _PCA = _KM = None

def archetype_label(state_row: Dict[str,float], models_dir: str = "models") -> int:
    if _PCA is None or _KM is None:
        _load_models(models_dir)
    if _PCA is None or _KM is None:
        return -1
    import numpy as np
    import pandas as pd
    df = pd.DataFrame([state_row])
    # PCA expects training feature set; missing keys become 0
    for c in _PCA.feature_names_in_:
        if c not in df.columns:
            df[c] = 0.0
    df = df[_PCA.feature_names_in_]
    z = _PCA.transform(df)
    return int(_KM.predict(z)[0])

def archetype_bias(action: str, label: int) -> float:
    if label < 0:
        return 0.0
    a = (action or "").upper()
    # crude mapping: tune as needed per cluster
    if label == 0:
        # objective-focused
        if a.startswith("TAKE_") or a.startswith("SETUP_"):
            return 0.01
    elif label == 1:
        # skirmish/aggressive
        if a.startswith("LOOK_FOR_PICK") or a.startswith("JOIN_FIGHT") or a.startswith("GANK_"):
            return 0.01
    elif label == 2:
        # farm/vision
        if a.startswith("FARM_") or a in ("CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS","DEEP_VISION_SWEEP"):
            return 0.01
    elif label == 3:
        # invade
        if a.startswith("INVADE_"):
            return 0.01
    elif label == 4:
        # split/push
        if a.startswith("PRESS_TOWER") or a.startswith("PRESS_INHIB"):
            return 0.01
    return 0.0

