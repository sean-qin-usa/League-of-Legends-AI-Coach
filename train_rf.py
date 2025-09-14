"""Train RandomForest classifier (bagging) with simple CV and optional calibration."""

import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from .metrics import print_metrics

IGNORE = {"game_id","role","window_type","action_space","action","reward"}

def _group_split(X, y, groups, test_size=0.2, random_state=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main(csv_path: str, out_model: str = "models/jungle_rf.joblib"):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df["win"].astype(int).values
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    X = df.drop(columns=[c for c in df.columns if c in IGNORE or c=="win"]).copy()
    if groups is not None:
        X_train, X_test, y_train, y_test = _group_split(X, y, groups)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    base = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42)
    grid = dict(max_features=["sqrt","log2", None])
    clf = GridSearchCV(base, grid, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    model = clf.best_estimator_
    if bool(int(os.getenv("RF_USE_CALIBRATION","1"))):
        model = CalibratedClassifierCV(model, method=os.getenv("CAL_METHOD","isotonic"), cv=3)
        model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:,1]
    print_metrics(y_test, p, label="rf")
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    joblib.dump(model, out_model)
    print("[rf] Saved", out_model)

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

