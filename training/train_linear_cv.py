"""Train Elastic-Net logistic regression with cross-validation and optional calibration."""

import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from .metrics import print_metrics

IGNORE = {"game_id","role","window_type","action_space","action","reward"}

def _group_split(X, y, groups, test_size=0.2, random_state=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main(csv_path: str, out_model: str = "models/jungle_linear_cv.joblib"):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df["win"].astype(int).values
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    X = df.drop(columns=[c for c in df.columns if c in IGNORE or c=="win"]).copy()
    if groups is not None:
        X_train, X_test, y_train, y_test = _group_split(X, y, groups)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    base = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegressionCV(
            Cs=10,
            cv=3,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
            max_iter=200,
            n_jobs=-1,
        ),
    )
    base.fit(X_train, y_train)
    clf = base
    if bool(int(os.getenv("LINCV_USE_CALIBRATION","1"))):
        clf = CalibratedClassifierCV(base, method=os.getenv("CAL_METHOD","isotonic"), cv=3)
        clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:,1]
    print_metrics(y_test, p, label="linear_cv")
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    joblib.dump(clf, out_model)
    print("[linear_cv] Saved", out_model)

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

