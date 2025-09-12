"""Train RL-style value and policy models on windows features.

- Saves only policy by default; value used internally.
"""

import os, json, joblib
import pandas as pd, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from .metrics import print_metrics
from xgboost import XGBClassifier

def _group_split(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, test_size: float=0.2, random_state: int=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main(csv_path: str, out_dir: str="models"):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    df = _downcast(df)
    y = df["win"].astype(int).values
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    X = df.drop(columns=[c for c in ["game_id","role","window_type","action_space","action","reward","win"] if c in df.columns])
    # Value model (scaled logistic regression)
    if groups is not None:
        X_train, X_test, y_train, y_test = _group_split(X, y, groups)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Optional value model (kept for internal use, no prints)
    V = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, n_jobs=-1))
    V.fit(X_train, y_train)
    # Policy with advantage weights
    base = y_train.mean()
    adv = (y_train - base)
    w = 1.0 + 2.0*adv
    w = np.clip(w, 0.1, 3.0)
    # XGBoost always available
    pos = float((y_train==1).sum()); neg = float((y_train==0).sum())
    spw = max(1.0, neg/max(1.0, pos))
    # CPU-only configuration (no GPU)
    tm = os.getenv("XGB_TREE_METHOD", "hist")
    eval_metric = os.getenv("XGB_EVAL_METRIC", "auc")
    params = dict(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", 600)),
        learning_rate=float(os.getenv("XGB_LR", 0.05)),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", 6)),
        subsample=float(os.getenv("XGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE", 0.8)),
        reg_lambda=float(os.getenv("XGB_LAMBDA", 1.0)),
        objective="binary:logistic",
        n_jobs=-1,
        tree_method=tm,
        eval_metric=eval_metric,
        scale_pos_weight=spw,
        random_state=42,
    )
    Pi = XGBClassifier(**params)
    Pi.fit(X_train, y_train, sample_weight=w)
    p = Pi.predict_proba(X_test)[:,1]
    metrics = print_metrics(y_test, p, label="rl_policy")
    os.makedirs(out_dir, exist_ok=True)
    pol_path = os.path.join(out_dir, "rl_policy.joblib")
    val_path = os.path.join(out_dir, "rl_value.joblib")
    joblib.dump(Pi, pol_path)
    # Suppress saving the value model to avoid extra artifact/confusion
    print("[rl_policy] Saved", pol_path)
def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='float')
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='integer')
    return df
    # Threshold tuning + meta
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
        pr, rc, th = precision_recall_curve(y_test, p); th = np.append(th, 1.0)
        f1 = 2*pr*rc/(pr+rc+1e-12); idx_f1 = int(np.nanargmax(f1)); bt_f1 = float(th[idx_f1])
        fpr, tpr, thr = roc_curve(y_test, p); j = tpr - fpr; idx_j = int(np.nanargmax(j)); bt_y = float(thr[idx_j])
        meta = {"metrics": {k: (None if isinstance(v,float) and (v!=v) else v) for k,v in metrics.items()},
                "thresholds": {"f1": bt_f1, "youden": bt_y}}
        with open(os.path.join(out_dir, "model_meta_rl.json"), 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    print("Saved RL policy/value to", out_dir)
