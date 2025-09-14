"""Offline actor–critic style training.

Policy: multiclass over ACTIONS via XGBoost softmax.
Value: logistic regression on win.
Advantage: (win - baseline) used as sample weights for policy.

Requires 'action' labels; skips gracefully if absent.
"""

import os, joblib
import pandas as pd, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from xgboost import XGBClassifier

IGNORE = {"game_id","role","window_type","action_space","reward","win"}

def _group_split(X, y, groups, test_size=0.2, random_state=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main(csv_path: str, out_dir: str = "models"):
    from util.recommendations import ACTIONS
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    if "action" not in df.columns:
        raise ValueError("Missing 'action' column for actor–critic policy training")
    df["action"] = df["action"].astype(str).str.strip()
    df = df[df["action"].isin(set(ACTIONS))]
    if df.empty:
        raise ValueError("No valid actions found for actor–critic training")

    y_win = df["win"].astype(int).values
    actions = df["action"].astype('category')
    y_act = actions.cat.codes.values
    classes = list(actions.cat.categories)
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    X = df.drop(columns=[c for c in df.columns if c in IGNORE or c=="action"]).copy()

    if groups is not None:
        X_train, X_test, yw_train, yw_test = _group_split(X, y_win, groups)
        _, _, ya_train, ya_test = _group_split(X, y_act, groups)
    else:
        X_train, X_test, yw_train, yw_test = train_test_split(X, y_win, test_size=0.2, random_state=42, stratify=y_win)
        _, _, ya_train, ya_test = train_test_split(X, y_act, test_size=0.2, random_state=42, stratify=y_act)

    # Value model
    V = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, n_jobs=-1))
    V.fit(X_train, yw_train)
    # Advantage
    b = V.predict_proba(X_train)[:,1]
    adv = (yw_train - b)
    w = np.clip(1.0 + 2.0*adv, 0.1, 3.0)

    # Policy model (multiclass XGB)
    Pi = XGBClassifier(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", 600)),
        learning_rate=float(os.getenv("XGB_LR", 0.05)),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", 6)),
        subsample=float(os.getenv("XGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE", 0.8)),
        reg_lambda=float(os.getenv("XGB_LAMBDA", 1.0)),
        objective="multi:softprob",
        n_jobs=-1,
        tree_method=os.getenv("XGB_TREE_METHOD","hist"),
        eval_metric=os.getenv("XGB_EVAL_METRIC","mlogloss"),
        random_state=42,
    )
    Pi.fit(X_train, ya_train, sample_weight=w)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"policy":Pi, "value":V, "classes":classes}, os.path.join(out_dir, "actor_critic.joblib"))
    print("[actor_critic] Saved", os.path.join(out_dir, "actor_critic.joblib"))

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

