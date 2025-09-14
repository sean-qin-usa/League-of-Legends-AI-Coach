"""Train a stacking meta-classifier combining existing model outputs.

Build meta-features from available models' predicted win probas:
  - classification, regression_ev (via mapper), rl_awr, rf, linear_cv, imitation (reduce via top-class prob)
Train a LogisticRegression meta-model on a held-out split.
"""

import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression


def _group_split(X, y, groups, test_size=0.2, random_state=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def main(csv_path: str, out_path: str="models/stacking_meta.joblib"):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df["win"].astype(int).values
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    X = df.drop(columns=[c for c in df.columns if c=="win"]).copy()

    # Load available models
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, ".."))
    models_dir = os.path.join(root, "models")
    paths = {
        "classification": os.path.join(models_dir, "jungle_bc.joblib"),
        "rl_awr": os.path.join(models_dir, "rl_policy.joblib"),
        "rf": os.path.join(models_dir, "jungle_rf.joblib"),
        "linear_cv": os.path.join(models_dir, "jungle_linear_cv.joblib"),
        "regression_ev": os.path.join(models_dir, "ev_multi.joblib"),
        "ev_win_mapper": os.path.join(models_dir, "ev_win_mapper.joblib"),
        "imitation": os.path.join(models_dir, "jungle_il.joblib"),
    }
    loaded = {k: (joblib.load(p) if os.path.exists(p) else None) for k,p in paths.items()}

    def proba_row(row_df):
        res = {}
        if loaded.get("classification") is not None:
            try: res["p_cls"] = float(loaded["classification"].predict_proba(row_df)[:,1][0])
            except Exception: res["p_cls"] = 0.5
        if loaded.get("rl_awr") is not None:
            try: res["p_rl"] = float(loaded["rl_awr"].predict_proba(row_df)[:,1][0])
            except Exception: res["p_rl"] = 0.5
        if loaded.get("rf") is not None:
            try: res["p_rf"] = float(loaded["rf"].predict_proba(row_df)[:,1][0])
            except Exception: res["p_rf"] = 0.5
        if loaded.get("linear_cv") is not None:
            try: res["p_lin"] = float(loaded["linear_cv"].predict_proba(row_df)[:,1][0])
            except Exception: res["p_lin"] = 0.5
        if loaded.get("regression_ev") is not None and loaded.get("ev_win_mapper") is not None:
            try:
                ev = loaded["regression_ev"].predict(row_df)[0].reshape(1, -1)
                res["p_ev"] = float(loaded["ev_win_mapper"].predict_proba(ev)[:,1][0])
            except Exception:
                res["p_ev"] = 0.5
        if loaded.get("imitation") is not None:
            try:
                pack = loaded["imitation"]
                m = pack.get("model") if isinstance(pack, dict) else pack
                le = pack.get("label_encoder") if isinstance(pack, dict) else None
                proba = m.predict_proba(row_df)[0]
                res["p_il"] = float(np.max(proba))
            except Exception:
                res["p_il"] = 0.5
        return res

    # Build meta-feature matrix (compute per row; sample subset for speed)
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    take = idx[:min(len(idx), 50000)]  # cap compute
    feats = []
    labels = []
    for i in take:
        row_df = X.iloc[[i]]
        r = proba_row(row_df)
        if not r:
            continue
        feats.append([r.get(k,0.5) for k in ["p_cls","p_rl","p_rf","p_lin","p_ev","p_il"]])
        labels.append(int(y[i]))
    if not feats:
        raise ValueError("No base model predictions available for stacking")
    F = pd.DataFrame(feats, columns=["p_cls","p_rl","p_rf","p_lin","p_ev","p_il"]).fillna(0.5)
    Y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(F, Y, test_size=0.2, random_state=42, stratify=Y)
    meta = LogisticRegression(max_iter=200)
    meta.fit(X_train, y_train)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(meta, out_path)
    print("[stacking] Saved", out_path)

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

