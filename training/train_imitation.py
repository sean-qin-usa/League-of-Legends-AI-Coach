"""Train a multiclass imitation (behavior cloning) policy over ACTIONS.

- Expects a windows CSV with an `action` column containing action names.
- Filters to actions present in util.recommendations.ACTIONS.
- Trains XGBoost multiclass classifier and saves model + label map.
"""

import os, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from .metrics import print_metrics


IGNORE = {"game_id","role","window_type","action_space","reward","win"}

def _group_split(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, test_size: float=0.2, random_state: int=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main(csv_path: str, out_model: str="models/jungle_il.joblib"):
    # Lazily import ACTIONS to avoid heavy imports when unused
    from util.recommendations import ACTIONS

    df = _load_df(csv_path).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Keep only rows with non-empty actions that are in our vocabulary
    if "action" not in df.columns:
        raise ValueError("CSV missing required 'action' column for imitation learning")
    df["action"] = (df["action"].astype(str)).str.strip()
    df = df[df["action"] != ""]
    df = df[df["action"].isin(set(ACTIONS))]
    if df.empty:
        raise ValueError("No valid action labels found for imitation learning after filtering.")

    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    y_str = df["action"].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    X = df.drop(columns=[c for c in df.columns if c in IGNORE or c=="action"]).copy()

    if groups is not None:
        X_train, X_test, y_train, y_test = _group_split(X, y, groups)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_class = int(len(le.classes_))
    # XGBoost multiclass configuration (CPU)
    tm = os.getenv("XGB_TREE_METHOD", "hist")
    eval_metric = os.getenv("XGB_EVAL_METRIC", "mlogloss")
    params = dict(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", 700)),
        learning_rate=float(os.getenv("XGB_LR", 0.05)),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", 6)),
        subsample=float(os.getenv("XGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE", 0.8)),
        reg_lambda=float(os.getenv("XGB_LAMBDA", 1.0)),
        objective="multi:softprob",
        num_class=num_class,
        n_jobs=-1,
        tree_method=tm,
        eval_metric=eval_metric,
        random_state=42,
    )
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)

    # Evaluate top-1 accuracy as a simple proxy
    acc = float((clf.predict(X_test) == y_test).mean())
    # Also compute a win-proxy: map predicted action to win via majority in train (if available)
    metrics = {"acc_top1": acc}

    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    # Save both model and label encoder
    joblib.dump({"model": clf, "label_encoder": le}, out_model)
    print("[imitation] Saved", out_model, "classes=", num_class, "acc=", f"{acc:.4f}")

    # Save meta
    try:
        meta = {"metrics": metrics, "num_classes": num_class}
        with open(os.path.join(os.path.dirname(out_model) or ".", "model_meta_il.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print("[train_imitation] meta save skipped:", e)


def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

