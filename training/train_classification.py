"""Train and evaluate the classification model.

- Group-aware split to avoid leakage
- Optional probability calibration toggle via `USE_CALIBRATION`
"""

import os
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_curve
from .metrics import print_metrics

from xgboost import XGBClassifier


IGNORE = {"game_id","role","window_type","action_space","action","reward"}

def _group_split(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, test_size: float=0.2, random_state: int=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def main(csv_path: str, out_model: str="models/jungle_bc.joblib"):
    df = _load_df(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = _downcast(df)
    y = df["win"].astype(int).values
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    X = df.drop(columns=[c for c in df.columns if c in IGNORE or c=="win"]).copy()

    if groups is not None:
        X_train, X_test, y_train, y_test = _group_split(X, y, groups)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost is the standard classifier
    use_calibration = bool(int(os.getenv("USE_CALIBRATION", "0")))

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    pos = float((y_train==1).sum()); neg = float((y_train==0).sum());
    spw = max(1.0, neg/max(1.0,pos))
    tm = os.getenv("XGB_TREE_METHOD", "hist")
    eval_metric = os.getenv("XGB_EVAL_METRIC", "auc")

    params = dict(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", 800)),
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
    clf = XGBClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    if use_calibration:
        # Calibrate XGB's probabilities on validation split
        cal = CalibratedClassifierCV(clf, method=os.getenv("CAL_METHOD", "isotonic"), cv="prefit")
        cal.fit(X_val, y_val)
        clf = cal
    p_val = clf.predict_proba(X_val)[:,1]
    p = clf.predict_proba(X_test)[:,1]

    metrics = print_metrics(y_test, p, label="classification")
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    joblib.dump(clf, out_model)
    print("[classification] Saved", out_model)

    # Save feature importances for inspection
    try:
        import pandas as _pd
        if hasattr(clf, "feature_importances_"):
            fi = _pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
            out_fi = os.path.join(os.path.dirname(out_model) or ".", "feature_importance.csv")
            fi.to_csv(out_fi, index=False)
            print("Saved", out_fi)
        # Save meta (basic)
        import json
        # Threshold tuning on validation
        def _best_thresh_f1(y_true, p_pred):
            pr, rc, th = precision_recall_curve(y_true, p_pred)
            th = np.append(th, 1.0)  # align lengths
            f1 = 2*pr*rc/(pr+rc+1e-12)
            idx = int(np.nanargmax(f1))
            return float(th[idx]), float(f1[idx])
        def _best_thresh_youden(y_true, p_pred):
            fpr, tpr, th = roc_curve(y_true, p_pred)
            j = tpr - fpr
            idx = int(np.nanargmax(j))
            return float(th[idx]), float(j[idx])
        try:
            bt_f1, f1_val = _best_thresh_f1(y_val, p_val)
            bt_y, j_val = _best_thresh_youden(y_val, p_val)
        except Exception:
            bt_f1, f1_val, bt_y, j_val = 0.5, float('nan'), 0.5, float('nan')
        meta = {
            "use_calibration": use_calibration,
            "metrics": {k: (None if isinstance(v, float) and (v!=v) else v) for k,v in metrics.items()},
            "thresholds": {"f1": bt_f1, "youden": bt_y},
            "prevalence_train": float(y_train.mean()),
        }
        with open(os.path.join(os.path.dirname(out_model) or ".", "model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print("[train_classification] meta save skipped:", e)
    
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
