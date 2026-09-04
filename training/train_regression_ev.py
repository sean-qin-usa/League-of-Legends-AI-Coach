"""Train multi-target EV regressor and a win-probability mapper.

- Produces two artifacts: EV model and EV->P(win) mapper
"""

import os, json, joblib
import pandas as pd, numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .metrics import print_metrics
from xgboost import XGBRegressor

TARGETS = ["dragon_diff","baron_diff","herald_diff","tower_diff","plate_diff"]

def _group_split(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, test_size: float=0.2, random_state: int=42):
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def main(csv_path: str, out_model_dir: str="models"):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    df = _downcast(df)
    df = df.sort_values(["game_id","team","time_s"])
    g = df.groupby(["game_id","team"], as_index=False)
    fut = g[TARGETS].shift(-12)  # 12*5s = 60s ahead for default window 5s
    # Build future and delta columns in a single concat to avoid fragmentation warnings
    fut_df = fut.rename(columns={k: f"{k}_fut" for k in TARGETS})
    dv_df = pd.DataFrame({f"{k}_dv": fut[k] - df[k] for k in TARGETS}, index=df.index)
    df = pd.concat([df, fut_df, dv_df], axis=1, copy=False)
    keep = [f"{k}_dv" for k in TARGETS]
    df = df.dropna(subset=keep)
    y_ev = df[keep].values
    X = df.drop(columns=["game_id","role","window_type","action_space","action","reward","win"] + [c for c in df.columns if c.endswith("_fut")] + keep)
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0)
    groups = df["game_id"].astype(str).values if "game_id" in df.columns else None
    if groups is not None:
        X_train, X_test, y_train, y_test = _group_split(X, y_ev, groups)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_ev, test_size=0.2, random_state=42)
    # XGBoost always available
    # CPU-only configuration (no GPU)
    tm = os.getenv("XGB_TREE_METHOD", "hist")
    params = dict(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", 600)),
        learning_rate=float(os.getenv("XGB_LR", 0.05)),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", 6)),
        subsample=float(os.getenv("XGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE", 0.8)),
        reg_lambda=float(os.getenv("XGB_LAMBDA", 1.0)),
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method=tm,
        random_state=42,
    )
    base = XGBRegressor(**params)
    ev = MultiOutputRegressor(base)
    ev.fit(X_train, y_train)
    # EV -> P(win) mapper (scaled logistic regression)
    Z_train = pd.DataFrame(y_train, columns=keep)
    y_train_win = df.loc[X_train.index, "win"].astype(int).values
    mapper = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=300, n_jobs=-1))
    mapper.fit(Z_train, y_train_win)
    Z_test = pd.DataFrame(ev.predict(X_test), columns=keep)
    p = mapper.predict_proba(Z_test)[:,1]
    y_true = df.loc[X_test.index, "win"].astype(int).values
    metrics = print_metrics(y_true, p, label="regression_ev")
    os.makedirs(out_model_dir, exist_ok=True)
    ev_path = os.path.join(out_model_dir, "ev_multi.joblib")
    mp_path = os.path.join(out_model_dir, "ev_win_mapper.joblib")
    joblib.dump(ev, ev_path)
    joblib.dump(mapper, mp_path)
    print("[regression_ev] Saved", ev_path)
    print("[regression_ev_mapper] Saved", mp_path)
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

    # Save basic meta with metrics and simple threshold suggestions
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
        pr, rc, th = precision_recall_curve(y_true, p); th = np.append(th, 1.0)
        f1 = 2*pr*rc/(pr+rc+1e-12); idx_f1 = int(np.nanargmax(f1)); bt_f1 = float(th[idx_f1])
        fpr, tpr, thr = roc_curve(y_true, p); j = tpr - fpr; idx_j = int(np.nanargmax(j)); bt_y = float(thr[idx_j])
        meta = {"metrics": {k: (None if isinstance(v,float) and (v!=v) else v) for k,v in metrics.items()},
                "thresholds": {"f1": bt_f1, "youden": bt_y}}
        with open(os.path.join(out_model_dir, "model_meta_ev.json"), 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    print("Saved models to", out_model_dir)
