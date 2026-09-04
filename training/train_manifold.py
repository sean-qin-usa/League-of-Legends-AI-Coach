"""Compute manifold embeddings (t-SNE) and density-based clustering (DBSCAN)."""

import os, joblib
import pandas as pd, numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

IGNORE = {"game_id","role","window_type","action_space","action","reward","win"}

def main(csv_path: str, out_dir: str = "models"):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    X = df.drop(columns=[c for c in df.columns if c in IGNORE]).copy()
    # Sample for speed
    n = min(len(X), int(os.getenv("TSNE_SAMPLES", 20000)))
    Xs = X.sample(n=n, random_state=42) if len(X) > n else X
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30, random_state=42)
    Z = tsne.fit_transform(Xs)
    db = DBSCAN(eps=3.0, min_samples=20)
    labels = db.fit_predict(Z)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({
        "X_index": Xs.index.values,
        "X_cols": list(Xs.columns),
        "X_sample": Xs.values,
        "Z": Z,
        "labels": labels
    }, os.path.join(out_dir, "manifold_tsne_dbscan.joblib"))
    print("[manifold] Saved t-SNE+DBSCAN to", out_dir)

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)
