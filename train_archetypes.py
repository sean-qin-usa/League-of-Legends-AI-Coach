"""Train playstyle archetypes via PCA + KMeans and save models."""

import os, joblib
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

IGNORE = {"game_id","role","window_type","action_space","action","reward","win"}

def main(csv_path: str, out_dir: str="models", n_components: int=20, n_clusters: int=5):
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    X = df.drop(columns=[c for c in df.columns if c in IGNORE]).copy()
    pca = PCA(n_components=min(n_components, max(2, min(X.shape)-1)), random_state=42)
    Z = pca.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(Z)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pca, os.path.join(out_dir, "archetypes_pca.joblib"))
    joblib.dump(km, os.path.join(out_dir, "archetypes_kmeans.joblib"))
    print("[archetypes] Saved PCA+KMeans to", out_dir)

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

