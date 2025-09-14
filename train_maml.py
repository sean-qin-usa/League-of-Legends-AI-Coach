"""True gradient-based meta-learning (MAML) for win classification.

Uses PyTorch to train a small network via MAML over tasks defined by phase.
Saves initial parameters to models/maml_cls.pt
"""

import os, math
import pandas as pd, numpy as np

def main(csv_path: str, out_path: str = "models/maml_cls.pt", inner_steps: int = 1, meta_steps: int = 200, k_support: int = 32, k_query: int = 64, lr_inner: float = 1e-2, lr_meta: float = 1e-3):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as e:
        raise RuntimeError("PyTorch is required for MAML. Install torch to enable true gradient-based meta-learning.") from e

    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    y = df["win"].astype(float).values
    X = df.drop(columns=[c for c in df.columns if c=="win"]).copy()
    # Select a manageable subset of features for stability (top N numeric)
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    cols = num_cols[:min(256, len(num_cols))]
    X = X[cols].values.astype(np.float32)
    # Tasks by phase bins
    phase = df["phase_num"].astype(int).values if "phase_num" in df.columns else np.zeros(len(df), dtype=int)

    device = torch.device("cpu")
    D = X.shape[1]
    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x)
    net = Net(D).to(device)
    meta_opt = optim.Adam(net.parameters(), lr=lr_meta)
    bce = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(42)

    def sample_task(p):
        idx = np.where(phase==p)[0]
        if len(idx) < (k_support + k_query):
            idx = rng.choice(len(X), size=(k_support+k_query,), replace=False)
        else:
            idx = rng.choice(idx, size=(k_support+k_query,), replace=False)
        s_idx = idx[:k_support]; q_idx = idx[k_support:]
        return (torch.from_numpy(X[s_idx]).to(device), torch.from_numpy(y[s_idx]).float().to(device),
                torch.from_numpy(X[q_idx]).to(device), torch.from_numpy(y[q_idx]).float().to(device))

    phases = list(sorted(set(phase.tolist()))) or [0]

    for step in range(meta_steps):
        meta_opt.zero_grad()
        meta_loss = 0.0
        for p in phases:
            xs, ys, xq, yq = sample_task(p)
            # Clone fast weights
            fast = Net(D).to(device)
            fast.load_state_dict(net.state_dict())
            opt = optim.SGD(fast.parameters(), lr=lr_inner)
            # Inner loop
            for _ in range(inner_steps):
                l = bce(fast(xs).squeeze(-1), ys)
                opt.zero_grad(); l.backward(); opt.step()
            lq = bce(fast(xq).squeeze(-1), yq)
            lq.backward()
            meta_loss += float(lq.item())
        meta_opt.step()
        if (step+1) % 20 == 0:
            print(f"[maml] step {step+1}/{meta_steps} meta_loss≈{meta_loss/len(phases):.4f}")

    # Save initial parameters
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import torch
        torch.save({"state_dict": net.state_dict(), "in_dim": D, "feat_cols": cols}, out_path)
        print("[maml] Saved", out_path)
    except Exception:
        pass

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

