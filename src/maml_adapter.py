"""MAML adapter for live inference with optional few-shot inner-loop.

If PyTorch or the saved params are unavailable, predict_proba falls back gracefully.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np

class MamlWrapper:
    def __init__(self, model_path: str, inner_steps: int = 1, lr_inner: float = 1e-2):
        self.model_path = model_path
        self.inner_steps = inner_steps
        self.lr_inner = lr_inner
        self.ready = False
        self._torch = None
        self._net = None
        self._feat_cols: List[str] = []
        self._in_dim: int = 0
        self._load()

    def _load(self):
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            pack = torch.load(self.model_path, map_location='cpu')
            self._feat_cols = list(pack.get('feat_cols') or [])
            self._in_dim = int(pack.get('in_dim') or 0)
            class Net(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
                def forward(self, x):
                    return self.net(x)
            net = Net(self._in_dim)
            net.load_state_dict(pack['state_dict'])
            net.eval()
            self._net = net
            self.ready = True
        except Exception:
            self.ready = False

    def _vec(self, features_df) -> np.ndarray:
        # Map incoming features to training cols; missing -> 0
        import pandas as pd
        if not self._feat_cols:
            # fallback to first N columns
            v = features_df.select_dtypes(include=[np.number]).values.astype(np.float32)
            return v[:, :min(v.shape[1], max(1, self._in_dim))]
        df = features_df.copy()
        for c in self._feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[self._feat_cols]
        return df.values.astype(np.float32)

    def predict_proba(self, features_df, support: Optional[List[Dict[str, Any]]] = None):
        # Returns array [[1-p, p]]
        if not self.ready:
            p = 0.5
            return np.array([[1.0 - p, p]], dtype=float)
        torch = self._torch
        nn = torch.nn
        with torch.no_grad():
            # Clone fast net for optional inner-loop
            net = self._net
            if support:
                fast = type(net)(self._in_dim)
                fast.load_state_dict(net.state_dict())
                fast.train()
                opt = torch.optim.SGD(fast.parameters(), lr=self.lr_inner)
                bce = nn.BCEWithLogitsLoss()
                # Build small support tensors
                xs = []
                ys = []
                for it in support:
                    row = it.get('x')
                    y = float(it.get('y', 0.0))
                    if row is None:
                        continue
                    v = self._vec(row)
                    if v.shape[1] != self._in_dim:
                        continue
                    xs.append(v[0])
                    ys.append(y)
                if xs and ys:
                    xs = torch.from_numpy(np.stack(xs).astype(np.float32))
                    ys = torch.from_numpy(np.array(ys, dtype=np.float32))
                    for _ in range(max(1, self.inner_steps)):
                        opt.zero_grad()
                        out = fast(xs).squeeze(-1)
                        loss = bce(out, ys)
                        loss.backward()
                        opt.step()
                use = fast.eval()
            else:
                use = net
            v = torch.from_numpy(self._vec(features_df))
            logit = use(v).squeeze(-1)
            p = torch.sigmoid(logit).detach().cpu().numpy()
        if np.ndim(p)==0:
            p = float(p)
        else:
            p = float(p[0])
        p = max(0.0, min(1.0, p))
        return np.array([[1.0 - p, p]], dtype=float)

