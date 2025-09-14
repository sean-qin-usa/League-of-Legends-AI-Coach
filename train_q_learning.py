"""Offline tabular Q-learning over discretized state and ACTIONS.

Requires 'action' and 'reward' columns. Skips if not present.
Saves a Q-table dict keyed by (state_tuple, action_str).
"""

import os, math, joblib
import pandas as pd, numpy as np

STATE_KEYS = [
    "phase_num","baron_live","dragon_live","skirmish_flag",
    "vision_delta","team_gold_diff","team_cc_ms_diff","team_ehp_proxy_diff",
]

def _disc(v, bins):
    try:
        x = float(v)
    except Exception:
        return 0
    # uniform bins around 0
    return int(np.digitize([x], bins)[0])

def _make_state(row):
    bins = {
        "vision_delta": [-1, 0, 1, 2],
        "team_gold_diff": [-2000, -500, 0, 500, 2000],
        "team_cc_ms_diff": [-200, 0, 200],
        "team_ehp_proxy_diff": [-500, 0, 500],
    }
    out = []
    for k in STATE_KEYS:
        if k in bins:
            out.append((k, _disc(row.get(k,0), bins[k])))
        else:
            out.append((k, int(row.get(k,0) or 0)))
    return tuple(out)

def main(csv_path: str, out_path: str = "models/q_table.joblib", gamma: float=0.95, alpha: float=0.3):
    from util.recommendations import ACTIONS
    df = _load_df(csv_path).replace([np.inf,-np.inf], np.nan).fillna(0)
    if "action" not in df.columns or "reward" not in df.columns:
        raise ValueError("Q-learning requires 'action' and 'reward' columns")
    df["action"] = df["action"].astype(str).str.strip()
    df = df[df["action"].isin(set(ACTIONS))]
    df = df.sort_values(["game_id","team","time_s"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid action/reward rows for Q-learning")

    Q = {}
    def Qget(s, a):
        return Q.get((s,a), 0.0)
    def Qset(s,a,v):
        Q[(s,a)] = v

    # Iterate trajectories per (game,team)
    for (gid, tm), g in df.groupby(["game_id","team"], sort=False):
        rows = g.to_dict("records")
        for t in range(len(rows)-1):
            s = _make_state(rows[t])
            a = rows[t]["action"]
            r = float(rows[t].get("reward", 0.0) or 0.0)
            s2 = _make_state(rows[t+1])
            # TD target
            max_next = max((Qget(s2, aa) for aa in ACTIONS), default=0.0)
            target = r + gamma * max_next
            qsa = Qget(s, a)
            Qset(s, a, qsa + alpha * (target - qsa))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"Q":Q, "state_keys":STATE_KEYS}, out_path)
    print("[q_learning] Saved", out_path, "| entries:", len(Q))

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

