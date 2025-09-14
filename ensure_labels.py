"""Ensure 'action' and 'reward' columns exist for datasets that need them.

If actions are missing/blank, label with heuristic recommendations (no model).
If rewards are missing, compute a shaped reward from deltas and objective diffs.
"""

import pandas as pd, numpy as np
from typing import Optional

ACTION_COL = "action"
REWARD_COL = "reward"

def _label_actions(df: pd.DataFrame) -> pd.DataFrame:
    # Use heuristic ranker from recommendations with base_p=0.5
    from util.recommendations import rank_by_strategy
    acts = []
    for i in range(len(df)):
        row = df.iloc[[i]]  # DataFrame
        try:
            ranked = rank_by_strategy("classification", model=None, features_df=row)
            a = ranked[0][0] if ranked else ""
        except Exception:
            a = ""
        acts.append(a)
    out = df.copy()
    out[ACTION_COL] = acts
    return out

def _compute_rewards(df: pd.DataFrame) -> pd.DataFrame:
    # Shaped reward: weighted sum of immediate objective deltas and econ delta.
    w_obj = 1.0; w_gold = 0.001; w_cc = 0.0005
    gcols = ["dragon_diff","herald_diff","baron_diff","tower_diff","plate_diff","ward_kill_diff"]
    out = df.copy()
    # Compute within-group diffs to approximate immediate change
    out = out.sort_values([c for c in ["game_id","team","time_s"] if c in out.columns])
    for k in gcols + ["team_gold_diff","team_cc_ms_diff"]:
        if k in out.columns:
            out[f"_d_{k}"] = out.groupby([c for c in ["game_id","team"] if c in out.columns])[k].diff().fillna(0.0)
        else:
            out[f"_d_{k}"] = 0.0
    obj = sum(out.get(f"_d_{k}", 0.0) for k in gcols)
    gold = out.get("_d_team_gold_diff", 0.0)
    cc = out.get("_d_team_cc_ms_diff", 0.0)
    r = w_obj*obj + w_gold*gold + w_cc*cc
    out[REWARD_COL] = r.astype(float)
    # Clean temp columns
    dropc = [c for c in out.columns if c.startswith("_d_")]
    return out.drop(columns=dropc)

def ensure_actions_rewards(csv_path: str, out_path: Optional[str]=None) -> str:
    df = _load_df(csv_path)
    need_actions = (ACTION_COL not in df.columns) or (df[ACTION_COL].astype(str).str.strip()=="").mean() > 0.5
    need_rewards = (REWARD_COL not in df.columns) or (df[REWARD_COL].isna().mean() > 0.5)
    if need_actions:
        df = _label_actions(df)
    if need_rewards:
        df = _compute_rewards(df)
    outp = out_path or csv_path
    if outp.lower().endswith('.parquet'):
        df.to_parquet(outp, index=False)
    else:
        df.to_csv(outp, index=False)
    return outp

def _load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

