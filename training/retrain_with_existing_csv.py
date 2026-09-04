import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

IGNORE = {"game_id","role","window_type","action_space","action","reward"}

ROLL_WINDOW_S = 30

def infer_window_s(df: pd.DataFrame) -> int:
    try:
        g = df.sort_values(["game_id","team","time_s"]).groupby(["game_id","team"])['time_s']
        diffs = g.diff().dropna()
        # use mode of positive diffs or min
        diffs = diffs[diffs > 0]
        val = int(diffs.mode().iloc[0]) if not diffs.mode().empty else int(diffs.min())
        return max(1, val)
    except Exception:
        return 5

def add_rolling_features(df: pd.DataFrame, window_s: int) -> pd.DataFrame:
    n = max(1, int(ROLL_WINDOW_S // max(1, window_s)))
    df = df.sort_values(["game_id","team","time_s"]).reset_index(drop=True)
    grp = df.groupby(["game_id","team"], sort=False)
    def _roll(col):
        if col not in df.columns:
            return None
        return grp[col].rolling(n, min_periods=1).sum().reset_index(level=[0,1], drop=True)
    mapping = {
        'team_kills_d': 'roll_kills_30s',
        'team_deaths_d': 'roll_deaths_30s',
        'vision_delta': 'roll_vision_delta_30s',
        'ward_kill_diff': 'roll_ward_kill_diff_30s',
        'tower_diff': 'roll_tower_diff_30s',
        'plate_diff': 'roll_plate_diff_30s',
        'dragon_diff': 'roll_dragon_diff_30s',
        'herald_diff': 'roll_herald_diff_30s',
        'baron_diff': 'roll_baron_diff_30s',
    }
    for src, dst in mapping.items():
        r = _roll(src)
        if r is not None:
            df[dst] = r
    return df

def train_and_save(df: pd.DataFrame, out_model: str):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['win'].astype(int).values
    X = df.drop(columns=[c for c in df.columns if c in IGNORE or c=='win'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pos = float((y_train==1).sum()); neg = float((y_train==0).sum())
    spw = max(1.0, neg/max(1.0, pos))
    tree_method = os.getenv("XGB_TREE_METHOD", "hist")
    eval_metric = os.getenv("XGB_EVAL_METRIC", "auc")
    clf = XGBClassifier(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", 800)),
        learning_rate=float(os.getenv("XGB_LR", 0.05)),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", 6)),
        subsample=float(os.getenv("XGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(os.getenv("XGB_COLSAMPLE", 0.8)),
        reg_lambda=float(os.getenv("XGB_LAMBDA", 1.0)),
        objective="binary:logistic",
        n_jobs=-1,
        tree_method=tree_method,
        eval_metric=eval_metric,
        scale_pos_weight=spw,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(out_model) or '.', exist_ok=True)
    joblib.dump(clf, out_model)
    print('Saved', out_model)

def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'windows_v7.csv')
    in_csv = os.path.abspath(in_csv)
    out_model = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'models', 'jungle_bc.joblib')
    print('Loading', in_csv)
    df = pd.read_csv(in_csv)
    ws = infer_window_s(df)
    print('Inferred window_s =', ws)
    df = add_rolling_features(df, ws)
    train_and_save(df, out_model)

if __name__ == '__main__':
    main()
