"""Strategy weight suggestions for Borda-style ensemble voting.

Currently supports ELO-profile presets: 'low', 'mid', 'high'.
Values are multiplicative weights applied to per-strategy Borda votes.
"""

from typing import Dict


def elo_borda_weights(profile: str) -> Dict[str, float]:
    """Return suggested strategy weights for a given ELO profile.

    Profiles:
      - low: safer, supervised/bagging favored; RL toned down
      - mid: balanced
      - high: proactive/tempo; RL/EV/MAML emphasized
    """
    p = (profile or "mid").strip().lower()
    if p in ("low", "safe", "farm", "scaling", "low_safe"):
        return {
            "classification": 1.20,
            "rf": 1.10,
            "linear_cv": 1.10,
            "regression_ev": 1.00,
            "stacking": 1.10,
            "imitation": 1.00,
            "rl_awr": 0.90,
            "actor_critic": 0.90,
            "q_table": 0.85,
            "maml": 1.00,
        }
    if p in ("high", "aggro", "aggressive", "proactive", "tempo", "high_aggressive"):
        return {
            "classification": 1.00,
            "rf": 0.95,
            "linear_cv": 0.95,
            "regression_ev": 1.10,
            "stacking": 1.10,
            "imitation": 1.00,
            "rl_awr": 1.20,
            "actor_critic": 1.15,
            "q_table": 1.05,
            "maml": 1.10,
        }
    # mid/balanced default
    return {
        "classification": 1.05,
        "rf": 1.00,
        "linear_cv": 1.00,
        "regression_ev": 1.05,
        "stacking": 1.05,
        "imitation": 1.00,
        "rl_awr": 1.05,
        "actor_critic": 1.00,
        "q_table": 1.00,
        "maml": 1.00,
    }

