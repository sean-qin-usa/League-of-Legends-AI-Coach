"""Meta-adaptive utilities for contextual ensemble weighting.

This provides a lightweight, heuristic meta-adapter that adjusts
strategy weights based on current game state (few-shot style adaptation
to context). It is not MAML, but offers meta-like behavior by
conditioning weights on features.
"""

from typing import Dict


def contextual_weights(state: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    """Compute per-strategy weights conditioned on current state.

    Inputs:
      - state: single-row feature dict (from features_from_live)
      - base: base weights dict (e.g., {'classification':1.0,'regression_ev':1.0,'rl_awr':1.0,'imitation':1.0})

    Returns a copy of base with small adjustments.
    """
    def _f(k, d=0.0):
        try:
            return float(state.get(k, d) or 0.0)
        except Exception:
            return d

    out = dict(base or {})
    t = _f("time_s", 0.0)
    phase = 1 if t < 8*60 else 2 if t < 14*60 else 3 if t < 20*60 else 4 if t < 30*60 else 5
    vision = _f("vision_delta", 0.0)
    skirm = int(state.get("skirmish_flag", 0))
    plates_left = int(state.get("plates_time_left", 0))
    baron_live = int(state.get("baron_live", 0))

    # Bias EV regression during objective-centric phases with vision
    if (phase >= 3 and (baron_live or _f("dragon_live", 1) > 0)) and vision >= 0:
        out["regression_ev"] = out.get("regression_ev", 1.0) + 0.2

    # Bias RL-style policy during active skirmishes
    if skirm:
        out["rl_awr"] = out.get("rl_awr", 1.0) + 0.2

    # Early game with plates: slightly up-weight imitation (if trained) and classification equally
    if phase in (1, 2) and plates_left > 0:
        out["imitation"] = out.get("imitation", out.get("il_bc", 1.0)) + 0.15
        out["classification"] = out.get("classification", 1.0) + 0.1

    # Low vision: downweight aggressive RL/EV suggestions a touch
    if vision < 0:
        out["rl_awr"] = max(0.1, out.get("rl_awr", 1.0) - 0.1)
        out["regression_ev"] = max(0.1, out.get("regression_ev", 1.0) - 0.05)

    return out

