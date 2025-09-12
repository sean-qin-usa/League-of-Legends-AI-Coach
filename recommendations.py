"""Action ranking + tactical add-ons and warnings.

- `recommend_addons(state, include_warnings=True, action=None)`: toggle warnings via parameter
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import pandas as pd

ADDONS = [
    "FIGHT_SLOW",  # stall, avoid flips, farm/vision first
    "ALL_IN",      # commit to decisive engage
    "POKE",        # chip before commit
    "WARNINGS",    # contextual risk warnings
]

ACTIONS = [
    "SETUP_DRAGON_90","TAKE_DRAGON",
    "SETUP_HERALD_60","TAKE_HERALD",
    "SETUP_BARON_120","TAKE_BARON",
    "PRESS_TOWER_TOP_T1","PRESS_TOWER_MID_T1","PRESS_TOWER_BOT_T1",
    "PRESS_TOWER_TOP_T2","PRESS_TOWER_MID_T2","PRESS_TOWER_BOT_T2",
    "PRESS_INHIB_TOP","PRESS_INHIB_MID","PRESS_INHIB_BOT",
    # Macro skirmish/pick
    "LOOK_FOR_PICK","JOIN_RIVER_FIGHT","COVER_COUNTERGANK",
    # Farming groupings
    "CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS",
    # Direct farm actions (kept for variety when useful)
    "FARM_BLUE","FARM_GROMP","FARM_WOLVES","FARM_RAPTORS","FARM_RED","FARM_KRUGS",
    "SECURE_SCUTTLE_TOP","SECURE_SCUTTLE_BOT",
    # Aggression
    "INVADE_TOP_CAMPS","INVADE_BOT_CAMPS",
    # Lane ganks (less specific)
    "GANK_TOP","GANK_MID","GANK_BOT",
    # Join nearby fights
    "JOIN_FIGHT_TOP","JOIN_FIGHT_MID","JOIN_FIGHT_BOT",
    "JOIN_FIGHT_TOP_JUNGLE","JOIN_FIGHT_BOT_JUNGLE",
    # Objective usage
    "USE_HERALD","use_herald",
    # Utility
    "RESET_BUY","DEEP_VISION_SWEEP",
]

def _phase_from_t(t: float) -> int:
    return 1 if t<8*60 else 2 if t<14*60 else 3 if t<20*60 else 4 if t<30*60 else 5

def action_candidates(state: Dict[str, float]) -> List[str]:
    t = float(state.get("time_s", 0))
    out = []
    if t < 14*60:
        out += [
            # grouped options
            "CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS",
            # individual farm
            "FARM_BLUE","FARM_GROMP","FARM_WOLVES","FARM_RAPTORS","FARM_RED","FARM_KRUGS","SECURE_SCUTTLE",
            # proactive
            "GANK_TOP","GANK_MID","GANK_BOT",
            "INVADE_TOP_CAMPS","INVADE_BOT_CAMPS",
        ]
        out += ["PRESS_TOWER_TOP_T1","PRESS_TOWER_MID_T1","PRESS_TOWER_BOT_T1"]
        out += ["SETUP_HERALD_60","TAKE_HERALD"]
        out += ["LOOK_FOR_PICK","COVER_COUNTERGANK"]
    elif t < 20*60:
        out += [
            "SETUP_DRAGON_90","TAKE_DRAGON","SETUP_HERALD_60","TAKE_HERALD",
            "PRESS_TOWER_MID_T1","LOOK_FOR_PICK","DEEP_VISION_SWEEP",
            # keep flexible aggression
            "GANK_TOP","GANK_MID","GANK_BOT",
            "INVADE_TOP_CAMPS","INVADE_BOT_CAMPS",
            # grouped farm
            "CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS",
        ]
    else:
        out += [
            "SETUP_BARON_120","TAKE_BARON","SETUP_DRAGON_90","TAKE_DRAGON",
            "LOOK_FOR_PICK","JOIN_RIVER_FIGHT","DEEP_VISION_SWEEP",
            # still valid occasionally
            "GANK_TOP","GANK_MID","GANK_BOT",
            "INVADE_TOP_CAMPS","INVADE_BOT_CAMPS",
        ]
        out += ["PRESS_TOWER_TOP_T2","PRESS_TOWER_MID_T2","PRESS_TOWER_BOT_T2","PRESS_INHIB_TOP","PRESS_INHIB_MID","PRESS_INHIB_BOT"]
    # Join fight options are broadly valid
    out += ["JOIN_FIGHT_TOP","JOIN_FIGHT_MID","JOIN_FIGHT_BOT","JOIN_FIGHT_TOP_JUNGLE","JOIN_FIGHT_BOT_JUNGLE"]
    # Herald usage if available
    try:
        if int(state.get("has_herald",0)) == 1:
            out += ["USE_HERALD","use_herald"]
    except Exception:
        pass
    out += ["RESET_BUY"]
    return list(dict.fromkeys(out))

def _compute_warnings(state: Dict[str, float], action: Optional[str]=None) -> List[str]:
    def _f(k, d=0.0):
        try:
            return float(state.get(k, d) or 0.0)
        except Exception:
            return d
    warns: List[str] = []
    # Execute threats
    if int(state.get("enemy_has_pyke", 0)) == 1:
        warns.append("Beware Pyke execute (R) resets")
    if int(state.get("enemy_has_chogath", 0)) == 1:
        warns.append("Beware Cho'Gath true-damage execute on low objectives/heroes")
    if int(state.get("enemy_has_urgot", 0)) == 1:
        warns.append("Beware Urgot execute (R) threshold")
    if int(state.get("enemy_has_darius", 0)) == 1:
        warns.append("Beware Darius Noxian Guillotine snowball")
    # Numbers and stats
    if _f("team_cc_ms_diff") < 0:
        warns.append("Enemy has higher crowd control")
    if _f("team_ehp_proxy_diff") < 0:
        warns.append("Enemy tankier in current window")
    if _f("enemy_fed_proxy") > 800:
        warns.append("Fed enemy detected (gold lead > 800)")
    if _f("vision_delta") < 0:
        warns.append("Vision deficit — place wards / sweep before committing")
    # Resist stacking awareness
    if int(state.get("enemy_high_mr", 0)) == 1:
        ch = state.get("enemy_high_mr_champ", "Enemy")
        warns.append(f"{ch} has high MR — mixed damage or magic pen needed")
    if int(state.get("enemy_high_armor", 0)) == 1:
        ch = state.get("enemy_high_armor_champ", "Enemy")
        warns.append(f"{ch} has high armor — consider armor pen or AP threat")
    # Action-specific extras
    if action:
        if action.startswith("TAKE_BARON") or action.startswith("TAKE_DRAGON"):
            if int(state.get("enemy_has_chogath", 0)) == 1:
                warns.append("Objective flip risk vs Cho'Gath true damage")
            if _f("vision_delta") < 0:
                warns.append("Low vision for objective — high flip risk")
            if _f("jng_smite_proxy") < 0:
                warns.append("Smite disadvantage — secure numbers/vision")
            if action.startswith("TAKE_DRAGON") and int(state.get("enemy_soul_point", 0)) == 1:
                warns.append("Enemy at soul point — don't flip")
        if action.startswith("INVADE") and _f("vision_delta") < 0:
            warns.append("Invading with vision disadvantage")
        if action.startswith("PRESS_TOWER") and _f("team_ehp_proxy_diff") < 0:
            warns.append("Dive risk — they are tankier")
    return warns

def recommend_addons(state: Dict[str, float], include_warnings: bool=True, action: Optional[str]=None) -> List[Tuple[str, float, str]]:
    """Compute tactical add-ons based on fight-impact features.
    Returns (addon, score, reason). Higher score is stronger recommendation.
    """
    def _f(k, d=0.0):
        try:
            return float(state.get(k, d) or 0.0)
        except Exception:
            return d
    vision = _f("vision_delta")
    skirm = int(state.get("skirmish_flag", 0))
    o = _f("team_offense_power_diff")
    c = _f("team_cc_ms_diff")
    e = _f("team_ehp_proxy_diff")
    h = _f("team_ability_haste_diff")
    pen = _f("team_penetration_proxy_diff")
    ten = _f("team_tenacity_proxy_diff")
    sustain = _f("team_sustain_vamp_diff")
    lvl = _f("team_lvl11_spike_diff") + _f("team_lvl16_spike_diff") + 0.5*_f("team_lvl6_spike_diff")
    mix = _f("resist_mismatch_proxy")
    ad_edge = _f("team_ad_power_diff")
    ap_edge = _f("team_ap_power_diff")

    ranked: List[Tuple[str, float, str]] = []
    # ALL_IN: strong engage stats and/or level spikes, not behind in EHP, non-negative vision
    score_allin = 0.0
    score_allin += 0.02*max(o, 0) + 0.0005*max(c, 0) + 0.02*max(lvl, 0) + 0.01*max(e, 0)
    score_allin += 0.01*max(mix, 0) + 0.005*max(ad_edge, 0) + 0.005*max(ap_edge, 0)
    if vision >= 0: score_allin += 0.01
    why_allin = []
    if o > 0: why_allin.append("offense lead")
    if c > 0: why_allin.append("cc lead")
    if lvl > 0: why_allin.append("level spike")
    if e > 0: why_allin.append("ehp lead")
    if mix > 0: why_allin.append("damage vs resist edge")
    if vision >= 0: why_allin.append("vision ok")
    ranked.append(("ALL_IN", float(score_allin), "+".join(why_allin)))

    # POKE: pen/haste and offense tempo, but squishier or uncertain vision
    score_poke = 0.0
    score_poke += 0.015*max(o, 0) + 0.01*max(h, 0) + 0.01*max(pen, 0)
    if e < 0: score_poke += 0.02
    if mix < 0: score_poke += 0.01
    if vision >= 0: score_poke += 0.005
    why_poke = []
    if pen > 0: why_poke.append("pen lead")
    if h > 0: why_poke.append("haste tempo")
    if o > 0: why_poke.append("dps edge")
    if e < 0: why_poke.append("squishy")
    if mix < 0: why_poke.append("poor damage match")
    ranked.append(("POKE", float(score_poke), "+".join(why_poke)))

    # FIGHT_SLOW: behind on ehp or levels or vision; rely on sustain/tenacity and avoid flips
    score_slow = 0.0
    if e < 0: score_slow += 0.03
    if lvl < 0: score_slow += 0.02
    if vision < 0: score_slow += 0.015
    score_slow += 0.01*max(sustain, 0) + 0.005*max(ten, 0)
    why_slow = []
    if e < 0: why_slow.append("ehp deficit")
    if lvl < 0: why_slow.append("level deficit")
    if vision < 0: why_slow.append("vision deficit")
    if sustain > 0: why_slow.append("sustain edge")
    if ten > 0: why_slow.append("tenacity edge")
    ranked.append(("FIGHT_SLOW", float(score_slow), "+".join(why_slow)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    if include_warnings:
        warns = _compute_warnings(state, action=action)
        # Severity score: count + presence of execute/fed/cc issues
        sev = 0.02*len(warns)
        if any("execute" in w.lower() for w in warns):
            sev += 0.03
        if any("fed" in w.lower() for w in warns):
            sev += 0.02
        if any("vision" in w.lower() for w in warns):
            sev += 0.01
        ranked.insert(0, ("WARNINGS", float(sev), "; ".join(warns)))
    return ranked

def list_all_actions() -> List[str]:
    return list(ACTIONS)

def rank_by_strategy(strategy: str, model, features_df: pd.DataFrame, ev_mapper=None) -> List[Tuple[str, float, str]]:
    row = features_df.iloc[0].to_dict()
    cand = action_candidates(row)
    base_p = 0.5
    if hasattr(model, "predict_proba"):
        try:
            base_p = float(model.predict_proba(features_df)[:,1][0])
        except Exception:
            base_p = 0.5
    elif strategy == "regression_ev" and model is not None and ev_mapper is not None:
        try:
            ev = model.predict(features_df)[0].reshape(1, -1)
            base_p = float(ev_mapper.predict_proba(ev)[:,1][0])
        except Exception:
            base_p = 0.5

    t = float(row.get("time_s",0))
    baron_live = int(row.get("baron_live",0)); dragon_live = int(row.get("dragon_live",1))
    vision = float(row.get("vision_delta",0))
    skirm = int(row.get("skirmish_flag",0))
    plates_left = int(row.get("plates_time_left",0))
    tower_mid_t1 = float(row.get("t1_mid_diff",0))
    herald_diff = float(row.get("herald_diff",0))
    dragon_diff = float(row.get("dragon_diff",0))
    dead_diff = int(row.get("dead_diff",0))
    roll_recent_kills = int(row.get("roll_recent_kills_30s", 0))
    roll_vision_30s = float(row.get("roll_vision_delta_30s", 0.0))
    ally_support_engage = int(row.get("ally_support_engage", 0))
    has_herald = int(row.get("has_herald",0))

    ranked = []
    # ELO profile bias (env ELO_PROFILE or default 'balanced')
    elo_profile = os.getenv("ELO_PROFILE", "balanced").lower()

    for a in cand:
        score = base_p
        why = []
        phase = _phase_from_t(t)
        # Objective setup and takes
        if a.startswith("SETUP_DRAGON") and dragon_live:
            score += 0.03 + 0.01*max(vision,0); why.append("dragon_live+vision")
        if a == "TAKE_DRAGON" and dragon_live and vision >= 1 and skirm==0:
            bonus = 0.05
            if dead_diff < 0: bonus += 0.02; why.append("numbers advantage")
            score += bonus; why.append("secure dragon")
        if a.startswith("SETUP_HERALD") and t < 14*60 and plates_left>0:
            score += 0.04; why.append("plates value")
        if a == "TAKE_HERALD" and t < 14*60 and vision>=0:
            score += 0.03; why.append("tempo herald")
        if a.startswith("SETUP_BARON") and baron_live:
            score += 0.02 + 0.01*max(vision,0); why.append("baron live")
        if a == "TAKE_BARON" and baron_live and vision>=2 and skirm==0:
            bonus = 0.06
            if dead_diff < 0: bonus += 0.03; why.append("numbers advantage")
            score += bonus; why.append("secure baron")
        if a.startswith("PRESS_TOWER_MID_T1") and plates_left>0 and tower_mid_t1<=0:
            score += 0.03; why.append("open mid T1")
        # Picks/ganks
        if a.startswith("LOOK_FOR_PICK") and skirm==0 and vision>=0:
            score += 0.01; why.append("pick opportunity")
        if a in ("GANK_TOP","GANK_MID","GANK_BOT") and skirm==0 and vision>=0:
            bump = 0.012
            if plates_left>0 and a in ("GANK_TOP","GANK_BOT"):
                bump += 0.006; why.append("plates")
            if ally_support_engage and a == "GANK_BOT":
                bump += 0.006; why.append("engage support")
            score += bump; why.append("lane gank")
        # Join fights
        if a in ("JOIN_FIGHT_TOP","JOIN_FIGHT_MID","JOIN_FIGHT_BOT","JOIN_FIGHT_TOP_JUNGLE","JOIN_FIGHT_BOT_JUNGLE"):
            if skirm:
                bump = 0.018 + 0.006*max(vision,0)
                if roll_recent_kills >= 2:
                    bump += 0.006; why.append("hot fight 30s")
                # phase-based lane weighting: herald side early, dragon side pre-20, mid always viable
                if a in ("JOIN_FIGHT_TOP","JOIN_FIGHT_TOP_JUNGLE") and phase in (1,2):
                    bump += 0.008; why.append("herald side fight")
                if a in ("JOIN_FIGHT_BOT","JOIN_FIGHT_BOT_JUNGLE") and phase in (2,3):
                    bump += 0.008; why.append("dragon side fight")
                if a == "JOIN_FIGHT_MID":
                    bump += 0.004; why.append("mid priority")
                if ally_support_engage and a in ("JOIN_FIGHT_BOT","JOIN_FIGHT_BOT_JUNGLE"):
                    bump += 0.006; why.append("engage support")
                score += bump; why.append("active fight")
            else:
                score += 0.0
        # Farming
        if a in ("FARM_BLUE","FARM_GROMP","FARM_WOLVES","FARM_RAPTORS","FARM_RED","FARM_KRUGS") and skirm==0 and plates_left>0:
            score += 0.005; why.append("safe farm")
        if a in ("CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS") and skirm==0:
            bump = 0.006
            # early herald bias top, mid/drag bias bot
            if a == "CLEAR_TOP_CAMPS" and t < 14*60:
                bump += 0.006; why.append("herald side")
            if a == "CLEAR_BOT_CAMPS" and t >= 8*60 and t < 20*60:
                bump += 0.006; why.append("dragon side")
            if plates_left>0:
                bump += 0.002
            score += bump; why.append("grouped farm")
        # Separate scuttle control
        if a in ("SECURE_SCUTTLE_TOP","SECURE_SCUTTLE_BOT") and skirm==0:
            bump = 0.008
            # prefer vision advantage for scuttle takes
            if vision >= 0: bump += 0.004; why.append("vision ok")
            # phase weighting: top scuttle earlier (herald side), bot scuttle pre-20 (dragon side)
            if a == "SECURE_SCUTTLE_TOP" and t < 14*60:
                bump += 0.006; why.append("top river early")
            if a == "SECURE_SCUTTLE_BOT" and t >= 8*60 and t < 20*60:
                bump += 0.006; why.append("bot river pre-20")
            # objective proximity headcount hints
            try:
                nb = float(state.get("near_baron_count_diff", 0))
                nd = float(state.get("near_dragon_count_diff", 0))
            except Exception:
                nb = nd = 0.0
            if a == "SECURE_SCUTTLE_TOP" and nb > 0:
                bump += 0.004; why.append("allys near top river")
            if a == "SECURE_SCUTTLE_BOT" and nd > 0:
                bump += 0.004; why.append("allys near bot river")
            score += bump; why.append("secure scuttle")
        # Invades
        if a in ("INVADE_TOP_CAMPS","INVADE_BOT_CAMPS") and skirm==0 and vision>=1:
            bump = 0.014
            if dead_diff < 0:
                bump += 0.008; why.append("man advantage")
            if roll_vision_30s > 0:
                bump += 0.004; why.append("vision swing 30s")
            if phase in (2,3) and a == "INVADE_BOT_CAMPS":
                bump += 0.006; why.append("dragon timing")
            if phase in (1,2) and a == "INVADE_TOP_CAMPS":
                bump += 0.006; why.append("herald timing")
            score += bump; why.append("vision lead invade")
        # Herald usage
        if a in ("USE_HERALD","use_herald") and has_herald:
            bump = 0.02
            if plates_left>0:
                bump += 0.02; why.append("plates value")
            score += bump; why.append("use herald")
        if a == "RESET_BUY" and skirm==0:
            score += 0.0; why.append("reset option")
        # Apply ELO-based action bias
        eb = _elo_bias_for_action(elo_profile, a)
        if eb != 0:
            score += eb
            why.append(f"elo:{elo_profile}")
        # Apply Win-Con bias
        wincon = os.getenv("WIN_CON_PROFILE", "balanced").lower()
        wb = _wincon_bias_for_action(wincon, a, row)
        if wb != 0:
            score += wb
            why.append(f"wincon:{wincon}")
        ranked.append((a, float(score), "+".join(why) if why else ""))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

def dynamic_risk_threshold(state: Dict[str, float], base_thresh: float, profile: str = "auto") -> Tuple[float, str]:
    """Compute dynamic decision threshold based on game state and phase.

    Toggle: pass profile='safe'|'balanced'|'flip' to force bias; 'auto' adapts.
    """
    def _f(k, d=0.0):
        try:
            return float(state.get(k, d) or 0.0)
        except Exception:
            return d
    t = _f("time_s", 0.0)
    phase = _phase_from_t(t)
    vision = _f("vision_delta", 0.0)
    fed = _f("enemy_fed_proxy", 0.0)
    ehp = _f("team_ehp_proxy_diff", 0.0)
    off = _f("team_offense_power_diff", 0.0)
    ccd = _f("team_cc_ms_diff", 0.0)
    mix = _f("resist_mismatch_proxy", 0.0)
    soul_enemy = int(state.get("enemy_soul_point", 0))
    baron_live = int(state.get("baron_live", 0))
    skirm = int(state.get("skirmish_flag", 0))

    saf = 0.0
    if fed > 800: saf += 0.04
    if vision < 0: saf += 0.02
    if ehp < 0: saf += 0.02
    if soul_enemy == 1 and t < 30*60: saf += 0.03
    if baron_live and vision < 0: saf += 0.02
    if phase in (1,2): saf += 0.01

    agr = 0.0
    if off > 0: agr += 0.02
    if ccd > 0: agr += 0.005
    if mix > 0: agr += 0.01
    if skirm and vision >= 0: agr += 0.01
    if phase in (4,5): agr += 0.01

    adj = max(-0.06, min(0.06, saf - agr))
    if profile == "safe":
        adj += 0.02
    elif profile == "flip":
        adj -= 0.02
    # 'balanced' or 'auto' keep adj as-is

    out = max(0.05, min(0.95, base_thresh + adj))
    label = f"auto(+{adj:.2f})" if profile == "auto" else f"{profile}(+{adj:.2f})"
    return out, label

def _elo_bias_for_action(profile: str, action: str) -> float:
    """ELO-profile bias for action categories.

    Profiles:
      - 'low' / 'low_safe' / 'farm-heavy': safer, farm/vision favored
      - 'balanced' / 'mid' : neutral
      - 'high' / 'high_aggressive' / 'aggressive' / 'proactive' / 'tempo': proactive/tempo favored
    Returns a small additive bump to score.
    """
    p = profile.lower()
    # Map profile aliases
    if p in ("low", "safe", "low_safe", "farm-heavy", "farm", "scaling"):
        cat = "low"
    elif p in ("high", "aggro", "aggressive", "high_aggressive", "proactive", "tempo"):
        cat = "high"
    else:
        cat = "balanced"

    a = action.upper()
    # Categorize action
    is_farm = a.startswith("FARM_") or a in ("CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS")
    is_scuttle = a.startswith("SECURE_SCUTTLE")
    is_invade = a.startswith("INVADE_")
    is_gank = a.startswith("GANK_")
    is_press = a.startswith("PRESS_")
    is_setup = a.startswith("SETUP_") or a in ("USE_HERALD","use_herald".upper())
    is_take = a.startswith("TAKE_")
    is_join = a.startswith("JOIN_")
    is_vision = a == "DEEP_VISION_SWEEP"
    is_reset = a == "RESET_BUY"

    bump = 0.0
    if cat == "low":
        if is_farm: bump += 0.012
        if is_scuttle: bump += 0.002
        if is_setup or is_take: bump += 0.006
        if is_vision: bump += 0.006
        if is_reset: bump += 0.006
        if is_invade: bump -= 0.010
        if is_gank: bump -= 0.006
        if is_press: bump -= 0.006
        if is_join: bump -= 0.006
    elif cat == "high":
        if is_invade: bump += 0.012
        if is_gank: bump += 0.010
        if is_press: bump += 0.010
        if is_join: bump += 0.010
        if is_setup or is_take: bump += 0.008
        if is_scuttle: bump += 0.006
        if is_farm: bump -= 0.008
        if is_reset: bump -= 0.006
    # balanced -> 0.0
    return bump

def _wincon_bias_for_action(profile: str, action: str, state: Dict[str, float]) -> float:
    """Win-condition bias for actions.

    Profiles (aliases):
      - split / 1-3-1 / 1-4: favor side pressure, herald/tower usage; de-emphasize river fights
      - pick: favor pick setup, deep vision, ganks; de-emphasize all-in teamfights
      - siege: favor press tower mid/side, herald usage, setup vision
      - objective / obj: favor setup/take objectives, scuttle control, river fights
      - scaling / farm: favor farm/reset, avoid high-variance invades and early flips
      - teamfight / 5v5: favor join fights, press mid, objective contests
      - balanced: neutral
    """
    p = profile.lower()
    if p in ("balanced", "none", "", "neutral"):
        return 0.0
    if p in ("split", "1-3-1", "1-4", "splitpush", "split-push"):
        mode = "split"
    elif p in ("pick", "pickcomp", "pick-comp"):
        mode = "pick"
    elif p in ("siege", "sieging"):
        mode = "siege"
    elif p in ("objective", "obj", "objectives"):
        mode = "objective"
    elif p in ("scaling", "farm", "farm-heavy"):
        mode = "scaling"
    elif p in ("teamfight", "5v5", "team-fight"):
        mode = "teamfight"
    else:
        mode = "balanced"

    a = action.upper()
    is_farm = a.startswith("FARM_") or a in ("CLEAR_TOP_CAMPS","CLEAR_BOT_CAMPS")
    is_scuttle = a.startswith("SECURE_SCUTTLE")
    is_invade = a.startswith("INVADE_")
    is_gank = a.startswith("GANK_") or a.startswith("LOOK_FOR_PICK")
    is_press = a.startswith("PRESS_")
    is_setup = a.startswith("SETUP_") or a in ("USE_HERALD","USE_HERALD".upper())
    is_take = a.startswith("TAKE_")
    is_join = a.startswith("JOIN_")
    is_vision = a == "DEEP_VISION_SWEEP"
    is_reset = a == "RESET_BUY"

    bump = 0.0
    if mode == "split":
        if is_press: bump += 0.012
        if is_setup or a in ("USE_HERALD",): bump += 0.008
        if is_join: bump -= 0.010
        if is_take and "DRAGON" in a: bump -= 0.006  # deprioritize early dragon brawls
        if is_farm: bump += 0.004
    elif mode == "pick":
        if is_gank: bump += 0.012
        if is_vision: bump += 0.010
        if is_invade: bump += 0.006
        if is_join and "MID" in a: bump -= 0.004
        if is_take and "BARON" in a: bump -= 0.004
    elif mode == "siege":
        if is_press: bump += 0.012
        if is_setup or a in ("USE_HERALD",): bump += 0.010
        if is_vision: bump += 0.006
        if is_join and ("TOP" in a or "BOT" in a): bump += 0.004  # collapse side
        if is_invade: bump -= 0.006
    elif mode == "objective":
        if is_setup or is_take: bump += 0.012
        if is_scuttle: bump += 0.006
        if is_vision: bump += 0.006
        if is_farm: bump -= 0.004
    elif mode == "scaling":
        if is_farm or is_reset: bump += 0.012
        if is_join or is_invade or is_take: bump -= 0.008
    elif mode == "teamfight":
        if is_join: bump += 0.012
        if is_setup or is_take: bump += 0.008
        if is_press and "MID" in a: bump += 0.006
        if is_invade: bump -= 0.004
    return bump
