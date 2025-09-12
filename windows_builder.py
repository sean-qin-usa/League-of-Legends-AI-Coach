"""Build per-team time-window features from Riot match + timeline.

Generates team, lane-standardized, positioning, and fight-impact features.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

ROLL_WINDOW_S = 30

RICH_COLUMNS = [
    "game_id","time_s","team","win",
    "role","window_type","action_space","action","reward",
    "gold_xp_delta","dragon_diff","baron_diff","vision_delta","herald_diff",
    "tower_diff","plate_diff","ward_kill_diff",
    "t1_top_diff","t1_mid_diff","t1_bot_diff","t2_top_diff","t2_mid_diff","t2_bot_diff",
    "t3_top_diff","t3_mid_diff","t3_bot_diff",
    "inh_top_diff","inh_mid_diff","inh_bot_diff",
    "team_kills_d","team_deaths_d","skirmish_flag","plates_time_left",
    "phase_num","baron_live","dragon_live",
    "flip_risk_proxy","setup_quality_score","recall_sync_score","crossmap_ev_proxy","smite_diff_proxy",
    # Additional aggregates for richer modeling
    "team_gold_diff","team_level_avg_diff","team_cs_diff",
    # Cumulative objective/vision pressure
    "dragon_diff_cum","herald_diff_cum","baron_diff_cum","tower_diff_cum","ward_kill_diff_cum",
    # Normalized economy deltas
    "team_gold_diff_per_min","team_cs_diff_per_min",
    # Rolling recent-pressure features (last 30s)
    "roll_kills_30s","roll_deaths_30s","roll_vision_delta_30s","roll_ward_kill_diff_30s",
    "roll_tower_diff_30s","roll_plate_diff_30s","roll_dragon_diff_30s","roll_herald_diff_30s","roll_baron_diff_30s",
    # Team composition signals
    "team_engage_support","team_engage_count","team_disengage_count",
    # Lane-standardized features (diff: team - opp per lane)
    "top_item_gold_diff","jng_item_gold_diff","mid_item_gold_diff","bot_item_gold_diff","sup_item_gold_diff",
    "top_cc_ms_diff","jng_cc_ms_diff","mid_cc_ms_diff","bot_cc_ms_diff","sup_cc_ms_diff",
    "top_cs_diff","jng_cs_diff","mid_cs_diff","bot_cs_diff","sup_cs_diff",
    "top_xp_diff","jng_xp_diff","mid_xp_diff","bot_xp_diff","sup_xp_diff",
    # Teamfight readiness and stats diffs
    "team_cc_ms_diff","team_offense_power_diff","team_resists_sum_diff","team_ehp_proxy_diff",
    "team_lvl6_spike_diff","team_lvl11_spike_diff","team_lvl16_spike_diff",
    "team_ability_haste_diff","team_penetration_proxy_diff",
    # Additional combat proxies
    "team_tenacity_proxy_diff","team_sustain_vamp_diff",
    # Enemy composition flags and fed proxy
    "enemy_has_pyke","enemy_has_chogath","enemy_has_urgot","enemy_has_darius",
    "team_max_gold_diff","enemy_fed_proxy","enemy_cc_ms",
    "enemy_max_armor","enemy_max_mr","enemy_high_armor","enemy_high_mr",
    "enemy_high_armor_champ","enemy_high_mr_champ",
    # Positioning and objective proximity
    "near_baron_count_diff","near_dragon_count_diff",
    "centroid_dist_baron_diff","centroid_dist_dragon_diff",
    "team_spread_std_diff",
    # Lane level diffs (avg level per lane)
    "top_level_diff","jng_level_diff","mid_level_diff","bot_level_diff","sup_level_diff",
]

def _pid_team_map(match: Dict[str, Any]):
    info = (match or {}).get("info") or {}
    m={}
    for p in info.get("participants") or []:
        pid = int(p.get("participantId") or 0)
        m[pid] = int(p.get("teamId") or 0)
    return m

def _team_win_map(match: Dict[str, Any]):
    info = (match or {}).get("info") or {}; wins={}
    for t in info.get("teams") or []:
        tid = int(t.get("teamId") or 0)
        wins[tid] = int(bool(t.get("win")))
    return wins

def _team_comp_features(match: Dict[str, Any]):
    info = (match or {}).get("info") or {}
    parts = info.get("participants") or []
    # Champion sets (lightweight, extensible)
    ENGAGE_SUPPORTS = {"Leona","Nautilus","Rell","Alistar","Rakan","Thresh","Blitzcrank","Braum","Pyke"}
    HARD_ENGAGE = {"Malphite","Sejuani","Wukong","Zac","Jarvan IV","Amumu","Rell","Nautilus","Leona","Rakan","Hecarim","Nocturne"}
    DISENGAGE = {"Janna","Gragas","Nami","Alistar","Braum","Thresh","Lulu"}
    # Initialize per-team aggregates
    comp = {100:{"engage_support":0,"engage_count":0,"disengage_count":0,
                 "has_pyke":0,"has_chogath":0,"has_urgot":0,"has_darius":0},
            200:{"engage_support":0,"engage_count":0,"disengage_count":0,
                 "has_pyke":0,"has_chogath":0,"has_urgot":0,"has_darius":0}}
    # Identify supports via teamPosition/individualPosition or lowest CS on team
    by_team = {100:[],200:[]}
    for p in parts:
        tid = int(p.get("teamId") or 0)
        if tid in by_team:
            by_team[tid].append(p)
    for tid, lst in by_team.items():
        # engage/disengage counts by team + specific champ flags
        for p in lst:
            champ = (p.get("championName") or "").strip()
            if champ in HARD_ENGAGE: comp[tid]["engage_count"] += 1
            if champ in DISENGAGE: comp[tid]["disengage_count"] += 1
            lc = champ.lower()
            if lc == "pyke": comp[tid]["has_pyke"] = 1
            if lc == "cho'gath" or lc == "chogath": comp[tid]["has_chogath"] = 1
            if lc == "urgot": comp[tid]["has_urgot"] = 1
            if lc == "darius": comp[tid]["has_darius"] = 1
        # support detection
        support = None
        # Prefer teamPosition/individualPosition
        for p in lst:
            pos = (p.get("teamPosition") or p.get("individualPosition") or "").upper()
            if pos in ("UTILITY","SUPPORT","DUO_SUPPORT"):
                support = p; break
        if support is None and lst:
            # Fallback to lowest CS
            support = min(lst, key=lambda q: int((q.get("totalMinionsKilled") or 0)))
        if support:
            champ = (support.get("championName") or "").strip()
            if champ in ENGAGE_SUPPORTS:
                comp[tid]["engage_support"] = 1
    return comp

def _event_buckets(events: List[dict]) -> Dict[str, Any]:
    k = defaultdict(int); d = defaultdict(int)
    team_obj = defaultdict(lambda: defaultdict(int))
    w_p = defaultdict(int); w_k = defaultdict(int)
    lane = defaultdict(lambda: defaultdict(int))
    for ev in events or []:
        et = ev.get("type")
        if et == "CHAMPION_KILL":
            vic = int(ev.get("victimId") or 0)
            kil = int(ev.get("killerId") or 0)
            if kil: k[kil]+=1
            if vic: d[vic]+=1
        elif et in ("ELITE_MONSTER_KILL","DRAGON_KILL"):
            mt = (ev.get("monsterType") or "").upper()
            if "DRAGON" in mt: tag="drag"
            elif "HERALD" in mt: tag="herald"
            elif "BARON" in mt: tag="baron"
            else: tag="elite"
            tid = ev.get("killerTeamId") or ev.get("teamId")
            if tid: team_obj[int(tid)][tag]+=1
        elif et == "BUILDING_KILL":
            b = (ev.get("buildingType") or "").upper()
            tid = ev.get("teamId")
            if not tid: 
                continue
            tid = int(tid)
            ltype = (ev.get("laneType") or "").upper()
            towerType = (ev.get("towerType") or "").upper()
            if b == "TOWER_BUILDING":
                team_obj[tid]["tower"] += 1
                key = None
                if "OUTER_TURRET" in towerType:
                    key = "t1_" + ("mid" if "MID" in ltype else "top" if "TOP" in ltype else "bot")
                elif "INNER_TURRET" in towerType:
                    key = "t2_" + ("mid" if "MID" in ltype else "top" if "TOP" in ltype else "bot")
                elif "BASE_TURRET" in towerType:
                    key = "t3_" + ("mid" if "MID" in ltype else "top" if "TOP" in ltype else "bot")
                if key: lane[tid][key] += 1
            elif b == "INHIBITOR_BUILDING":
                key = "inh_" + ("mid" if "MID" in ltype else "top" if "TOP" in ltype else "bot")
                lane[tid][key] += 1
        elif et == "TURRET_PLATE_DESTROYED":
            tid = ev.get("teamId")
            if tid: team_obj[int(tid)]["plate"]+=1
        elif et == "WARD_PLACED":
            pid = int(ev.get("creatorId") or 0); w_p[pid]+=1
        elif et == "WARD_KILL":
            pid = int(ev.get("killerId") or 0); w_k[pid]+=1
    return {"kills":dict(k),"deaths":dict(d),
            "team_obj":{k:dict(v) for k,v in team_obj.items()},
            "ward_place":dict(w_p), "ward_kill":dict(w_k),
            "lane": {k: dict(v) for k,v in lane.items()}}

def _phase_num(t: int) -> int:
    if t < 8*60: return 1
    if t < 14*60: return 2
    if t < 20*60: return 3
    if t < 30*60: return 4
    return 5

def _spawns(t: int):
    return int(t >= 20*60), 1

def _flip_risk_proxy(vision_delta: int, skirmish_flag: int, baron_live: int, dragon_live: int) -> float:
    base = 0.5*skirmish_flag + 0.4*baron_live + 0.2*dragon_live
    return float(base - 0.02*max(-vision_delta,0))

def _setup_quality(vision_delta: int, phase: int, plates_time_left: int) -> float:
    s = 0.03*vision_delta + (1 if phase in (2,3) else 0)*0.2 + (1 if plates_time_left>0 else 0)*0.1
    return float(max(min(s,1.0), -1.0))

def _recall_sync(skirmish_flag: int, tk: int, td: int) -> float:
    return float(0.3*skirmish_flag + 0.02*(tk-td))

def _crossmap_proxy(tower_diff: int, herald_diff: int) -> float:
    return float(0.1*tower_diff + 0.15*herald_diff)

def _aggregate_team_stats_from_pframes(pframes: Dict[str, Any], pid2team: Dict[int, int]):
    gold = {100:0.0, 200:0.0}; lvl = {100:[], 200:[]}; cs = {100:0, 200:0}; max_gold = {100:0.0, 200:0.0}
    for pid_str, pf in (pframes or {}).items():
        try:
            pid_i = int(pid_str)
        except Exception:
            pid_i = int((pf or {}).get("participantId") or 0)
        team_id = pid2team.get(pid_i)
        if team_id not in (100,200):
            continue
        try:
            tg = float((pf or {}).get("totalGold") or 0.0)
            gold[team_id] += tg
            if tg > max_gold[team_id]:
                max_gold[team_id] = tg
        except Exception:
            pass
        try:
            lvl[team_id].append(int((pf or {}).get("level") or 0))
        except Exception:
            pass
        try:
            cs_val = int((pf or {}).get("minionsKilled") or 0) + int((pf or {}).get("jungleMinionsKilled") or 0)
            cs[team_id] += cs_val
        except Exception:
            pass
    return gold, lvl, cs, max_gold

def build_windows_v7_for_match(match: Dict[str, Any], timeline: Dict[str, Any], window_s: int=5, champ_filter: Optional[str]=None) -> pd.DataFrame:
    if not match or not timeline:
        return pd.DataFrame(columns=RICH_COLUMNS)
    meta = (match or {}).get("metadata") or {}
    match_id = meta.get("matchId")
    frames = ((timeline or {}).get("info") or {}).get("frames") or []
    if not frames:
        return pd.DataFrame(columns=RICH_COLUMNS)
    # Participant team map
    info = (match or {}).get("info") or {}
    pid2team = {}
    for p in info.get("participants") or []:
        pid2team[int(p.get("participantId") or 0)] = int(p.get("teamId") or 0)
    # Participant lane map (standardized lanes)
    def _lane_tag(p):
        pos = (p.get("teamPosition") or p.get("individualPosition") or "").upper()
        if pos in ("TOP"):
            return "top"
        if pos in ("JUNGLE", "JUNGLE_SUPPORT"):
            return "jng"
        if pos in ("MIDDLE", "MID"):
            return "mid"
        if pos in ("BOTTOM", "ADC", "DUO_CARRY"):
            return "bot"
        if pos in ("UTILITY", "SUPPORT", "DUO_SUPPORT"):
            return "sup"
        return None
    pid2lane = {}
    for p in info.get("participants") or []:
        pid2lane[int(p.get("participantId") or 0)] = _lane_tag(p)
    # Participant champion map
    pid2champ = {}
    for p in info.get("participants") or []:
        pid2champ[int(p.get("participantId") or 0)] = (p.get("championName") or "").strip()
    # Optional filter: only include matches where a given champion appears
    if champ_filter:
        name = champ_filter.strip().lower()
        has = False
        for p in info.get("participants") or []:
            cn = (p.get("championName") or "").strip().lower()
            if name in cn:
                has = True; break
        if not has:
            return pd.DataFrame(columns=RICH_COLUMNS)
    # Team comp features (static across match)
    comp = _team_comp_features(match)
    # Win map
    wins = {}
    for t in info.get("teams") or []:
        wins[int(t.get("teamId") or 0)] = int(bool(t.get("win")))

    rows = []
    # cumulative trackers per team across frames
    cum = {100:{"drag":0,"herald":0,"baron":0,"tower":0,"ward_kill":0},
           200:{"drag":0,"herald":0,"baron":0,"tower":0,"ward_kill":0}}

    for idx, f in enumerate(frames):
        ts_ms = int(f.get("timestamp") or 0)
        next_ts_ms = int(frames[idx+1].get("timestamp") or (ts_ms+60000)) if (idx+1) < len(frames) else (ts_ms+60000)
        ev_all = f.get("events") or []
        pframes = (f.get("participantFrames") or {})
        gold, lvl, cs, max_gold = _aggregate_team_stats_from_pframes(pframes, pid2team)
        # Next frame aggregates for interpolation
        if (idx+1) < len(frames):
            pframes_n = (frames[idx+1].get("participantFrames") or {})
            gold_n, lvl_n, cs_n, max_gold_n = _aggregate_team_stats_from_pframes(pframes_n, pid2team)
        else:
            gold_n, lvl_n, cs_n, max_gold_n = gold, lvl, cs, max_gold

        # Iterate 5s bins within this frame range
        start_s = ts_ms//1000
        end_s = max(start_s+1, next_ts_ms//1000)
        # align to window_s grid
        grid_start = (start_s//window_s)*window_s
        for bin_start in range(grid_start, end_s, window_s):
            bin_end = min(end_s, bin_start + window_s)
            # Filter events belonging to this bin
            ev = [e for e in ev_all if bin_start <= int((e.get("timestamp") or 0)//1000) < bin_end]
            b = _event_buckets(ev)
            k_map, d_map = b["kills"], b["deaths"]
            team_obj, w_place, w_kill, lane = b["team_obj"], b["ward_place"], b["ward_kill"], b["lane"]
            # Lane aggregates from participantFrames
            lane_agg = {100: {"top": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "jng": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "mid": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "bot": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "sup": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0}},
                        200: {"top": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "jng": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "mid": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "bot": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0},
                               "sup": {"item":0.0, "cc":0.0, "cs":0, "xp":0, "gold":0.0}}}
            # Additional per-bin aggregations for positions and levels per lane
            lane_level_sum = {100:{"top":0.0,"jng":0.0,"mid":0.0,"bot":0.0,"sup":0.0},
                              200:{"top":0.0,"jng":0.0,"mid":0.0,"bot":0.0,"sup":0.0}}
            lane_level_cnt = {100:{"top":0,"jng":0,"mid":0,"bot":0,"sup":0},
                              200:{"top":0,"jng":0,"mid":0,"bot":0,"sup":0}}
            BARON_POS = (5007.0, 10471.0)
            DRAGON_POS = (9866.0, 4414.0)
            OBJ_R2 = float(2500.0**2)
            near_obj = {100:{"baron":0, "dragon":0}, 200:{"baron":0, "dragon":0}}
            centroid = {100:[0.0,0.0,0], 200:[0.0,0.0,0]}
            accum_pos = {100:[], 200:[]}
            team_cc = {100:0.0, 200:0.0}
            team_offense = {100:0.0, 200:0.0}
            team_resists = {100:0.0, 200:0.0}
            team_ad = {100:0.0, 200:0.0}
            team_ap = {100:0.0, 200:0.0}
            team_armor = {100:0.0, 200:0.0}
            team_mr = {100:0.0, 200:0.0}
            team_ehp = {100:0.0, 200:0.0}
            team_haste = {100:0.0, 200:0.0}
            team_pen = {100:0.0, 200:0.0}
            team_tenacity = {100:0.0, 200:0.0}
            team_vamp = {100:0.0, 200:0.0}
            # Track maximum per-team resist holders
            max_armor_val = {100:0.0, 200:0.0}
            max_mr_val = {100:0.0, 200:0.0}
            max_armor_champ = {100:"", 200:""}
            max_mr_champ = {100:"", 200:""}
            lvl6 = {100:0,200:0}; lvl11 = {100:0,200:0}; lvl16 = {100:0,200:0}
            for pid_str, pf in (pframes or {}).items():
                try:
                    pid = int(pid_str)
                except Exception:
                    pid = int((pf or {}).get("participantId") or 0)
                lane_tag = pid2lane.get(pid)
                team_id = pid2team.get(pid)
                if lane_tag is None or team_id not in (100,200):
                    continue
                try:
                    tot = float((pf or {}).get("totalGold") or 0.0)
                    cur = float((pf or {}).get("currentGold") or 0.0)
                    spent = max(0.0, tot - cur)  # proxy for item value owned
                except Exception:
                    spent = 0.0
                cc_ms = float((pf or {}).get("timeEnemySpentControlled") or 0.0)
                try:
                    cs_val = int((pf or {}).get("minionsKilled") or 0) + int((pf or {}).get("jungleMinionsKilled") or 0)
                except Exception:
                    cs_val = 0
                try:
                    xp_val = int((pf or {}).get("xp") or 0)
                except Exception:
                    xp_val = 0
                # Level and lane-level aggregates
                try:
                    lvlv = int((pf or {}).get("level") or 0)
                except Exception:
                    lvlv = 0
                lane_agg[team_id][lane_tag]["item"] += spent
                lane_agg[team_id][lane_tag]["cc"] += cc_ms
                lane_agg[team_id][lane_tag]["cs"] += cs_val
                lane_agg[team_id][lane_tag]["xp"] += xp_val
                try:
                    tg = float((pf or {}).get("totalGold") or 0.0)
                except Exception:
                    tg = 0.0
                lane_agg[team_id][lane_tag]["gold"] += tg
                if lane_tag in lane_level_sum[team_id]:
                    lane_level_sum[team_id][lane_tag] += lvlv
                    lane_level_cnt[team_id][lane_tag] += 1
                # Teamfight readiness stats from championStats
                csn = (pf or {}).get("championStats") or {}
                ad = float(csn.get("attackDamage") or 0.0)
                ap = float(csn.get("abilityPower") or 0.0)
                armor = float(csn.get("armor") or 0.0)
                mr = float(csn.get("magicResist") or 0.0)
                hpmax = float(csn.get("healthMax") or 0.0)
                haste = float(csn.get("abilityHaste") or 0.0)
                # Penetration proxies (percents only; flat may exist but is champ dependent)
                pen = float(csn.get("armorPenPercent") or 0.0) + float(csn.get("bonusArmorPenPercent") or 0.0) \
                      + float(csn.get("magicPenPercent") or 0.0) + float(csn.get("bonusMagicPenPercent") or 0.0)
                team_cc[team_id] += cc_ms
                team_offense[team_id] += (ad + 1.5*ap)
                team_resists[team_id] += (armor + mr)
                # Simple effective HP proxy vs mixed damage: hp*(1+ (armor+mr)/200)
                team_ehp[team_id] += hpmax * (1.0 + (armor + mr)/200.0)
                team_haste[team_id] += haste
                team_pen[team_id] += pen
                team_ad[team_id] += ad
                team_ap[team_id] += ap
                team_armor[team_id] += armor
                team_mr[team_id] += mr
                if armor > max_armor_val[team_id]:
                    max_armor_val[team_id] = armor
                    max_armor_champ[team_id] = pid2champ.get(pid, "")
                if mr > max_mr_val[team_id]:
                    max_mr_val[team_id] = mr
                    max_mr_champ[team_id] = pid2champ.get(pid, "")
                # Tenacity and sustain proxies
                ten = float(csn.get("ccReduction") or 0.0)
                ov = float(csn.get("omnivamp") or 0.0)
                ls = float(csn.get("lifesteal") or 0.0)
                team_tenacity[team_id] += ten
                team_vamp[team_id] += (ov + ls)
                if lvlv >= 6: lvl6[team_id] += 1
                if lvlv >= 11: lvl11[team_id] += 1
                if lvlv >= 16: lvl16[team_id] += 1
                # Position features
                pos = (pf or {}).get("position") or {}
                try:
                    x = float(pos.get("x") or 0.0); y = float(pos.get("y") or 0.0)
                except Exception:
                    x = y = 0.0
                if x or y:
                    dx_b = x-BARON_POS[0]; dy_b = y-BARON_POS[1]
                    if (dx_b*dx_b + dy_b*dy_b) <= OBJ_R2:
                        near_obj[team_id]["baron"] += 1
                    dx_d = x-DRAGON_POS[0]; dy_d = y-DRAGON_POS[1]
                    if (dx_d*dx_d + dy_d*dy_d) <= OBJ_R2:
                        near_obj[team_id]["dragon"] += 1
                    centroid[team_id][0] += x; centroid[team_id][1] += y; centroid[team_id][2] += 1
                    accum_pos[team_id].append((x,y))

            team_kd = {100:0,200:0}; team_dd={100:0,200:0}
            for pid, kd in k_map.items():
                t = pid2team.get(int(pid)); 
                if t in team_kd: team_kd[t]+=kd
            for pid, dd in d_map.items():
                t = pid2team.get(int(pid)); 
                if t in team_dd: team_dd[t]+=dd

            def _t(team,key): return int((team_obj.get(team, {}) or {}).get(key,0))
            drag = {100:_t(100,"drag"),200:_t(200,"drag")}
            baron= {100:_t(100,"baron"),200:_t(200,"baron")}
            herald={100:_t(100,"herald"),200:_t(200,"herald")}
            tower={100:_t(100,"tower"),200:_t(200,"tower")}
            plate={100:_t(100,"plate"),200:_t(200,"plate")}

            # Update cumulative counts by bin deltas
            for team in (100,200):
                cum[team]["drag"] += drag[team]
                cum[team]["herald"] += herald[team]
                cum[team]["baron"] += baron[team]
                cum[team]["tower"] += tower[team]

            def L(team, key): return int((lane.get(team, {}) or {}).get(key,0))

            wardp={100:0,200:0}; wardk={100:0,200:0}
            for pid,c in w_place.items():
                t = pid2team.get(int(pid)); 
                if t in wardp: wardp[t]+=int(c)
            for pid,c in w_kill.items():
                t = pid2team.get(int(pid)); 
                if t in wardk: wardk[t]+=int(c)
            for team in (100,200):
                cum[team]["ward_kill"] += wardk[team]

            # Interpolation factor within the minute
            span = max(1, end_s - start_s)
            alpha = float(min(max(bin_start - start_s, 0), span) / span)

            for team in (100,200):
                opp = 200 if team==100 else 100
                skirmish_flag = int((team_kd[team]+team_dd[team]+team_kd[opp]+team_dd[opp])>0)
                vision_delta = wardp[team]-wardp[opp]
                phase = _phase_num(bin_start)
                baron_live, dragon_live = _spawns(bin_start)
                # Derived diffs (copied across bins within minute)
                g_team = gold.get(team,0.0) + alpha*(gold_n.get(team,0.0) - gold.get(team,0.0))
                g_opp  = gold.get(opp,0.0)  + alpha*(gold_n.get(opp,0.0)  - gold.get(opp,0.0))
                gold_diff = float(g_team - g_opp)
                lvl_avg_team = (sum(lvl.get(team,[])) / max(1, len(lvl.get(team,[])))) if lvl.get(team) else 0.0
                lvl_avg_opp  = (sum(lvl.get(opp,[]))  / max(1, len(lvl.get(opp,[]))))  if lvl.get(opp) else 0.0
                level_avg_diff = float(lvl_avg_team - lvl_avg_opp)
                cs_team = cs.get(team,0) + alpha*(cs_n.get(team,0) - cs.get(team,0))
                cs_opp  = cs.get(opp,0)  + alpha*(cs_n.get(opp,0)  - cs.get(opp,0))
                cs_diff = int(round(cs_team - cs_opp))
                # Cumulative diffs
                dragon_diff_cum = int(cum[team]["drag"] - cum[opp]["drag"])
                herald_diff_cum = int(cum[team]["herald"] - cum[opp]["herald"])
                baron_diff_cum = int(cum[team]["baron"] - cum[opp]["baron"])
                tower_diff_cum = int(cum[team]["tower"] - cum[opp]["tower"])
                ward_kill_diff_cum = int(cum[team]["ward_kill"] - cum[opp]["ward_kill"])
                # Normalized economy
                gold_diff_per_min = float(60.0*gold_diff/max(1, bin_start))
                cs_diff_per_min = float(60.0*cs_diff/max(1, bin_start))

                # Max individual gold per team (interpolated like team gold)
                mg_team = max_gold.get(team,0.0) + alpha*(max_gold_n.get(team,0.0) - max_gold.get(team,0.0))
                mg_opp  = max_gold.get(opp,0.0)  + alpha*(max_gold_n.get(opp,0.0)  - max_gold.get(opp,0.0))
                team_max_gold_diff = float(mg_team - mg_opp)
                enemy_fed_proxy = float(max(0.0, mg_opp - mg_team))

                # Lane diffs (team - opp)
                def LDIFF(metric):
                    return {
                        "top": lane_agg[team]["top"][metric] - lane_agg[opp]["top"][metric],
                        "jng": lane_agg[team]["jng"][metric] - lane_agg[opp]["jng"][metric],
                        "mid": lane_agg[team]["mid"][metric] - lane_agg[opp]["mid"][metric],
                        "bot": lane_agg[team]["bot"][metric] - lane_agg[opp]["bot"][metric],
                        "sup": lane_agg[team]["sup"][metric] - lane_agg[opp]["sup"][metric],
                    }

                ld_item = LDIFF("item")
                ld_cc = LDIFF("cc")
                ld_cs = LDIFF("cs")
                ld_xp = LDIFF("xp")
                # Lane level diffs (average per lane)
                def LAVG(team_id_local, lane):
                    n = max(1, int(lane_level_cnt[team_id_local][lane]))
                    return float(lane_level_sum[team_id_local][lane]) / n
                lvl_diff = {
                    "top": LAVG(team,"top") - LAVG(opp,"top"),
                    "jng": LAVG(team,"jng") - LAVG(opp,"jng"),
                    "mid": LAVG(team,"mid") - LAVG(opp,"mid"),
                    "bot": LAVG(team,"bot") - LAVG(opp,"bot"),
                    "sup": LAVG(team,"sup") - LAVG(opp,"sup"),
                }
                # Objective proximity diffs
                near_baron_count_diff = int(near_obj[team]["baron"] - near_obj[opp]["baron"])
                near_dragon_count_diff = int(near_obj[team]["dragon"] - near_obj[opp]["dragon"])
                def CENTROID(team_id_local):
                    s = centroid[team_id_local]
                    if s[2] == 0:
                        return (0.0, 0.0)
                    return (s[0]/s[2], s[1]/s[2])
                c_team = CENTROID(team); c_opp = CENTROID(opp)
                def DIST(p, q):
                    dx = p[0]-q[0]; dy = p[1]-q[1]
                    return float((dx*dx + dy*dy)**0.5)
                centroid_dist_baron_diff = DIST(c_team, BARON_POS) - DIST(c_opp, BARON_POS)
                centroid_dist_dragon_diff = DIST(c_team, DRAGON_POS) - DIST(c_opp, DRAGON_POS)
                # Spread (std of distances to centroid proxy via RMS radius)
                def SPREAD(team_id_local, c):
                    pts = accum_pos[team_id_local]
                    if not pts:
                        return 0.0
                    import math
                    dsq = [(p[0]-c[0])**2 + (p[1]-c[1])**2 for p in pts]
                    mean = sum(dsq)/len(dsq)
                    return math.sqrt(max(mean, 0.0))
                spread_team = SPREAD(team, c_team)
                spread_opp = SPREAD(opp, c_opp)
                team_spread_std_diff = float(spread_team - spread_opp)

                row = dict(
                    game_id=match_id, time_s=int(bin_start), team=team, win=int(wins.get(team,0)),
                    role="JUNGLE", window_type=f"team{window_s}s", action_space="", action="", reward=np.nan,
                    gold_xp_delta=np.nan,
                    dragon_diff=drag[team]-drag[opp], baron_diff=baron[team]-baron[opp], herald_diff=herald[team]-herald[opp],
                    tower_diff=tower[team]-tower[opp], plate_diff=plate[team]-plate[opp], ward_kill_diff=wardk[team]-wardk[opp],
                    t1_top_diff=L(team,"t1_top")-L(opp,"t1_top"), t1_mid_diff=L(team,"t1_mid")-L(opp,"t1_mid"), t1_bot_diff=L(team,"t1_bot")-L(opp,"t1_bot"),
                    t2_top_diff=L(team,"t2_top")-L(opp,"t2_top"), t2_mid_diff=L(team,"t2_mid")-L(opp,"t2_mid"), t2_bot_diff=L(team,"t2_bot")-L(opp,"t2_bot"),
                    t3_top_diff=L(team,"t3_top")-L(opp,"t3_top"), t3_mid_diff=L(team,"t3_mid")-L(opp,"t3_mid"), t3_bot_diff=L(team,"t3_bot")-L(opp,"t3_bot"),
                    inh_top_diff=L(team,"inh_top")-L(opp,"inh_top"), inh_mid_diff=L(team,"inh_mid")-L(opp,"inh_mid"), inh_bot_diff=L(team,"inh_bot")-L(opp,"inh_bot"),
                    team_kills_d=team_kd[team], team_deaths_d=team_dd[team],
                    skirmish_flag=skirmish_flag,
                    plates_time_left=max(0, 14*60 - int(bin_start)),
                    phase_num=phase, baron_live=baron_live, dragon_live=dragon_live,
                    flip_risk_proxy=0.0, setup_quality_score=0.0, recall_sync_score=0.0, crossmap_ev_proxy=0.0, smite_diff_proxy=0.0,
                    vision_delta=vision_delta,
                    team_gold_diff=gold_diff, team_level_avg_diff=level_avg_diff, team_cs_diff=cs_diff,
                    dragon_diff_cum=dragon_diff_cum, herald_diff_cum=herald_diff_cum, baron_diff_cum=baron_diff_cum,
                    tower_diff_cum=tower_diff_cum, ward_kill_diff_cum=ward_kill_diff_cum,
                    team_gold_diff_per_min=gold_diff_per_min, team_cs_diff_per_min=cs_diff_per_min,
                    team_engage_support=int(comp.get(team,{}).get("engage_support",0)),
                    team_engage_count=int(comp.get(team,{}).get("engage_count",0)),
                    team_disengage_count=int(comp.get(team,{}).get("disengage_count",0)),
                    top_item_gold_diff=float(ld_item["top"]), jng_item_gold_diff=float(ld_item["jng"]), mid_item_gold_diff=float(ld_item["mid"]), bot_item_gold_diff=float(ld_item["bot"]), sup_item_gold_diff=float(ld_item["sup"]),
                    top_cc_ms_diff=float(ld_cc["top"]), jng_cc_ms_diff=float(ld_cc["jng"]), mid_cc_ms_diff=float(ld_cc["mid"]), bot_cc_ms_diff=float(ld_cc["bot"]), sup_cc_ms_diff=float(ld_cc["sup"]),
                    top_cs_diff=int(ld_cs["top"]), jng_cs_diff=int(ld_cs["jng"]), mid_cs_diff=int(ld_cs["mid"]), bot_cs_diff=int(ld_cs["bot"]), sup_cs_diff=int(ld_cs["sup"]),
                    top_xp_diff=int(ld_xp["top"]), jng_xp_diff=int(ld_xp["jng"]), mid_xp_diff=int(ld_xp["mid"]), bot_xp_diff=int(ld_xp["bot"]), sup_xp_diff=int(ld_xp["sup"]),
                    jng_gold_diff=float((lane_agg[team]["jng"]["gold"] - lane_agg[opp]["jng"]["gold"])) ,
                    top_level_diff=float(lvl_diff["top"]), jng_level_diff=float(lvl_diff["jng"]), mid_level_diff=float(lvl_diff["mid"]), bot_level_diff=float(lvl_diff["bot"]), sup_level_diff=float(lvl_diff["sup"]),
                    near_baron_count_diff=near_baron_count_diff,
                    near_dragon_count_diff=near_dragon_count_diff,
                    centroid_dist_baron_diff=centroid_dist_baron_diff,
                    centroid_dist_dragon_diff=centroid_dist_dragon_diff,
                    team_spread_std_diff=team_spread_std_diff,
                    team_cc_ms_diff=float(team_cc[team] - team_cc[opp]),
                    team_offense_power_diff=float(team_offense[team] - team_offense[opp]),
                    team_resists_sum_diff=float(team_resists[team] - team_resists[opp]),
                    team_ehp_proxy_diff=float(team_ehp[team] - team_ehp[opp]),
                    team_ad_power_diff=float(team_ad[team] - team_ad[opp]),
                    team_ap_power_diff=float(team_ap[team] - team_ap[opp]),
                    team_lvl6_spike_diff=int(lvl6[team] - lvl6[opp]),
                    team_lvl11_spike_diff=int(lvl11[team] - lvl11[opp]),
                    team_lvl16_spike_diff=int(lvl16[team] - lvl16[opp]),
                    team_ability_haste_diff=float(team_haste[team] - team_haste[opp]),
                    team_penetration_proxy_diff=float(team_pen[team] - team_pen[opp]),
                    team_tenacity_proxy_diff=float(team_tenacity[team] - team_tenacity[opp]),
                    team_sustain_vamp_diff=float(team_vamp[team] - team_vamp[opp]),
                    enemy_has_pyke=int(comp.get(opp,{}).get("has_pyke",0)),
                    enemy_has_chogath=int(comp.get(opp,{}).get("has_chogath",0)),
                    enemy_has_urgot=int(comp.get(opp,{}).get("has_urgot",0)),
                    enemy_has_darius=int(comp.get(opp,{}).get("has_darius",0)),
                    team_max_gold_diff=team_max_gold_diff,
                    enemy_fed_proxy=enemy_fed_proxy,
                    enemy_cc_ms=float(team_cc[opp]),
                    ally_dragons=int(cum[team]["drag"]), enemy_dragons=int(cum[opp]["drag"]),
                    ally_soul_point=int(cum[team]["drag"]>=3), enemy_soul_point=int(cum[opp]["drag"]>=3),
                    dragon_soul_taken=int(cum[team]["drag"]>=4 or cum[opp]["drag"]>=4),
                    jng_smite_proxy=float(5.0 * float(lvl_diff["jng"])) ,
                    kills_diff_bin=int(team_kd[team] - team_kd[opp]),
                    deaths_diff_bin=int(team_dd[team] - team_dd[opp]),
                )
                # Resist mismatch proxy (effective damage edge vs enemy resists)
                try:
                    eff_ad = (team_ad[team] - team_ad[opp]) / (1.0 + team_armor[opp]/100.0)
                    eff_ap = (team_ap[team] - team_ap[opp]) / (1.0 + team_mr[opp]/100.0)
                    row["resist_mismatch_proxy"] = float(eff_ad + eff_ap)
                except Exception:
                    row["resist_mismatch_proxy"] = 0.0
                rows.append(row)

    df = pd.DataFrame(rows, columns=RICH_COLUMNS)
    if not df.empty:
        df["flip_risk_proxy"] = df.apply(lambda r: _flip_risk_proxy(int(r["vision_delta"]), int(r["skirmish_flag"]), int(r["baron_live"]), int(r["dragon_live"])), axis=1)
        df["setup_quality_score"] = df.apply(lambda r: 0.03*int(r["vision_delta"]) + (1 if int(r["phase_num"]) in (2,3) else 0)*0.2 + (1 if int(r["plates_time_left"])>0 else 0)*0.1, axis=1)
        df["recall_sync_score"] = df.apply(lambda r: 0.3*int(r["skirmish_flag"]) + 0.02*(int(r["team_kills_d"])-int(r["team_deaths_d"])), axis=1)
        df["crossmap_ev_proxy"] = df.apply(lambda r: 0.1*int(r["tower_diff"]) + 0.15*int(r["herald_diff"]), axis=1)
        df = df.sort_values(["game_id","time_s","team"]).reset_index(drop=True)
        # Rolling window features per game_id+team over last 30s
        n = max(1, int(ROLL_WINDOW_S // max(1, window_s)))
        grp = df.groupby(["game_id","team"], sort=False)
        def _roll(col):
            return grp[col].rolling(n, min_periods=1).sum().reset_index(level=[0,1], drop=True)
        df["roll_kills_30s"] = _roll("team_kills_d")
        df["roll_deaths_30s"] = _roll("team_deaths_d")
        df["roll_vision_delta_30s"] = _roll("vision_delta")
        df["roll_ward_kill_diff_30s"] = _roll("ward_kill_diff")
        df["roll_tower_diff_30s"] = _roll("tower_diff")
        df["roll_plate_diff_30s"] = _roll("plate_diff")
        df["roll_dragon_diff_30s"] = _roll("dragon_diff")
        df["roll_herald_diff_30s"] = _roll("herald_diff")
        df["roll_baron_diff_30s"] = _roll("baron_diff")
        # Additional rolling momentum features
        if "team_cc_ms_diff" in df.columns:
            df["roll_cc_ms_30s_diff"] = _roll("team_cc_ms_diff")
        if "kills_diff_bin" in df.columns:
            df["roll_kills_diff_30s"] = _roll("kills_diff_bin")
        if "deaths_diff_bin" in df.columns:
            df["roll_deaths_diff_30s"] = _roll("deaths_diff_bin")
        # Additional rolling momentum features
        if "team_cc_ms_diff" in df.columns:
            df["roll_cc_ms_30s_diff"] = _roll("team_cc_ms_diff")
        if "kills_diff_bin" in df.columns:
            df["roll_kills_diff_30s"] = _roll("kills_diff_bin")
        if "deaths_diff_bin" in df.columns:
            df["roll_deaths_diff_30s"] = _roll("deaths_diff_bin")
    return df
