"""问题四：2022 卡塔尔世界杯真实赛程与 P2 优化赛程同口径评价。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import math
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from params import MASTER_SEED, SECURITY_BASE_COST_USD
from problem2 import compute_travel
from utils import FIGURES_DIR, OUTPUT_DIR, ROOT, load_all_sheets, minmax, runtime_metadata, set_all_seeds, write_json

FIFA_SCHEDULE_URL = "https://inside.fifa.com/tournaments/mens/worldcup/qatar2022/media-releases/final-match-schedule-for-the-fifa-world-cup-qatar-2022-tm-now-available-on"
FIFA_FIXTURE_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/qatar2022/scores-fixtures"
RAW_VALIDATION_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json"
TEAM_ELO_SOURCE_URL = "https://www.international-football.net/elo-ratings-table?confed=&day=12&month=11&old-team=&year=2022"
FIFA_STADIUM_SOURCE_URL = "https://inside.fifa.com/tournament-organisation/world-cup-2022-in-numbers/fifa-world-cup-qatar-2022-at-a-glance"
FIFA_STADIUM_ADDRESS_URL = "https://www.fifa.com/en/articles/stadium-addresses-en"
ACCESSED_AT = "2026-08-14"
ACTUAL_CLIMATE_SCENARIO = 0.12

STADIUMS = {
    "Al Bayt Stadium": ("Al Khor", 68895, 25.652, 51.488),
    "Al Thumama Stadium": ("Doha", 44400, 25.235, 51.532),
    "Khalifa International Stadium": ("Al Rayyan", 45857, 25.263, 51.448),
    "Ahmad bin Ali Stadium": ("Al Rayyan", 45032, 25.330, 51.342),
    "Ahmad Bin Ali Stadium": ("Al Rayyan", 45032, 25.330, 51.342),
    "Al Janoub Stadium": ("Al Wakrah", 44325, 25.160, 51.574),
    "Education City Stadium": ("Al Rayyan", 44667, 25.310, 51.424),
    "Stadium 974": ("Doha", 44089, 25.289, 51.567),
    "Lusail Stadium": ("Lusail", 88966, 25.420, 51.490),
    "Lusail Iconic Stadium": ("Lusail", 88966, 25.420, 51.490),
}


def _clean_ground(value: str) -> str:
    name = value.split(",")[0].strip()
    return name


def collect_actual_schedule() -> pd.DataFrame:
    """从已下载的开放结构化副本取数，并单列 FIFA 官方核验来源。"""
    raw_path = ROOT / "user_data" / "worldcup_2022_openfootball.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))["matches"]
    games = [x for x in raw if str(x.get("group", "")).startswith("Group ")]
    assert len(games) == 48
    df = pd.DataFrame(games)
    df["group_id"] = df["group"].str.replace("Group ", "", regex=False)
    df["local_dt"] = pd.to_datetime(df["date"] + " " + df["time"]).dt.tz_localize(ZoneInfo("Asia/Qatar"))
    df = df.sort_values(["group_id", "local_dt", "team1", "team2"], kind="mergesort").reset_index(drop=True)
    df["round_in_group"] = df.groupby("group_id").cumcount().floordiv(2) + 1
    rows = []
    for idx, r in df.iterrows():
        venue = _clean_ground(str(r.ground)); assert venue in STADIUMS, venue
        city, cap, lat, lon = STADIUMS[venue]
        score_obj = r["score"]
        score = score_obj.get("ft", [None, None]) if isinstance(score_obj, dict) else score_obj
        rows.append({
            "match_id": f"Q22-{idx+1:02d}", "competition": "FIFA World Cup Qatar 2022",
            "stage": "Group", "group_id": str(r.group_id), "round_in_group": int(r.round_in_group),
            "team_a": str(r.team1), "team_b": str(r.team2), "venue": venue,
            "city": city, "country": "Qatar", "date": str(r.date),
            "local_kickoff_time": str(r.time), "local_datetime": r.local_dt.isoformat(),
            "utc_datetime": r.local_dt.tz_convert("UTC").isoformat(), "capacity": cap,
            "latitude": lat, "longitude": lon, "goals_a": int(score[0]), "goals_b": int(score[1]),
            "source_url": RAW_VALIDATION_URL, "fixture_source_url": FIFA_FIXTURE_URL,
            "official_schedule_url": FIFA_SCHEDULE_URL, "stadium_source_url": FIFA_STADIUM_SOURCE_URL,
            "stadium_address_url": FIFA_STADIUM_ADDRESS_URL, "retrieval_date": ACCESSED_AT,
        })
    out = pd.DataFrame(rows)
    assert len(out) == 48 and out.match_id.is_unique and out.source_url.str.len().gt(0).all()
    for team in sorted(set(out.team_a) | set(out.team_b)):
        assert len(out[(out.team_a == team) | (out.team_b == team)]) == 3
    return out


def load_actual_team_strength(actual: pd.DataFrame) -> pd.DataFrame:
    """读取世界杯开赛前 Elo 快照；不允许以题内球队中位数静默补齐。"""
    path = ROOT / "user_data" / "worldcup_2022_pre_tournament_elo.csv"
    ratings = pd.read_csv(path)
    assert ratings["team_name"].is_unique and len(ratings) == 32
    actual_teams = set(actual["team_a"]) | set(actual["team_b"])
    assert actual_teams == set(ratings["team_name"]), sorted(actual_teams ^ set(ratings["team_name"]))
    assert ratings["elo_rating"].between(1000, 2500).all()
    assert ratings["rating_date"].eq("2022-11-12").all()
    assert ratings["source_url"].eq(TEAM_ELO_SOURCE_URL).all()
    return ratings.set_index("team_name")


def _match_features(elo_a: float, elo_b: float, pa: float, pd_: float, pb: float,
                    round_in_group: int) -> list[float]:
    """只使用现实与题内两侧都能在赛前取得的同构特征。"""
    return [pa, pd_, pb, 1-max(pa, pd_, pb), (elo_a+elo_b)/2,
            abs(elo_a-elo_b), float(round_in_group)]


def attach_match_proxies(actual, sheets, p1_group):
    teams, groups, base = sheets["teams"], sheets["groups_matches"], sheets["base_predictions"]
    actual_strength = load_actual_team_strength(actual)
    synthetic_strength = teams.set_index("team_name")["elo_rating"]
    attach = groups.merge(base, on=["match_id","group_id","round_in_group","team_a","team_b"], validate="one_to_one")
    attach = attach.merge(sheets["security_requirements"][["match_id","security_demand_score","required_security_level"]], on="match_id", validate="one_to_one")
    attach = attach.merge(p1_group[["match_id","predicted_tv_viewers"]], on="match_id", validate="one_to_one")
    X_train = []
    for r in attach.itertuples(index=False):
        elo_a, elo_b = float(synthetic_strength.loc[str(r.team_a)]), float(synthetic_strength.loc[str(r.team_b)])
        X_train.append(_match_features(elo_a, elo_b, float(r.p_a_win), float(r.p_draw),
                                       float(r.p_b_win), int(r.round_in_group)))
    X_actual = []
    pregame = []
    for r in actual.itertuples(index=False):
        elo_a = float(actual_strength.loc[str(r.team_a), "elo_rating"])
        elo_b = float(actual_strength.loc[str(r.team_b), "elo_rating"])
        # 仅用赛前 Elo 差构造 Bradley-Terry 概率，不读取实际赛果。
        win_a = 1/(1+10**((elo_b-elo_a)/400)); draw = 0.24
        pa, pb = (1-draw)*win_a, (1-draw)*(1-win_a)
        X_actual.append(_match_features(elo_a, elo_b, pa, draw, pb, int(r.round_in_group)))
        pregame.append((pa,draw,pb))
    scaler = StandardScaler().fit(np.asarray(X_train))
    nn = NearestNeighbors(n_neighbors=2).fit(scaler.transform(np.asarray(X_train)))
    _, neigh = nn.kneighbors(scaler.transform(np.asarray(X_actual)))
    rows = actual.copy()
    cols = ["expected_attendance_base","attractiveness_index","uncertainty_index","commercial_value_index",
            "security_demand_score","required_security_level","predicted_tv_viewers"]
    for c in cols:
        rows[c] = [attach.iloc[j][c] for j in neigh[:,0]]
        rows[f"proxy2_{c}"] = [attach.iloc[j][c] for j in neigh[:,1]]
    rows[["p_a_win","p_draw","p_b_win"]] = np.asarray(pregame)
    # U 属于可由同一组三向赛前概率直接计算的指标，不能从另一场比赛继承。
    rows["uncertainty_index"] = 1.0 - rows[["p_a_win", "p_draw", "p_b_win"]].max(axis=1)
    rows["proxy_match_id"] = [attach.iloc[j].match_id for j in neigh[:,0]]
    rows["proxy2_match_id"] = [attach.iloc[j].match_id for j in neigh[:,1]]
    rows["is_proxy"] = True
    return rows


def attach_venue_proxies(actual, venues):
    """仅按有官方来源的容量映射；安保等级作为可行域而不是距离特征。"""
    actual_feat = actual[["capacity"]].to_numpy(float)
    train_feat = venues[["capacity"]].to_numpy(float)
    scaler = StandardScaler().fit(train_feat)
    train_scaled = scaler.transform(train_feat)
    actual_scaled = scaler.transform(actual_feat)
    neigh = []
    for i, row in enumerate(actual.itertuples(index=False)):
        primary_legal = np.flatnonzero(venues["security_level"].to_numpy(int) >= int(row.required_security_level))
        second_legal = np.flatnonzero(venues["security_level"].to_numpy(int) >= int(row.proxy2_required_security_level))
        assert len(primary_legal) >= 1 and len(second_legal) >= 2
        primary_dist = np.square(train_scaled[primary_legal] - actual_scaled[i]).sum(axis=1)
        second_dist = np.square(train_scaled[second_legal] - actual_scaled[i]).sum(axis=1)
        primary = primary_legal[np.argsort(primary_dist, kind="mergesort")[0]]
        second = second_legal[np.argsort(second_dist, kind="mergesort")[1]]
        neigh.append([primary, second])
    neigh = np.asarray(neigh, dtype=int)
    out = actual.copy()
    for c in ["venue_id","setup_cost_musd","operation_cost_musd_per_match","security_level","security_cost_index"]:
        out[f"proxy_{c}"] = [venues.iloc[j][c] for j in neigh[:,0]]
        out[f"proxy2_{c}"] = [venues.iloc[j][c] for j in neigh[:,1]]
    return out


def _haversine(lat1, lon1, lat2, lon2):
    p1,p2=np.radians(lat1),np.radians(lat2); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 6371*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))


def map_prime(rows, slots):
    """按赛事相对进度日和 UTC 钟点唯一映射，消除跨日期同距的任意并列。"""
    slots = slots.copy()
    slots["utc"] = pd.to_datetime(slots.reference_utc_time, utc=True)
    slots["slot_date"] = pd.to_datetime(slots["date"]).dt.date
    actual_dates = sorted(pd.to_datetime(rows["date"]).dt.date.unique())
    slot_dates = sorted(slots["slot_date"].unique())
    actual_day_index = {day: idx for idx, day in enumerate(actual_dates)}
    values=[]; flags=[]; mapped_ids=[]
    prime_ids = set(slots.nlargest(20, "global_prime_score")["slot_id"].astype(str))
    for r in rows.itertuples(index=False):
        utc = pd.Timestamp(r.utc_datetime)
        day = pd.Timestamp(r.date).date()
        progress = actual_day_index[day] / max(len(actual_dates)-1, 1)
        mapped_day = slot_dates[int(round(progress * (len(slot_dates)-1)))]
        candidates = slots[slots["slot_date"].eq(mapped_day)].copy()
        clock = utc.hour + utc.minute/60
        candidate_clock = candidates["utc"].dt.hour + candidates["utc"].dt.minute/60
        raw_delta = (candidate_clock-clock).abs()
        candidates["_clock_delta"] = np.minimum(raw_delta, 24-raw_delta)
        nearest = candidates.sort_values(["_clock_delta", "slot_id"], kind="mergesort").iloc[0]
        values.append(float(nearest["global_prime_score"]))
        flags.append(str(nearest["slot_id"]) in prime_ids)
        mapped_ids.append(str(nearest["slot_id"]))
    return np.asarray(values, dtype=float), np.asarray(flags, dtype=bool), np.asarray(mapped_ids, dtype=object)


def build_p2_travel_context(sheets):
    """构造与 P2 完全相同的候选域、安保规则、三分项边界和旅行负担表。"""
    groups = sheets["groups_matches"].merge(
        sheets["security_requirements"][["match_id", "required_security_level"]],
        on="match_id", validate="one_to_one",
    ).reset_index(drop=True)
    venues = sheets["venues"].reset_index(drop=True)
    burden = compute_travel(groups, venues, sheets["security_requirements"], sheets["distance_matrix"])
    return {
        "burden": burden,
        "match_index": {str(m): int(i) for i, m in enumerate(groups["match_id"])},
        "venue_index": {str(v): int(i) for i, v in enumerate(venues["venue_id"])},
    }


def assign_p2_proxy_travel(rows, context):
    """将现实赛程的比赛/场馆整体映射到 P2 候选域后，复算同定义 D。"""
    values = []
    for row in rows.itertuples(index=False):
        i = context["match_index"][str(row.proxy_match_id)]
        v = context["venue_index"][str(row.proxy_venue_id)]
        key = (i, v)
        assert key in context["burden"], (row.proxy_match_id, row.proxy_venue_id)
        values.append(float(context["burden"][key]))
    return np.asarray(values, dtype=float)


def prepare_actual_rows(actual, sheets, travel_context):
    prime_score, prime, mapped_slot = map_prime(actual, sheets["time_slots"])
    z=actual.copy(); z["global_prime_score"]=prime_score; z["is_prime"]=prime.astype(int)
    z["prime_proxy_slot_id"] = mapped_slot
    large_stadiums=set(z.groupby("venue")["capacity"].first().nlargest(2).index)
    z["is_large_venue"]=z.venue.isin(large_stadiums).astype(int)
    # 真实场馆路径只保留为结构层公里量，不再冒充 P2 的复合旅行指标 D。
    travel=np.zeros(len(z)); team_last={}
    for idx,r in z.sort_values("utc_datetime").iterrows():
        vals=[]
        for team in (r.team_a,r.team_b):
            if team in team_last:
                prev=team_last[team]; vals.append(_haversine(prev[0],prev[1],r.latitude,r.longitude))
            else: vals.append(0.0)
            team_last[team]=(r.latitude,r.longitude)
        travel[idx]=np.mean(vals)
    z["path_travel_km_per_team"]=travel
    z["travel_cost_index"] = assign_p2_proxy_travel(z, travel_context)
    z["expected_attendance"]=np.minimum(z.expected_attendance_base,z.capacity)
    z["ticket_revenue_usd"]=[(80 if r==1 else 95 if r==2 else 110)*n for r,n in zip(z.round_in_group,z.expected_attendance)]
    unit={1:0.035,2:0.040,3:0.052}; sponsor={1:0.85,2:0.90,3:1.10}
    z["broadcast_value_usd"]=[tv*unit[r]*gp*sponsor[r] for tv,r,gp in zip(z.predicted_tv_viewers,z.round_in_group,z.global_prime_score)]
    z["risk_index"]=0.5*ACTUAL_CLIMATE_SCENARIO+0.3*z.expected_attendance/z.capacity+0.2*z.required_security_level/z.proxy_security_level
    return z


def prepare_optimized_rows(schedule, sheets):
    z=schedule.copy(); slots=sheets["time_slots"].set_index("slot_id"); venues=sheets["venues"].set_index("venue_id")
    prime_ids=set(sheets["time_slots"].nlargest(20,"global_prime_score").slot_id)
    large_ids=set(sheets["venues"].nlargest(4,"capacity").venue_id)
    z["is_prime"]=z.slot_id.isin(prime_ids).astype(int); z["is_large_venue"]=z.venue_id.isin(large_ids).astype(int)
    z["setup_cost_musd"]=[venues.loc[v,"setup_cost_musd"] for v in z.venue_id]
    z["operation_cost_musd_per_match"]=[venues.loc[v,"operation_cost_musd_per_match"] for v in z.venue_id]
    z["security_cost_index"]=[venues.loc[v,"security_cost_index"] for v in z.venue_id]
    z["uncertainty_index"]=sheets["base_predictions"].set_index("match_id").loc[z.match_id,"uncertainty_index"].to_numpy()
    z["attractiveness_index"]=sheets["base_predictions"].set_index("match_id").loc[z.match_id,"attractiveness_index"].to_numpy()
    z["team_a_id"]=sheets["groups_matches"].set_index("match_id").loc[z.match_id,"team_a_id"].to_numpy()
    z["team_b_id"]=sheets["groups_matches"].set_index("match_id").loc[z.match_id,"team_b_id"].to_numpy()
    for c in ["capacity", "latitude", "longitude", "timezone"]:
        z[f"venue_{c}"] = [venues.loc[v, c] for v in z.venue_id]
    return z


STRUCTURAL_DEFINITIONS = {
    "rest_min_hours": {"category": "休息", "unit": "小时", "preferred_direction": "higher",
                       "denominator": "全部球队相邻两轮间隔", "definition": "球队相邻比赛 UTC 间隔的最小值", "evidence": "direct"},
    "rest_mean_hours": {"category": "休息", "unit": "小时", "preferred_direction": "higher",
                        "denominator": "全部球队相邻两轮间隔", "definition": "球队相邻比赛 UTC 间隔的平均值", "evidence": "direct"},
    "timezone_crossings": {"category": "时区", "unit": "次", "preferred_direction": "lower",
                           "denominator": "全部球队相邻两轮间隔", "definition": "相邻比赛场馆 UTC 偏移发生变化的次数", "evidence": "direct"},
    "timezone_crossing_rate": {"category": "时区", "unit": "比例", "preferred_direction": "lower",
                               "denominator": "球队转场次数", "definition": "跨时区次数除以球队转场次数", "evidence": "direct"},
    "travel_mean_km": {"category": "旅行", "unit": "公里/次", "preferred_direction": "lower",
                       "denominator": "球队转场次数", "definition": "相邻比赛实际场馆大圆距离的平均值", "evidence": "direct"},
    "venue_change_rate": {"category": "旅行", "unit": "比例", "preferred_direction": "lower",
                          "denominator": "球队转场次数", "definition": "相邻轮更换场馆的球队转场比例", "evidence": "direct"},
    "venue_utilization_rate": {"category": "场馆", "unit": "比例", "preferred_direction": "higher",
                               "denominator": "启用场馆数乘比赛日期数", "definition": "发生比赛的场馆日占可用场馆日比例", "evidence": "direct"},
    "venue_daily_peak": {"category": "场馆", "unit": "场/场馆日", "preferred_direction": "lower",
                         "denominator": "单个场馆当地自然日", "definition": "单一场馆单日承办场次最大值", "evidence": "direct"},
    "capacity_occupancy_rate": {"category": "容量", "unit": "比例", "preferred_direction": "higher",
                                "denominator": "逐场容量后取场均", "definition": "预计现场观众与场馆容量之比的场均值", "evidence": "proxy_demand"},
    "prime_coverage_rate": {"category": "黄金时段", "unit": "比例", "preferred_direction": "higher",
                            "denominator": "比赛场次", "definition": "按赛事进度日和 UTC 钟点映射后落入前四分位时段的比例", "evidence": "time_mapping"},
    "expected_attendance_per_match": {"category": "观众", "unit": "人/场", "preferred_direction": "higher",
                                      "denominator": "比赛场次", "definition": "预计现场观众总量除以比赛场次", "evidence": "proxy_demand"},
}


def structural_metrics(rows: pd.DataFrame, kind: str) -> dict:
    """对两套赛程调用同一结构函数；总量均附明确规模分母。"""
    ta = "team_a" if kind == "actual" else "team_a_id"
    tb = "team_b" if kind == "actual" else "team_b_id"
    venue = "venue" if kind == "actual" else "venue_id"
    capacity = "capacity" if kind == "actual" else "venue_capacity"
    lat = "latitude" if kind == "actual" else "venue_latitude"
    lon = "longitude" if kind == "actual" else "venue_longitude"
    z = rows.copy()
    z["_utc"] = pd.to_datetime(z["utc_datetime"], utc=True)
    z["_local_date"] = [pd.Timestamp(x).date().isoformat() for x in z["local_datetime"]]
    z["_utc_offset"] = [pd.Timestamp(x).utcoffset().total_seconds() / 3600 for x in z["local_datetime"]]

    rest, travel, timezone_shift = [], [], []
    venue_changes = 0
    teams = sorted(set(z[ta]) | set(z[tb]))
    for team in teams:
        q = z[(z[ta] == team) | (z[tb] == team)].sort_values("_utc")
        records = q.to_dict(orient="records")
        for previous, current in zip(records[:-1], records[1:]):
            rest.append((current["_utc"] - previous["_utc"]).total_seconds() / 3600)
            travel.append(_haversine(previous[lat], previous[lon], current[lat], current[lon]))
            timezone_shift.append(abs(current["_utc_offset"] - previous["_utc_offset"]))
            venue_changes += int(previous[venue] != current[venue])
    transitions = len(rest)
    assert transitions > 0
    venue_days = z[[venue, "_local_date"]].drop_duplicates()
    n_venues = int(z[venue].nunique()); n_dates = int(z["_local_date"].nunique())
    crossings = int(np.sum(np.asarray(timezone_shift) > 1e-9))
    metrics = {
        "rest_min_hours": float(np.min(rest)),
        "rest_mean_hours": float(np.mean(rest)),
        "timezone_crossings": float(crossings),
        "timezone_crossing_rate": float(crossings / transitions),
        "travel_mean_km": float(np.mean(travel)),
        "venue_change_rate": float(venue_changes / transitions),
        "venue_utilization_rate": float(len(venue_days) / (n_venues * n_dates)),
        "venue_daily_peak": float(z.groupby([venue, "_local_date"]).size().max()),
        "capacity_occupancy_rate": float(np.mean(np.minimum(z["expected_attendance"].to_numpy(float) / z[capacity].to_numpy(float), 1.0))),
        "prime_coverage_rate": float(z["is_prime"].mean()),
        "expected_attendance_per_match": float(z["expected_attendance"].mean()),
        "match_count": int(len(z)), "team_transition_count": int(transitions),
        "active_venue_count": n_venues, "match_date_count": n_dates,
    }
    assert set(STRUCTURAL_DEFINITIONS).issubset(metrics)
    return metrics


def _comparison_row(name, category, actual, optimized, direction):
    a, o = float(actual), float(optimized)
    diff = o - a
    imp = None if abs(a) <= 1e-15 else ((o-a)/abs(a) if direction == "higher" else (a-o)/abs(a))
    return {"indicator_name": name, "indicator_category": category,
            "actual_schedule_value": a, "optimized_schedule_value": o,
            "absolute_difference": diff, "relative_improvement": imp,
            "preferred_direction": direction,
            "evaluation_result": "优化较优" if imp is not None and imp > 0 else "实际较优或持平"}


def structural_comparison_rows(actual_m, optimized_m):
    return [_comparison_row(k, f"结构-{spec['category']}", actual_m[k], optimized_m[k], spec["preferred_direction"])
            for k, spec in STRUCTURAL_DEFINITIONS.items()]


def validate_p4_c2(actual, optimized, actual_structure, optimized_structure, structure_comp):
    """P4-C2 机器闸：任一题定结构维度、单位、方向或分母缺失即失败。"""
    required = set(STRUCTURAL_DEFINITIONS)
    assert required == set(row["indicator_name"] for row in structure_comp)
    for metrics in (actual_structure, optimized_structure):
        assert required.issubset(metrics)
        assert np.isfinite([metrics[k] for k in required]).all()
    for spec in STRUCTURAL_DEFINITIONS.values():
        assert spec["unit"] and spec["preferred_direction"] in {"higher", "lower"} and spec["denominator"]
    recalculated_u = 1.0 - actual[["p_a_win", "p_draw", "p_b_win"]].max(axis=1).to_numpy(float)
    assert np.allclose(actual["uncertainty_index"].to_numpy(float), recalculated_u, atol=1e-12)
    assert "path_travel_km_per_team" in actual and "travel_cost_index" in actual
    assert not np.allclose(actual["path_travel_km_per_team"], actual["travel_cost_index"])
    assert len(actual) == 48 and len(optimized) == 72
    return {"status": "PASS", "required_dimensions": sorted(required),
            "actual_matches": 48, "optimized_matches": 72,
            "travel_contract": "physical_km_separate_from_P2_composite_D"}


def raw_schedule_metrics(rows: pd.DataFrame, kind: str) -> dict:
    """先计算未归一化的同口径指标，供共同边界一次性拟合。"""
    n=len(rows); preferred_direction={"T":"higher","B":"higher","U":"higher","H":"higher","C":"lower","D":"lower","F":"lower","R":"lower"}
    scale_denominator={"T":n,"B":n,"C":n,"D":n,"R":n,"U":n,"H":n,"F":"three_matches_per_team"}
    T0=float(rows.ticket_revenue_usd.sum()/n); B0=float(rows.broadcast_value_usd.sum()/n)
    if kind=="actual":
        byvenue=rows.groupby("venue"); setup=sum(float(g.proxy_setup_cost_musd.iloc[0])*1e6 for _,g in byvenue)
        op=sum(float(r.proxy_operation_cost_musd_per_match)*1e6+SECURITY_BASE_COST_USD*r.required_security_level*r.proxy_security_cost_index for r in rows.itertuples())
    else:
        byvenue=rows.groupby("venue_id"); setup=sum(float(g.setup_cost_musd.iloc[0])*1e6 for _,g in byvenue)
        op=sum(float(r.operation_cost_musd_per_match)*1e6+SECURITY_BASE_COST_USD*r.required_security_level*r.security_cost_index for r in rows.itertuples())
    C0=(setup+op)/n; U=float(rows.uncertainty_index.mean()); H=float(rows.attractiveness_index.mean()/100)
    D=float(rows.travel_cost_index.mean()); R=float(rows.risk_index.mean())
    ta="team_a" if kind=="actual" else "team_a_id"; tb="team_b" if kind=="actual" else "team_b_id"
    teams=sorted(set(rows[ta])|set(rows[tb])); pc=[];lc=[]
    for team in teams:
        q=rows[(rows[ta]==team)|(rows[tb]==team)]; pc.append(int(q.is_prime.sum()));lc.append(int(q.is_large_venue.sum()))
    F=0.5*(max(pc)-min(pc))/3+0.5*(max(lc)-min(lc))/3
    return {"T":T0,"B":B0,"U":U,"H":H,"C":C0,"D":D,"F":float(F),"R":R,
            "preferred_direction":preferred_direction,"scale_denominator":scale_denominator}


def common_comparison_bounds(base_bounds: dict, *raw_metrics: dict) -> dict:
    """用 P2 边界与全部已报告 P4 场景的原始值构造共同包络。"""
    bounds = {}
    for key in ("T", "B", "C"):
        observed = [float(metrics[key]) for metrics in raw_metrics]
        lower = min(float(base_bounds[key][0]), *observed)
        upper = max(float(base_bounds[key][1]), *observed)
        assert upper > lower
        bounds[key] = (lower, upper)
    return bounds


def evaluate_schedule(raw: dict, comparison_bounds: dict) -> dict:
    """在两方案和敏感性场景共用的边界下执行 Min-Max，结果严格属于 [0,1]。"""
    vals={"T":float(minmax(raw["T"],*comparison_bounds["T"])),
          "B":float(minmax(raw["B"],*comparison_bounds["B"])),
          "U":float(raw["U"]),"H":float(raw["H"]),
          "C":float(minmax(raw["C"],*comparison_bounds["C"])),
          "D":float(raw["D"]),"F":float(raw["F"]),"R":float(raw["R"])}
    assert all(-1e-12 <= vals[k] <= 1+1e-12 for k in ("T","B","U","H","C","D","F","R")), vals
    vals["Z4"]=(0.25*vals["T"]+0.25*vals["B"]+0.15*vals["U"]+0.10*vals["H"]
                -0.08*vals["C"]-0.07*vals["D"]-0.06*vals["F"]-0.04*vals["R"])
    vals.update({"T_raw_per_match":float(raw["T"]),"B_raw_per_match":float(raw["B"]),
                 "C_raw_per_match":float(raw["C"]),
                 "preferred_direction":raw["preferred_direction"],
                 "scale_denominator":raw["scale_denominator"],
                 "normalization_bounds":{k:list(v) for k,v in comparison_bounds.items()},
                 "scenario_extrapolation":False,"extrapolated_indicators":[]})
    return vals


def comparison_rows(actual_m, optimized_m):
    info={"T":("价值","higher"),"B":("价值","higher"),"U":("竞技","higher"),"H":("吸引力","higher"),
          "C":("成本","lower"),"D":("旅行","lower"),"F":("公平","lower"),"R":("风险","lower"),"Z4":("综合","higher")}
    rows=[]
    for k,(cat,direction) in info.items():
        rows.append(_comparison_row(k, f"代理-{cat}", actual_m[k], optimized_m[k], direction))
    return rows


def solve() -> dict:
    set_all_seeds(MASTER_SEED); start=time.time(); sheets=load_all_sheets()
    actual_schedule=collect_actual_schedule(); p1=pd.read_csv(OUTPUT_DIR/"result_1_match_prediction.csv")
    actual_proxy=attach_match_proxies(actual_schedule,sheets,p1); actual_proxy=attach_venue_proxies(actual_proxy,sheets["venues"])
    travel_context = build_p2_travel_context(sheets)
    actual=prepare_actual_rows(actual_proxy,sheets,travel_context)
    actual_proxy2=actual_proxy.copy()
    for c in ["expected_attendance_base","attractiveness_index","commercial_value_index",
              "security_demand_score","required_security_level","predicted_tv_viewers"]:
        actual_proxy2[c]=actual_proxy2[f"proxy2_{c}"]
    for c in ["venue_id","setup_cost_musd","operation_cost_musd_per_match","security_level","security_cost_index"]:
        actual_proxy2[f"proxy_{c}"]=actual_proxy2[f"proxy2_{c}"]
    actual_proxy2["proxy_match_id"] = actual_proxy2["proxy2_match_id"]
    actual_proxy2["proxy_venue_id"] = actual_proxy2["proxy2_venue_id"]
    actual_second=prepare_actual_rows(actual_proxy2,sheets,travel_context)
    schedule=pd.read_csv(OUTPUT_DIR/"result_2_group_schedule.csv"); optimized=prepare_optimized_rows(schedule,sheets)
    p2=json.loads((FIGURES_DIR/"problem_2_results.json").read_text(encoding="utf-8"))["normalization_bounds"]
    base_bounds={"T":(p2["T_min"]/72,p2["T_max"]/72),"B":(p2["B_min"]/72,p2["B_max"]/72),"C":(p2["C_min"]/72,p2["C_max"]/72)}
    actual_raw=raw_schedule_metrics(actual,"actual")
    optimized_raw=raw_schedule_metrics(optimized,"optimized")
    actual_second_raw=raw_schedule_metrics(actual_second,"actual")
    comparison_bounds=common_comparison_bounds(base_bounds,actual_raw,optimized_raw,actual_second_raw)
    actual_m=evaluate_schedule(actual_raw,comparison_bounds)
    optimized_m=evaluate_schedule(optimized_raw,comparison_bounds)
    actual_second_m=evaluate_schedule(actual_second_raw,comparison_bounds)
    actual_structure=structural_metrics(actual,"actual"); optimized_structure=structural_metrics(optimized,"optimized")
    proxy_comp=comparison_rows(actual_m,optimized_m)
    structure_comp=structural_comparison_rows(actual_structure,optimized_structure)
    capability_validation=validate_p4_c2(actual,optimized,actual_structure,optimized_structure,structure_comp)
    # 16个权重扰动：绝对值重归一且符号保留。
    base_w=np.array([.25,.25,.15,.10,-.08,-.07,-.06,-.04]); keys=["T","B","U","H","C","D","F","R"]
    robust=[]; base_direction=np.sign(optimized_m["Z4"]-actual_m["Z4"])
    for j in range(8):
        for factor in (0.9,1.1):
            w=base_w.copy();w[j]*=factor;w=np.sign(w)*np.abs(w)/np.abs(w).sum()
            za=sum(w[k]*actual_m[keys[k]] for k in range(8));zo=sum(w[k]*optimized_m[keys[k]] for k in range(8))
            robust.append({"perturbed":keys[j],"factor":factor,"actual":float(za),"optimized":float(zo),"direction":int(np.sign(zo-za))})
    direction_rate=float(np.mean([r["direction"]==base_direction for r in robust]))
    neighbor_direction=int(np.sign(optimized_m["Z4"]-actual_second_m["Z4"]))
    neighbor_preserved=bool(neighbor_direction==base_direction)
    overall_stable=bool(direction_rate>=0.9 and neighbor_preserved)
    if not overall_stable:
        next(row for row in proxy_comp if row["indicator_name"]=="Z4")["evaluation_result"]="代理敏感，不作总体判定"
    comp=structure_comp+proxy_comp
    actual_cols=pd.read_csv(ROOT/"user_data"/"actual_schedule_template.csv").columns
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    actual[list(actual_cols)].to_csv(OUTPUT_DIR/"actual_schedule_2022.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(comp).to_csv(OUTPUT_DIR/"result_4_schedule_comparison.csv",index=False,encoding="utf-8-sig")
    assert len(actual)==48 and actual.source_url.eq(RAW_VALIDATION_URL).all()
    assert all(np.isfinite([r["actual_schedule_value"],r["optimized_schedule_value"]]).all() for r in comp)
    assert all(0 <= actual_m[k] <= 1 and 0 <= optimized_m[k] <= 1 for k in ("T","B","U","H","C","D","F","R"))
    proxy_diagnostics={
        "actual_team_elo_coverage":"32/32",
        "primary_match_proxy_unique":int(actual["proxy_match_id"].nunique()),
        "primary_match_proxy_max_reuse":int(actual["proxy_match_id"].value_counts().max()),
        "primary_venue_proxy_unique":int(actual["proxy_venue_id"].nunique()),
        "primary_venue_proxy_max_reuse":int(actual["proxy_venue_id"].value_counts().max()),
        "prime_proxy_slot_unique":int(actual["prime_proxy_slot_id"].nunique()),
    }
    result={
        "metadata":{**runtime_metadata(MASTER_SEED),"elapsed_seconds":time.time()-start,"accessed_at":ACCESSED_AT},
        "method":"physical structure is evaluated directly; the conditional proxy layer uses pre-tournament Elo-complete match mapping, capacity-only venue mapping, deterministic tournament-progress time mapping, and one common envelope Min-Max domain",
        "sources":{"machine_read_schedule":RAW_VALIDATION_URL,"official_schedule_verification":FIFA_SCHEDULE_URL,
                   "official_scores_verification":FIFA_FIXTURE_URL,"pre_tournament_elo":TEAM_ELO_SOURCE_URL,
                   "official_stadium_capacity":FIFA_STADIUM_SOURCE_URL,"official_stadium_addresses":FIFA_STADIUM_ADDRESS_URL},
        "source_contract":{"machine_ingestion":"OpenFootball structured JSON snapshot",
                           "official_verification":"FIFA schedule, scores, stadium capacity and address pages"},
        "actual_schedule_rows":actual.to_dict(orient="records"),"actual_metrics":actual_m,"optimized_metrics":optimized_m,
        "common_normalization_bounds":{k:list(v) for k,v in comparison_bounds.items()},
        "structural_definitions":STRUCTURAL_DEFINITIONS,"actual_structural_metrics":actual_structure,
        "optimized_structural_metrics":optimized_structure,"structural_comparison":structure_comp,
        "capability_validation":{"P4-C2":capability_validation},
        "proxy_diagnostics":proxy_diagnostics,
        "proxy_comparison":proxy_comp,"comparison":comp,"weight_robustness":robust,"direction_robustness_rate":direction_rate,
        "weight_direction_stable":bool(direction_rate>=0.9),
        "overall_conclusion_stable":overall_stable,
        "overall_conclusion":"代理映射敏感，仅报告指标级取舍" if not overall_stable else "综合方向在已检验权重和邻居代理下稳定，但仍为条件性结论",
        "second_neighbor_sensitivity":{"actual_Z4":float(actual_second_m["Z4"]),
                                        "baseline_direction":int(base_direction),
                                        "direction_preserved":neighbor_preserved},
        "proxy_note":"结构层报告物理量；代理层用完整赛前Elo映射比赛、仅按容量和安保可行域映射场馆，黄金时段按赛事进度日与UTC钟点唯一映射；T/B/C在全部报告场景共用的包络边界上归一化。",
        "unobserved_constraints":["抽签结果","转播合同","场馆可用性","当地治理","商业谈判"],
        "seed":MASTER_SEED,"outputs":["output/actual_schedule_2022.csv","output/result_4_schedule_comparison.csv"],
    }
    write_json(FIGURES_DIR/"problem_4_results.json",result)
    print(f"[P4] validate_capability PASS actual_Z4={actual_m['Z4']:.6f} optimized_Z4={optimized_m['Z4']:.6f} robust={direction_rate:.1%}",flush=True)
    return result


if __name__=="__main__":
    solve()
