"""问题二：72 场比赛—场馆—时段三维联合 CP-SAT 排程。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

cp_model = None

from params import (
    CP_SAT_SCALE, GLOBAL_PRIME_FRACTION, HIGH_SECURITY_THRESHOLD,
    LARGE_VENUE_FRACTION, MASTER_SEED, MAX_PRIME_COUNT_RANGE,
    MINIMUM_REST_HOURS, P2_TIME_LIMIT_SECONDS, SECURITY_BASE_COST_USD,
)
from utils import FIGURES_DIR, OUTPUT_DIR, load_all_sheets, minmax, runtime_metadata, set_all_seeds, write_json


def _round_ticket(ticket: pd.DataFrame) -> dict[int, dict]:
    return {r: ticket.loc[ticket["match_stage"].eq(f"Group_Match_R{r}")].iloc[0].to_dict() for r in (1, 2, 3)}


def _distance_lookup(distance: pd.DataFrame):
    return {(str(r.origin_id), str(r.destination_id)): {
        "distance": float(r.distance_km), "time": float(r.travel_time_hour), "timezone": float(r.timezone_diff)
    } for r in distance.itertuples(index=False)}


def compute_travel(groups: pd.DataFrame, venues: pd.DataFrame, security: pd.DataFrame,
                   distance: pd.DataFrame) -> dict[tuple[int, int], float]:
    """严格按上一轮全部安保合格场馆均值计算候选旅行负担。"""
    if "required_security_level" in groups.columns:
        g = groups.copy()
    else:
        g = groups.merge(security[["match_id", "required_security_level"]], on="match_id", validate="one_to_one")
    venue_ids = venues["venue_id"].astype(str).tolist()
    venue_sec = venues.set_index("venue_id")["security_level"].to_dict()
    lookup = _distance_lookup(distance)
    team_round_match = {}
    for i, row in g.iterrows():
        for team_id in (str(row.team_a_id), str(row.team_b_id)):
            team_round_match[(team_id, int(row.round_in_group))] = i
    raw = {}
    legal_values = {q: [] for q in ("distance", "time", "timezone")}
    for i, row in g.iterrows():
        r = int(row.round_in_group)
        current_req = int(row.required_security_level)
        for team_id in (str(row.team_a_id), str(row.team_b_id)):
            for v in venue_ids:
                if int(venue_sec[v]) < current_req:
                    continue
                if r == 1:
                    vals = lookup[(team_id, v)]
                else:
                    prev_i = team_round_match[(team_id, r - 1)]
                    prev_req = int(g.iloc[prev_i].required_security_level)
                    origins = [u for u in venue_ids if int(venue_sec[u]) >= prev_req]
                    vals = {q: float(np.mean([lookup[(u, v)][q] for u in origins])) for q in legal_values}
                raw[(team_id, r, v)] = vals
                for q in legal_values:
                    legal_values[q].append(vals[q])
    bounds = {q: (min(values), max(values)) for q, values in legal_values.items()}
    burden = {}
    for i, row in g.iterrows():
        r = int(row.round_in_group)
        for vi, v in enumerate(venue_ids):
            if int(venue_sec[v]) < int(row.required_security_level):
                continue
            team_values = []
            for team_id in (str(row.team_a_id), str(row.team_b_id)):
                vals = raw[(team_id, r, v)]
                team_values.append(
                    0.5 * minmax(vals["distance"], *bounds["distance"])
                    + 0.3 * minmax(vals["time"], *bounds["time"])
                    + 0.2 * minmax(vals["timezone"], *bounds["timezone"])
                )
            burden[(i, vi)] = float(np.mean(team_values))
    return burden


def prepare_data():
    sheets = load_all_sheets()
    p1 = pd.read_csv(OUTPUT_DIR / "result_1_match_prediction.csv")
    groups = sheets["groups_matches"].merge(sheets["base_predictions"], on=["match_id", "group_id", "round_in_group", "team_a", "team_b"], validate="one_to_one")
    groups = groups.merge(sheets["security_requirements"][["match_id", "security_demand_score", "required_security_level"]], on="match_id", validate="one_to_one")
    groups = groups.merge(p1[["match_id", "predicted_tv_viewers"]], on="match_id", validate="one_to_one")
    groups = groups.reset_index(drop=True)
    venues = sheets["venues"].reset_index(drop=True)
    slots = sheets["time_slots"].reset_index(drop=True)
    slots["utc"] = pd.to_datetime(slots["reference_utc_time"], utc=True)
    t0 = slots["utc"].min()
    slots["utc_hour"] = ((slots["utc"] - t0).dt.total_seconds() / 3600).round().astype(int)
    limits = sheets["dynamic_resource_limits"].copy()
    limits["reference_date"] = pd.to_datetime(limits["reference_date"]).dt.date.astype(str)
    travel = compute_travel(groups, venues, sheets["security_requirements"], sheets["distance_matrix"])
    return sheets, groups, venues, slots, limits, travel


def build_model(groups, venues, slots, limits, travel, ticket, allowed_slots_by_round=None):
    global cp_model
    if cp_model is None:
        from ortools.sat.python import cp_model as _cp_model
        cp_model = _cp_model
    model = cp_model.CpModel()
    n_i, n_v, n_s = len(groups), len(venues), len(slots)
    x_ivs = {}
    by_match, by_venue_slot, by_venue = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in range(n_i):
        req = int(groups.loc[i, "required_security_level"])
        match_round = int(groups.loc[i, "round_in_group"])
        for v in range(n_v):
            if int(venues.loc[v, "security_level"]) < req:
                continue
            for s in range(n_s):
                if allowed_slots_by_round is not None and s not in allowed_slots_by_round[match_round]:
                    continue
                var = model.NewBoolVar(f"x_ivs_{i}_{v}_{s}")
                x_ivs[(i, v, s)] = var
                by_match[i].append(var)
                by_venue_slot[(v, s)].append(var)
                by_venue[v].append(var)
    for i in range(n_i):
        assert by_match[i], f"比赛 {i} 无安保合格候选"
        model.Add(cp_model.LinearExpr.Sum(by_match[i]) == 1)
    for vars_ in by_venue_slot.values():
        model.Add(cp_model.LinearExpr.Sum(vars_) <= 1)

    y_v = {}
    venue_totals = {}
    for v in range(n_v):
        y_v[v] = model.NewBoolVar(f"y_v_{v}")
        total = cp_model.LinearExpr.Sum(by_venue[v])
        venue_totals[v] = total
        lo, hi = int(venues.loc[v, "min_total_matches"]), int(venues.loc[v, "max_total_matches"])
        model.Add(total >= lo * y_v[v])
        model.Add(total <= hi * y_v[v])
        model.Add(y_v[v] <= total)
        # 题定 lo>=2，所有场馆必须启用；此断言使一次性 setup 成本逻辑透明。
        model.Add(y_v[v] == 1)

    # 场馆当地日负荷。
    for v in range(n_v):
        tz = ZoneInfo(str(venues.loc[v, "timezone"]))
        local_dates = [slots.loc[s, "utc"].to_pydatetime().astimezone(tz).date().isoformat() for s in range(n_s)]
        for day in sorted(set(local_dates)):
            vars_ = [x_ivs[(i, v, s)] for (i, vv, s) in x_ivs if vv == v and local_dates[s] == day]
            model.Add(cp_model.LinearExpr.Sum(vars_) <= int(venues.loc[v, "max_matches_per_day"]))

    # 每队轮次时刻及 60 小时休息。
    team_round_i = {}
    for i, row in groups.iterrows():
        team_round_i[(str(row.team_a_id), int(row.round_in_group))] = i
        team_round_i[(str(row.team_b_id), int(row.round_in_group))] = i
    for team in sorted({k[0] for k in team_round_i}):
        expressions = {}
        for r in (1, 2, 3):
            i = team_round_i[(team, r)]
            expressions[r] = cp_model.LinearExpr.Sum([
                int(slots.loc[s, "utc_hour"]) * x_ivs[(i, v, s)]
                for (ii, v, s) in x_ivs if ii == i
            ])
        model.Add(expressions[2] - expressions[1] >= MINIMUM_REST_HOURS)
        model.Add(expressions[3] - expressions[2] >= MINIMUM_REST_HOURS)

    prime_n = max(1, int(round(n_s * GLOBAL_PRIME_FRACTION)))
    prime_slots = set(slots.nlargest(prime_n, "global_prime_score").index.astype(int))
    large_n = max(1, int(round(n_v * LARGE_VENUE_FRACTION)))
    large_venues = set(venues.nlargest(large_n, "capacity").index.astype(int))
    team_prime, team_large = {}, {}
    for team in sorted({k[0] for k in team_round_i}):
        match_ids = {team_round_i[(team, r)] for r in (1, 2, 3)}
        team_prime[team] = cp_model.LinearExpr.Sum([
            var for (i, v, s), var in x_ivs.items() if i in match_ids and s in prime_slots
        ])
        team_large[team] = cp_model.LinearExpr.Sum([
            var for (i, v, s), var in x_ivs.items() if i in match_ids and v in large_venues
        ])
    pmax, pmin = model.NewIntVar(0, 3, "pmax"), model.NewIntVar(0, 3, "pmin")
    lmax, lmin = model.NewIntVar(0, 3, "lmax"), model.NewIntVar(0, 3, "lmin")
    for team in team_prime:
        model.Add(team_prime[team] <= pmax); model.Add(team_prime[team] >= pmin)
        model.Add(team_large[team] <= lmax); model.Add(team_large[team] >= lmin)
    model.Add(pmax - pmin <= MAX_PRIME_COUNT_RANGE)

    # 第三轮逐日高安保容量。
    limit_by_day = limits.set_index("reference_date")["high_security_capacity"].to_dict()
    for day, cap in limit_by_day.items():
        slot_ids = set(slots.index[slots["date"].astype(str).eq(day)].astype(int))
        vars_ = [var for (i, v, s), var in x_ivs.items()
                 if int(groups.loc[i, "round_in_group"]) == 3
                 and int(groups.loc[i, "required_security_level"]) >= HIGH_SECURITY_THRESHOLD and s in slot_ids]
        model.Add(cp_model.LinearExpr.Sum(vars_) <= int(cap))
    for s in range(n_s):
        model.Add(cp_model.LinearExpr.Sum([
            var for v in range(n_v) for var in by_venue_slot[(v, s)]
        ]) <= int(slots.loc[s, "broadcast_capacity"]))

    rt = _round_ticket(ticket)
    T_terms, B_terms, C_terms, D_terms, R_terms = [], [], [], [], []
    contributions = {}
    for (i, v, s), var in x_ivs.items():
        r = int(groups.loc[i, "round_in_group"])
        n_att = min(float(groups.loc[i, "expected_attendance_base"]), float(venues.loc[v, "capacity"]))
        tval = float(rt[r]["base_ticket_price_usd"]) * n_att
        bval = (float(groups.loc[i, "predicted_tv_viewers"]) * float(rt[r]["broadcast_unit_value_usd"])
                * float(slots.loc[s, "global_prime_score"]) * float(rt[r]["sponsor_weight"]))
        cval = float(venues.loc[v, "operation_cost_musd_per_match"]) * 1_000_000 + SECURITY_BASE_COST_USD * int(groups.loc[i, "required_security_level"]) * float(venues.loc[v, "security_cost_index"])
        dval = float(travel[(i, v)])
        rval = (0.5 * float(venues.loc[v, "climate_risk"])
                + 0.3 * n_att / float(venues.loc[v, "capacity"])
                + 0.2 * int(groups.loc[i, "required_security_level"]) / float(venues.loc[v, "security_level"]))
        contributions[(i, v, s)] = (tval, bval, cval, dval, rval, n_att)
        T_terms.append(round(tval) * var); B_terms.append(round(bval) * var)
        C_terms.append(round(cval) * var); D_terms.append(round(dval * CP_SAT_SCALE) * var)
        R_terms.append(round(rval * CP_SAT_SCALE) * var)
    for v, var in y_v.items():
        C_terms.append(round(float(venues.loc[v, "setup_cost_musd"]) * 1_000_000) * var)
    exprs = {"T": cp_model.LinearExpr.Sum(T_terms),
             "B": cp_model.LinearExpr.Sum(B_terms),
             "C": cp_model.LinearExpr.Sum(C_terms),
             "Dsum": cp_model.LinearExpr.Sum(D_terms),
             "Rsum": cp_model.LinearExpr.Sum(R_terms)}
    aux = {"x_ivs": x_ivs, "y_v": y_v, "pmax": pmax, "pmin": pmin,
           "lmax": lmax, "lmin": lmin, "prime_slots": prime_slots,
           "large_venues": large_venues, "contributions": contributions,
           "team_round_i": team_round_i}
    return model, exprs, aux


def _solver(seconds: float, gap: float = 0.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = MASTER_SEED
    solver.parameters.relative_gap_limit = float(gap)
    return solver


def _status_name(solver, status):
    return solver.StatusName(status)


def candidate_envelope_bounds(aux, groups: pd.DataFrame, venues: pd.DataFrame):
    """按附件“全部候选组合”口径构造确定性 Min--Max 包络。

    对每场比赛在其全部安保合格 (场馆, 时段) 候选中分别取最小值和
    最大值，再跨 72 场求和。任何满足联合硬约束的赛程都是该笛卡尔
    候选域的子集，故其 T/B/C 必位于该包络内。附件数据中每个场馆的
    min_total_matches 均大于 0，一次性启用成本因此是所有可行方案共有
    的常数，并在 C 的两端同时计入。
    """
    assert (venues["min_total_matches"].astype(int) > 0).all(), "当前数据并非全部场馆必启用"
    by_match = defaultdict(list)
    for (i, _v, _s), values in aux["contributions"].items():
        by_match[int(i)].append(values)
    assert set(by_match) == set(range(len(groups))) and all(by_match.values())

    setup = float(venues["setup_cost_musd"].sum() * 1_000_000)
    bounds = {
        "T_min": float(sum(min(x[0] for x in by_match[i]) for i in range(len(groups)))),
        "T_max": float(sum(max(x[0] for x in by_match[i]) for i in range(len(groups)))),
        "B_min": float(sum(min(x[1] for x in by_match[i]) for i in range(len(groups)))),
        "B_max": float(sum(max(x[1] for x in by_match[i]) for i in range(len(groups)))),
        "C_min": float(setup + sum(min(x[2] for x in by_match[i]) for i in range(len(groups)))),
        "C_max": float(setup + sum(max(x[2] for x in by_match[i]) for i in range(len(groups)))),
    }
    assert all(bounds[f"{key}_max"] > bounds[f"{key}_min"] for key in ("T", "B", "C"))
    meta = {
        "method": "per-match envelope over all legal match-venue-slot candidates",
        "candidate_count": int(len(aux["x_ivs"])),
        "match_count": int(len(groups)),
        "fixed_setup_cost_usd": setup,
        "all_venues_mandatory": True,
        "solver_used": False,
    }
    return bounds, meta


def extract_schedule(solver, groups, venues, slots, aux):
    rows = []
    for i, match in groups.iterrows():
        selected = [(v, s) for (ii, v, s), var in aux["x_ivs"].items() if ii == i and solver.Value(var)]
        assert len(selected) == 1
        v, s = selected[0]
        utc = slots.loc[s, "utc"].to_pydatetime()
        local = utc.astimezone(ZoneInfo(str(venues.loc[v, "timezone"])))
        tval, bval, _, dval, rval, n_att = aux["contributions"][(i, v, s)]
        rows.append({
            "match_id": str(match.match_id), "group_id": str(match.group_id),
            "round_in_group": int(match.round_in_group), "team_a": str(match.team_a), "team_b": str(match.team_b),
            "team_a_id": str(match.team_a_id), "team_b_id": str(match.team_b_id),
            "venue_id": str(venues.loc[v, "venue_id"]), "venue_index": int(v),
            "slot_id": str(slots.loc[s, "slot_id"]), "slot_index": int(s),
            "city": str(venues.loc[v, "city"]), "country": str(venues.loc[v, "country"]),
            "reference_date": str(slots.loc[s, "date"]),
            "reference_kickoff_time": str(slots.loc[s, "reference_kickoff_time"]),
            "local_datetime": local.isoformat(), "utc_datetime": utc.isoformat(),
            "required_security_level": int(match.required_security_level),
            "venue_security_level": int(venues.loc[v, "security_level"]),
            "venue_capacity": int(venues.loc[v, "capacity"]),
            "expected_attendance": float(n_att), "expected_tv_viewers": int(match.predicted_tv_viewers),
            "ticket_revenue_usd": float(tval), "broadcast_value_usd": float(bval),
            "travel_cost_index": float(dval), "risk_index": float(rval),
            "is_prime": int(s in aux["prime_slots"]), "is_large_venue": int(v in aux["large_venues"]),
        })
    return rows


def evaluate_rows(rows, groups, venues, bounds):
    df = pd.DataFrame(rows)
    venue_used = set(df["venue_id"])
    setup = venues.loc[venues["venue_id"].isin(venue_used), "setup_cost_musd"].sum() * 1_000_000
    vmap = venues.set_index("venue_id")
    op = 0.0
    for r in rows:
        v = vmap.loc[r["venue_id"]]
        op += float(v.operation_cost_musd_per_match) * 1_000_000 + SECURITY_BASE_COST_USD * r["required_security_level"] * float(v.security_cost_index)
    T0, B0, C0 = float(df.ticket_revenue_usd.sum()), float(df.broadcast_value_usd.sum()), float(setup + op)
    T = float(minmax(T0, bounds["T_min"], bounds["T_max"]))
    B = float(minmax(B0, bounds["B_min"], bounds["B_max"]))
    C = float(minmax(C0, bounds["C_min"], bounds["C_max"]))
    U = float(groups["uncertainty_index"].mean())
    H = float(groups["attractiveness_index"].mean() / 100.0)
    D, R = float(df.travel_cost_index.mean()), float(df.risk_index.mean())
    teams = sorted(set(df.team_a_id) | set(df.team_b_id))
    prime_counts, large_counts = [], []
    for t in teams:
        z = df[(df.team_a_id == t) | (df.team_b_id == t)]
        prime_counts.append(int(z.is_prime.sum())); large_counts.append(int(z.is_large_venue.sum()))
    F = 0.5 * (max(prime_counts) - min(prime_counts)) / 3.0 + 0.5 * (max(large_counts) - min(large_counts)) / 3.0
    Z2 = 0.25*T + 0.25*B + 0.15*U + 0.10*H - 0.08*C - 0.07*D - 0.06*F - 0.04*R
    return {"T_raw": T0, "B_raw": B0, "C_raw": C0, "T": T, "B": B, "U": U, "H": H,
            "C": C, "D": D, "F": float(F), "R": R, "Z2": float(Z2),
            "prime_count_range": max(prime_counts)-min(prime_counts),
            "large_count_range": max(large_counts)-min(large_counts)}


def validate_schedule(rows, groups, venues, slots, limits):
    df = pd.DataFrame(rows)
    violations = []
    if len(df) != 72 or not df.match_id.is_unique: violations.append("每场唯一")
    if df.duplicated(["venue_id", "utc_datetime"]).any(): violations.append("场馆UTC冲突")
    for team in sorted(set(df.team_a_id) | set(df.team_b_id)):
        z = df[(df.team_a_id == team) | (df.team_b_id == team)].sort_values("round_in_group")
        hours = pd.to_datetime(z.utc_datetime, utc=True).diff().dropna().dt.total_seconds() / 3600
        if len(z) != 3 or (hours < MINIMUM_REST_HOURS - 1e-9).any(): violations.append(f"休息:{team}")
    vmap = venues.set_index("venue_id")
    for vid, z in df.groupby("venue_id"):
        if not (int(vmap.loc[vid, "min_total_matches"]) <= len(z) <= int(vmap.loc[vid, "max_total_matches"])):
            violations.append(f"场馆总量:{vid}")
        local_days = z.local_datetime.astype(str).str.slice(0, 10)
        if local_days.value_counts().max() > int(vmap.loc[vid, "max_matches_per_day"]): violations.append(f"当地日:{vid}")
    if (df.required_security_level > df.venue_security_level).any(): violations.append("安保资格")
    slot_caps = slots.set_index("slot_id")["broadcast_capacity"].to_dict()
    for sid, z in df.groupby("slot_id"):
        if len(z) > int(slot_caps[sid]): violations.append(f"转播容量:{sid}")
    lim = limits.set_index("reference_date")["high_security_capacity"].to_dict()
    high = df[(df.round_in_group == 3) & (df.required_security_level >= HIGH_SECURITY_THRESHOLD)]
    for day, z in high.groupby("reference_date"):
        if len(z) > int(lim[str(day)]): violations.append(f"三轮高安保:{day}")
    prime = []
    for team in sorted(set(df.team_a_id) | set(df.team_b_id)):
        z = df[(df.team_a_id == team) | (df.team_b_id == team)]
        prime.append(int(z.is_prime.sum()))
    if max(prime) - min(prime) > MAX_PRIME_COUNT_RANGE: violations.append("黄金公平")
    assert not violations, "P2硬约束失败: " + ",".join(violations[:5])
    rest_values = []
    for team in sorted(set(df.team_a_id) | set(df.team_b_id)):
        z = df[(df.team_a_id == team) | (df.team_b_id == team)].sort_values("round_in_group")
        rest_values.extend((pd.to_datetime(z.utc_datetime, utc=True).diff().dropna().dt.total_seconds()/3600).tolist())
    return {"pass": True, "n_constraints": 9, "n_violations": 0,
            "min_rest_hours": float(min(rest_values))}


def solve() -> dict:
    set_all_seeds(MASTER_SEED)
    start = time.time()
    sheets, groups, venues, slots, limits, travel = prepare_data()
    model, exprs, aux = build_model(groups, venues, slots, limits, travel, sheets["ticket_broadcast"])
    print(f"[P2] x_ivs={len(aux['x_ivs'])} 开始寻找解析可行基线", flush=True)
    # 仅为 full-domain 求解器构造 incumbent hint；最终模型仍包含全部 80 个时段。
    hint_windows = {1: set(range(0, 20)), 2: set(range(28, 48)), 3: set(range(60, 80))}
    hint_model, _, hint_aux = build_model(groups, venues, slots, limits, travel, sheets["ticket_broadcast"], hint_windows)
    hint_model.Maximize(0)
    baseline_solver = _solver(180.0)
    status = baseline_solver.Solve(hint_model)
    print(f"[P2] hint status={baseline_solver.StatusName(status)}", flush=True)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), "P2无可行基线"
    feasible_baseline = extract_schedule(baseline_solver, groups, venues, slots, hint_aux)
    validate_schedule(feasible_baseline, groups, venues, slots, limits)
    chosen_hint = {(r["match_id"], r["venue_id"], r["slot_id"]) for r in feasible_baseline}
    for (i, v, s), var in aux["x_ivs"].items():
        key = (str(groups.loc[i, "match_id"]), str(venues.loc[v, "venue_id"]), str(slots.loc[s, "slot_id"]))
        if key in chosen_hint:
            model.AddHint(var, 1)

    bounds, normalization_meta = candidate_envelope_bounds(aux, groups, venues)
    print(f"[P2] deterministic candidate envelope {bounds}", flush=True)

    # 将浮点归一化目标一次性缩放为整数；U/H 为固定常数，求解后再加回。
    coeffs = []
    for key, var in aux["x_ivs"].items():
        t, b, c, d, r, _ = aux["contributions"][key]
        coef = (0.25 * t / (bounds["T_max"] - bounds["T_min"])
                + 0.25 * b / (bounds["B_max"] - bounds["B_min"])
                - 0.08 * c / (bounds["C_max"] - bounds["C_min"])
                - 0.07 * d / 72.0 - 0.04 * r / 72.0)
        coeffs.append(round(coef * CP_SAT_SCALE) * var)
    for v, var in aux["y_v"].items():
        setup = float(venues.loc[v, "setup_cost_musd"]) * 1_000_000
        coef = -0.08 * setup / (bounds["C_max"] - bounds["C_min"])
        coeffs.append(round(coef * CP_SAT_SCALE) * var)
    coeffs.append(round(-0.06 * 0.5 / 3.0 * CP_SAT_SCALE) * (aux["pmax"] - aux["pmin"] + aux["lmax"] - aux["lmin"]))
    model.Maximize(cp_model.LinearExpr.Sum(coeffs))
    solver = _solver(P2_TIME_LIMIT_SECONDS, gap=5e-3)
    print("[P2] main CP-SAT solve", flush=True)
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"P2主求解失败 {solver.StatusName(status)}"
    rows = extract_schedule(solver, groups, venues, slots, aux)
    audit = validate_schedule(rows, groups, venues, slots, limits)
    metrics = evaluate_rows(rows, groups, venues, bounds)
    baseline_metrics = evaluate_rows(feasible_baseline, groups, venues, bounds)
    assert all(-1e-8 <= metrics[k] <= 1+1e-8 for k in ["T","B","U","H","C","D","F","R"])
    assert -0.25-1e-8 <= metrics["Z2"] <= 0.75+1e-8

    out = pd.DataFrame(rows)
    out["fairness_penalty"] = metrics["F"]
    out["total_objective_value"] = np.nan
    out.loc[0, "total_objective_value"] = metrics["Z2"]
    template_cols = pd.read_csv(Path(__file__).resolve().parent.parent / "user_data" / "result_2_template.csv").columns
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out[list(template_cols)].to_csv(OUTPUT_DIR / "result_2_group_schedule.csv", index=False, encoding="utf-8-sig")
    result = {
        "metadata": {**runtime_metadata(MASTER_SEED), "elapsed_seconds": time.time()-start},
        "method": "three-dimensional match-venue-slot CP-SAT",
        "solver": {"status": solver.StatusName(status), "incumbent_scaled": solver.ObjectiveValue(),
                   "best_bound_scaled": solver.BestObjectiveBound(), "wall_time": solver.WallTime(),
                   "workers": 1, "relative_gap_limit": 0.005},
        "normalization_bounds": bounds, "normalization_meta": normalization_meta,
        "metrics": metrics, "schedule": rows, "constraint_audit": audit,
        "feasible_baseline": {"schedule": feasible_baseline, "metrics": baseline_metrics,
                              "constraint_audit": {"pass": True, "n_violations": 0}},
        "venue_load": out.groupby("venue_id").size().to_dict(),
        "seed": MASTER_SEED,
        "outputs": ["output/result_2_group_schedule.csv"],
    }
    write_json(FIGURES_DIR / "problem_2_results.json", result)
    print(f"[P2] validate_capability PASS Z2={metrics['Z2']:.6f} status={solver.StatusName(status)}", flush=True)
    return result


if __name__ == "__main__":
    solve()
