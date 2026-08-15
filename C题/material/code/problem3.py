"""问题三：共同截面晋级模拟与四资源日级多选择 CP-SAT。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from params import (
    ATTENDANCE_FLOOR_RATIO, CP_SAT_SCALE, DELTA_INJURY_COEF,
    DELTA_STATE_COEF, DIRECT_INJURY_COEF, GOAL_MEAN_CLIP, MASTER_SEED,
    MC_DRAWS, P3_TIME_LIMIT_SECONDS, RAW_FEEDBACK_CLIP, RED_CARD_DECAY,
    STATE_WEIGHTS, STATE_XG_SCALE, TEAM_FEEDBACK_CLIP,
)
from utils import FIGURES_DIR, OUTPUT_DIR, load_all_sheets, minmax, runtime_metadata, set_all_seeds, write_json


def team_snapshot(live: pd.DataFrame, teams: list[str]) -> dict[str, dict]:
    stats = {t: {"games": 0, "points": 0.0, "gf": 0.0, "ga": 0.0,
                 "xgf": 0.0, "xga": 0.0, "red": 0.0, "injuries": []} for t in teams}
    for r in live.itertuples(index=False):
        ga, gb = int(r.goals_a), int(r.goals_b)
        pa, pb = (3, 0) if ga > gb else ((0, 3) if ga < gb else (1, 1))
        for t, gf, gc, xgf, xga, red, pts in [
            (r.team_a, ga, gb, r.xg_a, r.xg_b, r.red_cards_a, pa),
            (r.team_b, gb, ga, r.xg_b, r.xg_a, r.red_cards_b, pb),
        ]:
            s = stats[str(t)]
            s["games"] += 1; s["points"] += pts; s["gf"] += gf; s["ga"] += gc
            s["xgf"] += float(xgf); s["xga"] += float(xga); s["red"] += float(red)
            s["injuries"].append(float(r.injury_impact_level))
    for t, s in stats.items():
        assert s["games"] == 2, f"{t} 前两轮场数不等于2"
        g = s["games"]
        h = float(np.clip(np.mean(s["injuries"]), 0, 3))
        state = (STATE_WEIGHTS["points"] * s["points"] / (3 * g)
                 + STATE_WEIGHTS["xg_form"] * (0.5 + 0.5 * np.tanh(((s["xgf"]-s["xga"])/g) / STATE_XG_SCALE))
                 + STATE_WEIGHTS["discipline"] * np.exp(-RED_CARD_DECAY * s["red"] / g))
        s["injury"] = h; s["state"] = float(np.clip(state, 0, 1)); s["gd"] = s["gf"] - s["ga"]
    return stats


def updated_lambdas(round3: pd.DataFrame, snapshot: dict[str, dict]):
    la, lb = [], []
    for r in round3.itertuples(index=False):
        a, b = snapshot[str(r.team_a)], snapshot[str(r.team_b)]
        delta = DELTA_STATE_COEF * (a["state"] - b["state"]) + DELTA_INJURY_COEF * (b["injury"] - a["injury"])
        la.append(float(np.clip(r.expected_goals_a * np.exp(delta - DIRECT_INJURY_COEF*a["injury"]), *GOAL_MEAN_CLIP)))
        lb.append(float(np.clip(r.expected_goals_b * np.exp(-delta - DIRECT_INJURY_COEF*b["injury"]), *GOAL_MEAN_CLIP)))
    return np.asarray(la), np.asarray(lb)


def _blocks(order, metrics):
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and metrics[order[end]] == metrics[order[start]]:
            end += 1
        yield start, end, order[start:end]
        start = end


def simulate_qualification(round3: pd.DataFrame, membership: pd.DataFrame,
                           snapshot: dict[str, dict], lam_a: np.ndarray, lam_b: np.ndarray,
                           seed: int, keep_scores: bool = True):
    """一个固定 snapshot 上一次性生成 MC_DRAWS×24，并对完全同分块等比例分配。"""
    rng = np.random.Generator(np.random.PCG64(seed))
    score_a = rng.poisson(lam=np.broadcast_to(lam_a, (MC_DRAWS, len(round3))))
    score_b = rng.poisson(lam=np.broadcast_to(lam_b, (MC_DRAWS, len(round3))))
    teams = membership.sort_values(["group_id", "team_id"])["team_name"].astype(str).tolist()
    team_idx = {t: i for i, t in enumerate(teams)}
    groups = {str(g): [team_idx[str(t)] for t in z.team_name] for g, z in membership.groupby("group_id", sort=True)}
    match_idx = [(team_idx[str(r.team_a)], team_idx[str(r.team_b)]) for r in round3.itertuples(index=False)]
    base_pts = np.array([snapshot[t]["points"] for t in teams], float)
    base_gd = np.array([snapshot[t]["gd"] for t in teams], float)
    base_gf = np.array([snapshot[t]["gf"] for t in teams], float)
    adv = np.zeros((MC_DRAWS, len(teams)), dtype=np.float64)
    mass = np.zeros(MC_DRAWS, dtype=np.float64)
    for m in range(MC_DRAWS):
        pts, gd, gf = base_pts.copy(), base_gd.copy(), base_gf.copy()
        for k, (a, b) in enumerate(match_idx):
            xa, xb = int(score_a[m, k]), int(score_b[m, k])
            gf[a] += xa; gf[b] += xb; gd[a] += xa-xb; gd[b] += xb-xa
            if xa > xb: pts[a] += 3
            elif xa < xb: pts[b] += 3
            else: pts[a] += 1; pts[b] += 1
        candidate_weight = np.zeros(len(teams), float)
        metrics = [(float(pts[i]), float(gd[i]), float(gf[i])) for i in range(len(teams))]
        for members in groups.values():
            order = sorted(members, key=lambda i: metrics[i], reverse=True)
            for start, end, block in _blocks(order, metrics):
                direct_slots = max(0, min(end, 2) - start)
                if direct_slots:
                    adv[m, block] += direct_slots / len(block)
                if start <= 2 < end:
                    candidate_weight[block] = 1.0 / len(block)
        candidates = [i for i, w in enumerate(candidate_weight) if w > 0]
        order = sorted(candidates, key=lambda i: metrics[i], reverse=True)
        remaining = 8.0
        for _, _, block in _blocks(order, metrics):
            block_mass = float(candidate_weight[block].sum())
            take = min(remaining, block_mass)
            if take > 0:
                adv[m, block] += candidate_weight[block] * (take / block_mass)
                remaining -= take
            if remaining <= 1e-12:
                break
        mass[m] = float(adv[m].sum())
    assert np.max(np.abs(mass - 32.0)) <= 1e-6, "晋级名额质量不守恒"
    result = {"teams": teams, "probability": adv.mean(axis=0), "mass": mass}
    if keep_scores:
        result.update({"score_a": score_a, "score_b": score_b, "adv": adv})
    return result


def conditional_probabilities(sim, round3: pd.DataFrame):
    idx = {t: i for i, t in enumerate(sim["teams"])}
    rows = []
    for k, r in enumerate(round3.itertuples(index=False)):
        draw = sim["score_a"][:, k] == sim["score_b"][:, k]
        awin = sim["score_a"][:, k] > sim["score_b"][:, k]
        bwin = sim["score_b"][:, k] > sim["score_a"][:, k]
        ia, ib = idx[str(r.team_a)], idx[str(r.team_b)]
        def avg(mask, j): return float(sim["adv"][mask, j].mean()) if int(mask.sum()) else np.nan
        rows.append({
            "match_id": str(r.match_id), "p_a_draw": avg(draw, ia), "p_b_draw": avg(draw, ib),
            "p_a_cond_win": avg(awin, ia), "p_b_cond_win": avg(bwin, ib),
            "n_draw": int(draw.sum()), "n_a_win": int(awin.sum()), "n_b_win": int(bwin.sum()),
        })
    return pd.DataFrame(rows)


def feedback_by_team(live, base, p1_group):
    z = live.merge(base[["match_id", "expected_attendance_base"]], on="match_id", validate="one_to_one")
    z = z.merge(p1_group[["match_id", "predicted_tv_viewers"]], on="match_id", validate="one_to_one")
    values = defaultdict(lambda: {"N": [], "V": []})
    for r in z.itertuples(index=False):
        rn = float(np.clip(r.attendance / r.expected_attendance_base, *RAW_FEEDBACK_CLIP))
        rv = float(np.clip(r.tv_viewers / r.predicted_tv_viewers, *RAW_FEEDBACK_CLIP))
        for t in (str(r.team_a), str(r.team_b)):
            values[t]["N"].append(rn); values[t]["V"].append(rv)
    return {t: {k: float(np.clip(np.mean(v), *TEAM_FEEDBACK_CLIP)) for k, v in d.items()} for t, d in values.items()}


def price_common_upper(epsilon: float, day_increase: float) -> float:
    """共同价格域解析上界；与静态/动态状态无关。"""
    return float(min(day_increase, 0.88 ** (1 / epsilon) - 1))


def resource_maps(costs: pd.DataFrame):
    return {(str(r.resource_type), int(r.resource_level)): {
        "cost": float(r.unit_cost_index), "demand": float(r.demand_multiplier), "risk": float(r.risk_multiplier)
    } for r in costs.itertuples(index=False)}


def derive_match_environment(round3, schedule, venues, ticket, limits, snapshot, feedback,
                             team_prob, cond):
    sched = schedule[schedule["round_in_group"].eq(3)].copy()
    z = round3.merge(sched[["match_id", "venue_id", "slot_id", "reference_date"]], on="match_id", validate="one_to_one")
    z = z.merge(venues[["venue_id", "capacity", "security_level"]], on="venue_id", validate="many_to_one")
    z = z.merge(cond, on="match_id", validate="one_to_one")
    trow = ticket.loc[ticket["match_stage"].eq("Group_Match_R3")].iloc[0]
    pmap = team_prob
    rows = []
    for r in z.itertuples(index=False):
        a, b = snapshot[str(r.team_a)], snapshot[str(r.team_b)]
        pa, pb = pmap[str(r.team_a)], pmap[str(r.team_b)]
        Q = (4*pa*(1-pa) + 4*pb*(1-pb)) / 2.0
        fN = float(np.sqrt(feedback[str(r.team_a)]["N"] * feedback[str(r.team_b)]["N"]))
        fV = float(np.sqrt(feedback[str(r.team_a)]["V"] * feedback[str(r.team_b)]["V"]))
        S = (a["state"] + b["state"]) / 2.0; H = (a["injury"] + b["injury"]) / 6.0
        A0 = float(r.attractiveness_index) / 100.0
        F = float(np.clip((0.5*fN + 0.5*fV - 0.75) / 0.50, 0, 1))
        A = float(np.clip(0.50*A0 + 0.25*Q + 0.12*F + 0.08*S + 0.05*(1-H), 0, 1))
        Ntilde = float(r.expected_attendance_base * fN * (0.88+0.27*Q) * (0.94+0.12*S) * (1-0.10*H))
        Vtilde = float(r.predicted_tv_viewers * fV * (0.87+0.30*Q) * (0.90+0.20*A) * (1-0.06*H))
        Rstake = float(0.5*(2*pa-1)**2 + 0.5*(2*pb-1)**2)
        D = min(float(r.p_a_draw), float(r.p_b_draw))
        G = 0.5*max(float(r.p_a_cond_win)-float(r.p_a_draw), 0) + 0.5*max(float(r.p_b_cond_win)-float(r.p_b_draw), 0)
        Rcol = float(np.clip(D * (1-np.clip(G, 0, 1)) * (1-0.35*Rstake), 0, 1))
        occupancy = float(np.clip(Ntilde / float(r.capacity), 0, 1))
        dsec = float(np.clip(0.45*r.security_demand_score + 0.25*occupancy + 0.15*A + 0.15*(Rstake+Rcol)/2, 0, 1))
        rows.append({
            **r._asdict(), "p_a": pa, "p_b": pb, "Q": Q, "fN": fN, "fV": fV,
            "S": S, "H": H, "A0": A0, "A": A, "Ntilde": Ntilde, "Vtilde": Vtilde,
            "Rstake": Rstake, "Rcol": Rcol, "dsec": dsec,
            "base_ticket_price": float(trow.base_ticket_price_usd), "epsilon": float(trow.price_elasticity),
            "broadcast_unit_value": float(trow.broadcast_unit_value_usd),
        })
    return pd.DataFrame(rows)


def make_action_table(env: pd.DataFrame, costs: pd.DataFrame, limits: pd.DataFrame, dynamic: bool):
    maps = resource_maps(costs)
    lim = limits.copy(); lim["reference_date"] = lim["reference_date"].astype(str)
    daymap = lim.set_index("reference_date").to_dict(orient="index")
    action_table = []
    for i, r in env.iterrows():
        if dynamic:
            Nbase, Vbase, attractiveness = float(r.Ntilde), float(r.Vtilde), float(r.A)
            stake, col, dsec = float(r.Rstake), float(r.Rcol), float(r.dsec)
        else:
            Nbase, Vbase, attractiveness = float(r.expected_attendance_base), float(r.predicted_tv_viewers), float(r.A0)
            stake, col, dsec = 1-float(r.uncertainty_index), 0.0, float(r.security_demand_score)
        day_limit = daymap[str(r.reference_date)]
        delta = price_common_upper(float(r.epsilon), float(day_limit["max_ticket_increase_rate"]))
        for b in (1, 2, 3):
            for q in range(int(r.required_security_level), int(r.security_level)+1):
                for l in (1, 2, 3):
                    bm, qm, lm = maps[("broadcast", b)], maps[("security", q)], maps[("transport", l)]
                    at_zero = min(float(r.capacity), Nbase * lm["demand"])
                    attendance = min(float(r.capacity), Nbase * lm["demand"] * (1+delta) ** float(r.epsilon))
                    assert attendance + 1e-8 >= ATTENDANCE_FLOOR_RATIO * at_zero
                    tv = Vbase * bm["demand"]
                    TV = float(r.base_ticket_price) * (1+delta) * attendance
                    BV = float(r.broadcast_unit_value) * tv
                    risk = 0.40*stake + 0.40*col + 0.20*dsec*qm["risk"]
                    breakdown = {"broadcast": bm["cost"], "security": qm["cost"], "transport": lm["cost"]}
                    action_table.append({
                        "match_index": i, "match_id": str(r.match_id), "reference_date": str(r.reference_date),
                        "b": b, "q": q, "l": l, "delta": delta, "attendance": attendance,
                        "attendance_at_zero": at_zero, "tv": tv, "TV": TV, "BV": BV,
                        "A": attractiveness, "C": sum(breakdown.values()), "R": float(np.clip(risk, 0, 1)),
                        "cost_breakdown": breakdown,
                    })
    return action_table


def action_bounds(actions):
    return {k: (min(a[k] for a in actions), max(a[k] for a in actions)) for k in ("TV", "BV", "A", "C", "R")}


def action_value(a, bounds):
    return (0.35*minmax(a["TV"], *bounds["TV"]) + 0.35*minmax(a["BV"], *bounds["BV"])
            + 0.10*minmax(a["A"], *bounds["A"]) - 0.10*minmax(a["C"], *bounds["C"])
            - 0.10*minmax(a["R"], *bounds["R"]))


def optimize_actions(action_table, limits, bounds, label):
    model = cp_model.CpModel()
    z_ia = [model.NewBoolVar(f"z_ia_{label}_{j}") for j in range(len(action_table))]
    by_match, by_day = defaultdict(list), defaultdict(list)
    for j, a in enumerate(action_table):
        by_match[a["match_index"]].append(j); by_day[a["reference_date"]].append(j)
    for idx in sorted(by_match): model.Add(sum(z_ia[j] for j in by_match[idx]) == 1)
    lim = limits.copy(); lim["reference_date"] = lim["reference_date"].astype(str)
    daily_resource_budget = lim.set_index("reference_date").to_dict(orient="index")
    for day, js in by_day.items():
        caps = daily_resource_budget[day]
        model.Add(sum(z_ia[j] for j in js if action_table[j]["b"] == 3) <= int(caps["high_broadcast_capacity"]))
        model.Add(sum(z_ia[j] for j in js if action_table[j]["q"] >= 3) <= int(caps["high_security_capacity"]))
        model.Add(sum(z_ia[j] for j in js if action_table[j]["l"] == 3) <= int(caps["enhanced_transport_capacity"]))
        model.Add(sum(round(action_table[j]["C"] * 1000) * z_ia[j] for j in js) <= round(float(caps["daily_resource_budget_index"])*1000))
    model.Maximize(sum(round(action_value(a, bounds)*CP_SAT_SCALE)*z_ia[j] for j, a in enumerate(action_table)))
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = min(P3_TIME_LIMIT_SECONDS, 300)
    solver.parameters.relative_gap_limit = 1e-3; solver.parameters.num_search_workers = 1; solver.parameters.random_seed = MASTER_SEED
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    selected = [action_table[j] for j, var in enumerate(z_ia) if solver.Value(var)]
    assert len(selected) == 24
    return selected, {"status": solver.StatusName(status), "incumbent": solver.ObjectiveValue()/CP_SAT_SCALE,
                      "best_bound": solver.BestObjectiveBound()/CP_SAT_SCALE, "wall_time": solver.WallTime()}


def daily_audit(selected, limits):
    lim = limits.copy(); lim["reference_date"] = lim["reference_date"].astype(str)
    lim = lim.set_index("reference_date").to_dict(orient="index")
    rows = []
    for day in sorted({a["reference_date"] for a in selected}):
        z = [a for a in selected if a["reference_date"] == day]
        use = {"broadcast3": sum(a["b"] == 3 for a in z), "security3plus": sum(a["q"] >= 3 for a in z),
               "transport3": sum(a["l"] == 3 for a in z), "budget": sum(a["C"] for a in z)}
        cap = lim[day]
        ok = (use["broadcast3"] <= cap["high_broadcast_capacity"] and use["security3plus"] <= cap["high_security_capacity"]
              and use["transport3"] <= cap["enhanced_transport_capacity"] and use["budget"] <= cap["daily_resource_budget_index"]+1e-9)
        rows.append({"date": day, **use, "caps": cap, "pass": bool(ok)})
    assert all(r["pass"] for r in rows)
    return rows


def solve() -> dict:
    set_all_seeds(MASTER_SEED); start = time.time(); sheets = load_all_sheets()
    groups = sheets["groups_matches"]
    base = groups.merge(sheets["base_predictions"], on=["match_id","group_id","round_in_group","team_a","team_b"], validate="one_to_one")
    base = base.merge(sheets["security_requirements"][["match_id","security_demand_score","required_security_level"]], on="match_id", validate="one_to_one")
    p1 = pd.read_csv(OUTPUT_DIR / "result_1_match_prediction.csv")
    base = base.merge(p1[["match_id","predicted_tv_viewers"]], on="match_id", validate="one_to_one")
    round3 = base[base["round_in_group"].eq(3)].reset_index(drop=True)
    teams = sheets["group_membership"]["team_name"].astype(str).tolist()
    snapshot = team_snapshot(sheets["live_group_results"], teams)
    lam_a, lam_b = updated_lambdas(round3, snapshot)
    print(f"[P3] joint Monte Carlo shape=({MC_DRAWS},{len(round3)})", flush=True)
    sim = simulate_qualification(round3, sheets["group_membership"], snapshot, lam_a, lam_b, MASTER_SEED)
    team_prob = {t: float(sim["probability"][i]) for i, t in enumerate(sim["teams"])}
    assert abs(sum(team_prob.values()) - 32.0) <= 1e-5 and all(0 <= p <= 1 for p in team_prob.values())
    cond = conditional_probabilities(sim, round3)
    feedback = feedback_by_team(sheets["live_group_results"], sheets["base_predictions"], p1)
    schedule = pd.read_csv(OUTPUT_DIR / "result_2_group_schedule.csv")
    env = derive_match_environment(round3, schedule, sheets["venues"], sheets["ticket_broadcast"],
                                   sheets["dynamic_resource_limits"], snapshot, feedback, team_prob, cond)
    static_actions = make_action_table(env, sheets["dynamic_resource_costs"], sheets["dynamic_resource_limits"], False)
    dynamic_actions = make_action_table(env, sheets["dynamic_resource_costs"], sheets["dynamic_resource_limits"], True)
    static_bounds = action_bounds(static_actions)
    static_selected, static_solver = optimize_actions(static_actions, sheets["dynamic_resource_limits"], static_bounds, "static")
    static_action_snapshot = {(a["match_id"]): (a["b"],a["q"],a["l"],a["delta"]) for a in static_selected}
    shared_eval_bounds = action_bounds(dynamic_actions)
    dynamic_selected, dynamic_solver = optimize_actions(dynamic_actions, sheets["dynamic_resource_limits"], shared_eval_bounds, "dynamic")
    dyn_lookup = {(a["match_id"],a["b"],a["q"],a["l"]): a for a in dynamic_actions}
    static_reeval = [dyn_lookup[(mid,*decision[:3])] for mid, decision in static_action_snapshot.items()]
    assert static_action_snapshot == {(a["match_id"]):(a["b"],a["q"],a["l"],a["delta"]) for a in static_selected}
    z_static = sum(action_value(a, shared_eval_bounds) for a in static_reeval)
    z_dynamic = sum(action_value(a, shared_eval_bounds) for a in dynamic_selected)
    assert z_dynamic + 1e-4 >= z_static
    daily_static = daily_audit(static_reeval, sheets["dynamic_resource_limits"])
    daily_dynamic = daily_audit(dynamic_selected, sheets["dynamic_resource_limits"])

    envmap = env.set_index("match_id").to_dict(orient="index")
    smap = {a["match_id"]:a for a in static_reeval}; dmap = {a["match_id"]:a for a in dynamic_selected}
    rows = []
    for mid in env["match_id"]:
        e, s, d = envmap[mid], smap[mid], dmap[mid]
        sv, dv = action_value(s, shared_eval_bounds), action_value(d, shared_eval_bounds)
        rows.append({
            "match_id": mid, "group_id": e["group_id"], "team_a": e["team_a"], "team_b": e["team_b"],
            "venue_id": e["venue_id"], "slot_id": e["slot_id"], "reference_date": e["reference_date"],
            "updated_p_team_a_advance": e["p_a"], "updated_p_team_b_advance": e["p_b"],
            "updated_expected_attendance": d["attendance"], "updated_expected_tv_viewers": d["tv"],
            "stakeless_risk": e["Rstake"], "collusion_risk": e["Rcol"], "updated_attractiveness": e["A"],
            "recommended_broadcast_priority": d["b"], "recommended_security_level": d["q"],
            "recommended_transport_level": d["l"], "recommended_ticket_adjustment": d["delta"],
            "updated_ticket_revenue_usd": d["TV"], "updated_broadcast_value_usd": d["BV"],
            "resource_cost_index": d["C"], "risk_exposure_index": d["R"],
            "static_net_value": sv, "dynamic_net_value": dv,
            "improvement_rate": None if abs(sv) <= 1e-15 else (dv-sv)/abs(sv),
            "static_decision": {"b":s["b"],"q":s["q"],"l":s["l"],"delta":s["delta"]},
            "dynamic_decision": {"b":d["b"],"q":d["q"],"l":d["l"],"delta":d["delta"]},
            "attendance_at_zero": d["attendance_at_zero"], "capacity": e["capacity"],
            "required_security_level": e["required_security_level"], "venue_security_level": e["security_level"],
        })
    out = pd.DataFrame(rows)
    template_cols = pd.read_csv(Path(__file__).resolve().parent.parent / "user_data" / "result_3_template.csv").columns
    out[list(template_cols)].to_csv(OUTPUT_DIR / "result_3_dynamic_strategy.csv", index=False, encoding="utf-8-sig")
    assert len(out) == 24 and out.match_id.is_unique
    assert all(r["venue_id"] == schedule.set_index("match_id").loc[r["match_id"],"venue_id"] for r in rows)
    assert all(r["updated_expected_attendance"] + 1e-8 >= ATTENDANCE_FLOOR_RATIO*r["attendance_at_zero"] for r in rows)
    assert all(r["required_security_level"] <= r["recommended_security_level"] <= r["venue_security_level"] for r in rows)

    # 四个附加固定种子各跑完整 20000 联合样本，并把更新状态传入完整动态重优化。
    stability = []
    base_dynamic_snapshot = {a["match_id"]:(a["b"],a["q"],a["l"],a["delta"]) for a in dynamic_selected}
    decision_stability = [{"seed":MASTER_SEED,"Z3_static":float(z_static),"Z3_dynamic":float(z_dynamic),
                           "improvement":float(z_dynamic-z_static),"decision_changes_vs_base":0,
                           "solver_status":dynamic_solver["status"]}]
    for seed in range(MASTER_SEED+1, MASTER_SEED+5):
        print(f"[P3] stability seed={seed}", flush=True)
        ss = simulate_qualification(round3, sheets["group_membership"], snapshot, lam_a, lam_b, seed, keep_scores=True)
        stability.append(ss["probability"].astype(float))
        seed_prob = {t:float(ss["probability"][i]) for i,t in enumerate(ss["teams"])}
        seed_cond = conditional_probabilities(ss, round3)
        seed_env = derive_match_environment(round3, schedule, sheets["venues"], sheets["ticket_broadcast"],
                                            sheets["dynamic_resource_limits"], snapshot, feedback,
                                            seed_prob, seed_cond)
        seed_actions = make_action_table(seed_env, sheets["dynamic_resource_costs"], sheets["dynamic_resource_limits"], True)
        seed_bounds = action_bounds(seed_actions)
        seed_selected, seed_solver = optimize_actions(seed_actions, sheets["dynamic_resource_limits"], seed_bounds, f"dynamic_{seed}")
        seed_lookup = {(a["match_id"],a["b"],a["q"],a["l"]):a for a in seed_actions}
        seed_static = [seed_lookup[(mid,*decision[:3])] for mid,decision in static_action_snapshot.items()]
        seed_z_static = sum(action_value(a,seed_bounds) for a in seed_static)
        seed_z_dynamic = sum(action_value(a,seed_bounds) for a in seed_selected)
        assert seed_z_dynamic + 1e-4 >= seed_z_static
        seed_snapshot = {a["match_id"]:(a["b"],a["q"],a["l"],a["delta"]) for a in seed_selected}
        changes = sum(seed_snapshot[mid] != base_dynamic_snapshot[mid] for mid in base_dynamic_snapshot)
        decision_stability.append({"seed":seed,"Z3_static":float(seed_z_static),"Z3_dynamic":float(seed_z_dynamic),
                                   "improvement":float(seed_z_dynamic-seed_z_static),
                                   "decision_changes_vs_base":int(changes),"solver_status":seed_solver["status"]})
    prob_stack = np.vstack([sim["probability"].astype(float), *stability])
    result = {
        "metadata": {**runtime_metadata(MASTER_SEED), "elapsed_seconds": time.time()-start},
        "method": "PCG64 joint Poisson Monte Carlo + action_table + daily multiple-choice CP-SAT",
        "mc_shape": [MC_DRAWS, 24], "common_snapshot": True,
        "qualifier_mass_by_draw": sim["mass"].astype(float).tolist(),
        "team_advancement": [{"team":t,"probability":team_prob[t],"five_seed_std":float(prob_stack[:,i].std())} for i,t in enumerate(sim["teams"])],
        "five_seed_decision_stability":decision_stability,
        "lambdas": [{"match_id":round3.loc[i,"match_id"],"lambda_a":float(lam_a[i]),"lambda_b":float(lam_b[i])} for i in range(24)],
        "conditional_probabilities": cond.to_dict(orient="records"),
        "action_table": dynamic_actions, "shared_eval_bounds": shared_eval_bounds,
        "static_action_snapshot": static_action_snapshot,
        "static_solver": static_solver, "dynamic_solver": dynamic_solver,
        "Z3_static_reeval": float(z_static), "Z3_dynamic": float(z_dynamic), "improvement": float(z_dynamic-z_static),
        "rows": rows, "daily_static_audit": daily_static, "daily_dynamic_audit": daily_dynamic,
        "source_unit_value": {"broadcast_cost": "dynamic_resource_costs", "security_cost": "dynamic_resource_costs", "transport_cost": "dynamic_resource_costs"},
        "seed": MASTER_SEED, "outputs": ["output/result_3_dynamic_strategy.csv"],
    }
    write_json(FIGURES_DIR / "problem_3_results.json", result)
    print(f"[P3] validate_capability PASS Z3_dynamic={z_dynamic:.6f} static={z_static:.6f}", flush=True)
    return result


if __name__ == "__main__":
    solve()
