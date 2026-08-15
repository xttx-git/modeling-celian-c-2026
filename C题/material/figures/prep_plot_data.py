"""Prepare figure-only datasets from the persisted model outputs and source workbook.

All derived values are deterministic.  The script never changes the modelling
outputs; it only gathers the exact fields needed by the publication figures.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from problem1 import _fit_fixed, _predict, build_features  # noqa: E402
from problem3 import simulate_qualification, team_snapshot  # noqa: E402


def read_json(name: str):
    return json.loads((ROOT / "figures" / name).read_text(encoding="utf-8"))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def load_workbook() -> dict[str, pd.DataFrame]:
    candidates = sorted((ROOT / "user_data").glob("*.xlsx"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one source workbook, found {len(candidates)}")
    return pd.read_excel(candidates[0], sheet_name=None)


def q1_data(sheets, p1):
    hist, features, _, _ = build_features(sheets["historical_matches"], sheets["teams"])
    train_mask = hist["dataset_split"].eq("train")
    train = hist.loc[train_mask].reset_index(drop=True)
    x_train = features.loc[train_mask].reset_index(drop=True)
    y = train["tv_viewers"].to_numpy(float) / 1_000_000.0

    pair = {
        "观看人数": y,
        "平均Elo": x_train["elo_mean"].to_numpy(float),
        "平均排名": x_train["strength_rank_mean"].to_numpy(float),
        "平均球迷基础": x_train["fan_base_index_mean"].to_numpy(float),
        "赔率熵": x_train["odds_entropy"].to_numpy(float),
    }

    # The final selected model is re-fitted on the first 80% of the fixed train
    # set, and permutation effects are evaluated only on the later 20%.
    cut = int(np.floor(len(x_train) * 0.8))
    model_name = p1["selected_model"]
    if model_name == "MeanBaseline":
        importance_rows = []
    else:
        fitted = _fit_fixed(model_name, x_train.iloc[:cut], y[:cut], p1["selected_params"])
        x_valid = x_train.iloc[cut:].reset_index(drop=True)
        y_valid = y[cut:]
        imp = permutation_importance(
            fitted,
            x_valid,
            y_valid,
            scoring="neg_mean_squared_error",
            n_repeats=12,
            random_state=int(p1["seed"]),
        )
        order = np.argsort(imp.importances_mean)[::-1][:12]
        original_prediction = _predict(model_name, fitted, x_valid)
        rng = np.random.default_rng(int(p1["seed"]))
        importance_rows = []
        for index in order:
            feature = str(x_valid.columns[index])
            shuffled = x_valid.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            shuffled_prediction = _predict(model_name, fitted, shuffled)
            effect = original_prediction - shuffled_prediction
            series = x_valid[feature]
            if pd.api.types.is_numeric_dtype(series):
                feature_value = series.rank(pct=True).fillna(0.5).to_numpy(float)
            else:
                codes = pd.Categorical(series.astype(str)).codes.astype(float)
                feature_value = codes / max(float(codes.max()), 1.0)
            importance_rows.append({
                "feature": feature,
                "delta_mse": float(imp.importances_mean[index]),
                "delta_mse_std": float(imp.importances_std[index]),
                "effect": effect.astype(float).tolist(),
                "feature_value": feature_value.astype(float).tolist(),
            })

    return {
        "target_million": y.tolist(),
        "stage": x_train["stage"].astype(str).tolist(),
        "timezone_pair": x_train["timezone_pair"].astype(str).tolist(),
        "pairplot": {k: np.asarray(v, float).tolist() for k, v in pair.items()},
        "fold_metrics": p1["fold_metrics"],
        "model_summary": p1["model_summary"],
        "oof": p1["oof"],
        "importance": importance_rows,
        "selected_model": model_name,
    }


def q2_constraint_rows(sheets, p2):
    schedule = pd.DataFrame(p2["schedule"])
    venues = sheets["venues"].set_index("venue_id")
    slots = sheets["time_slots"].set_index("slot_id")
    limits = sheets["dynamic_resource_limits"].copy()
    limits["reference_date"] = limits["reference_date"].astype(str)
    limit_map = limits.set_index("reference_date")

    loads = schedule.groupby("venue_id").size()
    lower_slacks = [loads[v] - int(venues.loc[v, "min_total_matches"]) for v in loads.index]
    upper_slacks = [int(venues.loc[v, "max_total_matches"]) - loads[v] for v in loads.index]
    local_day_use = schedule.groupby(["venue_id", schedule["local_datetime"].str[:10]]).size()
    local_day_ratios = [
        float(use) / max(float(venues.loc[venue_id, "max_matches_per_day"]), 1.0)
        for (venue_id, _), use in local_day_use.items()
    ]
    local_day_max_ratio = max(local_day_ratios)
    security_slack = (schedule["venue_security_level"] - schedule["required_security_level"]).min()
    slot_use = schedule.groupby("slot_id").size()
    broadcast_slacks = [int(slots.loc[s, "broadcast_capacity"]) - slot_use[s] for s in slot_use.index]
    high = schedule[(schedule["round_in_group"] == 3) & (schedule["required_security_level"] >= 3)]
    ratios = []
    for day, rows in high.groupby("reference_date"):
        cap = int(limit_map.loc[str(day), "high_security_capacity"])
        ratios.append((cap - len(rows)) / max(cap, 1))
    prime_margin = (2 - int(p2["metrics"]["prime_count_range"])) / 2

    return [
        {"name": "最小轮间休息", "actual": float(p2["constraint_audit"]["min_rest_hours"]), "limit": 60.0,
         "unit": "小时", "margin_pct": 100 * (float(p2["constraint_audit"]["min_rest_hours"]) - 60.0) / 60.0},
        {"name": "场馆承办下界", "actual": float(min(lower_slacks)), "limit": 0.0,
         "unit": "剩余场", "margin_pct": 100 * min(lower_slacks) / max(float(venues["min_total_matches"].max()), 1.0)},
        {"name": "场馆承办上界", "actual": float(min(upper_slacks)), "limit": 0.0,
         "unit": "剩余场", "margin_pct": 100 * min(upper_slacks) / max(float(venues["max_total_matches"].max()), 1.0)},
        {"name": "场馆单日本地容量", "actual": float(local_day_max_ratio), "limit": 1.0,
         "unit": "最大利用率", "margin_pct": 100 * (1.0 - local_day_max_ratio)},
        {"name": "场馆安保等级", "actual": float(security_slack), "limit": 0.0,
         "unit": "等级余量", "margin_pct": 100 * security_slack / max(float(venues["security_level"].max()), 1.0)},
        {"name": "时段转播容量", "actual": float(min(broadcast_slacks)), "limit": 0.0,
         "unit": "剩余场", "margin_pct": 100 * min(broadcast_slacks) / max(float(slots["broadcast_capacity"].max()), 1.0)},
        {"name": "第三轮高安保日容量", "actual": float(min(ratios) if ratios else 1.0), "limit": 0.0,
         "unit": "比例余量", "margin_pct": 100 * (min(ratios) if ratios else 1.0)},
        {"name": "黄金时段公平极差", "actual": float(p2["metrics"]["prime_count_range"]), "limit": 2.0,
         "unit": "场", "margin_pct": 100 * prime_margin},
    ]


def q2_data(sheets, p2, sensitivity):
    schedule = pd.DataFrame(p2["schedule"])
    venues = sheets["venues"].copy()
    venue_rows = venues[["venue_id", "city", "country", "latitude", "longitude", "capacity"]].to_dict("records")
    venue_lookup = venues.set_index("venue_id")

    transitions = Counter()
    all_teams = sorted(set(schedule["team_a_id"]) | set(schedule["team_b_id"]))
    for team in all_teams:
        rows = schedule[(schedule["team_a_id"] == team) | (schedule["team_b_id"] == team)].sort_values("round_in_group")
        ids = rows["venue_id"].astype(str).tolist()
        for left, right in zip(ids[:-1], ids[1:]):
            if left != right:
                transitions[(left, right)] += 1
    edge_rows = []
    for (left, right), count in transitions.items():
        edge_rows.append({
            "source": left,
            "target": right,
            "count": count,
            "source_lon": float(venue_lookup.loc[left, "longitude"]),
            "source_lat": float(venue_lookup.loc[left, "latitude"]),
            "target_lon": float(venue_lookup.loc[right, "longitude"]),
            "target_lat": float(venue_lookup.loc[right, "latitude"]),
        })

    weights = {"T": 0.25, "B": 0.25, "U": 0.15, "H": 0.10,
               "C": -0.08, "D": -0.07, "F": -0.06, "R": -0.04}
    objective = [
        {"indicator": key, "value": float(weights[key] * p2["metrics"][key]),
         "kind": "固定项" if key in {"U", "H"} else "决策项"}
        for key in weights
    ]
    return {
        "schedule": p2["schedule"],
        "venues": venue_rows,
        "venue_load": p2["venue_load"],
        "travel_edges": edge_rows,
        "constraint_margins": q2_constraint_rows(sheets, p2),
        "objective_contributions": objective,
        "objective_total": float(p2["metrics"]["Z2"]),
        "weight_surface": sensitivity["p2_weight_surface"],
        "time_slots": sheets["time_slots"][["slot_id", "date", "reference_kickoff_time", "broadcast_capacity"]].to_dict("records"),
    }


def q3_baseline_advancement(sheets, p3):
    groups = sheets["groups_matches"]
    base = groups.merge(
        sheets["base_predictions"],
        on=["match_id", "group_id", "round_in_group", "team_a", "team_b"],
        validate="one_to_one",
    )
    round3 = base[base["round_in_group"].eq(3)].reset_index(drop=True)
    teams = sheets["group_membership"]["team_name"].astype(str).tolist()
    snapshot = team_snapshot(sheets["live_group_results"], teams)
    baseline = simulate_qualification(
        round3,
        sheets["group_membership"],
        snapshot,
        round3["expected_goals_a"].to_numpy(float),
        round3["expected_goals_b"].to_numpy(float),
        int(p3["seed"]),
        keep_scores=False,
    )
    updated = {row["team"]: row["probability"] for row in p3["team_advancement"]}
    return [
        {"team": team, "baseline": float(baseline["probability"][i]), "updated": float(updated[team])}
        for i, team in enumerate(baseline["teams"])
    ]


def q3_data(sheets, p3, sensitivity):
    transitions = []
    resource_names = {"b": "转播", "q": "安保", "l": "交通"}
    for resource, label in resource_names.items():
        counts = Counter()
        for row in p3["rows"]:
            counts[(int(row["static_decision"][resource]), int(row["dynamic_decision"][resource]))] += 1
        transitions.extend([
            {"resource": label, "source": left, "target": right, "count": count}
            for (left, right), count in sorted(counts.items())
        ])

    bubble_rows = []
    for row in p3["rows"]:
        importance = 0.5 * (
            4 * row["updated_p_team_a_advance"] * (1 - row["updated_p_team_a_advance"])
            + 4 * row["updated_p_team_b_advance"] * (1 - row["updated_p_team_b_advance"])
        )
        bubble_rows.append({
            "match_id": row["match_id"],
            "risk": float(row["risk_exposure_index"]),
            "commercial_musd": float((row["updated_ticket_revenue_usd"] + row["updated_broadcast_value_usd"]) / 1_000_000),
            "importance": float(importance),
            "resource_cost": float(row["resource_cost_index"]),
        })
    return {
        "advancement_shift": q3_baseline_advancement(sheets, p3),
        "resource_transitions": transitions,
        "rows": p3["rows"],
        "risk_value": bubble_rows,
        "sensitivity_surface": sensitivity["p3_state_injury_surface"],
        "daily_static_audit": p3["daily_static_audit"],
        "daily_dynamic_audit": p3["daily_dynamic_audit"],
        "action_table": p3["action_table"],
        "Z3_static": p3["Z3_static_reeval"],
        "Z3_dynamic": p3["Z3_dynamic"],
    }


def q4_data(p4):
    proxy_metrics = [row for row in p4["proxy_comparison"] if row["indicator_name"] != "Z4"]
    structural_metrics = [row for row in p4["structural_comparison"] if row["indicator_name"] != "timezone_crossings"]
    differences = []
    radar = []
    for row in structural_metrics:
        actual = float(row["actual_schedule_value"])
        optimized = float(row["optimized_schedule_value"])
        higher = row["preferred_direction"] == "higher"
        denominator = max(abs(actual), abs(optimized), 1e-12)
        preferred_change = (optimized - actual if higher else actual - optimized) / denominator
        differences.append({**row, "preferred_change": float(preferred_change)})
    for row in proxy_metrics:
        actual = float(row["actual_schedule_value"])
        optimized = float(row["optimized_schedule_value"])
        higher = row["preferred_direction"] == "higher"
        # P4 已在共同边界上把八项代理指标归一到 [0,1]；雷达图直接使用该共同尺度，
        # 成本/旅行/公平/风险只做方向反转，不再逐指标二次缩放。
        if higher:
            actual_score, optimized_score = actual, optimized
        else:
            actual_score, optimized_score = 1-actual, 1-optimized
        radar.append({"indicator": row["indicator_name"], "actual": float(actual_score), "optimized": float(optimized_score)})
    return {
        "differences": differences,
        "radar": radar,
        "comparison": p4["comparison"],
        "weight_robustness": p4["weight_robustness"],
        "proxy_note": p4["proxy_note"],
    }


def main():
    print("[prep] loading source workbook and persisted results", flush=True)
    sheets = load_workbook()
    p1 = read_json("problem_1_results.json")
    p2 = read_json("problem_2_results.json")
    p3 = read_json("problem_3_results.json")
    p4 = read_json("problem_4_results.json")
    sensitivity = read_json("sensitivity_results.json")
    output = {
        "metadata": {
            "source_results": [
                "problem_1_results.json", "problem_2_results.json", "problem_3_results.json",
                "problem_4_results.json", "sensitivity_results.json",
            ],
            "seed": p1["seed"],
            "baseline_advancement_definition": "same observed two-round snapshot, pre-update round-three scoring rates",
        },
        "q1": q1_data(sheets, p1),
        "q2": q2_data(sheets, p2, sensitivity),
        "q3": q3_data(sheets, p3, sensitivity),
        "q4": q4_data(p4),
    }
    target = ROOT / "figures" / "_plot_data.json"
    target.write_text(json.dumps(jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prep] wrote {target} ({target.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
