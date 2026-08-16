"""Reproducible models and independent metric recomputation for Problem C."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ATTACH_DIR = ROOT / "2026年度“策联杯”数学建模精英联赛-C题-附件" / "2026年度“策联杯”数学建模精英联赛-C题-附件"
INPUT_XLSX = ATTACH_DIR / "input_data" / "C题_数据附件.xlsx"
TEMPLATE_DIR = ATTACH_DIR / "output_result"
MATERIAL_DIR = ROOT / "material" / "user_data"
P1_MODEL_SEED = 20260816
P3_PRIMARY_SEED = 20260816
P3_STABILITY_SEEDS = tuple(P3_PRIMARY_SEED + offset for offset in range(5))
P3_CONVERGENCE_SEED = 20260816
P4_WEIGHT_SENSITIVITY_SEED = 20260816
RNG_SEED = P3_PRIMARY_SEED
RANDOM_SEEDS = {
    "problem1_random_models": P1_MODEL_SEED,
    "problem3_primary_simulation": P3_PRIMARY_SEED,
    "problem3_five_seed_stability": list(P3_STABILITY_SEEDS),
    "problem3_convergence": P3_CONVERGENCE_SEED,
    "problem4_weight_perturbation": P4_WEIGHT_SENSITIVITY_SEED,
}
P1_NUMERIC_FEATURES = [
    "neutral", "rank_mean", "rank_min", "rank_max", "rank_absdiff",
    "elo_mean", "elo_min", "elo_max", "elo_absdiff",
    "p_draw", "win_prob_min", "win_prob_max", "win_prob_absdiff",
    "prob_entropy",
    "market_value_musd_mean", "market_value_musd_min", "market_value_musd_max", "market_value_musd_absdiff",
    "avg_age_mean", "avg_age_min", "avg_age_max", "avg_age_absdiff",
    "star_index_mean", "star_index_min", "star_index_max", "star_index_absdiff",
    "fan_base_index_mean", "fan_base_index_min", "fan_base_index_max", "fan_base_index_absdiff",
    "style_attack_mean", "style_attack_min", "style_attack_max", "style_attack_absdiff",
    "style_defense_mean", "style_defense_min", "style_defense_max", "style_defense_absdiff",
    "host_flag_mean", "host_flag_min", "host_flag_max", "host_flag_absdiff",
    "strength_score_mean", "strength_score_min", "strength_score_max", "strength_score_absdiff",
    "month", "weekday_num",
]
P1_CATEGORICAL_FEATURES = ["competition", "stage_group"]
P2_WEIGHTS = {"T": 0.25, "B": 0.25, "U": 0.15, "H": 0.10, "C": -0.08, "D": -0.07, "F": -0.06, "R": -0.04}
P3_FIELDS = ["ticket_value", "broadcast_value", "attractiveness", "resource_cost", "risk"]


def read_data() -> dict[str, pd.DataFrame]:
    book = pd.ExcelFile(INPUT_XLSX)
    return {sheet: pd.read_excel(INPUT_XLSX, sheet_name=sheet) for sheet in book.sheet_names}


def clip(value, low=0.0, high=1.0):
    return np.minimum(np.maximum(value, low), high)


def minmax_value(value: float, bounds: tuple[float, float] | list[float]) -> float:
    low, high = map(float, bounds)
    return 0.0 if abs(high - low) < 1e-12 else float((value - low) / (high - low))


def utc_timestamp(value) -> pd.Timestamp:
    stamp = pd.to_datetime(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def stage_key(round_in_group: int) -> str:
    return f"Group_Match_R{int(round_in_group)}"


def distribution(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


# ---------------------------------------------------------------------------
# Problem 1


def _stage_group(value: object) -> str:
    text = str(value).lower()
    if "group" in text:
        return "Group"
    if "friendly" in text:
        return "Friendly"
    if any(word in text for word in ["final", "quarter", "semi", "round of", "knockout"]):
        return "Knockout"
    return "Other"


def normalize_competition_name(value: object) -> str:
    """Return a deterministic competition category shared by every P1 split."""
    if pd.isna(value):
        return "Other"
    text = re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(value).strip()))
    key = text.casefold()
    world_cup_key = re.sub(r"\b(?:19|20)\d{2}\b", "", key)
    world_cup_key = re.sub(r"\s+", " ", world_cup_key).strip()
    if re.fullmatch(r"(?:fifa\s+)?world cup", world_cup_key):
        return "World Cup"
    canonical = {
        "world cup qualifier": "World Cup Qualifier",
        "fifa world cup qualifier": "World Cup Qualifier",
        "friendly": "Friendly",
        "continental cup": "Continental Cup",
        "nations league": "Nations League",
        "other": "Other",
    }
    return canonical.get(key, text.title())


def raw_competition_series(matches: pd.DataFrame) -> pd.Series:
    default = pd.Series("FIFA World Cup", index=matches.index, dtype=object)
    return matches["competition"].copy() if "competition" in matches else default


def competition_normalization_diagnostics(
    train: pd.DataFrame, test: pd.DataFrame, future: pd.DataFrame,
) -> dict:
    frames = {"train": train, "test": test, "future72": future}
    normalized: dict[str, pd.Series] = {}
    result: dict[str, dict] = {}
    for name, frame in frames.items():
        raw = raw_competition_series(frame).fillna("Other").astype(str)
        canonical = raw.map(normalize_competition_name)
        normalized[name] = canonical
        result[name] = {
            "before_categories": sorted(raw.unique().tolist()),
            "after_categories": sorted(canonical.unique().tolist()),
            "before_counts": {str(k): int(v) for k, v in raw.value_counts().sort_index().items()},
            "after_counts": {str(k): int(v) for k, v in canonical.value_counts().sort_index().items()},
        }
    train_categories = set(normalized["train"])
    for name in ("test", "future72"):
        unseen_mask = ~normalized[name].isin(train_categories)
        result[name]["unseen_categories"] = sorted(normalized[name][unseen_mask].unique().tolist())
        result[name]["unseen_unique_count"] = len(result[name]["unseen_categories"])
        result[name]["unseen_row_count"] = int(unseen_mask.sum())
    result["normalizer"] = "normalize_competition_name_v1"
    result["world_cup_canonical_category"] = "World Cup"
    return result


def build_prediction_features(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One symmetric pre-match feature builder used for train, test and future."""
    frame = matches.copy()
    team_index = teams.set_index("team_name")
    for side in ("a", "b"):
        rank_col = f"strength_rank_{side}"
        elo_col = f"elo_{side}"
        names = frame[f"team_{side}"]
        if rank_col not in frame:
            frame[rank_col] = names.map(team_index["strength_rank"])
        if elo_col not in frame:
            frame[elo_col] = names.map(team_index["elo_rating"])

    if {"p_a_win", "p_draw", "p_b_win"}.issubset(frame.columns):
        probs = frame[["p_a_win", "p_draw", "p_b_win"]].astype(float).copy()
        probs = probs.div(probs.sum(axis=1), axis=0)
    else:
        inverse = 1.0 / frame[["odds_a", "odds_draw", "odds_b"]].astype(float).clip(lower=1.000001)
        probs = inverse.div(inverse.sum(axis=1), axis=0)
        probs.columns = ["p_a_win", "p_draw", "p_b_win"]

    out = pd.DataFrame(index=frame.index)
    out["neutral"] = frame.get("neutral", pd.Series(True, index=frame.index)).astype(int)
    for source, target in [("strength_rank", "rank"), ("elo", "elo")]:
        a = frame[f"{source}_a"].astype(float)
        b = frame[f"{source}_b"].astype(float)
        out[f"{target}_mean"] = (a + b) / 2
        out[f"{target}_min"] = np.minimum(a, b)
        out[f"{target}_max"] = np.maximum(a, b)
        out[f"{target}_absdiff"] = np.abs(a - b)
    for attribute in (
        "market_value_musd", "avg_age", "star_index", "fan_base_index",
        "style_attack", "style_defense", "host_flag", "strength_score",
    ):
        a = frame["team_a"].map(team_index[attribute]).astype(float)
        b = frame["team_b"].map(team_index[attribute]).astype(float)
        out[f"{attribute}_mean"] = (a + b) / 2
        out[f"{attribute}_min"] = np.minimum(a, b)
        out[f"{attribute}_max"] = np.maximum(a, b)
        out[f"{attribute}_absdiff"] = np.abs(a - b)
    pa, pd_, pb = probs["p_a_win"], probs["p_draw"], probs["p_b_win"]
    out["p_draw"] = pd_
    out["win_prob_min"] = np.minimum(pa, pb)
    out["win_prob_max"] = np.maximum(pa, pb)
    out["win_prob_absdiff"] = np.abs(pa - pb)
    out["prob_entropy"] = -(probs * np.log(probs.clip(lower=1e-12))).sum(axis=1) / math.log(3)
    dates = pd.to_datetime(frame.get("date", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
    out["month"] = dates.dt.month.fillna(6).astype(int)
    out["weekday_num"] = dates.dt.dayofweek.fillna(0).astype(int)
    out["competition"] = raw_competition_series(frame).fillna("Other").map(normalize_competition_name)
    if "stage" in frame:
        out["stage_group"] = frame["stage"].map(_stage_group)
    else:
        out["stage_group"] = "Group"
    return out[P1_NUMERIC_FEATURES + P1_CATEGORICAL_FEATURES]


def swap_match_sides(matches: pd.DataFrame) -> pd.DataFrame:
    swapped = matches.copy()
    pairs = [
        ("team_a", "team_b"), ("team_a_id", "team_b_id"),
        ("strength_rank_a", "strength_rank_b"), ("elo_a", "elo_b"),
        ("odds_a", "odds_b"), ("p_a_win", "p_b_win"),
        ("expected_goals_a", "expected_goals_b"),
    ]
    for left, right in pairs:
        if left in swapped and right in swapped:
            swapped[left], swapped[right] = matches[right].copy(), matches[left].copy()
    return swapped


def make_p1_model(name: str) -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), P1_NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), P1_CATEGORICAL_FEATURES),
    ])
    if name == "mean":
        regressor = DummyRegressor(strategy="mean")
    elif name == "elastic_net":
        regressor = ElasticNet(alpha=0.02, l1_ratio=0.10, max_iter=20000, random_state=P1_MODEL_SEED)
    elif name == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=3, random_state=P1_MODEL_SEED, n_jobs=-1
        )
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def expanding_validation(features: pd.DataFrame, target_log: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    order = np.argsort(pd.to_datetime(dates).to_numpy())
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fraction in (0.60, 0.70, 0.80):
        cut = int(len(order) * fraction)
        end = min(len(order), cut + max(30, int(len(order) * 0.10)))
        folds.append((order[:cut], order[cut:end]))
    rows = []
    for name in ("mean", "elastic_net", "extra_trees"):
        fold_metrics = []
        for train_idx, valid_idx in folds:
            model = make_p1_model(name)
            model.fit(features.iloc[train_idx], target_log[train_idx])
            prediction = np.expm1(model.predict(features.iloc[valid_idx])).clip(min=0)
            truth = np.expm1(target_log[valid_idx])
            mse = mean_squared_error(truth, prediction)
            fold_metrics.append((mse, math.sqrt(mse), mean_absolute_error(truth, prediction)))
        values = np.asarray(fold_metrics)
        rows.append({
            "model": name,
            "mse": float(values[:, 0].mean()),
            "rmse": float(values[:, 1].mean()),
            "mae": float(values[:, 2].mean()),
        })
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def feature_distribution_diagnostics(train: pd.DataFrame, test: pd.DataFrame, future: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}
    for column in P1_NUMERIC_FEATURES:
        train_values = train[column].astype(float)
        mean = float(train_values.mean())
        std = float(train_values.std(ddof=1))
        limits = (mean - 4 * std, mean + 4 * std) if std > 1e-12 else (mean, mean)
        result[column] = {
            "train": distribution(train_values),
            "test": distribution(test[column]),
            "future72": distribution(future[column]),
            "train_mean_minus_4sd": limits[0],
            "train_mean_plus_4sd": limits[1],
            "test_outside_4sd_count": int(((test[column] < limits[0]) | (test[column] > limits[1])).sum()),
            "future72_outside_4sd_count": int(((future[column] < limits[0]) | (future[column] > limits[1])).sum()),
        }
    return result


def solve_problem1(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    historical = data["historical_matches"].copy()
    train = historical[historical["dataset_split"].eq("train") & historical["tv_viewers"].notna()].copy()
    test = historical[historical["dataset_split"].eq("test")].copy()
    teams = data["teams"]
    train_features = build_prediction_features(train, teams)
    test_features = build_prediction_features(test, teams)
    target_log = np.log1p(train["tv_viewers"].to_numpy(float) / 1_000_000)
    metrics = expanding_validation(train_features, target_log, train["date"])
    selected = str(metrics.iloc[0]["model"])
    elastic_mse = float(metrics.loc[metrics["model"].eq("elastic_net"), "mse"].iloc[0])
    if elastic_mse <= float(metrics.iloc[0]["mse"]) * 1.02:
        selected = "elastic_net"
    model = make_p1_model(selected)
    model.fit(train_features, target_log)
    test_prediction = np.expm1(model.predict(test_features)).clip(min=0)

    future_raw = data["groups_matches"].merge(
        data["base_predictions"].drop(columns=["group_id", "round_in_group", "team_a", "team_b"], errors="ignore"),
        on="match_id", how="left",
    )
    future_features = build_prediction_features(future_raw, teams)
    future_prediction_million = np.expm1(model.predict(future_features)).clip(min=0)
    swapped_features = build_prediction_features(swap_match_sides(future_raw), teams)
    swapped_prediction = np.expm1(model.predict(swapped_features)).clip(min=0)

    result_test = pd.DataFrame({
        "match_id_test": test["match_id"].to_numpy(),
        "predicted_test_tv_viewers": np.round(test_prediction, 6),
    })
    result_match = data["groups_matches"][["match_id", "team_a", "team_b"]].copy()
    result_match["predicted_tv_viewers"] = np.rint(future_prediction_million * 1_000_000).astype(np.int64)
    train_fitted = np.expm1(model.predict(train_features)).clip(min=0)
    diagnostics = {
        "problem1_model_metrics": metrics.to_dict(orient="records"),
        "problem1_selected_model": selected,
        "problem1_feature_columns": {"numeric": P1_NUMERIC_FEATURES, "categorical": P1_CATEGORICAL_FEATURES},
        "problem1_feature_builder": "build_prediction_features_v3_symmetric_competition_normalized",
        "problem1_model_seed": P1_MODEL_SEED,
        "problem1_competition_normalization": competition_normalization_diagnostics(train, test, future_raw),
        "problem1_prediction_distributions": {
            "train_fitted_million": distribution(train_fitted),
            "test_prediction_million": distribution(test_prediction),
            "future72_prediction_million": distribution(future_prediction_million),
        },
        "problem1_swap_invariance_max_abs_error_million": float(np.max(np.abs(future_prediction_million - swapped_prediction))),
        "problem1_feature_distributions": feature_distribution_diagnostics(train_features, test_features, future_features),
        "problem1_forbidden_features": [
            "attractiveness_index", "commercial_value_index",
            "expected_attendance_base", "uncertainty_index",
        ],
    }
    return result_test, result_match, diagnostics


# ---------------------------------------------------------------------------
# Problem 2


def exact_top_quartile_ids(frame: pd.DataFrame, score: str, id_column: str) -> set[str]:
    count = int(math.ceil(len(frame) * 0.25))
    ordered = frame.sort_values([score, id_column], ascending=[False, True], kind="mergesort")
    return set(ordered.head(count)[id_column].astype(str))


def distance_lookup(distance: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    return {
        (str(r.origin_id), str(r.destination_id)): (
            float(r.distance_km), float(r.travel_time_hour), abs(float(r.timezone_diff))
        )
        for r in distance.itertuples(index=False)
    }


@dataclass
class P2Context:
    matches: pd.DataFrame
    venue_candidates: pd.DataFrame
    slot_candidates: pd.DataFrame
    prime_slots: set[str]
    large_venues: set[str]
    bounds: dict[str, list[float]]


def build_p2_context(data: dict[str, pd.DataFrame], p1_match: pd.DataFrame) -> P2Context:
    gm = data["groups_matches"].copy()
    matches = gm.merge(
        data["base_predictions"].drop(columns=["group_id", "round_in_group", "team_a", "team_b"], errors="ignore"),
        on="match_id", how="left",
    ).merge(
        data["security_requirements"][["match_id", "required_security_level", "security_demand_score"]],
        on="match_id", how="left",
    ).merge(p1_match[["match_id", "predicted_tv_viewers"]], on="match_id", how="left")
    venues = data["venues"]
    slots = data["time_slots"]
    ticket = data["ticket_broadcast"].set_index("match_stage")
    distance = distance_lookup(data["distance_matrix"])
    venue_rows = []

    previous_match: dict[tuple[str, int], pd.Series] = {}
    for row in matches.itertuples(index=False):
        previous_match[(row.team_a, int(row.round_in_group))] = pd.Series(row._asdict())
        previous_match[(row.team_b, int(row.round_in_group))] = pd.Series(row._asdict())

    def travel_components(match_row, venue_row, team_name: str, team_id: str) -> tuple[float, float, float]:
        round_no = int(match_row.round_in_group)
        if round_no == 1:
            return distance[(str(team_id), str(venue_row.venue_id))]
        prior = previous_match[(team_name, round_no - 1)]
        required = int(prior["required_security_level"])
        eligible = venues.loc[venues["security_level"].ge(required), "venue_id"].astype(str)
        values = [distance[(origin, str(venue_row.venue_id))] for origin in eligible]
        return tuple(np.mean(np.asarray(values, dtype=float), axis=0))

    for match in matches.itertuples(index=False):
        tb = ticket.loc[stage_key(match.round_in_group)]
        for venue in venues.itertuples(index=False):
            if int(venue.security_level) < int(match.required_security_level):
                continue
            ta = travel_components(match, venue, match.team_a, match.team_a_id)
            tb_team = travel_components(match, venue, match.team_b, match.team_b_id)
            components = (np.asarray(ta) + np.asarray(tb_team)) / 2
            attendance = min(float(match.expected_attendance_base), float(venue.capacity))
            venue_rows.append({
                "match_id": match.match_id,
                "venue_id": venue.venue_id,
                "expected_attendance": attendance,
                "ticket_revenue_usd": float(tb["base_ticket_price_usd"]) * attendance,
                "match_cost_musd": float(venue.operation_cost_musd_per_match)
                    + 0.1 * int(match.required_security_level) * float(venue.security_cost_index),
                "distance_km": float(components[0]),
                "travel_time_hour": float(components[1]),
                "timezone_diff": float(components[2]),
                "risk_index": 0.5 * float(venue.climate_risk)
                    + 0.3 * attendance / float(venue.capacity)
                    + 0.2 * int(match.required_security_level) / float(venue.security_level),
            })
    venue_candidates = pd.DataFrame(venue_rows)
    component_bounds = {}
    for column in ("distance_km", "travel_time_hour", "timezone_diff"):
        low, high = float(venue_candidates[column].min()), float(venue_candidates[column].max())
        component_bounds[column] = [low, high]
        venue_candidates[column + "_norm"] = venue_candidates[column].map(lambda value: minmax_value(value, (low, high)))
    venue_candidates["travel_burden"] = (
        0.5 * venue_candidates["distance_km_norm"]
        + 0.3 * venue_candidates["travel_time_hour_norm"]
        + 0.2 * venue_candidates["timezone_diff_norm"]
    )

    slot_rows = []
    for match in matches.itertuples(index=False):
        tb = ticket.loc[stage_key(match.round_in_group)]
        for slot in slots.itertuples(index=False):
            slot_rows.append({
                "match_id": match.match_id,
                "slot_id": slot.slot_id,
                "broadcast_value_usd": float(match.predicted_tv_viewers)
                    * float(tb["broadcast_unit_value_usd"])
                    * float(slot.global_prime_score)
                    * float(tb["sponsor_weight"]),
            })
    slot_candidates = pd.DataFrame(slot_rows)
    bounds = {
        "ticket_revenue_usd": [float(venue_candidates["ticket_revenue_usd"].min()), float(venue_candidates["ticket_revenue_usd"].max())],
        "broadcast_value_usd": [float(slot_candidates["broadcast_value_usd"].min()), float(slot_candidates["broadcast_value_usd"].max())],
        "match_cost_musd": [float(venue_candidates["match_cost_musd"].min()), float(venue_candidates["match_cost_musd"].max())],
        "travel_burden": [0.0, 1.0],
        "risk_index": [float(venue_candidates["risk_index"].min()), float(venue_candidates["risk_index"].max())],
        "travel_components": component_bounds,
    }
    for column in ("ticket_revenue_usd", "match_cost_musd", "risk_index"):
        venue_candidates[column + "_norm"] = venue_candidates[column].map(lambda value: minmax_value(value, bounds[column]))
    slot_candidates["broadcast_value_usd_norm"] = slot_candidates["broadcast_value_usd"].map(
        lambda value: minmax_value(value, bounds["broadcast_value_usd"])
    )
    return P2Context(
        matches=matches,
        venue_candidates=venue_candidates,
        slot_candidates=slot_candidates,
        prime_slots=exact_top_quartile_ids(slots, "global_prime_score", "slot_id"),
        large_venues=exact_top_quartile_ids(venues, "capacity", "venue_id"),
        bounds=bounds,
    )


def _milp_result_diagnostics(result, started: float) -> dict:
    gap = float(getattr(result, "mip_gap", np.nan))
    if result.status == 0:
        status = "optimal_within_mip_tolerance"
        interpretation = "在设定 MIP 容差内求得最优解"
    elif result.status == 1:
        status = "limit_reached"
        interpretation = "达到求解限制，保留当前可行解与实际 gap"
    else:
        status = "failed"
        interpretation = "求解失败"
    return {
        "status_code": int(result.status),
        "solver_status": "optimal" if result.status == 0 else "limit_reached" if result.status == 1 else "failed",
        "status": status,
        "interpretation": interpretation,
        "message": str(result.message),
        "runtime_seconds": float(time.perf_counter() - started),
        "best_bound": float(getattr(result, "mip_dual_bound", np.nan)),
        "mip_gap": gap,
        "node_count": int(getattr(result, "mip_node_count", 0)),
    }


def solve_p2_slots(
    ctx: P2Context, data: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    weights = P2_WEIGHTS if weights is None else weights
    matches, slots = ctx.matches.reset_index(drop=True), data["time_slots"].reset_index(drop=True)
    n_match, n_slot = len(matches), len(slots)
    n_x = n_match * n_slot
    pmax, pmin, n_var = n_x, n_x + 1, n_x + 2
    slot_index = {str(row.slot_id): i for i, row in slots.iterrows()}
    match_index = {str(row.match_id): i for i, row in matches.iterrows()}
    score = ctx.slot_candidates.pivot(index="match_id", columns="slot_id", values="broadcast_value_usd_norm")
    c = np.zeros(n_var)
    for match_id, i in match_index.items():
        for slot_id, j in slot_index.items():
            c[i * n_slot + j] = -weights["B"] * float(score.loc[match_id, slot_id]) / n_match
    c[pmax], c[pmin] = -weights["F"] / 6, weights["F"] / 6
    rows, lower, upper = [], [], []

    def add(entries: dict[int, float], low: float, high: float):
        row = np.zeros(n_var)
        for index, value in entries.items():
            row[index] = value
        rows.append(row); lower.append(low); upper.append(high)

    for i in range(n_match):
        add({i * n_slot + j: 1 for j in range(n_slot)}, 1, 1)
    for j, slot in slots.iterrows():
        add({i * n_slot + j: 1 for i in range(n_match)}, 0, int(slot.broadcast_capacity))
    times = np.array([(utc_timestamp(value) - utc_timestamp(slots["reference_utc_time"].iloc[0])).total_seconds() / 3600 for value in slots["reference_utc_time"]])
    all_teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
    for team in all_teams:
        team_matches = matches[(matches["team_a"].eq(team)) | (matches["team_b"].eq(team))].sort_values("round_in_group")
        ids = [match_index[mid] for mid in team_matches["match_id"]]
        for first, second in zip(ids, ids[1:]):
            entries = {first * n_slot + j: -times[j] for j in range(n_slot)}
            entries.update({second * n_slot + j: times[j] for j in range(n_slot)})
            add(entries, 60, np.inf)
    # Stronger group boundary: every match in round r+1 starts at least
    # 60 hours after every match in round r for the same group.
    for _, group_matches in matches.groupby("group_id"):
        for round_no in (1, 2):
            previous_ids = [
                match_index[mid]
                for mid in group_matches.loc[group_matches["round_in_group"].eq(round_no), "match_id"]
            ]
            next_ids = [
                match_index[mid]
                for mid in group_matches.loc[group_matches["round_in_group"].eq(round_no + 1), "match_id"]
            ]
            for previous in previous_ids:
                for following in next_ids:
                    entries = {previous * n_slot + j: -times[j] for j in range(n_slot)}
                    entries.update({following * n_slot + j: times[j] for j in range(n_slot)})
                    add(entries, 60, np.inf)
    limits = data["dynamic_resource_limits"].set_index("reference_date")
    for day, day_slots in slots.groupby("date"):
        high_matches = matches.index[matches["round_in_group"].eq(3) & matches["required_security_level"].ge(3)]
        entries = {i * n_slot + j: 1 for i in high_matches for j in day_slots.index}
        add(entries, 0, int(limits.loc[day, "high_security_capacity"]))
    for team in all_teams:
        team_ids = matches.index[(matches["team_a"].eq(team)) | (matches["team_b"].eq(team))]
        entries = {
            i * n_slot + j: 1
            for i in team_ids for j, slot in slots.iterrows()
            if str(slot.slot_id) in ctx.prime_slots
        }
        add({**entries, pmax: -1}, -np.inf, 0)
        add({**entries, pmin: -1}, 0, np.inf)
    add({pmax: 1, pmin: -1}, -np.inf, 2)
    lower_bounds = np.zeros(n_var)
    upper_bounds = np.r_[np.ones(n_x), 3, 3]
    started = time.perf_counter()
    result = milp(
        c=c, integrality=np.ones(n_var),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 180, "mip_rel_gap": 0.0005, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Problem 2 slot MILP failed: {result.message}")
    chosen = []
    for i, match in matches.iterrows():
        values = result.x[i * n_slot:(i + 1) * n_slot]
        chosen.append({"match_id": match.match_id, "slot_id": slots.iloc[int(np.argmax(values))].slot_id})
    diagnostics = _milp_result_diagnostics(result, started)
    diagnostics.update({"variable_count": n_var, "constraint_count": len(rows)})
    return pd.DataFrame(chosen), diagnostics


def solve_p2_venues(
    ctx: P2Context, data: dict[str, pd.DataFrame], match_slots: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    weights = P2_WEIGHTS if weights is None else weights
    matches = ctx.matches.merge(match_slots, on="match_id").reset_index(drop=True)
    venues = data["venues"].reset_index(drop=True)
    slots = data["time_slots"].set_index("slot_id")
    pairs = ctx.venue_candidates[["match_id", "venue_id"]].drop_duplicates().reset_index(drop=True)
    pairs = pairs.merge(match_slots, on="match_id", how="left")
    pair_metric = ctx.venue_candidates.set_index(["match_id", "venue_id"])
    n_pair = len(pairs)
    lmax, lmin, n_var = n_pair, n_pair + 1, n_pair + 2
    c = np.zeros(n_var)
    for k, pair in pairs.iterrows():
        metric = pair_metric.loc[(pair.match_id, pair.venue_id)]
        value = (
            weights["T"] * metric.ticket_revenue_usd_norm
            + weights["C"] * metric.match_cost_musd_norm
            + weights["D"] * metric.travel_burden
            + weights["R"] * metric.risk_index_norm
        ) / len(matches)
        c[k] = -float(value)
    c[lmax], c[lmin] = -weights["F"] / 6, weights["F"] / 6
    rows, lower, upper = [], [], []

    def add(indices: list[int], coefficients: list[float] | None, low: float, high: float, extras: dict[int, float] | None = None):
        row = np.zeros(n_var)
        row[indices] = 1 if coefficients is None else coefficients
        for index, value in (extras or {}).items():
            row[index] = value
        rows.append(row); lower.append(low); upper.append(high)

    for match_id, indices in pairs.groupby("match_id").groups.items():
        add(list(indices), None, 1, 1)
    venue_index = venues.set_index("venue_id")
    for venue_id, indices in pairs.groupby("venue_id").groups.items():
        venue = venue_index.loc[venue_id]
        add(list(indices), None, int(venue.min_total_matches), int(venue.max_total_matches))
    local_groups: dict[tuple[str, str], list[int]] = {}
    utc_groups: dict[tuple[str, str], list[int]] = {}
    for k, pair in pairs.iterrows():
        venue = venue_index.loc[pair.venue_id]
        utc = utc_timestamp(slots.loc[pair.slot_id, "reference_utc_time"])
        local_day = utc.tz_convert(ZoneInfo(str(venue.timezone))).date().isoformat()
        local_groups.setdefault((pair.venue_id, local_day), []).append(k)
        utc_groups.setdefault((pair.venue_id, utc.isoformat()), []).append(k)
    for (venue_id, _), indices in local_groups.items():
        add(indices, None, 0, int(venue_index.loc[venue_id, "max_matches_per_day"]))
    for indices in utc_groups.values():
        if len(indices) > 1:
            add(indices, None, 0, 1)
    all_teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
    match_team = matches.set_index("match_id")[["team_a", "team_b"]]
    for team in all_teams:
        indices = [
            k for k, pair in pairs.iterrows()
            if pair.venue_id in ctx.large_venues
            and team in (match_team.loc[pair.match_id, "team_a"], match_team.loc[pair.match_id, "team_b"])
        ]
        add(indices, None, -np.inf, 0, {lmax: -1})
        add(indices, None, 0, np.inf, {lmin: -1})
    started = time.perf_counter()
    result = milp(
        c=c, integrality=np.ones(n_var),
        bounds=Bounds(np.zeros(n_var), np.r_[np.ones(n_pair), 3, 3]),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 180, "mip_rel_gap": 0.0005, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Problem 2 venue MILP failed: {result.message}")
    chosen = pairs.loc[np.where(result.x[:n_pair] > 0.5)[0], ["match_id", "venue_id", "slot_id"]].copy()
    diagnostics = _milp_result_diagnostics(result, started)
    diagnostics.update({"variable_count": n_var, "constraint_count": len(rows)})
    return chosen, diagnostics


def p2_fairness(schedule: pd.DataFrame, prime_slots: set[str], large_venues: set[str]) -> tuple[float, dict]:
    teams = sorted(set(schedule["team_a"]) | set(schedule["team_b"]))
    prime = {team: 0 for team in teams}
    large = {team: 0 for team in teams}
    for row in schedule.itertuples(index=False):
        for team in (row.team_a, row.team_b):
            prime[team] += int(str(row.slot_id) in prime_slots)
            large[team] += int(str(row.venue_id) in large_venues)
    prime_range = max(prime.values()) - min(prime.values())
    large_range = max(large.values()) - min(large.values())
    value = 0.5 * prime_range / 3 + 0.5 * large_range / 3
    return float(value), {
        "prime_counts": prime,
        "large_venue_counts": large,
        "prime_range": int(prime_range),
        "large_venue_range": int(large_range),
    }


def p2_metrics(schedule: pd.DataFrame, ctx: P2Context, data: dict[str, pd.DataFrame]) -> dict:
    base = data["base_predictions"].set_index("match_id")
    venues = data["venues"]
    selected_venues = schedule[["match_id", "venue_id"]].merge(
        ctx.venue_candidates, on=["match_id", "venue_id"], how="left"
    )
    selected_slots = schedule[["match_id", "slot_id"]].merge(
        ctx.slot_candidates, on=["match_id", "slot_id"], how="left"
    )
    fairness, fairness_detail = p2_fairness(schedule, ctx.prime_slots, ctx.large_venues)
    setup = float(venues.loc[venues["venue_id"].isin(schedule["venue_id"].unique()), "setup_cost_musd"].sum())
    raw = {
        "T": float(selected_venues["ticket_revenue_usd"].sum()),
        "B": float(selected_slots["broadcast_value_usd"].sum()),
        "U": float(base.loc[schedule["match_id"], "uncertainty_index"].mean()),
        "H": float((base.loc[schedule["match_id"], "attractiveness_index"] / 100).mean()),
        "C": setup + float(selected_venues["match_cost_musd"].sum()),
        "D": float(selected_venues["travel_burden"].mean()),
        "F": fairness,
        "R": float(selected_venues["risk_index"].mean()),
    }
    n = len(schedule)
    bounds = {
        "T": [n * ctx.bounds["ticket_revenue_usd"][0], n * ctx.bounds["ticket_revenue_usd"][1]],
        "B": [n * ctx.bounds["broadcast_value_usd"][0], n * ctx.bounds["broadcast_value_usd"][1]],
        "U": [0.0, 1.0],
        "H": [0.0, 1.0],
        "C": [setup + n * ctx.bounds["match_cost_musd"][0], setup + n * ctx.bounds["match_cost_musd"][1]],
        "D": [0.0, 1.0],
        "F": [0.0, 1.0],
        "R": ctx.bounds["risk_index"],
    }
    normalized = {key: minmax_value(raw[key], bounds[key]) for key in raw}
    contributions = {key: P2_WEIGHTS[key] * normalized[key] for key in raw}
    return {
        "raw": raw,
        "bounds": bounds,
        "normalized": normalized,
        "weighted_contributions": contributions,
        "Z2": float(sum(contributions.values())),
        "fairness_detail": fairness_detail,
        "setup_cost_musd": setup,
    }


def p2_hard_constraint_audit(schedule: pd.DataFrame, ctx: P2Context, data: dict[str, pd.DataFrame]) -> dict:
    venues = data["venues"].set_index("venue_id")
    slots = data["time_slots"].set_index("slot_id")
    limits = data["dynamic_resource_limits"].set_index("reference_date")
    results: dict[str, object] = {}
    results["unique_matches"] = int(schedule["match_id"].nunique()) == 72
    results["venue_utc_conflicts"] = int(schedule.duplicated(["venue_id", "utc_datetime"]).sum())
    rest_values = []
    order_ok = True
    for team in sorted(set(schedule["team_a"]) | set(schedule["team_b"])):
        subset = schedule[(schedule["team_a"].eq(team)) | (schedule["team_b"].eq(team))].sort_values("round_in_group")
        order_ok &= subset["round_in_group"].tolist() == [1, 2, 3]
        differences = pd.to_datetime(subset["utc_datetime"], utc=True).diff().dropna().dt.total_seconds() / 3600
        rest_values.extend(differences.tolist())
    results["team_round_order"] = bool(order_ok)
    results["minimum_rest_hours"] = float(min(rest_values))
    group_round_gaps = {}
    for group_id, group_matches in schedule.groupby("group_id"):
        group_times = group_matches.assign(_utc=pd.to_datetime(group_matches["utc_datetime"], utc=True))
        for round_no in (1, 2):
            previous_max = group_times.loc[group_times["round_in_group"].eq(round_no), "_utc"].max()
            next_min = group_times.loc[group_times["round_in_group"].eq(round_no + 1), "_utc"].min()
            key = f"{group_id}:R{round_no}-R{round_no + 1}"
            group_round_gaps[key] = float((next_min - previous_max).total_seconds() / 3600)
    results["group_round_gaps_hours"] = group_round_gaps
    results["group_round_minimum_gap_hours"] = float(min(group_round_gaps.values()))
    results["group_round_violation_count"] = int(sum(gap < 60 - 1e-9 for gap in group_round_gaps.values()))
    results["group_round_boundary_passed"] = results["group_round_violation_count"] == 0
    local_counts = schedule.assign(local_day=pd.to_datetime(schedule["local_datetime"]).dt.date).groupby(["venue_id", "local_day"]).size()
    results["venue_local_day_max_violation"] = int(sum(
        count > int(venues.loc[venue_id, "max_matches_per_day"])
        for (venue_id, _), count in local_counts.items()
    ))
    total_counts = schedule.groupby("venue_id").size()
    results["venue_total_violation"] = int(sum(
        not (int(row.min_total_matches) <= int(total_counts.get(venue_id, 0)) <= int(row.max_total_matches))
        for venue_id, row in venues.iterrows()
    ))
    results["broadcast_capacity_violation"] = int(sum(
        count > int(slots.loc[slot_id, "broadcast_capacity"])
        for slot_id, count in schedule.groupby("slot_id").size().items()
    ))
    results["security_violation"] = int(sum(
        int(row.required_security_level) > int(venues.loc[row.venue_id, "security_level"])
        for row in schedule.itertuples(index=False)
    ))
    third_high = schedule[schedule["round_in_group"].eq(3) & schedule["required_security_level"].ge(3)].groupby("reference_date").size()
    results["third_round_high_security_violation"] = int(sum(
        count > int(limits.loc[day, "high_security_capacity"]) for day, count in third_high.items()
    ))
    results["prime_range"] = p2_fairness(schedule, ctx.prime_slots, ctx.large_venues)[1]["prime_range"]
    results["all_passed"] = bool(
        results["unique_matches"] and results["team_round_order"]
        and results["minimum_rest_hours"] >= 60 - 1e-9
        and results["group_round_boundary_passed"]
        and all(results[key] == 0 for key in [
            "venue_utc_conflicts", "venue_local_day_max_violation", "venue_total_violation",
            "broadcast_capacity_violation", "security_violation", "third_round_high_security_violation",
        ])
        and results["prime_range"] <= 2
    )
    return results


def solve_problem2(data: dict[str, pd.DataFrame], p1_match: pd.DataFrame) -> tuple[pd.DataFrame, dict, P2Context]:
    context = build_p2_context(data, p1_match)
    match_slots, slot_solver = solve_p2_slots(context, data)
    assignments, venue_solver = solve_p2_venues(context, data, match_slots)
    venues = data["venues"].set_index("venue_id")
    slots = data["time_slots"].set_index("slot_id")
    schedule = context.matches.merge(assignments, on="match_id", how="left")
    venue_metric = context.venue_candidates.set_index(["match_id", "venue_id"])
    slot_metric = context.slot_candidates.set_index(["match_id", "slot_id"])
    rows = []
    for row in schedule.itertuples(index=False):
        venue = venues.loc[row.venue_id]
        slot = slots.loc[row.slot_id]
        vm = venue_metric.loc[(row.match_id, row.venue_id)]
        sm = slot_metric.loc[(row.match_id, row.slot_id)]
        utc = utc_timestamp(slot.reference_utc_time)
        local = utc.tz_convert(ZoneInfo(str(venue.timezone)))
        rows.append({
            "match_id": row.match_id, "group_id": row.group_id,
            "round_in_group": int(row.round_in_group), "team_a": row.team_a, "team_b": row.team_b,
            "venue_id": row.venue_id, "slot_id": row.slot_id,
            "city": venue.city, "country": venue.country,
            "reference_date": pd.to_datetime(slot.date).date().isoformat(),
            "reference_kickoff_time": str(slot.reference_kickoff_time),
            "local_datetime": local.strftime("%Y-%m-%d %H:%M"),
            "utc_datetime": utc.strftime("%Y-%m-%d %H:%M"),
            "required_security_level": int(row.required_security_level),
            "expected_attendance": round(float(vm.expected_attendance), 2),
            "expected_tv_viewers": int(row.predicted_tv_viewers),
            "ticket_revenue_usd": round(float(vm.ticket_revenue_usd), 2),
            "broadcast_value_usd": round(float(sm.broadcast_value_usd), 2),
            "travel_cost_index": round(float(vm.travel_burden), 8),
            "fairness_penalty": np.nan,
            "risk_index": round(float(vm.risk_index), 8),
            "total_objective_value": "",
        })
    output = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    fairness = p2_fairness(output, context.prime_slots, context.large_venues)[0]
    output["fairness_penalty"] = round(fairness, 8)
    metrics = p2_metrics(output, context, data)
    output.loc[0, "total_objective_value"] = f"{metrics['Z2']:.10f}"
    audit = p2_hard_constraint_audit(output, context, data)
    diagnostics = {
        "problem2_total_objective": metrics["Z2"],
        "problem2_indicator_details": metrics,
        "problem2_candidate_normalization": context.bounds,
        "problem2_prime_slots": sorted(context.prime_slots),
        "problem2_large_capacity_venues": sorted(context.large_venues),
        "problem2_solver": {
            "method": "decomposed_slot_MILP_then_conditional_venue_MILP",
            "global_optimality_claim": False,
            "slot_subproblem": slot_solver,
            "venue_subproblem_given_slots": venue_solver,
        },
        "problem2_hard_constraint_audit": audit,
    }
    return output, diagnostics, context


# ---------------------------------------------------------------------------
# Problem 3


def standings_from_live(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    members = data["group_membership"]
    stats = {
        row.team_name: {
            "team_id": row.team_id, "group_id": row.group_id, "G": 0, "P": 0,
            "GF": 0, "GA": 0, "xGF": 0.0, "xGA": 0.0, "RC": 0, "injuries": [],
        }
        for row in members.itertuples(index=False)
    }
    for row in data["live_group_results"].itertuples(index=False):
        for side, opponent in (("a", "b"), ("b", "a")):
            team = getattr(row, f"team_{side}")
            goals_for = int(getattr(row, f"goals_{side}"))
            goals_against = int(getattr(row, f"goals_{opponent}"))
            item = stats[team]
            item["G"] += 1
            item["P"] += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
            item["GF"] += goals_for; item["GA"] += goals_against
            item["xGF"] += float(getattr(row, f"xg_{side}")); item["xGA"] += float(getattr(row, f"xg_{opponent}"))
            item["RC"] += int(getattr(row, f"red_cards_{side}"))
            item["injuries"].append(float(row.injury_impact_level))
    rows = []
    for team, item in stats.items():
        games = item["G"]
        state = (
            0.45 * item["P"] / (3 * games)
            + 0.35 * (0.5 + 0.5 * math.tanh(((item["xGF"] - item["xGA"]) / games) / 1.25))
            + 0.20 * math.exp(-0.55 * item["RC"] / games)
        )
        rows.append({
            **{key: value for key, value in item.items() if key != "injuries"},
            "team_name": team, "state": float(clip(state)),
            "injury": float(clip(np.mean(item["injuries"]), 0, 3)),
        })
    return pd.DataFrame(rows)


def _rank_key(team_index: int, points: np.ndarray, goals_for: np.ndarray, goals_against: np.ndarray) -> tuple[int, int, int]:
    return int(points[team_index]), int(goals_for[team_index] - goals_against[team_index]), int(goals_for[team_index])


def fractional_advancement_weights(
    points: np.ndarray, goals_for: np.ndarray, goals_against: np.ndarray,
    groups: dict[str, list[int]],
) -> np.ndarray:
    """Exact equal weights at group top-two/third and best-third cutoffs."""
    advancement = np.zeros(len(points), dtype=float)
    third_candidates: list[tuple[int, tuple[int, int, int], float]] = []
    for team_ids in groups.values():
        blocks: dict[tuple[int, int, int], list[int]] = {}
        for team_id in team_ids:
            blocks.setdefault(_rank_key(team_id, points, goals_for, goals_against), []).append(team_id)
        position = 0
        for key in sorted(blocks, reverse=True):
            block = blocks[key]
            block_size = len(block)
            top_slots = max(0, min(position + block_size, 2) - position)
            if top_slots:
                advancement[block] += top_slots / block_size
            if position <= 2 < position + block_size:
                third_candidates.extend((team_id, key, 1.0 / block_size) for team_id in block)
            position += block_size

    by_key: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
    for team_id, key, identity_weight in third_candidates:
        by_key.setdefault(key, []).append((team_id, identity_weight))
    remaining = 8.0
    for key in sorted(by_key, reverse=True):
        candidates = by_key[key]
        mass = sum(weight for _, weight in candidates)
        accepted = min(remaining, mass)
        if accepted > 0:
            ratio = accepted / mass
            for team_id, identity_weight in candidates:
                advancement[team_id] += identity_weight * ratio
        remaining -= accepted
        if remaining <= 1e-12:
            break
    if abs(float(advancement.sum()) - 32.0) > 1e-9:
        raise AssertionError(f"Advancement weights sum to {advancement.sum()}, expected 32")
    return advancement


def simulate_advancement(
    data: dict[str, pd.DataFrame], standings: pd.DataFrame,
    n_sim: int = 20000, seed: int = RNG_SEED,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    groups_frame = data["group_membership"].sort_values("team_id")
    team_names = groups_frame["team_name"].tolist()
    index = {team: i for i, team in enumerate(team_names)}
    groups = {group: [index[name] for name in names] for group, names in groups_frame.groupby("group_id")["team_name"]}
    state = standings.set_index("team_name").reindex(team_names)
    third = data["groups_matches"].query("round_in_group == 3").reset_index(drop=True)
    base = data["base_predictions"].set_index("match_id")
    lambda_a, lambda_b = [], []
    for row in third.itertuples(index=False):
        sa, sb = float(state.loc[row.team_a, "state"]), float(state.loc[row.team_b, "state"])
        ha, hb = float(state.loc[row.team_a, "injury"]), float(state.loc[row.team_b, "injury"])
        delta = 0.32 * (sa - sb) - 0.055 * (ha - hb)
        base_row = base.loc[row.match_id]
        lambda_a.append(float(clip(float(base_row.expected_goals_a) * math.exp(delta - 0.04 * ha), 0.15, 4.50)))
        lambda_b.append(float(clip(float(base_row.expected_goals_b) * math.exp(-delta - 0.04 * hb), 0.15, 4.50)))
    goals_a = rng.poisson(np.asarray(lambda_a), size=(n_sim, len(third)))
    goals_b = rng.poisson(np.asarray(lambda_b), size=(n_sim, len(third)))
    points = np.tile(state["P"].to_numpy(int), (n_sim, 1))
    goals_for = np.tile(state["GF"].to_numpy(int), (n_sim, 1))
    goals_against = np.tile(state["GA"].to_numpy(int), (n_sim, 1))
    outcomes = np.empty((n_sim, len(third)), dtype=np.int8)
    for j, row in enumerate(third.itertuples(index=False)):
        ia, ib = index[row.team_a], index[row.team_b]
        ga, gb = goals_a[:, j], goals_b[:, j]
        goals_for[:, ia] += ga; goals_against[:, ia] += gb
        goals_for[:, ib] += gb; goals_against[:, ib] += ga
        a_win, b_win = ga > gb, gb > ga
        draw = ~(a_win | b_win)
        points[a_win, ia] += 3; points[b_win, ib] += 3
        points[draw, ia] += 1; points[draw, ib] += 1
        outcomes[:, j] = np.where(a_win, 1, np.where(b_win, -1, 0))
    advancement = np.empty((n_sim, len(team_names)), dtype=float)
    for simulation in range(n_sim):
        advancement[simulation] = fractional_advancement_weights(
            points[simulation], goals_for[simulation], goals_against[simulation], groups
        )
    probabilities = advancement.mean(axis=0)
    match_rows = []
    for j, row in enumerate(third.itertuples(index=False)):
        ia, ib = index[row.team_a], index[row.team_b]
        pa, pb = float(probabilities[ia]), float(probabilities[ib])
        masks = {"draw": outcomes[:, j] == 0, "a_win": outcomes[:, j] == 1, "b_win": outcomes[:, j] == -1}

        def conditional(team_id: int, key: str) -> float:
            mask = masks[key]
            return float(advancement[mask, team_id].mean()) if mask.any() else float(probabilities[team_id])

        pa_draw, pb_draw = conditional(ia, "draw"), conditional(ib, "draw")
        pa_win, pb_win = conditional(ia, "a_win"), conditional(ib, "b_win")
        stakeless = 0.5 * (2 * pa - 1) ** 2 + 0.5 * (2 * pb - 1) ** 2
        draw_benefit = min(pa_draw, pb_draw)
        gain = 0.5 * max(pa_win - pa_draw, 0) + 0.5 * max(pb_win - pb_draw, 0)
        collusion = float(clip(draw_benefit * (1 - clip(gain)) * (1 - 0.35 * stakeless)))
        match_rows.append({
            "match_id": row.match_id, "lambda_a": lambda_a[j], "lambda_b": lambda_b[j],
            "p_team_a_advance": pa, "p_team_b_advance": pb,
            "stakeless_risk": float(stakeless), "collusion_risk": collusion,
        })
    diagnostics = {
        "seed": int(seed), "simulations": int(n_sim),
        "tie_method": "exact_equal_fractional_weights_without_team_id_tiebreak",
        "advancement_probs": {team: float(probabilities[index[team]]) for team in team_names},
        "advancement_probability_sum": float(probabilities.sum()),
        "per_simulation_weight_sum_min": float(advancement.sum(axis=1).min()),
        "per_simulation_weight_sum_max": float(advancement.sum(axis=1).max()),
    }
    return diagnostics, pd.DataFrame(match_rows)


def dynamic_p3_environment(
    data: dict[str, pd.DataFrame], p2: pd.DataFrame, simulation: pd.DataFrame,
    standings: pd.DataFrame,
) -> pd.DataFrame:
    third = p2[p2["round_in_group"].eq(3)].merge(simulation, on="match_id", how="left")
    base = data["base_predictions"].set_index("match_id")
    live = data["live_group_results"]
    state = standings.set_index("team_name")
    p2_index = p2.set_index("match_id")
    feedback = {}
    for team in sorted(set(third["team_a"]) | set(third["team_b"])):
        prior = live[(live["team_a"].eq(team)) | (live["team_b"].eq(team))]
        attendance_ratios, viewer_ratios = [], []
        for row in prior.itertuples(index=False):
            attendance_ratios.append(float(clip(
                float(row.attendance) / float(p2_index.loc[row.match_id, "expected_attendance"]), 0.60, 1.50
            )))
            viewer_ratios.append(float(clip(
                float(row.tv_viewers) / float(p2_index.loc[row.match_id, "expected_tv_viewers"]), 0.60, 1.50
            )))
        feedback[team] = {
            "attendance": float(clip(np.mean(attendance_ratios), 0.75, 1.25)),
            "viewers": float(clip(np.mean(viewer_ratios), 0.75, 1.25)),
        }
    rows = []
    for row in third.itertuples(index=False):
        base_row = base.loc[row.match_id]
        sa, sb = float(state.loc[row.team_a, "state"]), float(state.loc[row.team_b, "state"])
        ha, hb = float(state.loc[row.team_a, "injury"]), float(state.loc[row.team_b, "injury"])
        state_mean, injury_scaled = (sa + sb) / 2, (ha + hb) / 6
        attendance_feedback = math.sqrt(feedback[row.team_a]["attendance"] * feedback[row.team_b]["attendance"])
        viewer_feedback = math.sqrt(feedback[row.team_a]["viewers"] * feedback[row.team_b]["viewers"])
        importance = (
            4 * row.p_team_a_advance * (1 - row.p_team_a_advance)
            + 4 * row.p_team_b_advance * (1 - row.p_team_b_advance)
        ) / 2
        base_attractiveness = float(base_row.attractiveness_index) / 100
        feedback_strength = float(clip((0.5 * attendance_feedback + 0.5 * viewer_feedback - 0.75) / 0.50))
        attractiveness = float(clip(
            0.50 * base_attractiveness + 0.25 * importance + 0.12 * feedback_strength
            + 0.08 * state_mean + 0.05 * (1 - injury_scaled)
        ))
        n_tilde = (
            float(row.expected_attendance) * attendance_feedback * (0.88 + 0.27 * importance)
            * (0.94 + 0.12 * state_mean) * (1 - 0.10 * injury_scaled)
        )
        v_tilde = (
            float(row.expected_tv_viewers) * viewer_feedback * (0.87 + 0.30 * importance)
            * (0.90 + 0.20 * attractiveness) * (1 - 0.06 * injury_scaled)
        )
        rows.append({
            **row._asdict(), "Q_i": float(importance), "S_i": state_mean, "H_i": injury_scaled,
            "feedback_i": feedback_strength, "attractiveness": attractiveness,
            "n_base": float(n_tilde), "v_base": float(v_tilde),
            "information_state": "after_round2_common_snapshot",
        })
    return pd.DataFrame(rows)


def static_p3_environment(data: dict[str, pd.DataFrame], p2: pd.DataFrame) -> pd.DataFrame:
    base = data["base_predictions"].set_index("match_id")
    rows = []
    for row in p2[p2["round_in_group"].eq(3)].itertuples(index=False):
        base_row = base.loc[row.match_id]
        rows.append({
            **row._asdict(),
            "p_team_a_advance": np.nan, "p_team_b_advance": np.nan,
            "stakeless_risk": float(1 - base_row.uncertainty_index),
            "collusion_risk": 0.0,
            "attractiveness": float(base_row.attractiveness_index) / 100,
            "n_base": float(row.expected_attendance),
            "v_base": float(row.expected_tv_viewers),
            "information_state": "pre_tournament_static",
        })
    return pd.DataFrame(rows)


def p3_resource_tables(data: dict[str, pd.DataFrame]) -> dict[tuple[str, int], dict[str, float]]:
    return {
        (row.resource_type, int(row.resource_level)): {
            "cost": float(row.unit_cost_index),
            "demand": float(row.demand_multiplier),
            "risk": float(row.risk_multiplier),
        }
        for row in data["dynamic_resource_costs"].itertuples(index=False)
    }


def p3_attendance(n_base: float, capacity: float, transport_multiplier: float, elasticity: float, delta: float) -> float:
    return float(min(capacity, n_base * transport_multiplier * (1 + delta) ** elasticity))


def p3_price_candidates(
    n_base: float, capacity: float, transport_multiplier: float, elasticity: float,
    max_discount: float, max_increase: float,
) -> list[float]:
    low, high = -float(max_discount), float(max_increase)
    candidates = {low, 0.0, high}
    if elasticity != 0 and n_base * transport_multiplier > 0:
        capacity_boundary = (capacity / (n_base * transport_multiplier)) ** (1 / elasticity) - 1
        if low <= capacity_boundary <= high:
            candidates.add(float(capacity_boundary))
    n0 = p3_attendance(n_base, capacity, transport_multiplier, elasticity, 0.0)
    # Bisection locates the exact truncated-demand boundary N(delta)=0.88N(0).
    if p3_attendance(n_base, capacity, transport_multiplier, elasticity, high) < 0.88 * n0:
        left, right = 0.0, high
        for _ in range(80):
            middle = (left + right) / 2
            if p3_attendance(n_base, capacity, transport_multiplier, elasticity, middle) >= 0.88 * n0:
                left = middle
            else:
                right = middle
        candidates.add(float(left))
    return sorted(
        delta for delta in candidates
        if low - 1e-12 <= delta <= high + 1e-12
        and 1 + delta > 0
        and p3_attendance(n_base, capacity, transport_multiplier, elasticity, delta) >= 0.88 * n0 - 1e-8
    )


def evaluate_p3_action(
    data: dict[str, pd.DataFrame], environment_row: pd.Series | object,
    broadcast_level: int, security_level: int, transport_level: int,
    delta: float,
) -> dict:
    row = environment_row if isinstance(environment_row, pd.Series) else pd.Series(environment_row._asdict())
    venues = data["venues"].set_index("venue_id")
    ticket = data["ticket_broadcast"].set_index("match_stage")
    resources = p3_resource_tables(data)
    security = data["security_requirements"].set_index("match_id")
    venue = venues.loc[row["venue_id"]]
    ticket_row = ticket.loc[stage_key(int(row["round_in_group"]))]
    broadcast = resources[("broadcast", int(broadcast_level))]
    security_resource = resources[("security", int(security_level))]
    transport = resources[("transport", int(transport_level))]
    attendance = p3_attendance(
        float(row["n_base"]), float(venue.capacity), transport["demand"],
        float(ticket_row.price_elasticity), float(delta),
    )
    attendance_zero = p3_attendance(
        float(row["n_base"]), float(venue.capacity), transport["demand"],
        float(ticket_row.price_elasticity), 0.0,
    )
    if attendance < 0.88 * attendance_zero - 1e-7:
        raise ValueError(f"Ticket demand retention violated for {row['match_id']}")
    ticket_value = float(ticket_row.base_ticket_price_usd) * (1 + float(delta)) * attendance
    viewers = float(row["v_base"]) * broadcast["demand"]
    broadcast_value = viewers * float(ticket_row.broadcast_unit_value_usd)
    pre_resource_occupancy = float(clip(float(row["n_base"]) / float(venue.capacity)))
    base_security = float(security.loc[row["match_id"], "security_demand_score"])
    security_demand = float(clip(
        0.45 * base_security + 0.25 * pre_resource_occupancy
        + 0.15 * float(row["attractiveness"])
        + 0.15 * (float(row["stakeless_risk"]) + float(row["collusion_risk"])) / 2
    ))
    risk = (
        0.40 * float(row["stakeless_risk"]) + 0.40 * float(row["collusion_risk"])
        + 0.20 * security_demand * security_resource["risk"]
    )
    return {
        "match_id": row["match_id"], "reference_date": row["reference_date"],
        "broadcast": int(broadcast_level), "security": int(security_level), "transport": int(transport_level),
        "delta": float(delta), "attendance": attendance, "attendance_at_zero_delta": attendance_zero,
        "broadcast_viewers": viewers, "ticket_value": ticket_value, "broadcast_value": broadcast_value,
        "attractiveness": float(row["attractiveness"]),
        "resource_cost": broadcast["cost"] + security_resource["cost"] + transport["cost"],
        "security_demand": security_demand, "pre_resource_occupancy": pre_resource_occupancy,
        "risk": float(risk), "information_state": row["information_state"],
    }


def build_p3_actions(data: dict[str, pd.DataFrame], environment: pd.DataFrame) -> pd.DataFrame:
    venues = data["venues"].set_index("venue_id")
    ticket = data["ticket_broadcast"].set_index("match_stage")
    limits = data["dynamic_resource_limits"].set_index("reference_date")
    resources = p3_resource_tables(data)
    rows = []
    for environment_row in environment.itertuples(index=False):
        venue = venues.loc[environment_row.venue_id]
        ticket_row = ticket.loc[stage_key(int(environment_row.round_in_group))]
        limit = limits.loc[environment_row.reference_date]
        for broadcast_level in (1, 2, 3):
            for security_level in range(int(environment_row.required_security_level), int(venue.security_level) + 1):
                for transport_level in (1, 2, 3):
                    transport_multiplier = resources[("transport", transport_level)]["demand"]
                    deltas = p3_price_candidates(
                        float(environment_row.n_base), float(venue.capacity), transport_multiplier,
                        float(ticket_row.price_elasticity), float(limit.max_ticket_discount_rate),
                        float(limit.max_ticket_increase_rate),
                    )
                    evaluated = [
                        evaluate_p3_action(
                            data, environment_row, broadcast_level, security_level, transport_level, delta
                        )
                        for delta in deltas
                    ]
                    rows.append(max(evaluated, key=lambda item: item["ticket_value"]))
    return pd.DataFrame(rows)


def p3_normalize(actions: pd.DataFrame, bounds: dict[str, list[float]] | None = None) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    output = actions.copy()
    if bounds is None:
        bounds = {field: [float(output[field].min()), float(output[field].max())] for field in P3_FIELDS}
    for field in P3_FIELDS:
        output[field + "_norm"] = output[field].map(lambda value: minmax_value(value, bounds[field]))
    output["net_value"] = (
        0.35 * output["ticket_value_norm"] + 0.35 * output["broadcast_value_norm"]
        + 0.10 * output["attractiveness_norm"] - 0.10 * output["resource_cost_norm"]
        - 0.10 * output["risk_norm"]
    )
    return output, bounds


def solve_p3_actions(actions: pd.DataFrame, limits: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    actions = actions.reset_index(drop=True)
    n = len(actions)
    rows, lower, upper = [], [], []

    def add(indices: list[int], values: np.ndarray | None, low: float, high: float):
        row = np.zeros(n)
        row[indices] = 1 if values is None else values
        rows.append(row); lower.append(low); upper.append(high)

    for indices in actions.groupby("match_id").groups.values():
        add(list(indices), None, 1, 1)
    limit_index = limits.set_index("reference_date")
    for day, indices in actions.groupby("reference_date").groups.items():
        indices = list(indices)
        limit = limit_index.loc[day]
        for field, capacity in [
            ("broadcast", "high_broadcast_capacity"),
            ("security", "high_security_capacity"),
            ("transport", "enhanced_transport_capacity"),
        ]:
            add(indices, actions.loc[indices, field].ge(3).to_numpy(float), 0, float(limit[capacity]))
        add(indices, actions.loc[indices, "resource_cost"].to_numpy(float), 0, float(limit.daily_resource_budget_index))
    started = time.perf_counter()
    result = milp(
        c=-actions["net_value"].to_numpy(float), integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120, "mip_rel_gap": 0.0001},
    )
    if result.x is None:
        raise RuntimeError(f"Problem 3 MILP failed: {result.message}")
    return actions.loc[np.where(result.x > 0.5)[0]].copy(), _milp_result_diagnostics(result, started)


def evaluate_fixed_p3(
    data: dict[str, pd.DataFrame], updated_environment: pd.DataFrame,
    fixed_decisions: pd.DataFrame, updated_bounds: dict[str, list[float]],
) -> pd.DataFrame:
    environment = updated_environment.set_index("match_id")
    rows = []
    for decision in fixed_decisions.itertuples(index=False):
        environment_row = environment.loc[decision.match_id].copy()
        environment_row["match_id"] = decision.match_id
        evaluated = evaluate_p3_action(
            data, environment_row, int(decision.broadcast), int(decision.security),
            int(decision.transport), float(decision.delta),
        )
        evaluated["decision_information_state"] = "pre_tournament_static"
        rows.append(evaluated)
    result, _ = p3_normalize(pd.DataFrame(rows), updated_bounds)
    return result


def p3_constraints_audit(selected: pd.DataFrame, data: dict[str, pd.DataFrame], p2: pd.DataFrame) -> dict:
    limits = data["dynamic_resource_limits"].set_index("reference_date")
    p2_index = p2.set_index("match_id")
    venues = data["venues"].set_index("venue_id")
    violations = []
    for row in selected.itertuples(index=False):
        p2_row = p2_index.loc[row.match_id]
        venue = venues.loc[p2_row.venue_id]
        if not (int(p2_row.required_security_level) <= int(row.security) <= int(venue.security_level)):
            violations.append(f"security:{row.match_id}")
        if float(row.attendance) > float(venue.capacity) + 1e-7:
            violations.append(f"capacity:{row.match_id}")
        if float(row.attendance) < 0.88 * float(row.attendance_at_zero_delta) - 1e-7:
            violations.append(f"ticket:{row.match_id}")
    for day, subset in selected.groupby("reference_date"):
        limit = limits.loc[day]
        if subset["broadcast"].ge(3).sum() > int(limit.high_broadcast_capacity):
            violations.append(f"broadcast:{day}")
        if subset["security"].ge(3).sum() > int(limit.high_security_capacity):
            violations.append(f"security_day:{day}")
        if subset["transport"].ge(3).sum() > int(limit.enhanced_transport_capacity):
            violations.append(f"transport:{day}")
        if subset["resource_cost"].sum() > float(limit.daily_resource_budget_index) + 1e-7:
            violations.append(f"budget:{day}")
    return {"all_passed": not violations, "violations": violations}


def solve_problem3(data: dict[str, pd.DataFrame], p2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    standings = standings_from_live(data)
    static_environment = static_p3_environment(data, p2)
    static_actions, static_bounds = p3_normalize(build_p3_actions(data, static_environment))
    static_decisions, static_solver = solve_p3_actions(static_actions, data["dynamic_resource_limits"])

    seeds = list(P3_STABILITY_SEEDS)
    runs = []
    primary = None
    for seed in seeds:
        simulation_diagnostics, simulation = simulate_advancement(data, standings, 20000, seed)
        dynamic_environment = dynamic_p3_environment(data, p2, simulation, standings)
        dynamic_actions, updated_bounds = p3_normalize(build_p3_actions(data, dynamic_environment))
        dynamic_decisions, dynamic_solver = solve_p3_actions(dynamic_actions, data["dynamic_resource_limits"])
        static_evaluation = evaluate_fixed_p3(data, dynamic_environment, static_decisions, updated_bounds)
        dynamic_total = float(dynamic_decisions["net_value"].sum())
        static_total = float(static_evaluation["net_value"].sum())
        signature = {
            row.match_id: [int(row.broadcast), int(row.security), int(row.transport), round(float(row.delta), 8)]
            for row in dynamic_decisions.itertuples(index=False)
        }
        runs.append({
            "seed": seed, "advancement_probs": simulation_diagnostics["advancement_probs"],
            "dynamic_total_net": dynamic_total, "static_total_net": static_total,
            "overall_improvement_rate": None if abs(static_total) < 1e-12 else (dynamic_total - static_total) / abs(static_total),
            "decision_signature": signature,
        })
        if primary is None:
            primary = (
                simulation_diagnostics, dynamic_environment, dynamic_decisions,
                static_evaluation, updated_bounds, dynamic_solver,
            )
    assert primary is not None
    simulation_diagnostics, environment, dynamic, static_eval, updated_bounds, dynamic_solver = primary
    dynamic_index, static_index, environment_index = (
        dynamic.set_index("match_id"), static_eval.set_index("match_id"), environment.set_index("match_id")
    )
    rows = []
    for match_id in sorted(dynamic_index.index):
        dyn, sta, env = dynamic_index.loc[match_id], static_index.loc[match_id], environment_index.loc[match_id]
        static_net = float(sta.net_value)
        improvement = np.nan if abs(static_net) < 1e-12 else (float(dyn.net_value) - static_net) / abs(static_net)
        rows.append({
            "match_id": match_id, "group_id": env.group_id, "team_a": env.team_a, "team_b": env.team_b,
            "updated_p_team_a_advance": round(float(env.p_team_a_advance), 8),
            "updated_p_team_b_advance": round(float(env.p_team_b_advance), 8),
            "updated_expected_attendance": round(float(dyn.attendance), 2),
            "updated_expected_tv_viewers": round(float(dyn.broadcast_viewers), 2),
            "stakeless_risk": round(float(env.stakeless_risk), 8),
            "collusion_risk": round(float(env.collusion_risk), 8),
            "updated_attractiveness": round(float(env.attractiveness), 8),
            "recommended_broadcast_priority": int(dyn.broadcast),
            "recommended_security_level": int(dyn.security),
            "recommended_transport_level": int(dyn.transport),
            "recommended_ticket_adjustment": round(float(dyn.delta), 8),
            "updated_ticket_revenue_usd": round(float(dyn.ticket_value), 2),
            "updated_broadcast_value_usd": round(float(dyn.broadcast_value), 2),
            "resource_cost_index": round(float(dyn.resource_cost), 6),
            "risk_exposure_index": round(float(dyn.risk), 8),
            "static_net_value": round(static_net, 10),
            "dynamic_net_value": round(float(dyn.net_value), 10),
            "improvement_rate": np.nan if pd.isna(improvement) else round(float(improvement), 10),
        })
    primary_signature = runs[0]["decision_signature"]
    teams = sorted(runs[0]["advancement_probs"])
    probability_stability = {
        team: {
            "mean": float(np.mean([run["advancement_probs"][team] for run in runs])),
            "std": float(np.std([run["advancement_probs"][team] for run in runs], ddof=1)),
            "min": float(np.min([run["advancement_probs"][team] for run in runs])),
            "max": float(np.max([run["advancement_probs"][team] for run in runs])),
        }
        for team in teams
    }
    decision_stability = [
        float(np.mean([run["decision_signature"][match_id] == primary_signature[match_id] for match_id in primary_signature]))
        for run in runs
    ]
    output = pd.DataFrame(rows)
    diagnostics = {
        **simulation_diagnostics,
        "problem3_normalization_bounds": updated_bounds,
        "problem3_static_pre_normalization_bounds": static_bounds,
        "problem3_dynamic_total_net": float(dynamic["net_value"].sum()),
        "problem3_static_total_net": float(static_eval["net_value"].sum()),
        "problem3_overall_improvement_rate": (
            float(dynamic["net_value"].sum() - static_eval["net_value"].sum()) / abs(float(static_eval["net_value"].sum()))
            if abs(float(static_eval["net_value"].sum())) > 1e-12 else None
        ),
        "problem3_static_decisions": {
            row.match_id: {
                "broadcast": int(row.broadcast), "security": int(row.security), "transport": int(row.transport),
                "delta": float(row.delta), "selected_using": "pre_tournament_static_information_only",
            }
            for row in static_decisions.itertuples(index=False)
        },
        "problem3_solver": {"dynamic": dynamic_solver, "static_pre": static_solver},
        "problem3_constraint_audit": {
            "dynamic": p3_constraints_audit(dynamic, data, p2),
            "static_fixed_in_updated_environment": p3_constraints_audit(static_eval, data, p2),
        },
        "problem3_robustness": {
            "seeds": seeds, "runs": runs,
            "advancement_probability_by_team": probability_stability,
            "dynamic_decision_match_fraction_vs_primary": decision_stability,
            "dynamic_total_net_range": [min(run["dynamic_total_net"] for run in runs), max(run["dynamic_total_net"] for run in runs)],
            "static_total_net_range": [min(run["static_total_net"] for run in runs), max(run["static_total_net"] for run in runs)],
            "overall_improvement_rate_range": [
                min(run["overall_improvement_rate"] for run in runs),
                max(run["overall_improvement_rate"] for run in runs),
            ],
        },
    }
    return output, diagnostics


# ---------------------------------------------------------------------------
# Problem 4


def read_entity_mappings() -> dict:
    return json.loads((MATERIAL_DIR / "entity_mappings.json").read_text(encoding="utf-8"))


def actual_schedule_2022() -> pd.DataFrame:
    source = json.loads((MATERIAL_DIR / "worldcup_2022_openfootball.json").read_text(encoding="utf-8"))
    mappings = read_entity_mappings()
    aliases = mappings["team_aliases"]
    stadium_aliases = mappings["stadium_aliases"]
    stadiums = mappings["stadiums"]
    source_url = mappings["sources"]["openfootball_file"]
    matches = [match for match in source["matches"] if "group" in match]
    rows = []
    team_round_count: dict[str, int] = {}
    for sequence, match in enumerate(sorted(matches, key=lambda item: (item["date"], item["time"], item["group"]))):
        team_a = aliases.get(match["team1"], match["team1"])
        team_b = aliases.get(match["team2"], match["team2"])
        round_no = max(team_round_count.get(team_a, 0), team_round_count.get(team_b, 0)) + 1
        team_round_count[team_a] = round_no; team_round_count[team_b] = round_no
        raw_stadium = match["ground"].split(",")[0].strip()
        stadium = stadium_aliases.get(raw_stadium, raw_stadium)
        metadata = stadiums[stadium]
        local = datetime.fromisoformat(f"{match['date']}T{match['time']}:00").replace(tzinfo=ZoneInfo("Asia/Qatar"))
        utc = local.astimezone(timezone.utc)
        score_value = match["score"]
        score = score_value["ft"] if isinstance(score_value, dict) else score_value
        rows.append({
            "match_id": f"WC2022_{sequence + 1:02d}", "competition": "FIFA World Cup 2022",
            "stage": "Group", "group_id": match["group"].replace("Group ", ""),
            "round_in_group": round_no, "team_a": team_a, "team_b": team_b,
            "venue": stadium, "city": metadata["city"], "country": "Qatar",
            "date": local.date().isoformat(), "local_kickoff_time": local.strftime("%H:%M"),
            "utc_datetime": utc.strftime("%Y-%m-%d %H:%M"),
            "goals_a": int(score[0]), "goals_b": int(score[1]),
            "source_url": source_url, "retrieval_date": mappings["retrieval_date"],
        })
    return pd.DataFrame(rows).sort_values(["group_id", "round_in_group", "utc_datetime"]).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def map_actual_matches(
    actual: pd.DataFrame, data: dict[str, pd.DataFrame], context: P2Context,
    neighbour_rank: int,
) -> pd.DataFrame:
    mappings = read_entity_mappings()
    stadiums = mappings["stadiums"]
    elo = pd.read_csv(MATERIAL_DIR / "worldcup_2022_pre_tournament_elo.csv").set_index("team_name")
    if "USA" in elo.index and "United States" not in elo.index:
        elo.loc["United States"] = elo.loc["USA"]
    teams = data["teams"].set_index("team_name")
    template = context.matches.copy()
    template["elo_mean"] = template.apply(lambda row: (teams.loc[row.team_a, "elo_rating"] + teams.loc[row.team_b, "elo_rating"]) / 2, axis=1)
    template["elo_absdiff"] = template.apply(lambda row: abs(teams.loc[row.team_a, "elo_rating"] - teams.loc[row.team_b, "elo_rating"]), axis=1)
    venues = data["venues"]
    slots = data["time_slots"].copy()
    slot_hours = pd.to_datetime(slots["reference_utc_time"], utc=True).dt.hour + pd.to_datetime(slots["reference_utc_time"], utc=True).dt.minute / 60
    rows = []
    for row in actual.itertuples(index=False):
        elo_a = float(elo.loc[row.team_a, "elo_rating"])
        elo_b = float(elo.loc[row.team_b, "elo_rating"])
        mean_elo, diff_elo = (elo_a + elo_b) / 2, abs(elo_a - elo_b)
        candidates = template[template["round_in_group"].eq(int(row.round_in_group))].copy()
        candidates["mapping_distance"] = abs(candidates["elo_mean"] - mean_elo) / 200 + abs(candidates["elo_absdiff"] - diff_elo) / 200
        matched = candidates.sort_values(["mapping_distance", "match_id"], kind="mergesort").iloc[min(neighbour_rank, len(candidates) - 1)]
        capacity = float(stadiums[row.venue]["capacity"])
        venue_candidates = venues[venues["security_level"].ge(int(matched.required_security_level))].copy()
        venue_candidates = venue_candidates.assign(
            mapping_distance=(venue_candidates["capacity"] - capacity).abs()
        ).sort_values(["mapping_distance", "venue_id"], kind="mergesort")
        mapped_venue = venue_candidates.iloc[min(neighbour_rank, len(venue_candidates) - 1)]
        actual_utc = pd.to_datetime(row.utc_datetime, utc=True)
        actual_hour = actual_utc.hour + actual_utc.minute / 60
        circular = np.minimum(abs(slot_hours - actual_hour), 24 - abs(slot_hours - actual_hour))
        slot_candidates = slots.assign(mapping_distance=circular).sort_values(
            ["mapping_distance", "global_prime_score", "slot_id"], ascending=[True, False, True], kind="mergesort"
        )
        mapped_slot = slot_candidates.iloc[min(neighbour_rank, len(slot_candidates) - 1)]
        rows.append({
            "actual_match_id": row.match_id, "mapped_match_id": matched.match_id,
            "mapped_venue_id": mapped_venue.venue_id, "mapped_slot_id": mapped_slot.slot_id,
            "match_mapping_distance": float(matched.mapping_distance),
            "venue_capacity_difference": float(mapped_venue.mapping_distance),
            "slot_hour_difference": float(mapped_slot.mapping_distance),
        })
    return pd.DataFrame(rows)


def actual_proxy_metrics(
    actual: pd.DataFrame, mapping: pd.DataFrame, p1_match: pd.DataFrame,
    context: P2Context, data: dict[str, pd.DataFrame],
) -> dict:
    mapped = actual.merge(mapping, left_on="match_id", right_on="actual_match_id")
    base = data["base_predictions"].set_index("match_id")
    ticket = data["ticket_broadcast"].set_index("match_stage")
    venues = data["venues"].set_index("venue_id")
    slots = data["time_slots"].set_index("slot_id")
    security = data["security_requirements"].set_index("match_id")
    viewers = p1_match.set_index("match_id")
    venue_metric = context.venue_candidates.set_index(["match_id", "venue_id"])
    rows = []
    for row in mapped.itertuples(index=False):
        matched_id, venue_id, slot_id = row.mapped_match_id, row.mapped_venue_id, row.mapped_slot_id
        match = context.matches.set_index("match_id").loc[matched_id]
        base_row, venue, slot = base.loc[matched_id], venues.loc[venue_id], slots.loc[slot_id]
        ticket_row = ticket.loc[stage_key(int(match.round_in_group))]
        vm = venue_metric.xs((matched_id, venue_id))
        rows.append({
            "actual_match_id": row.match_id, "team_a": row.team_a, "team_b": row.team_b,
            "T": float(vm.ticket_revenue_usd),
            "B": float(viewers.loc[matched_id, "predicted_tv_viewers"]) * float(ticket_row.broadcast_unit_value_usd)
                * float(slot.global_prime_score) * float(ticket_row.sponsor_weight),
            "U": float(base_row.uncertainty_index), "H": float(base_row.attractiveness_index) / 100,
            "C_match": float(venue.operation_cost_musd_per_match)
                + 0.1 * int(security.loc[matched_id, "required_security_level"]) * float(venue.security_cost_index),
            "D": float(vm.travel_burden), "R": float(vm.risk_index),
            "mapped_venue_id": venue_id, "mapped_slot_id": slot_id,
        })
    detail = pd.DataFrame(rows)
    fairness, fairness_detail = p2_fairness(
        detail.rename(columns={"mapped_venue_id": "venue_id", "mapped_slot_id": "slot_id"}),
        context.prime_slots, context.large_venues,
    )
    setup = float(venues.loc[detail["mapped_venue_id"].unique(), "setup_cost_musd"].sum())
    n = len(detail)
    raw = {
        "T": float(detail["T"].sum()), "B": float(detail["B"].sum()),
        "U": float(detail["U"].mean()), "H": float(detail["H"].mean()),
        "C": setup + float(detail["C_match"].sum()), "D": float(detail["D"].mean()),
        "F": fairness, "R": float(detail["R"].mean()),
    }
    bounds = {
        "T": [n * context.bounds["ticket_revenue_usd"][0], n * context.bounds["ticket_revenue_usd"][1]],
        "B": [n * context.bounds["broadcast_value_usd"][0], n * context.bounds["broadcast_value_usd"][1]],
        "U": [0, 1], "H": [0, 1],
        "C": [setup + n * context.bounds["match_cost_musd"][0], setup + n * context.bounds["match_cost_musd"][1]],
        "D": [0, 1], "F": [0, 1], "R": context.bounds["risk_index"],
    }
    normalized = {key: minmax_value(raw[key], bounds[key]) for key in raw}
    contributions = {key: P2_WEIGHTS[key] * normalized[key] for key in raw}
    return {
        "raw": raw, "per_match": {"T": raw["T"] / n, "B": raw["B"] / n, "C": raw["C"] / n},
        "bounds": bounds, "normalized": normalized, "weighted_contributions": contributions,
        "Z2": float(sum(contributions.values())), "fairness_detail": fairness_detail,
        "detail": detail.to_dict(orient="records"),
    }


def schedule_structure_metrics(
    schedule: pd.DataFrame, venue_column: str, date_column: str, utc_column: str,
    venue_metadata: dict[str, dict], prime_flags: dict[str, bool],
) -> dict[str, float]:
    frame = schedule.copy()
    frame["_utc"] = pd.to_datetime(frame[utc_column], utc=True)
    rests, travel, timezone_transitions = [], 0.0, 0
    teams = sorted(set(frame["team_a"]) | set(frame["team_b"]))
    for team in teams:
        subset = frame[(frame["team_a"].eq(team)) | (frame["team_b"].eq(team))].sort_values("_utc")
        rests.extend((subset["_utc"].diff().dropna().dt.total_seconds() / 3600).tolist())
        previous = None
        for row in subset.itertuples(index=False):
            metadata = venue_metadata[getattr(row, venue_column)]
            if previous is not None:
                travel += haversine_km(previous["latitude"], previous["longitude"], metadata["latitude"], metadata["longitude"])
                timezone_transitions += int(previous["timezone"] != metadata["timezone"])
            previous = metadata
    dates = pd.to_datetime(frame[date_column])
    span = int((dates.max() - dates.min()).days + 1)
    venue_count = frame[venue_column].nunique()
    active_venue_days = frame.groupby([venue_column, date_column]).ngroups
    prime_counts = {team: 0 for team in teams}
    for row in frame.itertuples(index=False):
        if prime_flags[getattr(row, "match_id")]:
            prime_counts[row.team_a] += 1; prime_counts[row.team_b] += 1
    return {
        "match_count": float(len(frame)), "team_count": float(len(teams)), "venue_count": float(venue_count),
        "schedule_span_days": float(span), "total_travel_km": float(travel),
        "avg_travel_per_team_km": float(travel / len(teams)),
        "avg_travel_per_match_km": float(travel / len(frame)),
        "cross_timezone_transitions": float(timezone_transitions),
        "average_rest_hours": float(np.mean(rests)), "minimum_rest_hours": float(np.min(rests)),
        "rest_hours_std": float(np.std(rests, ddof=1)),
        "matches_per_venue": float(len(frame) / venue_count),
        "matches_per_active_venue_day": float(len(frame) / active_venue_days),
        "venue_active_day_utilization": float(active_venue_days / (venue_count * span)),
        "prime_time_share": float(np.mean([prime_flags[mid] for mid in frame["match_id"]])),
        "prime_matches_per_team": float(sum(prime_counts.values()) / len(teams)),
        "prime_count_range": float(max(prime_counts.values()) - min(prime_counts.values())),
    }


def _comparison_row(name: str, category: str, actual_value: float, optimized_value: float, direction: str) -> dict:
    difference = float(optimized_value) - float(actual_value)
    if direction == "larger_better":
        relative = np.nan if abs(actual_value) < 1e-12 else difference / abs(actual_value)
        result = "优化赛程更优" if difference > 1e-12 else "实际赛程更优" if difference < -1e-12 else "基本相同"
    elif direction == "smaller_better":
        relative = np.nan if abs(actual_value) < 1e-12 else -difference / abs(actual_value)
        result = "优化赛程更优" if difference < -1e-12 else "实际赛程更优" if difference > 1e-12 else "基本相同"
    else:
        relative, result = np.nan, "规模或描述指标，不作直接优劣判定"
    return {
        "indicator_name": name, "indicator_category": category,
        "actual_schedule_value": actual_value, "optimized_schedule_value": optimized_value,
        "absolute_difference": difference, "relative_improvement": relative,
        "preferred_direction": direction, "evaluation_result": result,
    }


def solve_problem4(
    data: dict[str, pd.DataFrame], p1_match: pd.DataFrame, p2: pd.DataFrame,
    context: P2Context,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    actual = actual_schedule_2022()
    first_mapping = map_actual_matches(actual, data, context, 0)
    second_mapping = map_actual_matches(actual, data, context, 1)
    first_proxy = actual_proxy_metrics(actual, first_mapping, p1_match, context, data)
    second_proxy = actual_proxy_metrics(actual, second_mapping, p1_match, context, data)
    optimized_proxy = p2_metrics(p2, context, data)
    mappings = read_entity_mappings()
    actual_venues = mappings["stadiums"]
    optimized_venues = {
        row.venue_id: {
            "latitude": float(row.latitude), "longitude": float(row.longitude), "timezone": str(row.timezone),
            "capacity": int(row.capacity),
        }
        for row in data["venues"].itertuples(index=False)
    }
    first_slot_flags = dict(zip(
        first_mapping["actual_match_id"], first_mapping["mapped_slot_id"].astype(str).isin(context.prime_slots)
    ))
    second_slot_flags = dict(zip(
        second_mapping["actual_match_id"], second_mapping["mapped_slot_id"].astype(str).isin(context.prime_slots)
    ))
    optimized_slot_flags = dict(zip(p2["match_id"], p2["slot_id"].astype(str).isin(context.prime_slots)))
    actual_structure = schedule_structure_metrics(
        actual, "venue", "date", "utc_datetime", actual_venues, first_slot_flags
    )
    optimized_structure = schedule_structure_metrics(
        p2, "venue_id", "reference_date", "utc_datetime", optimized_venues, optimized_slot_flags
    )
    base = data["base_predictions"].set_index("match_id")
    actual_meta = actual.set_index("match_id")
    actual_capacity_ratios, actual_attendance = [], []
    for mapping in first_mapping.itertuples(index=False):
        expected = float(base.loc[mapping.mapped_match_id, "expected_attendance_base"])
        capacity = float(actual_venues[actual_meta.loc[mapping.actual_match_id, "venue"]]["capacity"])
        actual_attendance.append(min(expected, capacity))
        actual_capacity_ratios.append(min(expected, capacity) / capacity)
    optimized_capacity = p2.merge(data["venues"][["venue_id", "capacity"]], on="venue_id")
    actual_structure["capacity_match_ratio"] = float(np.mean(actual_capacity_ratios))
    actual_structure["expected_attendance_per_match"] = float(np.mean(actual_attendance))
    optimized_structure["capacity_match_ratio"] = float((optimized_capacity["expected_attendance"] / optimized_capacity["capacity"]).mean())
    optimized_structure["expected_attendance_per_match"] = float(p2["expected_attendance"].mean())
    directions = {
        "match_count": "neutral", "team_count": "neutral", "venue_count": "neutral", "schedule_span_days": "neutral",
        "total_travel_km": "smaller_better", "avg_travel_per_team_km": "smaller_better",
        "avg_travel_per_match_km": "smaller_better", "cross_timezone_transitions": "smaller_better",
        "average_rest_hours": "neutral", "minimum_rest_hours": "larger_better", "rest_hours_std": "smaller_better",
        "matches_per_venue": "neutral", "matches_per_active_venue_day": "neutral",
        "venue_active_day_utilization": "larger_better", "prime_time_share": "larger_better",
        "prime_matches_per_team": "neutral", "prime_count_range": "smaller_better",
        "capacity_match_ratio": "larger_better", "expected_attendance_per_match": "larger_better",
    }
    rows = [
        _comparison_row(name, "structure", actual_structure[name], optimized_structure[name], directions[name])
        for name in directions
    ]
    proxy_names = {
        "T": "proxy_ticket_value_per_match_usd", "B": "proxy_broadcast_value_per_match_usd",
        "U": "proxy_uncertainty_mean", "H": "proxy_attractiveness_mean",
        "C": "proxy_cost_per_match_musd", "D": "proxy_travel_burden",
        "F": "proxy_fairness_penalty", "R": "proxy_execution_risk",
    }
    for key, name in proxy_names.items():
        if key in ("T", "B", "C"):
            actual_value = first_proxy["per_match"][key]
            optimized_value = optimized_proxy["raw"][key] / len(p2)
        else:
            actual_value = first_proxy["raw"][key]
            optimized_value = optimized_proxy["raw"][key]
        direction = "larger_better" if key in ("T", "B", "U", "H") else "smaller_better"
        rows.append(_comparison_row(name, "proxy_nearest", actual_value, optimized_value, direction))
    rows.append(_comparison_row("proxy_Z2", "proxy_nearest", first_proxy["Z2"], optimized_proxy["Z2"], "larger_better"))
    rows.append(_comparison_row("proxy_Z2_second_nearest", "proxy_sensitivity", second_proxy["Z2"], optimized_proxy["Z2"], "larger_better"))
    second_prime_share = float(np.mean([second_slot_flags[mid] for mid in actual["match_id"]]))
    rows.append(_comparison_row(
        "prime_time_share_second_nearest", "proxy_sensitivity",
        second_prime_share, optimized_structure["prime_time_share"], "larger_better",
    ))
    diagnostics = {
        "problem4_sources": mappings["sources"],
        "problem4_entity_mapping_file": str(MATERIAL_DIR / "entity_mappings.json"),
        "problem4_mapping_method": "same-round nearest Elo mean/absolute-difference; venue by capacity; slot by circular UTC hour",
        "problem4_first_neighbour_mapping": first_mapping.to_dict(orient="records"),
        "problem4_second_neighbour_mapping": second_mapping.to_dict(orient="records"),
        "problem4_structure": {"actual": actual_structure, "optimized": optimized_structure},
        "problem4_proxy": {
            "actual_nearest": first_proxy, "actual_second_nearest": second_proxy,
            "optimized": optimized_proxy,
            "direction_reversal": bool(
                np.sign(optimized_proxy["Z2"] - first_proxy["Z2"])
                != np.sign(optimized_proxy["Z2"] - second_proxy["Z2"])
            ),
        },
        "problem4_scale_basis": {
            "T_B_C": "per_match_for_cross_size_display; each schedule normalized with the same candidate bounds",
            "travel": "within-tournament venue-to-venue transitions, excluding travel to first match",
            "rest": "per-team consecutive-match UTC differences",
        },
    }
    return actual, pd.DataFrame(rows), diagnostics
