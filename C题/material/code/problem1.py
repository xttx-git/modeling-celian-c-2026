"""问题一：固定题定 split 的无泄漏观看人数时序预测。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from params import MASTER_SEED, N_TIME_SPLITS, TARGET_SCALE
from utils import OUTPUT_DIR, FIGURES_DIR, load_all_sheets, runtime_metadata, set_all_seeds, write_json

POSTMATCH_FIELDS = ["goals", "xg", "shots", "possession", "attendance", "tv_viewers"]
POSTMATCH_DIRECT = {
    "goals_a", "goals_b", "xg_a", "xg_b", "shots_a", "shots_b",
    "possession_a", "possession_b", "attendance", "tv_viewers",
}
OMITTED_PREGAME = {"neutral"}
FORBIDDEN_DIRECT = POSTMATCH_DIRECT | OMITTED_PREGAME


def _team_metadata(teams: pd.DataFrame) -> dict[str, dict]:
    return teams.set_index("team_name").to_dict(orient="index")


def _pre_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    inv = np.column_stack([
        1.0 / frame["odds_a"].to_numpy(float),
        1.0 / frame["odds_draw"].to_numpy(float),
        1.0 / frame["odds_b"].to_numpy(float),
    ])
    prob = inv / inv.sum(axis=1, keepdims=True)
    out = pd.DataFrame(prob, columns=["p_a", "p_draw", "p_b"], index=frame.index)
    out["odds_entropy"] = -(prob * np.log(np.clip(prob, 1e-15, 1.0))).sum(axis=1)
    return out


def _empty_hist() -> dict[str, deque]:
    return {f: deque(maxlen=30) for f in POSTMATCH_FIELDS}


def _history_features(hist: dict[str, deque], prefix: str) -> dict[str, float]:
    ans: dict[str, float] = {}
    for field in POSTMATCH_FIELDS:
        values = np.asarray(hist[field], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            ans[f"{prefix}_{field}_roll5"] = np.nan
            ans[f"{prefix}_{field}_ewm5"] = np.nan
        else:
            recent = values[-5:]
            weights = np.power(0.5, np.arange(len(recent) - 1, -1, -1) / 5.0)
            ans[f"{prefix}_{field}_roll5"] = float(recent.mean())
            ans[f"{prefix}_{field}_ewm5"] = float(np.average(recent, weights=weights))
    return ans


def build_features(hist: pd.DataFrame, teams: pd.DataFrame):
    """逐场先提取过去、后更新状态，实现等价 shift(1) 的血缘安全特征。"""
    frame = hist.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    prob = _pre_probabilities(frame)
    meta = _team_metadata(teams)
    histories: defaultdict[str, dict[str, deque]] = defaultdict(_empty_hist)
    rows: list[dict] = []
    source_cutoff: list[str] = []
    last_date: dict[str, pd.Timestamp] = {}
    numeric_meta = [
        "market_value_musd", "avg_age", "star_index", "fan_base_index",
        "style_attack", "style_defense", "host_flag", "strength_score",
    ]
    # 同日可能出现重复记录；整日先提特征、后统一更新，保证严格 date_m < date_j。
    for current_date, day in frame.groupby("date", sort=True):
        pending = []
        for idx, row in day.iterrows():
            a, b = str(row["team_a"]), str(row["team_b"])
            feat: dict[str, object] = {
                "competition": str(row["competition"]),
                "stage": str(row["stage"]),
                "timezone_pair": "|".join(sorted([
                    str(meta.get(a, {}).get("home_timezone_region", "Unknown")),
                    str(meta.get(b, {}).get("home_timezone_region", "Unknown")),
                ])),
                "strength_rank_mean": (float(row["strength_rank_a"]) + float(row["strength_rank_b"])) / 2.0,
                "strength_rank_gap": abs(float(row["strength_rank_a"]) - float(row["strength_rank_b"])),
                "elo_mean": (float(row["elo_a"]) + float(row["elo_b"])) / 2.0,
                "elo_gap": abs(float(row["elo_a"]) - float(row["elo_b"])),
                "p_a": float(prob.loc[idx, "p_a"]),
                "p_draw": float(prob.loc[idx, "p_draw"]),
                "p_b": float(prob.loc[idx, "p_b"]),
                "odds_entropy": float(prob.loc[idx, "odds_entropy"]),
            }
            for col in numeric_meta:
                av = float(meta.get(a, {}).get(col, np.nan))
                bv = float(meta.get(b, {}).get(col, np.nan))
                feat[f"{col}_mean"] = np.nanmean([av, bv]) if np.isfinite([av, bv]).any() else np.nan
                feat[f"{col}_gap"] = abs(av - bv) if np.isfinite(av) and np.isfinite(bv) else np.nan
            feat.update(_history_features(histories[a], "a_hist"))
            feat.update(_history_features(histories[b], "b_hist"))
            feat["worldcup_stage"] = float(str(row["competition"]) == "World Cup")
            feat["fan_stage_interaction"] = feat["fan_base_index_mean"] * feat["worldcup_stage"]
            feat["star_stage_interaction"] = feat["star_index_mean"] * feat["worldcup_stage"]
            feat["strength_entropy_interaction"] = feat["strength_score_mean"] * feat["odds_entropy"]
            rows.append(feat)
            prior_dates = [last_date[t] for t in (a, b) if t in last_date]
            source_cutoff.append(max(prior_dates).isoformat() if prior_dates else "none")
            pending.append((a, b, row))
        for a, b, row in pending:
            for team, suffix in ((a, "a"), (b, "b")):
                values = {
                    "goals": row[f"goals_{suffix}"], "xg": row[f"xg_{suffix}"],
                    "shots": row[f"shots_{suffix}"], "possession": row[f"possession_{suffix}"],
                    "attendance": row["attendance"], "tv_viewers": row["tv_viewers"],
                }
                for field, value in values.items():
                    if pd.notna(value):
                        histories[team][field].append(float(value))
                last_date[team] = current_date
    X = pd.DataFrame(rows)
    assert not (set(X.columns) & FORBIDDEN_DIRECT), "同场赛后字段进入直接特征"
    for cutoff, kickoff in zip(source_cutoff, frame["date"]):
        if cutoff != "none":
            assert pd.Timestamp(cutoff) < kickoff, "发现 future-row 历史聚合"
    return frame, X, histories, last_date


def features_with_validation_targets_masked(hist: pd.DataFrame, teams: pd.DataFrame,
                                            validation_ids: list[str]) -> pd.DataFrame:
    """模拟整段验证集观看标签均未知，但允许较早比赛的其他赛后事实滚动进入。"""
    masked = hist.copy()
    target_rows = masked["match_id"].astype(str).isin(set(validation_ids))
    assert int(target_rows.sum()) == len(validation_ids)
    masked.loc[target_rows, "tv_viewers"] = np.nan
    rebuilt_hist, rebuilt_X, _, _ = build_features(masked, teams)
    assert rebuilt_hist["match_id"].astype(str).tolist() == hist["match_id"].astype(str).tolist()
    return rebuilt_X.loc[rebuilt_hist["dataset_split"].eq("train")].reset_index(drop=True)


def build_group_features(groups: pd.DataFrame, base: pd.DataFrame, teams: pd.DataFrame,
                         histories: dict[str, dict[str, deque]]) -> pd.DataFrame:
    merged = groups.merge(base, on=["match_id", "group_id", "round_in_group", "team_a", "team_b"], validate="one_to_one")
    meta = _team_metadata(teams)
    numeric_meta = [
        "market_value_musd", "avg_age", "star_index", "fan_base_index",
        "style_attack", "style_defense", "host_flag", "strength_score",
    ]
    rows = []
    for _, row in merged.iterrows():
        a, b = str(row["team_a"]), str(row["team_b"])
        p = np.array([row["p_a_win"], row["p_draw"], row["p_b_win"]], dtype=float)
        p = p / p.sum()
        feat: dict[str, object] = {
            "competition": "World Cup", "stage": "Group",
            "timezone_pair": "|".join(sorted([str(meta[a]["home_timezone_region"]), str(meta[b]["home_timezone_region"])])),
            "strength_rank_mean": (float(meta[a]["strength_rank"]) + float(meta[b]["strength_rank"])) / 2.0,
            "strength_rank_gap": abs(float(meta[a]["strength_rank"]) - float(meta[b]["strength_rank"])),
            "elo_mean": (float(meta[a]["elo_rating"]) + float(meta[b]["elo_rating"])) / 2.0,
            "elo_gap": abs(float(meta[a]["elo_rating"]) - float(meta[b]["elo_rating"])),
            "p_a": float(p[0]), "p_draw": float(p[1]), "p_b": float(p[2]),
            "odds_entropy": float(-(p * np.log(np.clip(p, 1e-15, 1.0))).sum()),
        }
        for col in numeric_meta:
            av, bv = float(meta[a][col]), float(meta[b][col])
            feat[f"{col}_mean"] = (av + bv) / 2.0
            feat[f"{col}_gap"] = abs(av - bv)
        feat.update(_history_features(histories[a], "a_hist"))
        feat.update(_history_features(histories[b], "b_hist"))
        feat["worldcup_stage"] = 1.0
        feat["fan_stage_interaction"] = feat["fan_base_index_mean"]
        feat["star_stage_interaction"] = feat["star_index_mean"]
        feat["strength_entropy_interaction"] = feat["strength_score_mean"] * feat["odds_entropy"]
        rows.append(feat)
    return pd.DataFrame(rows)


def _columns(X: pd.DataFrame):
    categorical = list(X.select_dtypes(include=["object", "category"]).columns)
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def _pipeline(model, X: pd.DataFrame, scale: bool) -> Pipeline:
    numeric, categorical = _columns(X)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def _inner_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    cut = max(20, int(math.floor(n * 0.8)))
    cut = min(cut, n - 1)
    return np.arange(cut), np.arange(cut, n)


def _fit_elastic(X: pd.DataFrame, y: np.ndarray):
    tr, va = _inner_split(len(X))
    best = None
    for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        for rho in [0.1, 0.5, 0.9]:
            model = _pipeline(ElasticNet(alpha=alpha, l1_ratio=rho, max_iter=20000, random_state=MASTER_SEED), X, True)
            model.fit(X.iloc[tr], y[tr])
            mse = mean_squared_error(y[va], model.predict(X.iloc[va]))
            if best is None or mse < best[0]:
                best = (mse, alpha, rho)
    final = _pipeline(ElasticNet(alpha=best[1], l1_ratio=best[2], max_iter=20000, random_state=MASTER_SEED), X, True)
    final.fit(X, y)
    return final, {"alpha": best[1], "l1_ratio": best[2]}


def _fit_extra(X: pd.DataFrame, y: np.ndarray):
    tr, va = _inner_split(len(X))
    best = None
    for leaf in [3, 5, 8]:
        for max_features in [0.6, 0.8, 1.0]:
            est = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=leaf,
                                      max_features=max_features, random_state=MASTER_SEED, n_jobs=-1)
            model = _pipeline(est, X, False)
            model.fit(X.iloc[tr], y[tr])
            mse = mean_squared_error(y[va], model.predict(X.iloc[va]))
            if best is None or mse < best[0]:
                best = (mse, leaf, max_features)
    est = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=best[1],
                              max_features=best[2], random_state=MASTER_SEED, n_jobs=-1)
    final = _pipeline(est, X, False)
    final.fit(X, y)
    return final, {"min_samples_leaf": best[1], "max_features": best[2]}


def _cat_data(X: pd.DataFrame, medians: pd.Series | None = None):
    numeric, categorical = _columns(X)
    z = X.copy()
    if medians is None:
        medians = z[numeric].median()
    z[numeric] = z[numeric].fillna(medians).astype(float)
    z[categorical] = z[categorical].fillna("<missing>").astype(str)
    return z, categorical, medians


def _fit_cat(X: pd.DataFrame, y: np.ndarray):
    tr, va = _inner_split(len(X))
    best = None
    for depth in [4, 5, 6]:
        for l2 in [3, 10]:
            ztr, cats, med = _cat_data(X.iloc[tr])
            zva, _, _ = _cat_data(X.iloc[va], med)
            model = CatBoostRegressor(
                loss_function="RMSE", eval_metric="RMSE", has_time=True,
                iterations=3000, learning_rate=3e-2, depth=depth,
                l2_leaf_reg=l2, random_seed=MASTER_SEED, verbose=False,
                thread_count=1, allow_writing_files=False,
            )
            model.fit(ztr, y[tr], cat_features=cats, eval_set=(zva, y[va]),
                      early_stopping_rounds=150, verbose=False)
            mse = mean_squared_error(y[va], model.predict(zva))
            iters = max(50, int(model.get_best_iteration()) + 1)
            if best is None or mse < best[0]:
                best = (mse, depth, l2, iters)
    z, cats, med = _cat_data(X)
    final = CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE", has_time=True,
        iterations=best[3], learning_rate=3e-2, depth=best[1],
        l2_leaf_reg=best[2], random_seed=MASTER_SEED, verbose=False,
        thread_count=1, allow_writing_files=False,
    )
    final.fit(z, y, cat_features=cats, verbose=False)
    return (final, med), {"depth": best[1], "l2_leaf_reg": best[2], "iterations": best[3]}


def _predict(model_name: str, fitted, X: pd.DataFrame) -> np.ndarray:
    if model_name == "CatBoost":
        model, med = fitted
        z, _, _ = _cat_data(X, med)
        return model.predict(z)
    return fitted.predict(X)


def _fit(model_name: str, X: pd.DataFrame, y: np.ndarray):
    if model_name == "ElasticNet":
        return _fit_elastic(X, y)
    if model_name == "ExtraTrees":
        return _fit_extra(X, y)
    if model_name == "CatBoost":
        return _fit_cat(X, y)
    raise ValueError(model_name)


def _fit_fixed(model_name: str, X: pd.DataFrame, y: np.ndarray, params: dict):
    """用最早训练窗内选出的冻结参数重拟合，避免在每个外折重复整套网格。"""
    if model_name == "ElasticNet":
        model = _pipeline(ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                                     max_iter=20000, random_state=MASTER_SEED), X, True)
        model.fit(X, y)
        return model
    if model_name == "ExtraTrees":
        est = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=params["min_samples_leaf"],
                                  max_features=params["max_features"], random_state=MASTER_SEED, n_jobs=-1)
        model = _pipeline(est, X, False)
        model.fit(X, y)
        return model
    if model_name == "CatBoost":
        z, cats, med = _cat_data(X)
        model = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE", has_time=True,
            iterations=params["iterations"], learning_rate=3e-2, depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"], random_seed=MASTER_SEED, verbose=False,
            thread_count=1, allow_writing_files=False,
        )
        model.fit(z, y, cat_features=cats, verbose=False)
        return model, med
    raise ValueError(model_name)


def solve() -> dict:
    set_all_seeds(MASTER_SEED)
    t0 = time.time()
    sheets = load_all_sheets()
    hist, X_all, histories, _ = build_features(sheets["historical_matches"], sheets["teams"])
    train_mask = hist["dataset_split"].eq("train")
    test_mask = hist["dataset_split"].eq("test")
    assert int(train_mask.sum()) == 560 and int(test_mask.sum()) == 140
    assert hist.loc[test_mask, "tv_viewers"].isna().all()
    train_ids = hist.loc[train_mask, "match_id"].astype(str).tolist()
    test_ids = hist.loc[test_mask, "match_id"].astype(str).tolist()
    assert set(train_ids).isdisjoint(test_ids)
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    y = hist.loc[train_mask, "tv_viewers"].to_numpy(float) / TARGET_SCALE

    folds = list(TimeSeriesSplit(n_splits=5).split(X_train))
    for tr, va in folds:
        assert hist.loc[train_mask].iloc[tr]["date"].max() < hist.loc[train_mask].iloc[va]["date"].min()
    model_names = ["MeanBaseline", "ElasticNet", "ExtraTrees", "CatBoost"]
    fold_metrics: list[dict] = []
    oof = {m: np.full(len(y), np.nan) for m in model_names}
    chosen_params: dict[str, list[dict]] = defaultdict(list)
    # 所有超参数只用最早外折的训练段选择，之后冻结；test 和任何外折验证标签均不参与。
    earliest_train = folds[0][0]
    _, inner_valid = _inner_split(len(earliest_train))
    inner_valid_global = earliest_train[inner_valid]
    inner_valid_ids = np.asarray(train_ids)[inner_valid_global].tolist()
    X_inner_masked = features_with_validation_targets_masked(hist, sheets["teams"], inner_valid_ids)
    X_earliest = X_inner_masked.iloc[earliest_train].reset_index(drop=True)
    frozen_params = {}
    for model_name in model_names[1:]:
        print(f"[P1] selecting {model_name} on earliest expanding train window", flush=True)
        _, frozen_params[model_name] = _fit(model_name, X_earliest, y[earliest_train])
    for fold_no, (tr, va) in enumerate(folds, 1):
        print(f"[P1] fold {fold_no}/5 train={len(tr)} valid={len(va)}", flush=True)
        validation_ids = np.asarray(train_ids)[va].tolist()
        X_fold = features_with_validation_targets_masked(hist, sheets["teams"], validation_ids)
        mean_pred = np.full(len(va), y[tr].mean())
        oof["MeanBaseline"][va] = mean_pred
        for model_name in model_names:
            if model_name == "MeanBaseline":
                pred = mean_pred
                params = {}
            else:
                params = frozen_params[model_name]
                fitted = _fit_fixed(model_name, X_train.iloc[tr], y[tr], params)
                pred = _predict(model_name, fitted, X_fold.iloc[va])
                oof[model_name][va] = pred
                chosen_params[model_name].append(params)
            fold_metrics.append({
                "fold": fold_no, "model": model_name,
                "mse": float(mean_squared_error(y[va], pred)),
                "rmse": float(mean_squared_error(y[va], pred) ** 0.5),
                "mae": float(mean_absolute_error(y[va], pred)),
                "train_end": hist.loc[train_mask].iloc[tr]["date"].max().isoformat(),
                "valid_start": hist.loc[train_mask].iloc[va]["date"].min().isoformat(),
            })

    summary = []
    for name in model_names:
        rows = [r for r in fold_metrics if r["model"] == name]
        summary.append({"model": name,
                        "mean_mse": float(np.mean([r["mse"] for r in rows])),
                        "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
                        "mean_mae": float(np.mean([r["mae"] for r in rows]))})
    summary.sort(key=lambda r: (r["mean_mse"], model_names.index(r["model"])))
    best_mse = summary[0]["mean_mse"]
    eligible = [r for r in summary if r["mean_mse"] <= best_mse * 1.01]
    selected = min(eligible, key=lambda r: model_names.index(r["model"]))["model"]
    print(f"[P1] selected={selected} mean_mse={best_mse:.6f}", flush=True)
    if selected == "MeanBaseline":
        fitted = float(y.mean())
        test_pred = np.full(len(X_test), fitted)
        final_params = {}
    else:
        final_params = frozen_params[selected]
        fitted = _fit_fixed(selected, X_train, y, final_params)
        test_pred = _predict(selected, fitted, X_test)

    X_group = build_group_features(sheets["groups_matches"], sheets["base_predictions"], sheets["teams"], histories)
    if selected == "MeanBaseline":
        group_pred = np.full(len(X_group), fitted)
    else:
        group_pred = _predict(selected, fitted, X_group)
    test_pred = np.maximum(test_pred, 1e-6)
    group_pred = np.maximum(group_pred, 1e-6)
    valid = np.isfinite(oof[selected])
    residuals = y[valid] - oof[selected][valid]
    q05, q95 = np.quantile(residuals, [0.05, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_out = pd.DataFrame({"match_id_test": test_ids, "predicted_test_tv_viewers": test_pred})
    group_info = sheets["groups_matches"]
    group_out = pd.DataFrame({
        "match_id": group_info["match_id"], "team_a": group_info["team_a"], "team_b": group_info["team_b"],
        "predicted_tv_viewers": np.rint(group_pred * TARGET_SCALE).astype(np.int64),
    })
    test_out.to_csv(OUTPUT_DIR / "result_1_test_prediction.csv", index=False, encoding="utf-8-sig")
    group_out.to_csv(OUTPUT_DIR / "result_1_match_prediction.csv", index=False, encoding="utf-8-sig")
    assert len(test_out) == 140 and test_out["match_id_test"].is_unique
    assert len(group_out) == 72 and group_out["match_id"].is_unique
    assert np.isfinite(test_pred).all() and (test_pred > 0).all()

    lineage = []
    for col in X_train.columns:
        history = "_hist_" in col
        lineage.append({
            "feature_name": col,
            "source_table": "historical_matches/teams/base_predictions" if not history else "historical_matches",
            "source_column": col,
            "source_time_rule": "strictly_before_kickoff_shift1" if history else "pregame_available",
            "fitted_on_ids": "outer_fold_train_only; final=560 train ids",
        })
    selected_oof = oof[selected][valid]
    result = {
        "metadata": {**runtime_metadata(MASTER_SEED), "elapsed_seconds": time.time() - t0},
        "method": "fixed split + TimeSeriesSplit(n_splits=5) + ElasticNet/ExtraTrees/CatBoost",
        "dataset": {"sheets": 15, "total_rows": 2241, "train_rows": 560, "test_rows": 140, "group_rows": 72},
        "feature_lineage": lineage,
        "forbidden_direct_features": sorted(FORBIDDEN_DIRECT),
        "postmatch_direct_features": sorted(POSTMATCH_DIRECT),
        "intentionally_omitted_pregame_features": sorted(OMITTED_PREGAME),
        "validation_target_history_policy": "mask every validation-fold tv_viewers label before rebuilding lag features",
        "fold_metrics": fold_metrics,
        "model_summary": summary,
        "selected_model": selected,
        "selected_params": final_params,
        "oof": {
            "match_id": np.asarray(train_ids)[valid].tolist(),
            "date": hist.loc[train_mask].iloc[np.where(valid)[0]]["date"].astype(str).tolist(),
            "actual_million": y[valid].tolist(),
            "predicted_million": selected_oof.tolist(),
            "residual_million": residuals.tolist(),
        },
        "prediction_interval_residual_quantiles": {"q05": float(q05), "q95": float(q95)},
        "test_prediction_summary_million": {"min": float(test_pred.min()), "mean": float(test_pred.mean()), "max": float(test_pred.max())},
        "group_prediction_summary_person": {"min": int(group_out.predicted_tv_viewers.min()), "mean": float(group_out.predicted_tv_viewers.mean()), "max": int(group_out.predicted_tv_viewers.max())},
        "outputs": ["output/result_1_test_prediction.csv", "output/result_1_match_prediction.csv"],
        "seed": MASTER_SEED,
    }
    write_json(FIGURES_DIR / "problem_1_results.json", result)
    print("[P1] validate_capability PASS: 固定split、赛前血缘、同折三模型、140+72交付", flush=True)
    return result


if __name__ == "__main__":
    solve()
