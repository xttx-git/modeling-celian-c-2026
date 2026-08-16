#!/usr/bin/env python3
"""Generate reproducible paper-only analyses without changing formal outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pipeline_core as pc  # noqa: E402


FIGURE_DIR: Path
TABLE_DIR: Path
COLORS = {
    "blue": "#2563EB", "teal": "#0F766E", "orange": "#D97706",
    "red": "#B91C1C", "gray": "#64748B", "green": "#15803D",
    "purple": "#7E22CE", "light": "#E2E8F0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Formal output directory.")
    parser.add_argument("--generated-dir", default="paper/generated", help="Paper-only generated directory.")
    parser.add_argument("--skip-p2-sensitivity", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def configure_plotting() -> None:
    for path in (
        "/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyhbd.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/times.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"],
        "font.size": 9.0, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 320,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            FIGURE_DIR / f"{name}.{suffix}", bbox_inches="tight",
            metadata={"Creator": "code/paper_analysis.py", "Title": name},
        )
    plt.close(fig)


def write_table(frame: pd.DataFrame, name: str, column_format: str | None = None) -> None:
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    def escape(value: object) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.6f}"
        else:
            text = str(value)
        replacements = {
            "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        }
        return re.sub(r"[\\&%$#_{}]", lambda match: replacements[match.group(0)], text)

    alignment = column_format or "l" + "r" * (len(frame.columns) - 1)
    lines = [rf"\begin{{tabular}}{{{alignment}}}", r"\toprule"]
    lines.append(" & ".join(escape(column) for column in frame.columns) + r" \\")
    lines.append(r"\midrule")
    lines.extend(" & ".join(escape(value) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    latex = "\n".join(lines)
    (TABLE_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")


def clean_feature_name(name: str) -> str:
    name = name.replace("numeric__", "").replace("categorical__", "")
    replacements = {
        "competition_": "赛事=", "stage_group_": "阶段=", "fan_base_index": "球迷基础",
        "market_value_musd": "市值", "strength_score": "综合实力", "style_attack": "进攻风格",
        "style_defense": "防守风格", "star_index": "明星指数", "rank": "排名",
        "elo": "Elo", "avg_age": "平均年龄", "host_flag": "东道主标记",
        "win_prob": "胜率", "prob_entropy": "概率熵", "p_draw": "平局概率",
        "neutral": "中立场", "weekday_num": "星期", "month": "月份",
        "_mean": "均值", "_min": "最小", "_max": "最大", "_absdiff": "绝对差",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def analyze_problem1(data: dict[str, pd.DataFrame]) -> dict:
    historical = data["historical_matches"].copy()
    train = historical[historical["dataset_split"].eq("train") & historical["tv_viewers"].notna()].copy()
    test = historical[historical["dataset_split"].eq("test")].copy()
    features = pc.build_prediction_features(train, data["teams"])
    target_log = np.log1p(train["tv_viewers"].to_numpy(float) / 1_000_000)
    order = np.argsort(pd.to_datetime(train["date"]).to_numpy())
    folds = []
    for fold_no, fraction in enumerate((0.60, 0.70, 0.80), start=1):
        cut = int(len(order) * fraction)
        end = min(len(order), cut + max(30, int(len(order) * 0.10)))
        folds.append((fold_no, order[:cut], order[cut:end]))

    metric_rows, elastic_rows = [], []
    oof_rows = []
    for model_name in ("mean", "elastic_net", "extra_trees"):
        for fold_no, train_idx, valid_idx in folds:
            model = pc.make_p1_model(model_name)
            model.fit(features.iloc[train_idx], target_log[train_idx])
            prediction = np.expm1(model.predict(features.iloc[valid_idx])).clip(min=0)
            truth = np.expm1(target_log[valid_idx])
            mse = mean_squared_error(truth, prediction)
            row = {
                "model": model_name, "fold": fold_no,
                "train_n": len(train_idx), "valid_n": len(valid_idx),
                "train_start": pd.to_datetime(train.iloc[train_idx]["date"]).min().date().isoformat(),
                "train_end": pd.to_datetime(train.iloc[train_idx]["date"]).max().date().isoformat(),
                "valid_start": pd.to_datetime(train.iloc[valid_idx]["date"]).min().date().isoformat(),
                "valid_end": pd.to_datetime(train.iloc[valid_idx]["date"]).max().date().isoformat(),
                "mse": float(mse), "rmse": float(math.sqrt(mse)),
                "mae": float(mean_absolute_error(truth, prediction)),
            }
            metric_rows.append(row)
            if model_name == "elastic_net":
                elastic_rows.append(row)
                oof_rows.extend({
                    "fold": fold_no, "match_id": train.iloc[idx]["match_id"],
                    "actual_million": float(actual), "predicted_million": float(pred),
                    "residual_million": float(actual - pred),
                } for idx, actual, pred in zip(valid_idx, truth, prediction))

    final_model = pc.make_p1_model("elastic_net")
    final_model.fit(features, target_log)
    names = final_model.named_steps["preprocess"].get_feature_names_out()
    coefficients = final_model.named_steps["regressor"].coef_
    coefficient_table = pd.DataFrame({
        "feature_raw": names, "feature": [clean_feature_name(name) for name in names],
        "coefficient": coefficients, "absolute_coefficient": np.abs(coefficients),
    }).sort_values("absolute_coefficient", ascending=False)

    future_raw = data["groups_matches"].merge(
        data["base_predictions"].drop(columns=["group_id", "round_in_group", "team_a", "team_b"], errors="ignore"),
        on="match_id", how="left",
    )
    future_features = pc.build_prediction_features(future_raw, data["teams"])
    train_competitions = sorted(features["competition"].astype(str).unique())
    future_competitions = sorted(future_features["competition"].astype(str).unique())
    transform = final_model.named_steps["preprocess"]
    transformed_train = transform.transform(features)
    transformed_future = transform.transform(future_features)
    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()
        transformed_future = transformed_future.toarray()
    mean_shift = (np.asarray(transformed_future).mean(axis=0) - np.asarray(transformed_train).mean(axis=0)) * coefficients
    shift_table = pd.DataFrame({
        "feature": [clean_feature_name(name) for name in names],
        "mean_log_prediction_shift": mean_shift,
    }).assign(abs_shift=lambda x: x["mean_log_prediction_shift"].abs()).sort_values("abs_shift", ascending=False)

    metrics = pd.DataFrame(metric_rows)
    elastic = pd.DataFrame(elastic_rows)
    oof = pd.DataFrame(oof_rows)
    write_table(elastic[[
        "fold", "train_n", "valid_n", "train_start", "train_end", "valid_start", "valid_end", "mse", "rmse", "mae",
    ]], "q1_expanding_folds")
    write_table(coefficient_table.head(15)[["feature", "coefficient", "absolute_coefficient"]], "q1_top_coefficients")
    write_table(shift_table.head(10)[["feature", "mean_log_prediction_shift"]], "q1_future_mean_shift")
    oof_summary = {
        "correlation": float(oof[["actual_million", "predicted_million"]].corr().iloc[0, 1]),
        "mean_residual_million": float(oof["residual_million"].mean()),
        "residual_std_million": float(oof["residual_million"].std(ddof=1)),
        "lowest_actual_decile_mean_residual": float(oof.nsmallest(max(1, len(oof) // 10), "actual_million")["residual_million"].mean()),
        "highest_actual_decile_mean_residual": float(oof.nlargest(max(1, len(oof) // 10), "actual_million")["residual_million"].mean()),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    ax = axes[0, 0]
    width = 0.24
    x = np.arange(3)
    for offset, metric, color in [(-width, "rmse", COLORS["blue"]), (0, "mae", COLORS["teal"])]:
        ax.bar(x + offset, elastic[metric], width=width, label=metric.upper(), color=color)
    ax.set_xticks(x - width / 2, [f"第 {i} 折" for i in elastic["fold"]])
    ax.set_ylabel("误差（百万人）")
    ax.set_title("(a) Elastic Net 扩窗验证误差")
    ax.legend(frameon=False)

    top = coefficient_table.head(15).sort_values("coefficient")
    axes[0, 1].barh(top["feature"], top["coefficient"], color=np.where(top["coefficient"] >= 0, COLORS["blue"], COLORS["orange"]))
    axes[0, 1].axvline(0, color="black", linewidth=0.7)
    axes[0, 1].set_xlabel("模型系数")
    axes[0, 1].set_title("(b) 系数绝对值前 15（数值特征已标准化）")

    axes[1, 0].scatter(oof["actual_million"], oof["predicted_million"], s=18, alpha=0.58, color=COLORS["blue"], edgecolors="none")
    low = min(oof["actual_million"].min(), oof["predicted_million"].min())
    high = max(oof["actual_million"].max(), oof["predicted_million"].max())
    axes[1, 0].plot([low, high], [low, high], "--", color=COLORS["gray"], linewidth=1)
    axes[1, 0].set(xlabel="真实观看人数（百万人）", ylabel="预测观看人数（百万人）")
    axes[1, 0].set_title(f"(c) 扩窗验证预测（n={len(oof)}）")

    axes[1, 1].scatter(oof["predicted_million"], oof["residual_million"], s=18, alpha=0.58, color=COLORS["teal"], edgecolors="none")
    axes[1, 1].axhline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="预测值（百万人）", ylabel="残差：真实值-预测值")
    axes[1, 1].set_title("(d) 残差与预测值")
    save_figure(fig, "fig_q1_validation_coefficients_residuals")

    return {
        "fold_metrics": metric_rows,
        "elastic_fold_summary": {
            key: {"mean": float(elastic[key].mean()), "std": float(elastic[key].std(ddof=1)),
                  "min": float(elastic[key].min()), "max": float(elastic[key].max())}
            for key in ("mse", "rmse", "mae")
        },
        "top_coefficients": coefficient_table.head(15).to_dict(orient="records"),
        "future_mean_shift_top": shift_table.head(10).to_dict(orient="records"),
        "competition_categories": {
            "train": train_competitions, "future72": future_competitions,
            "future_unseen_in_train": sorted(set(future_competitions) - set(train_competitions)),
        },
        "competition_normalization": pc.competition_normalization_diagnostics(train, test, future_raw),
        "model_seed": pc.P1_MODEL_SEED,
        "oof_summary": oof_summary,
        "oof_n": len(oof),
    }


def draw_information_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.set_axis_off()
    boxes = [
        (0.02, "问题一\n赛前收视预测", "历史赛前特征\nElastic Net"),
        (0.27, "问题二\n赛程与场馆", "72 场收视预测\n分解 MILP"),
        (0.52, "问题三\n动态资源", "固定第三轮赛程\n第二轮后快照"),
        (0.77, "问题四\n实际赛程比较", "优化赛程\n2022 结构与代理"),
    ]
    for x, title, subtitle in boxes:
        ax.add_patch(plt.Rectangle((x, 0.28), 0.20, 0.48, facecolor="#F8FAFC", edgecolor=COLORS["blue"], linewidth=1.4))
        ax.text(x + 0.10, 0.59, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + 0.10, 0.37, subtitle, ha="center", va="center", fontsize=8, color="#334155")
    for x in (0.22, 0.47, 0.72):
        ax.annotate("", xy=(x + 0.045, 0.52), xytext=(x, 0.52), arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": COLORS["teal"]})
    ax.text(0.5, 0.08, "信息只沿时间向后传递；第三轮 24 场共用开赛前同一截面", ha="center", color="#334155")
    save_figure(fig, "fig_information_flow")


def materialize_schedule(assignments: pd.DataFrame, ctx: pc.P2Context, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    venues = data["venues"].set_index("venue_id")
    slots = data["time_slots"].set_index("slot_id")
    venue_metric = ctx.venue_candidates.set_index(["match_id", "venue_id"])
    schedule = ctx.matches.merge(assignments, on="match_id", how="left")
    rows = []
    for row in schedule.itertuples(index=False):
        venue, slot = venues.loc[row.venue_id], slots.loc[row.slot_id]
        metric = venue_metric.loc[(row.match_id, row.venue_id)]
        utc = pc.utc_timestamp(slot.reference_utc_time)
        local = utc.tz_convert(pc.ZoneInfo(str(venue.timezone)))
        rows.append({
            **row._asdict(), "city": venue.city, "country": venue.country,
            "reference_date": pd.to_datetime(slot.date).date().isoformat(),
            "reference_kickoff_time": str(slot.reference_kickoff_time),
            "utc_datetime": utc.strftime("%Y-%m-%d %H:%M"),
            "local_datetime": local.strftime("%Y-%m-%d %H:%M"),
            "expected_attendance": float(metric.expected_attendance),
        })
    return pd.DataFrame(rows)


def perturbed_weights(indicator: str, factor: float) -> dict[str, float]:
    result = dict(pc.P2_WEIGHTS)
    result[indicator] *= factor
    scale = sum(abs(value) for value in result.values())
    return {key: value / scale for key, value in result.items()}


def weighted_p2_metrics(schedule: pd.DataFrame, ctx: pc.P2Context, data: dict[str, pd.DataFrame], weights: dict[str, float]) -> dict:
    base = pc.p2_metrics(schedule, ctx, data)
    contributions = {key: weights[key] * base["normalized"][key] for key in weights}
    return {**base, "weighted_contributions": contributions, "Z2": float(sum(contributions.values()))}


def p2_model_sizes(ctx: pc.P2Context, data: dict[str, pd.DataFrame], formal: pd.DataFrame) -> dict:
    matches, slots = ctx.matches.reset_index(drop=True), data["time_slots"].reset_index(drop=True)
    slot_constraints = len(matches) + len(slots) + 48 * 2 + 12 * 2 * 4 + 20 + 48 * 2 + 1
    pairs = ctx.venue_candidates[["match_id", "venue_id"]].drop_duplicates().merge(formal[["match_id", "slot_id"]], on="match_id")
    venue_index = data["venues"].set_index("venue_id")
    slot_index = data["time_slots"].set_index("slot_id")
    local_groups, utc_groups = {}, {}
    for k, pair in pairs.iterrows():
        venue = venue_index.loc[pair.venue_id]
        utc = pc.utc_timestamp(slot_index.loc[pair.slot_id, "reference_utc_time"])
        local_day = utc.tz_convert(pc.ZoneInfo(str(venue.timezone))).date().isoformat()
        local_groups.setdefault((pair.venue_id, local_day), []).append(k)
        utc_groups.setdefault((pair.venue_id, utc.isoformat()), []).append(k)
    venue_constraints = 72 + pairs["venue_id"].nunique() + len(local_groups) + sum(len(v) > 1 for v in utc_groups.values()) + 48 * 2
    return {
        "slot": {"variables": len(matches) * len(slots) + 2, "constraints": slot_constraints},
        "venue": {"variables": len(pairs) + 2, "constraints": int(venue_constraints)},
    }


def analyze_problem2(data: dict[str, pd.DataFrame], formal_dir: Path, diagnostics: dict, skip_sensitivity: bool) -> tuple[dict, pc.P2Context]:
    p1 = pd.read_csv(formal_dir / "result_1_match_prediction.csv")
    formal = pd.read_csv(formal_dir / "result_2_group_schedule.csv")
    ctx = pc.build_p2_context(data, p1)
    sizes = p2_model_sizes(ctx, data, formal)
    solver_rows = []
    for key, label in (("slot_subproblem", "时段 MILP"), ("venue_subproblem_given_slots", "条件场馆 MILP")):
        values = diagnostics["problem2_solver"][key]
        size = sizes["slot" if key == "slot_subproblem" else "venue"]
        solver_rows.append({
            "subproblem": label, "variables": size["variables"], "constraints": size["constraints"],
            "runtime_seconds": values["runtime_seconds"], "mip_gap": values["mip_gap"],
            "status": values["interpretation"],
        })
    write_table(pd.DataFrame(solver_rows), "q2_solver_scale")

    formal_index = formal.set_index("match_id")
    sensitivity_rows = [{
        "scenario": "baseline", "indicator": "baseline", "factor": 1.0,
        "Z2_scenario": diagnostics["problem2_total_objective"],
        "Z2_original_weights": diagnostics["problem2_total_objective"], "slot_change_rate": 0.0,
        "venue_change_rate": 0.0, "assignment_change_rate": 0.0,
        **{key: diagnostics["problem2_indicator_details"]["raw"][key] for key in ("T", "B", "C", "D", "F", "R")},
        "all_constraints_passed": True,
    }]
    if not skip_sensitivity:
        for indicator in ("T", "B", "C", "D", "F", "R"):
            for factor in (0.8, 1.2):
                weights = perturbed_weights(indicator, factor)
                slots, slot_diag = pc.solve_p2_slots(ctx, data, weights)
                assignments, venue_diag = pc.solve_p2_venues(ctx, data, slots, weights)
                schedule = materialize_schedule(assignments, ctx, data)
                metrics = weighted_p2_metrics(schedule, ctx, data, weights)
                original_weight_metrics = weighted_p2_metrics(schedule, ctx, data, pc.P2_WEIGHTS)
                audit = pc.p2_hard_constraint_audit(schedule, ctx, data)
                candidate = schedule.set_index("match_id")
                slot_change = float((candidate["slot_id"] != formal_index["slot_id"]).mean())
                venue_change = float((candidate["venue_id"] != formal_index["venue_id"]).mean())
                both = float(((candidate["slot_id"] != formal_index["slot_id"]) | (candidate["venue_id"] != formal_index["venue_id"])).mean())
                sensitivity_rows.append({
                    "scenario": f"{indicator}_{factor:.1f}", "indicator": indicator, "factor": factor,
                    "Z2_scenario": metrics["Z2"], "Z2_original_weights": original_weight_metrics["Z2"],
                    "slot_change_rate": slot_change,
                    "venue_change_rate": venue_change, "assignment_change_rate": both,
                    **{key: metrics["raw"][key] for key in ("T", "B", "C", "D", "F", "R")},
                    "all_constraints_passed": audit["all_passed"],
                    "slot_runtime_seconds": slot_diag["runtime_seconds"], "venue_runtime_seconds": venue_diag["runtime_seconds"],
                })
                print(f"P2 sensitivity {indicator} x {factor:.1f}: Z2={metrics['Z2']:.6f}, changed={both:.3f}", flush=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_table(sensitivity, "q2_weight_sensitivity")

    # Date by UTC kickoff occupancy.
    heat = formal.assign(
        date=pd.to_datetime(formal["utc_datetime"], utc=True).dt.strftime("%m-%d"),
        hour=pd.to_datetime(formal["utc_datetime"], utc=True).dt.strftime("%H:%M"),
    ).pivot_table(index="date", columns="hour", values="match_id", aggfunc="count", fill_value=0)
    heat = heat.reindex(sorted(heat.columns, key=lambda value: int(value[:2]) * 60 + int(value[3:])), axis=1)
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    sns.heatmap(heat, cmap="YlGnBu", linewidths=0.35, annot=True, fmt=".0f", cbar_kws={"label": "比赛场数"}, ax=ax)
    ax.set(xlabel="UTC 开球时刻", ylabel="参考日期", title="72 场比赛的日期—UTC 时段分布")
    save_figure(fig, "fig_q2_schedule_heatmap")

    venue_counts = formal.groupby("venue_id").size().sort_values()
    rests = []
    for team in sorted(set(formal["team_a"]) | set(formal["team_b"])):
        subset = formal[(formal["team_a"].eq(team)) | (formal["team_b"].eq(team))].sort_values("round_in_group")
        rests.extend(pd.to_datetime(subset["utc_datetime"], utc=True).diff().dropna().dt.total_seconds().div(3600).tolist())
    contribution = pd.Series(diagnostics["problem2_indicator_details"]["weighted_contributions"])
    gap = pd.Series(diagnostics["problem2_hard_constraint_audit"]["group_round_gaps_hours"]).sort_values()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    axes[0, 0].barh(venue_counts.index, venue_counts.values, color=COLORS["teal"])
    axes[0, 0].set(xlabel="承办场次", title="(a) 16 个场馆的负荷")
    axes[0, 1].hist(rests, bins=np.arange(60, max(rests) + 25, 24), color=COLORS["blue"], edgecolor="white")
    axes[0, 1].axvline(60, color=COLORS["red"], linestyle="--", label="60 小时下限")
    axes[0, 1].set(xlabel="同队相邻比赛间隔（小时）", ylabel="频数", title=f"(b) 48 队的 96 个相邻比赛间隔（最小 {min(rests):.0f}）")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(contribution.index, contribution.values, color=[COLORS["blue"] if value >= 0 else COLORS["orange"] for value in contribution])
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    axes[1, 0].set(ylabel="对 $Z_2$ 的加权贡献", title="(c) 八项指标的目标贡献")
    axes[1, 1].barh(gap.index, gap.values, color=np.where(gap.values < 100, COLORS["teal"], COLORS["gray"]))
    axes[1, 1].axvline(60, color=COLORS["red"], linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="轮次整体边界间隔（小时）", title="(d) 24 个同组相邻轮次边界")
    save_figure(fig, "fig_q2_load_rest_contributions_gaps")

    if len(sensitivity) > 1:
        plot = sensitivity.iloc[1:].copy()
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
        x = np.arange(len(plot))
        axes[0].bar(x, plot["Z2_original_weights"], color=COLORS["blue"])
        axes[0].axhline(sensitivity.iloc[0]["Z2_original_weights"], color=COLORS["red"], linestyle="--", label="正式方案")
        axes[0].set_xticks(x, plot["scenario"], rotation=45, ha="right")
        axes[0].set(ylabel="按正式权重回评的 $Z_2$", title="(a) 汇总指标稳定性")
        axes[0].legend(frameon=False)
        width = 0.25
        axes[1].bar(x - width, 100 * plot["slot_change_rate"], width, label="时段", color=COLORS["blue"])
        axes[1].bar(x, 100 * plot["venue_change_rate"], width, label="场馆", color=COLORS["teal"])
        axes[1].bar(x + width, 100 * plot["assignment_change_rate"], width, label="完整指派", color=COLORS["orange"])
        axes[1].set_xticks(x, plot["scenario"], rotation=45, ha="right")
        axes[1].set(ylabel="变化比例（%）", title="(b) 具体排程替代性")
        axes[1].legend(frameon=False, ncol=3, fontsize=8)
        save_figure(fig, "fig_q2_weight_sensitivity")

    return {
        "model_sizes": sizes, "solver": solver_rows,
        "venue_match_counts": venue_counts.to_dict(),
        "team_rest_hours": pc.distribution(rests),
        "sensitivity": sensitivity_rows,
    }, ctx


def analyze_problem3(data: dict[str, pd.DataFrame], formal_dir: Path) -> dict:
    p2 = pd.read_csv(formal_dir / "result_2_group_schedule.csv")
    formal = pd.read_csv(formal_dir / "result_3_dynamic_strategy.csv")
    standings = pc.standings_from_live(data)
    static_environment = pc.static_p3_environment(data, p2)
    static_actions, _ = pc.p3_normalize(pc.build_p3_actions(data, static_environment))
    static_decisions, _ = pc.solve_p3_actions(static_actions, data["dynamic_resource_limits"])

    runs = []
    primary = None
    for n_sim in (2000, 5000, 10000, 20000):
        simulation_diag, simulation = pc.simulate_advancement(data, standings, n_sim, pc.P3_CONVERGENCE_SEED)
        environment = pc.dynamic_p3_environment(data, p2, simulation, standings)
        actions, bounds = pc.p3_normalize(pc.build_p3_actions(data, environment))
        dynamic, _ = pc.solve_p3_actions(actions, data["dynamic_resource_limits"])
        static_eval = pc.evaluate_fixed_p3(data, environment, static_decisions, bounds)
        signature = {row.match_id: (int(row.broadcast), int(row.security), int(row.transport), round(float(row.delta), 8)) for row in dynamic.itertuples(index=False)}
        runs.append({
            "simulations": n_sim, "advancement_probs": simulation_diag["advancement_probs"],
            "dynamic_total_net": float(dynamic["net_value"].sum()),
            "static_total_net": float(static_eval["net_value"].sum()),
            "improvement_rate": float((dynamic["net_value"].sum() - static_eval["net_value"].sum()) / abs(static_eval["net_value"].sum())),
            "signature": signature,
        })
        if n_sim == 20000:
            primary = (simulation, environment, dynamic, static_eval)
        print(f"P3 convergence n={n_sim}: dynamic={runs[-1]['dynamic_total_net']:.6f}", flush=True)
    assert primary is not None
    reference = runs[-1]
    for run in runs:
        differences = [abs(run["advancement_probs"][team] - reference["advancement_probs"][team]) for team in reference["advancement_probs"]]
        run["advancement_max_abs_diff_vs_20000"] = float(max(differences))
        run["advancement_mean_abs_diff_vs_20000"] = float(np.mean(differences))
        run["decision_match_rate_vs_20000"] = float(np.mean([run["signature"][mid] == reference["signature"][mid] for mid in reference["signature"]]))

    simulation, environment, dynamic, static_eval = primary
    env_index, dynamic_index, static_index = environment.set_index("match_id"), dynamic.set_index("match_id"), static_eval.set_index("match_id")
    static_decision_index = static_decisions.set_index("match_id")
    formal_index = formal.set_index("match_id")
    case_ids = sorted(formal.loc[
        (formal["dynamic_net_value"] - formal["static_net_value"]).abs().gt(5e-10), "match_id"
    ].astype(str).tolist())
    case_rows = []
    for match_id in case_ids:
        env, dyn, sta = env_index.loc[match_id], dynamic_index.loc[match_id], static_index.loc[match_id]
        pre = static_decision_index.loc[match_id]
        out = formal_index.loc[match_id]
        case_rows.append({
            "match_id": match_id, "match": f"{env.team_a}-{env.team_b}",
            "p_a": env.p_team_a_advance, "p_b": env.p_team_b_advance, "importance": env.Q_i,
            "stakeless": env.stakeless_risk, "collusion": env.collusion_risk,
            "attendance_dynamic": dyn.attendance, "viewers_dynamic": dyn.broadcast_viewers,
            "static_action": f"B{int(pre.broadcast)}/S{int(pre.security)}/T{int(pre.transport)}/{100*pre.delta:.3f}%",
            "dynamic_action": f"B{int(dyn.broadcast)}/S{int(dyn.security)}/T{int(dyn.transport)}/{100*dyn.delta:.3f}%",
            "static_net": sta.net_value, "dynamic_net": dyn.net_value,
            "improvement_rate": out.improvement_rate,
        })
    cases = pd.DataFrame(case_rows)
    write_table(cases, "q3_changed_cases")
    write_table(cases[["match_id", "p_a", "p_b", "importance", "stakeless", "collusion"]].rename(columns={
        "match_id": "比赛", "p_a": "A队晋级", "p_b": "B队晋级", "importance": "重要性",
        "stakeless": "无激励", "collusion": "默契",
    }), "q3_changed_cases_risk")
    write_table(cases[["match_id", "static_action", "dynamic_action", "static_net", "dynamic_net", "improvement_rate"]].rename(columns={
        "match_id": "比赛", "static_action": "静态动作", "dynamic_action": "动态动作",
        "static_net": "静态净值", "dynamic_net": "动态净值", "improvement_rate": "改善率",
    }), "q3_changed_cases_actions")

    limits = data["dynamic_resource_limits"].set_index("reference_date")
    daily_rows = []
    for day, subset in dynamic.groupby("reference_date"):
        limit = limits.loc[day]
        daily_rows.append({
            "date": day,
            "high_broadcast_used": int(subset["broadcast"].ge(3).sum()), "high_broadcast_capacity": int(limit.high_broadcast_capacity),
            "high_security_used": int(subset["security"].ge(3).sum()), "high_security_capacity": int(limit.high_security_capacity),
            "enhanced_transport_used": int(subset["transport"].ge(3).sum()), "enhanced_transport_capacity": int(limit.enhanced_transport_capacity),
            "budget_used": float(subset["resource_cost"].sum()), "budget_capacity": float(limit.daily_resource_budget_index),
        })
    daily = pd.DataFrame(daily_rows).sort_values("date")
    for resource in ("high_broadcast", "high_security", "enhanced_transport"):
        daily[resource + "_binding"] = daily[resource + "_used"].eq(daily[resource + "_capacity"])
    daily["budget_ratio"] = daily["budget_used"] / daily["budget_capacity"]
    daily["budget_near_binding"] = daily["budget_ratio"].ge(0.95)
    write_table(daily, "q3_daily_resources")
    daily_compact = pd.DataFrame({
        "日期": pd.to_datetime(daily["date"]).dt.strftime("%m-%d"),
        "高转播_用量/容量": daily["high_broadcast_used"].astype(str) + "/" + daily["high_broadcast_capacity"].astype(str),
        "高安保_用量/容量": daily["high_security_used"].astype(str) + "/" + daily["high_security_capacity"].astype(str),
        "强交通_用量/容量": daily["enhanced_transport_used"].astype(str) + "/" + daily["enhanced_transport_capacity"].astype(str),
        "预算使用率%": 100 * daily["budget_ratio"],
    })
    write_table(daily_compact, "q3_daily_resources_compact")
    convergence = pd.DataFrame([{key: value for key, value in run.items() if key not in ("advancement_probs", "signature")} for run in runs])
    write_table(convergence, "q3_monte_carlo_convergence")
    write_table(convergence[[
        "simulations", "advancement_max_abs_diff_vs_20000", "decision_match_rate_vs_20000",
        "dynamic_total_net", "static_total_net", "improvement_rate",
    ]].rename(columns={
        "simulations": "模拟次数", "advancement_max_abs_diff_vs_20000": "晋级概率最大差",
        "decision_match_rate_vs_20000": "动作一致率", "dynamic_total_net": "动态总值",
        "static_total_net": "静态总值", "improvement_rate": "总改善率",
    }), "q3_monte_carlo_convergence_compact")

    risk = environment[["match_id", "p_team_a_advance", "p_team_b_advance", "Q_i", "stakeless_risk", "collusion_risk"]].copy()
    risk = risk.sort_values("Q_i", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    x = np.arange(len(risk))
    axes[0, 0].scatter(x - 0.12, risk["p_team_a_advance"], color=COLORS["blue"], s=22, label="球队 A")
    axes[0, 0].scatter(x + 0.12, risk["p_team_b_advance"], color=COLORS["teal"], s=22, label="球队 B")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_title("双方晋级概率")
    axes[0, 0].set_ylabel("概率")
    axes[0, 0].legend(frameon=False, ncol=2)
    panels = [
        (axes[0, 1], "Q_i", "晋级重要性", COLORS["purple"]),
        (axes[1, 0], "stakeless_risk", "无激励风险", COLORS["orange"]),
        (axes[1, 1], "collusion_risk", "默契风险", COLORS["red"]),
    ]
    for ax, column, label, color in panels:
        ax.scatter(x, risk[column], color=color, s=25)
        ax.vlines(x, 0, risk[column], color=color, alpha=0.35, linewidth=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_title(label)
        ax.set_ylabel("指数/ 概率")
    for ax in axes[1]:
        ax.set_xticks(x, risk["match_id"], rotation=70, fontsize=7)
    fig.suptitle("按晋级重要性排序的 24 场第三轮比赛", y=1.01)
    save_figure(fig, "fig_q3_advancement_and_risk")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1))
    x = np.arange(len(daily)); labels = pd.to_datetime(daily["date"]).dt.strftime("%m-%d")
    width = 0.24
    for offset, resource, label, color in [
        (-width, "high_broadcast", "高转播", COLORS["blue"]),
        (0, "high_security", "高安保", COLORS["orange"]),
        (width, "enhanced_transport", "强化交通", COLORS["teal"]),
    ]:
        axes[0, 0].bar(x + offset, daily[resource + "_used"], width=width, label=label, color=color)
        axes[0, 0].scatter(x + offset, daily[resource + "_capacity"], marker="_", s=180, color="black", linewidth=1.2)
    axes[0, 0].set_xticks(x, labels, rotation=45)
    axes[0, 0].set(ylabel="使用量（横线为容量）", title="(a) 每日三类高级资源")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0, 1].bar(x, 100 * daily["budget_ratio"], color=np.where(daily["budget_near_binding"], COLORS["red"], COLORS["gray"]))
    axes[0, 1].axhline(100, color="black", linestyle="--")
    axes[0, 1].set_xticks(x, labels, rotation=45)
    axes[0, 1].set(ylabel="预算使用率（%）", title="(b) 每日资源预算")
    axes[1, 0].plot(convergence["simulations"], convergence["advancement_max_abs_diff_vs_20000"], "o-", color=COLORS["blue"], label="晋级概率最大绝对差")
    axes[1, 0].plot(convergence["simulations"], 1 - convergence["decision_match_rate_vs_20000"], "s-", color=COLORS["orange"], label="动作不一致比例")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel="联合模拟次数", ylabel="与 20000 次的差异", title="(c) Monte Carlo 收敛")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(convergence["simulations"], convergence["dynamic_total_net"], "o-", label="动态", color=COLORS["blue"])
    axes[1, 1].plot(convergence["simulations"], convergence["static_total_net"], "s-", label="静态固定动作", color=COLORS["gray"])
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(xlabel="联合模拟次数", ylabel="总净值", title="(d) 总净值随模拟次数的变化")
    axes[1, 1].legend(frameon=False)
    save_figure(fig, "fig_q3_daily_resources_and_convergence")

    return {
        "convergence_seed": pc.P3_CONVERGENCE_SEED,
        "convergence_sequence_relation": (
            "same_seed_reinitialized_for_each_sample_size; not strict prefixes because "
            "home/away Poisson arrays are generated in separate calls"
        ),
        "changed_case_ids": case_ids,
        "convergence": [{key: value for key, value in run.items() if key not in ("advancement_probs", "signature")} for run in runs],
        "changed_cases": case_rows,
        "daily_resources": daily_rows,
        "binding_dates": {
            resource: daily.loc[daily[resource + "_binding"], "date"].tolist()
            for resource in ("high_broadcast", "high_security", "enhanced_transport")
        },
        "budget_near_binding_dates": daily.loc[daily["budget_near_binding"], "date"].tolist(),
    }


def describe_mapping(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for column, unit in [
        ("match_mapping_distance", "标准化 Elo 距离"),
        ("venue_capacity_difference", "人"), ("slot_hour_difference", "小时"),
    ]:
        values = frame[column]
        rows.append({
            "mapping": label, "metric": column, "unit": unit,
            "mean": values.mean(), "median": values.median(), "p75": values.quantile(0.75),
            "p90": values.quantile(0.90), "p95": values.quantile(0.95), "max": values.max(),
        })
    return rows


def analyze_problem4(data: dict[str, pd.DataFrame], formal_dir: Path, ctx: pc.P2Context, diagnostics: dict) -> dict:
    actual = pd.read_csv(formal_dir / "actual_schedule_2022.csv")
    p1 = pd.read_csv(formal_dir / "result_1_match_prediction.csv")
    p2 = pd.read_csv(formal_dir / "result_2_group_schedule.csv")
    nearest = pc.map_actual_matches(actual, data, ctx, 0)
    second = pc.map_actual_matches(actual, data, ctx, 1)
    mapping_stats = pd.DataFrame(describe_mapping(nearest, "nearest") + describe_mapping(second, "second_nearest"))
    write_table(mapping_stats, "q4_mapping_quality")
    compare = nearest.merge(second, on="actual_match_id", suffixes=("_nearest", "_second"))
    change_counts = {
        "match": int((compare["mapped_match_id_nearest"] != compare["mapped_match_id_second"]).sum()),
        "venue": int((compare["mapped_venue_id_nearest"] != compare["mapped_venue_id_second"]).sum()),
        "slot": int((compare["mapped_slot_id_nearest"] != compare["mapped_slot_id_second"]).sum()),
    }

    actual_proxy = diagnostics["problem4_proxy"]["actual_nearest"]
    second_proxy = diagnostics["problem4_proxy"]["actual_second_nearest"]
    optimized_proxy = diagnostics["problem4_proxy"]["optimized"]
    rng = np.random.default_rng(pc.P4_WEIGHT_SENSITIVITY_SEED)
    weight_rows = []
    for run in range(2000):
        multipliers = rng.uniform(0.8, 1.2, len(pc.P2_WEIGHTS))
        weights = {key: value * multipliers[i] for i, (key, value) in enumerate(pc.P2_WEIGHTS.items())}
        scale = sum(abs(value) for value in weights.values())
        weights = {key: value / scale for key, value in weights.items()}
        actual_z = sum(weights[key] * actual_proxy["normalized"][key] for key in weights)
        second_z = sum(weights[key] * second_proxy["normalized"][key] for key in weights)
        optimized_z = sum(weights[key] * optimized_proxy["normalized"][key] for key in weights)
        weight_rows.append({
            "run": run, "actual_Z2": actual_z, "second_Z2": second_z, "optimized_Z2": optimized_z,
            "gap_nearest": optimized_z - actual_z, "gap_second": optimized_z - second_z,
            "reversal_nearest": optimized_z < actual_z, "reversal_second": optimized_z < second_z,
        })
    weight_sensitivity = pd.DataFrame(weight_rows)
    weight_summary = pd.DataFrame([{
        "runs": len(weight_sensitivity),
        "nearest_gap_min": weight_sensitivity["gap_nearest"].min(),
        "nearest_gap_median": weight_sensitivity["gap_nearest"].median(),
        "nearest_gap_max": weight_sensitivity["gap_nearest"].max(),
        "nearest_reversal_count": int(weight_sensitivity["reversal_nearest"].sum()),
        "second_gap_min": weight_sensitivity["gap_second"].min(),
        "second_gap_median": weight_sensitivity["gap_second"].median(),
        "second_gap_max": weight_sensitivity["gap_second"].max(),
        "second_reversal_count": int(weight_sensitivity["reversal_second"].sum()),
    }])
    write_table(weight_summary, "q4_weight_sensitivity")

    structure = diagnostics["problem4_structure"]
    selected_structure = [
        ("avg_travel_per_team_km", "队均旅行"), ("minimum_rest_hours", "最小休息"),
        ("rest_hours_std", "休息标准差"), ("venue_active_day_utilization", "场馆活跃日利用率"),
        ("prime_count_range", "黄金时段极差"),
    ]
    auxiliary = [
        ("prime_time_share", "黄金时段占比"), ("capacity_match_ratio", "容量匹配率"),
        ("expected_attendance_per_match", "题内预计场均观众"),
    ]
    proxy_keys = list(pc.P2_WEIGHTS)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1))
    labels = [label for _, label in selected_structure]
    actual_structure_values = np.asarray([structure["actual"][key] for key, _ in selected_structure])
    optimized_structure_values = np.asarray([structure["optimized"][key] for key, _ in selected_structure])
    structure_max = np.maximum(actual_structure_values, optimized_structure_values)
    y = np.arange(len(selected_structure))
    axes[0].scatter(actual_structure_values / structure_max, y, label="2022 实际", color=COLORS["gray"])
    axes[0].scatter(optimized_structure_values / structure_max, y, label="优化赛程", color=COLORS["teal"])
    axes[0].set_yticks(y, labels)
    axes[0].set(xlabel="组内相对值（仅供作图）", title="(a) 直接结构事实")
    axes[0].legend(frameon=False)
    aux_labels = [label for _, label in auxiliary]
    actual_aux = [structure["actual"][key] for key, _ in auxiliary]
    optimized_aux = [structure["optimized"][key] for key, _ in auxiliary]
    # Each auxiliary is scaled by its pair maximum only for display.
    pair_max = np.maximum(actual_aux, optimized_aux)
    y = np.arange(len(auxiliary))
    axes[1].scatter(np.asarray(actual_aux) / pair_max, y, label="2022 映射", color=COLORS["gray"])
    axes[1].scatter(np.asarray(optimized_aux) / pair_max, y, label="优化赛程", color=COLORS["blue"])
    axes[1].set_yticks(y, aux_labels)
    axes[1].set(xlabel="组内相对值（仅供作图）", title="(b) 映射辅助指标")
    axes[1].legend(frameon=False)
    y = np.arange(len(proxy_keys))
    axes[2].scatter([actual_proxy["normalized"][k] for k in proxy_keys], y, label="2022 最近邻", color=COLORS["gray"])
    axes[2].scatter([optimized_proxy["normalized"][k] for k in proxy_keys], y, label="优化赛程", color=COLORS["blue"])
    axes[2].set_yticks(y, proxy_keys)
    axes[2].set(xlabel="题内统一归一化值", title="(c) 完整题内代理")
    axes[2].legend(frameon=False)
    save_figure(fig, "fig_q4_three_layer_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    axes[0, 0].hist(nearest["match_mapping_distance"], bins=12, color=COLORS["blue"], edgecolor="white")
    axes[0, 0].set(xlabel="标准化 Elo 映射距离", ylabel="场次", title="(a) 比赛最近邻距离")
    axes[0, 1].hist(nearest["venue_capacity_difference"], bins=12, color=COLORS["teal"], edgecolor="white")
    axes[0, 1].set(xlabel="场馆容量绝对差（人）", ylabel="场次", title="(b) 场馆容量映射")
    axes[1, 0].hist(nearest["slot_hour_difference"], bins=np.arange(-0.25, nearest["slot_hour_difference"].max() + 0.75, 0.5), color=COLORS["orange"], edgecolor="white")
    axes[1, 0].set(xlabel="UTC 循环小时差", ylabel="场次", title="(c) 开球时刻映射")
    actual_contrib = pd.Series(actual_proxy["weighted_contributions"])
    optimized_contrib = pd.Series(optimized_proxy["weighted_contributions"])
    x = np.arange(len(proxy_keys)); width = 0.36
    axes[1, 1].bar(x - width / 2, actual_contrib[proxy_keys], width, label="2022 最近邻", color=COLORS["gray"])
    axes[1, 1].bar(x + width / 2, optimized_contrib[proxy_keys], width, label="优化赛程", color=COLORS["blue"])
    axes[1, 1].axhline(0, color="black", linewidth=0.7)
    axes[1, 1].set_xticks(x, proxy_keys)
    axes[1, 1].set(ylabel="加权贡献", title="(d) 代理综合值的贡献来源")
    axes[1, 1].legend(frameon=False)
    save_figure(fig, "fig_q4_mapping_quality_and_contributions")

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.hist(weight_sensitivity["gap_nearest"], bins=30, color=COLORS["purple"], alpha=0.72, edgecolor="white", label="最近邻")
    ax.hist(weight_sensitivity["gap_second"], bins=30, color=COLORS["teal"], alpha=0.58, edgecolor="white", label="第二近邻")
    ax.axvline(0, color=COLORS["red"], linestyle="--", label="方向反转界线")
    ax.set(xlabel="$Z_2^{opt}-Z_2^{2022}$", ylabel="权重扰动样本数", title="2000 组±20% 权重扰动下的代理综合值差")
    ax.legend(frameon=False)
    save_figure(fig, "fig_q4_weight_sensitivity")

    return {
        "weight_sensitivity_seed": pc.P4_WEIGHT_SENSITIVITY_SEED,
        "weight_sensitivity_scope": "fixed schedules, mappings and indicators; re-evaluation only",
        "mapping_quality": mapping_stats.to_dict(orient="records"),
        "nearest_second_change_counts": change_counts,
        "weight_sensitivity": weight_summary.iloc[0].to_dict(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    global FIGURE_DIR, TABLE_DIR
    args = parse_args()
    formal_dir, generated_dir = resolve(args.output_dir), resolve(args.generated_dir)
    FIGURE_DIR, TABLE_DIR = generated_dir / "figures", generated_dir / "tables"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    data = pc.read_data()
    diagnostics = json.loads((formal_dir / "diagnostics.json").read_text(encoding="utf-8"))

    draw_information_flow()
    results = {
        "random_seeds": pc.RANDOM_SEEDS,
        "problem1": analyze_problem1(data),
    }
    results["problem2"], context = analyze_problem2(data, formal_dir, diagnostics, args.skip_p2_sensitivity)
    results["problem3"] = analyze_problem3(data, formal_dir)
    results["problem4"] = analyze_problem4(data, formal_dir, context, diagnostics)

    generated_files = sorted(path for path in generated_dir.rglob("*") if path.is_file())
    results["traceability"] = {
        "formal_output_directory": str(formal_dir),
        "formal_diagnostics_sha256": sha256(formal_dir / "diagnostics.json"),
        "formal_csv_sha256": {path.name: sha256(path) for path in sorted(formal_dir.glob("*.csv"))},
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated_files": [str(path.relative_to(ROOT)) for path in generated_files],
    }
    (generated_dir / "paper_analysis_results.json").write_text(
        json.dumps(
            results, ensure_ascii=False, indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    print(f"Paper analyses written to {generated_dir}")


if __name__ == "__main__":
    main()
