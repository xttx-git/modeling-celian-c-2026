"""Recipe-adapted publication plots for the CUMCM C results.

The twenty tiny ``gen_fig_*.py`` entry points call the functions in this module.
All numerical content is read from ``figures/_plot_data.json``.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from scipy.stats import gaussian_kde

import _utils.plot_utils as pu

pu.setup_style("nature")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "figures" / "_plot_data.json"

FEATURE_ZH = {
    "competition": "赛事类别",
    "stage": "赛事阶段",
    "timezone_pair": "时区组合",
    "strength_rank_mean": "平均实力排名",
    "strength_rank_gap": "实力排名差",
    "elo_mean": "平均Elo",
    "elo_gap": "Elo差",
    "p_a": "A队胜率",
    "p_draw": "平局概率",
    "p_b": "B队胜率",
    "odds_entropy": "赔率熵",
    "market_value_musd_mean": "平均身价",
    "market_value_musd_gap": "身价差",
    "avg_age_mean": "平均年龄",
    "avg_age_gap": "年龄差",
    "star_index_mean": "平均球星指数",
    "star_index_gap": "球星指数差",
    "fan_base_index_mean": "平均球迷基础",
    "fan_base_index_gap": "球迷基础差",
    "style_attack_mean": "平均进攻风格",
    "style_attack_gap": "进攻风格差",
    "style_defense_mean": "平均防守风格",
    "style_defense_gap": "防守风格差",
    "host_flag_mean": "东道主比例",
    "host_flag_gap": "东道主差异",
    "strength_score_mean": "平均实力得分",
    "strength_score_gap": "实力得分差",
    "worldcup_stage": "世界杯标记",
    "fan_stage_interaction": "球迷基础-世界杯交互",
    "star_stage_interaction": "球星-世界杯交互",
    "strength_entropy_interaction": "实力-赔率熵交互",
}
METRIC_ZH = {
    "T": "票务价值T", "B": "转播价值B", "U": "不确定性U", "H": "吸引力H",
    "C": "组织成本C", "D": "旅行负担D", "F": "公平惩罚F", "R": "执行风险R",
    "rest_min_hours": "最小休息", "rest_mean_hours": "平均休息",
    "timezone_crossing_rate": "跨时区率", "travel_mean_km": "场馆旅行",
    "venue_change_rate": "换馆率", "venue_utilization_rate": "场馆日利用率",
    "venue_daily_peak": "场馆日峰值", "capacity_occupancy_rate": "容量匹配",
    "prime_coverage_rate": "黄金覆盖", "expected_attendance_per_match": "场均观众",
}


def _data():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _output(name: str):
    return str(ROOT / "figures" / f"{name}.pdf")


def _seq_cmap():
    return LinearSegmentedColormap.from_list(
        "nature_sequence", [pu.COLORS["light"], pu.PALETTE[1], pu.PALETTE[0]]
    )


def _panel_label(ax, label):
    ax.set_title(label, loc="left", pad=3, fontsize=10, fontweight="bold")


def fig_q1_target_raincloud(d):
    values = np.asarray(d["q1"]["target_million"], float)
    rng = np.random.default_rng(int(d["metadata"]["seed"]))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    parts = ax.violinplot(values, positions=[0.15], vert=False, widths=0.6,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        vertices = body.get_paths()[0].vertices
        vertices[:, 1] = np.maximum(vertices[:, 1], 0.15)
        body.set_facecolor(pu.PALETTE[0])
        body.set_edgecolor(pu.PALETTE[0])
        body.set_alpha(0.30)
        body.set_linewidth(1.2)
    jitter = -0.18 + rng.normal(0, 0.025, len(values))
    ax.scatter(values, jitter, s=10, color=pu.PALETTE[1], alpha=0.34,
               edgecolor=pu.COLORS["light"], linewidth=0.25, rasterized=True)
    ax.boxplot(values, positions=[-0.02], vert=False, widths=0.12, patch_artist=True,
               showfliers=False,
               boxprops={"facecolor": pu.COLORS["light"], "edgecolor": pu.PALETTE[0], "linewidth": 1.2},
               whiskerprops={"color": pu.PALETTE[0]}, capprops={"color": pu.PALETTE[0]},
               medianprops={"color": pu.PALETTE[3], "linewidth": 1.6})
    mean = float(values.mean())
    ax.scatter([mean], [-0.02], marker="D", s=46, color=pu.PALETTE[3],
               edgecolor=pu.COLORS["light"], linewidth=0.7, zorder=5)
    ax.annotate(f"均值 {mean:.1f}", xy=(mean, -0.02), xytext=(8, 16),
                textcoords="offset points", fontsize=8, color=pu.COLORS["text"],
                arrowprops={"arrowstyle": "-", "color": pu.COLORS["ref_line"], "lw": 0.8})
    ax.set_xlabel("观看人数（百万人）")
    ax.set_yticks([])
    ax.set_ylim(-0.34, 0.52)
    ax.set_xlim(values.min() - 5, values.max() + 5)
    pu.save_fig(fig, _output("fig_q1_target_raincloud"))


def fig_q1_feature_pairplot(d):
    frame = pd.DataFrame(d["q1"]["pairplot"])
    cols = list(frame.columns)
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(8, 8), squeeze=False)
    cmap = _seq_cmap()
    for i, ycol in enumerate(cols):
        for j, xcol in enumerate(cols):
            ax = axes[i, j]
            x = frame[xcol].to_numpy(float)
            y = frame[ycol].to_numpy(float)
            if i == j:
                ax.hist(x, bins=22, density=True, color=pu.PALETTE[i % len(pu.PALETTE)],
                        alpha=0.35, edgecolor=pu.COLORS["light"], linewidth=0.4)
                if np.std(x) > 1e-12:
                    grid = np.linspace(x.min(), x.max(), 160)
                    kde = gaussian_kde(x)
                    ax.plot(grid, kde(grid), color=pu.PALETTE[i % len(pu.PALETTE)], lw=1.1)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(x, y, s=6, color=pu.PALETTE[0], alpha=0.22, edgecolor="none", rasterized=True)
                if np.std(x) > 1e-12:
                    coef = np.polyfit(x, y, 1)
                    grid = np.linspace(x.min(), x.max(), 80)
                    ax.plot(grid, np.polyval(coef, grid), color=pu.PALETTE[1], lw=0.9)
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
                background_score = (corr + 1) / 2
                ax.set_facecolor(cmap(background_score))
                ax.text(0.5, 0.53, f"r={corr:.2f}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, fontweight="bold",
                        color=pu.COLORS["light"] if background_score > 0.52 else pu.COLORS["text"])
                ax.text(0.5, 0.32, "正相关" if corr >= 0 else "负相关", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6.5,
                        color=pu.COLORS["light"] if background_score > 0.52 else pu.COLORS["text"])
                ax.set_xticks([]); ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(xcol, fontsize=7.5)
                ax.tick_params(axis="x", labelsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ycol, fontsize=7.5)
                ax.tick_params(axis="y", labelsize=6)
            else:
                ax.set_yticklabels([])
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    pu.save_fig(fig, _output("fig_q1_feature_pairplot"))


def _violin_panel(ax, labels, values, max_groups=6):
    counts = Counter(labels)
    chosen = [name for name, _ in counts.most_common(max_groups)]
    data = [np.asarray([v for label, v in zip(labels, values) if label == name], float) for name in chosen]
    positions = np.arange(len(chosen))
    parts = ax.violinplot(data, positions=positions, widths=0.82, showmeans=False,
                          showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(pu.PALETTE[i % len(pu.PALETTE)])
        body.set_edgecolor(pu.PALETTE[i % len(pu.PALETTE)])
        body.set_alpha(0.30)
        body.set_linewidth(1.1)
    for i, arr in enumerate(data):
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        ax.vlines(i, q1, q3, color=pu.PALETTE[i % len(pu.PALETTE)], lw=3.5, zorder=4)
        ax.scatter([i], [med], s=25, color=pu.COLORS["text"], edgecolor=pu.COLORS["light"], zorder=5)
    return chosen


def fig_q1_stage_timezone_violin(d):
    values = np.asarray(d["q1"]["target_million"], float)
    stages = d["q1"]["stage"]
    zones = [z.replace("|", "/") for z in d["q1"]["timezone_pair"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    stage_labels = _violin_panel(axes[0], stages, values, 6)
    zone_labels = _violin_panel(axes[1], zones, values, 6)
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    axes[0].set_xticks(range(len(stage_labels)), stage_labels, rotation=25, ha="right")
    axes[1].set_xticks(range(len(zone_labels)), zone_labels, rotation=25, ha="right")
    axes[0].set_ylabel("观看人数（百万人）")
    axes[0].set_xlabel("赛事阶段（按样本量前6类）")
    axes[1].set_xlabel("主队/客队时区组合（按样本量前6类）")
    pu.save_fig(fig, _output("fig_q1_stage_timezone_violin"))


def fig_q1_model_accuracy_heatmap(d):
    rows = d["q1"]["fold_metrics"]
    models = sorted({r["model"] for r in rows}, key=lambda x: ["MeanBaseline", "ElasticNet", "ExtraTrees", "CatBoost"].index(x))
    summary = {r["model"]: r for r in d["q1"]["model_summary"]}
    labels = [f"折{i} MSE" for i in range(1, 6)] + ["平均RMSE", "平均MAE"]
    values = np.zeros((len(models), len(labels)))
    for i, model in enumerate(models):
        by_fold = {int(r["fold"]): r for r in rows if r["model"] == model}
        values[i, :5] = [by_fold[k]["mse"] for k in range(1, 6)]
        values[i, 5] = summary[model]["mean_rmse"]
        values[i, 6] = summary[model]["mean_mae"]
    scores = np.zeros_like(values)
    for j in range(values.shape[1]):
        lo, hi = values[:, j].min(), values[:, j].max()
        scores[:, j] = 1 - (values[:, j] - lo) / max(hi - lo, 1e-12)
    display_models = {"MeanBaseline": "均值", "ElasticNet": "弹性网", "ExtraTrees": "极端树", "CatBoost": "提升树"}
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    image = ax.imshow(scores, cmap=_seq_cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = pu.COLORS["light"] if scores[i, j] > 0.63 else pu.COLORS["text"]
            text = f"{values[i, j]:.1f}" if j < 5 else f"{values[i, j]:.2f}"
            weight = "bold" if values[i, j] == values[:, j].min() else "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color, fontweight=weight)
    selected = d["q1"]["selected_model"]
    selected_i = models.index(selected)
    ax.add_patch(patches.Rectangle((-0.49, selected_i - 0.49), len(labels) - 0.02, 0.98,
                                   fill=False, edgecolor=pu.PALETTE[3], linewidth=1.8))
    ax.set_xticks(range(len(labels)), labels, rotation=22, ha="right")
    ax.set_yticks(range(len(models)), [display_models[m] for m in models])
    ax.set_xlabel("时序验证误差指标（颜色越深表示列内相对越优）", labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.label.set_size(8)
    ax.set_ylabel("")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_title("相对得分", fontsize=7, pad=3)
    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.24, top=0.96)
    pu.save_fig(fig, _output("fig_q1_model_accuracy_heatmap"))


def fig_q1_prediction_fit(d):
    oof = d["q1"]["oof"]
    actual = np.asarray(oof["actual_million"], float)
    predicted = np.asarray(oof["predicted_million"], float)
    residual = actual - predicted
    r2 = 1 - np.sum(residual ** 2) / np.sum((actual - actual.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.scatter(actual, predicted, s=18, color=pu.PALETTE[0], alpha=0.50,
               edgecolor=pu.COLORS["light"], linewidth=0.35, rasterized=True)
    lo = min(actual.min(), predicted.min()); hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], color=pu.COLORS["ref_line"], lw=1.2, linestyle="--", label="45度参考线")
    ax.set_xlim(lo - 4, hi + 4); ax.set_ylim(lo - 4, hi + 4)
    ax.set_xlabel("真实观看人数（百万人）"); ax.set_ylabel("预测观看人数（百万人）")
    ax.text(0.05, 0.92, f"$R^2$={r2:.3f}\nRMSE={rmse:.2f} 百万", transform=ax.transAxes,
            va="top", fontsize=8.5, color=pu.COLORS["text"],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": pu.COLORS["bg_box"], "edgecolor": pu.PALETTE[0], "alpha": 0.9})
    threshold = np.quantile(actual, 0.82)
    inset = ax.inset_axes([0.57, 0.08, 0.38, 0.35])
    mask = actual >= threshold
    inset.scatter(actual[mask], predicted[mask], s=12, color=pu.PALETTE[1], alpha=0.65, edgecolor="none")
    inset.plot([actual[mask].min(), actual[mask].max()], [actual[mask].min(), actual[mask].max()],
               color=pu.COLORS["ref_line"], lw=0.8, linestyle="--")
    inset.tick_params(labelsize=5.5)
    inset.set_xlabel("高观看区真实值", fontsize=6)
    inset.set_ylabel("预测值", fontsize=6)
    _panel_label(ax, "(a)")

    ax = axes[1]
    ax.hist(residual, bins=22, density=True, color=pu.PALETTE[1], alpha=0.32,
            edgecolor=pu.PALETTE[1], linewidth=0.6)
    grid = np.linspace(residual.min(), residual.max(), 220)
    kde = gaussian_kde(residual)
    ax.fill_between(grid, kde(grid), color=pu.PALETTE[0], alpha=0.18)
    ax.plot(grid, kde(grid), color=pu.PALETTE[0], lw=1.4, label="核密度")
    ax.axvline(0, color=pu.COLORS["ref_line"], linestyle="--", lw=1.0)
    ax.set_xlabel("残差（百万人）"); ax.set_ylabel("概率密度")
    ax.legend(loc="best")
    _panel_label(ax, "(b)")
    pu.save_fig(fig, _output("fig_q1_prediction_fit"))


def fig_q1_residual_diagnostics(d):
    oof = d["q1"]["oof"]
    fitted = np.asarray(oof["predicted_million"], float)
    residual = np.asarray(oof["residual_million"], float)
    dates = pd.to_datetime(oof["date"], utc=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    axes[0, 0].scatter(fitted, residual, s=13, color=pu.PALETTE[0], alpha=0.45, edgecolor="none", rasterized=True)
    coef = np.polyfit(fitted, residual, 2)
    grid = np.linspace(fitted.min(), fitted.max(), 160)
    axes[0, 0].plot(grid, np.polyval(coef, grid), color=pu.PALETTE[1], lw=1.3)
    axes[0, 0].axhline(0, color=pu.COLORS["ref_line"], linestyle="--", lw=0.9)
    axes[0, 0].set_xlabel("拟合值（百万人）"); axes[0, 0].set_ylabel("残差（百万人）")

    theoretical, ordered = stats.probplot(residual, dist="norm", fit=False)
    axes[0, 1].scatter(theoretical, ordered, s=13, color=pu.PALETTE[2], alpha=0.55, edgecolor="none")
    line = np.polyfit(theoretical, ordered, 1)
    axes[0, 1].plot(theoretical, np.polyval(line, theoretical), color=pu.PALETTE[3], lw=1.1)
    axes[0, 1].set_xlabel("理论正态分位数"); axes[0, 1].set_ylabel("样本残差分位数（百万人）")

    axes[1, 0].hist(residual, bins=24, density=True, color=pu.PALETTE[1], alpha=0.33,
                    edgecolor=pu.PALETTE[1], linewidth=0.5)
    xgrid = np.linspace(residual.min(), residual.max(), 200)
    axes[1, 0].plot(xgrid, stats.norm.pdf(xgrid, residual.mean(), residual.std(ddof=1)),
                    color=pu.PALETTE[3], lw=1.3)
    axes[1, 0].set_xlabel("残差（百万人）"); axes[1, 0].set_ylabel("概率密度")

    axes[1, 1].plot(dates, residual, color=pu.PALETTE[0], lw=0.75, alpha=0.65)
    axes[1, 1].scatter(dates, residual, s=8, color=pu.PALETTE[0], alpha=0.42, edgecolor="none")
    axes[1, 1].axhline(0, color=pu.COLORS["ref_line"], linestyle="--", lw=0.9)
    axes[1, 1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1, 1].set_xlabel("验证样本比赛日期"); axes[1, 1].set_ylabel("残差（百万人）")
    for i, ax in enumerate(axes.flat):
        _panel_label(ax, f"({chr(97 + i)})")
    pu.save_fig(fig, _output("fig_q1_residual_diagnostics"))


def fig_q1_shap_summary(d):
    rows = d["q1"]["importance"]
    if not rows:
        raise RuntimeError("No permutation importance data available")
    rows = sorted(rows, key=lambda row: row["delta_mse"])
    bar_rows = rows[-12:]
    swarm_rows = bar_rows[-8:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [0.9, 1.35]})
    labels = [FEATURE_ZH.get(row["feature"], row["feature"]) for row in bar_rows]
    values = np.asarray([row["delta_mse"] for row in bar_rows], float)
    colors = [pu.PALETTE[(i + 1) % len(pu.PALETTE)] for i in range(len(values))]
    bars = axes[0].barh(np.arange(len(values)), values, color=colors, alpha=0.30,
                        edgecolor=colors, linewidth=1.1)
    for bar, value in zip(bars, values):
        axes[0].text(max(value, 0) + max(values.max(), 1e-6) * 0.02,
                     bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                     va="center", fontsize=7, color=pu.COLORS["text"])
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlabel("置换重要性（验证集MSE增量）")
    _panel_label(axes[0], "(a)")

    rng = np.random.default_rng(int(d["metadata"]["seed"]))
    for yidx, row in enumerate(swarm_rows):
        effect = np.asarray(row["effect"], float)
        feature_value = np.asarray(row["feature_value"], float)
        order = np.argsort(np.abs(effect))[::-1][:100]
        jitter = rng.normal(0, 0.09, len(order))
        axes[1].scatter(effect[order], yidx + jitter, c=feature_value[order], cmap="coolwarm",
                        vmin=0, vmax=1, s=13, alpha=0.62,
                        edgecolor=pu.COLORS["light"], linewidth=0.2, rasterized=True)
    axes[1].axvline(0, color=pu.COLORS["ref_line"], linestyle="--", lw=0.9)
    axes[1].set_yticks(range(len(swarm_rows)), [FEATURE_ZH.get(r["feature"], r["feature"]) for r in swarm_rows])
    axes[1].set_xlabel("单样本置换预测影响（百万人）")
    _panel_label(axes[1], "(b)")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=pu.PALETTE[0], label="特征值较低", markersize=5),
        Line2D([0], [0], marker="o", linestyle="none", color=pu.PALETTE[1], label="特征值较高", markersize=5),
    ]
    axes[1].legend(handles=legend_handles, loc="best", frameon=True)
    pu.save_fig(fig, _output("fig_q1_shap_summary"))


def fig_q2_schedule_gantt(d):
    frame = pd.DataFrame(d["q2"]["schedule"])
    frame["utc"] = pd.to_datetime(frame["utc_datetime"], utc=True).dt.tz_convert(None)
    venues = sorted(frame["venue_id"].unique())
    ypos = {venue: i for i, venue in enumerate(venues)}
    fig = plt.figure(figsize=(11, 6.8))
    layout = fig.add_gridspec(1, 2, width_ratios=[18, 2], wspace=0.03)
    ax = fig.add_subplot(layout[0, 0])
    legend_ax = fig.add_subplot(layout[0, 1])
    for row in frame.itertuples(index=False):
        start = mdates.date2num(row.utc.to_pydatetime())
        color = pu.PALETTE[(int(row.round_in_group) - 1) % len(pu.PALETTE)]
        ax.barh(ypos[row.venue_id], 2 / 24, left=start, height=0.55,
                color=color, alpha=0.42, edgecolor=color, linewidth=0.8)
    ax.set_yticks(range(len(venues)), venues)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel("比赛开球时间（UTC，2026年）")
    ax.set_ylabel("场馆编号")
    handles = [patches.Patch(facecolor=pu.PALETTE[i], edgecolor=pu.PALETTE[i], alpha=0.42, label=f"小组第{i + 1}轮") for i in range(3)]
    legend_ax.legend(handles=handles, loc="center", frameon=True, fontsize=8)
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    pu.save_fig(fig, _output("fig_q2_schedule_gantt"))


def fig_q2_venue_travel_map(d):
    venues = pd.DataFrame(d["q2"]["venues"])
    edges = sorted(d["q2"]["travel_edges"], key=lambda row: row["count"])
    loads = d["q2"]["venue_load"]
    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    ax.set_facecolor(pu.COLORS["bg_box"])
    for edge in edges:
        ax.plot([edge["source_lon"], edge["target_lon"]], [edge["source_lat"], edge["target_lat"]],
                color=pu.PALETTE[1], alpha=0.12 + 0.08 * edge["count"],
                lw=0.45 + 0.55 * edge["count"], solid_capstyle="round", zorder=1)
    sizes = np.asarray([loads[str(v)] for v in venues["venue_id"]], float)
    ax.scatter(venues["longitude"], venues["latitude"], s=45 + 18 * sizes,
               color=pu.PALETTE[0], alpha=0.78, edgecolor=pu.COLORS["light"], linewidth=0.9, zorder=3)
    for row in venues.itertuples(index=False):
        ax.annotate(str(row.venue_id), (row.longitude, row.latitude), xytext=(4, 4),
                    textcoords="offset points", fontsize=6.5, color=pu.COLORS["text"], zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": pu.COLORS["light"], "edgecolor": pu.COLORS["grid"], "alpha": 0.85})
    ax.text(-127, 51.0, "加拿大", color=pu.COLORS["ref_line"], fontsize=9)
    ax.text(-108, 31.0, "墨西哥", color=pu.COLORS["ref_line"], fontsize=9)
    ax.text(-101, 42.5, "美国", color=pu.COLORS["ref_line"], fontsize=10)
    ax.set_xlim(-130, -70); ax.set_ylim(18, 56)
    ax.set_xlabel("经度（度）"); ax.set_ylabel("纬度（度）")
    ax.text(0.02, 0.04, "节点面积：承办场次；连线宽度：球队跨轮移动次数",
            transform=ax.transAxes, fontsize=7.5, color=pu.COLORS["text"],
            bbox={"boxstyle": "round,pad=0.3", "facecolor": pu.COLORS["light"], "edgecolor": pu.PALETTE[0], "alpha": 0.9})
    pu.save_fig(fig, _output("fig_q2_venue_travel_map"))


def fig_q2_venue_slot_matrix(d):
    frame = pd.DataFrame(d["q2"]["schedule"])
    dates = sorted(frame["reference_date"].astype(str).unique())
    venues = sorted(frame["venue_id"].astype(str).unique())
    matrix = np.zeros((len(venues), len(dates)), int)
    vi = {v: i for i, v in enumerate(venues)}; di = {date: i for i, date in enumerate(dates)}
    for row in frame.itertuples(index=False):
        matrix[vi[str(row.venue_id)], di[str(row.reference_date)]] += 1
    fig, ax = plt.subplots(figsize=(11, 5.8))
    image = ax.imshow(matrix, cmap=_seq_cmap(), vmin=0, vmax=max(matrix.max(), 1), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                norm = matrix[i, j] / max(matrix.max(), 1)
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6.2,
                        color=pu.COLORS["light"] if norm > 0.6 else pu.COLORS["text"])
    ax.set_yticks(range(len(venues)), venues)
    ax.set_xticks(range(len(dates)), [date[5:] for date in dates], rotation=40, ha="right")
    ax.set_xlabel("参考日期（2026年，月-日）", labelpad=8); ax.set_ylabel("场馆编号")
    ax.xaxis.set_label_position("top")
    ax.xaxis.label.set_size(8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("场次")
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.25, top=0.97)
    pu.save_fig(fig, _output("fig_q2_venue_slot_matrix"))


def fig_q2_objective_waterfall(d):
    rows = d["q2"]["objective_contributions"]
    values = np.asarray([r["value"] for r in rows], float)
    labels = [r["indicator"] for r in rows]
    starts = np.r_[0.0, np.cumsum(values)[:-1]]
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    for i, (row, value, start) in enumerate(zip(rows, values, starts)):
        bottom = min(start, start + value)
        color = pu.PALETTE[2] if value >= 0 else pu.PALETTE[1]
        hatch = "//" if row["kind"] == "固定项" else None
        ax.bar(i, abs(value), bottom=bottom, width=0.68, color=color, alpha=0.30,
               edgecolor=color, linewidth=1.2, hatch=hatch)
        end = start + value
        ax.text(i, end + (0.018 if value >= 0 else -0.025), f"{value:+.3f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=7.2, color=pu.COLORS["text"])
        if i < len(values) - 1:
            ax.plot([i + 0.34, i + 0.66], [end, end], color=pu.COLORS["ref_line"], lw=0.8)
    total = float(d["q2"]["objective_total"])
    ax.bar(len(values), total, width=0.68, color=pu.PALETTE[0], alpha=0.35,
           edgecolor=pu.PALETTE[0], linewidth=1.4)
    ax.text(len(values), total + 0.018, f"{total:.3f}", ha="center", va="bottom", fontsize=8,
            color=pu.COLORS["text"], fontweight="bold")
    ax.axhline(0, color=pu.COLORS["ref_line"], lw=0.9)
    ax.set_xticks(range(len(labels) + 1), labels + ["Z2"])
    ax.set_xlabel("目标函数指标（斜线为赛程固定项）")
    ax.set_ylabel("对目标函数的加权贡献")
    pu.save_fig(fig, _output("fig_q2_objective_waterfall"))


def fig_q2_constraint_margins_lollipop(d):
    rows = sorted(d["q2"]["constraint_margins"], key=lambda row: row["margin_pct"])
    labels = [row["name"] for row in rows]
    values = np.asarray([row["margin_pct"] for row in rows], float)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for yi, value in zip(y, values):
        color = pu.PALETTE[1] if value <= 1e-10 else pu.PALETTE[0]
        ax.hlines(yi, 0, value, color=color, lw=2.1, alpha=0.75)
        ax.scatter(value, yi, s=55 + 2 * value, color=color, edgecolor=pu.COLORS["light"], linewidth=0.8, zorder=3)
        ax.text(value + max(values.max(), 5) * 0.025, yi, "紧约束" if value <= 1e-10 else f"{value:.1f}%",
                va="center", fontsize=7.5, color=pu.COLORS["text"])
    ax.axvline(0, color=pu.COLORS["ref_line"], lw=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("最小归一化可行裕度（%）")
    ax.set_ylabel("约束族")
    ax.set_xlim(-1, max(values.max() * 1.20, 8))
    pu.save_fig(fig, _output("fig_q2_constraint_margins_lollipop"))


def fig_q2_weight_sensitivity_contour(d):
    frame = pd.DataFrame(d["q2"]["weight_surface"])
    xs = sorted(frame["commercial_multiplier"].unique())
    ys = sorted(frame["penalty_multiplier"].unique())
    matrix = frame.pivot(index="penalty_multiplier", columns="commercial_multiplier", values="fixed_schedule_Z2").loc[ys, xs].to_numpy(float)
    xx, yy = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    filled = ax.contourf(xx, yy, matrix, levels=14, cmap="YlOrRd")
    lines = ax.contour(xx, yy, matrix, levels=8, colors=[pu.COLORS["text"]], linewidths=0.55, alpha=0.70)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.4f")
    index = np.unravel_index(np.argmax(matrix), matrix.shape)
    ax.scatter([xx[index]], [yy[index]], marker="*", s=120, color=pu.PALETTE[3],
               edgecolor=pu.COLORS["light"], linewidth=0.7, zorder=4)
    ax.annotate(f"固定方案最高值 {matrix[index]:.4f}", xy=(xx[index], yy[index]), xytext=(-104, 25),
                textcoords="offset points", fontsize=7.5, color=pu.COLORS["text"],
                bbox={"boxstyle": "round,pad=0.25", "facecolor": pu.COLORS["light"], "edgecolor": pu.PALETTE[3], "alpha": 0.88},
                arrowprops={"arrowstyle": "->", "color": pu.PALETTE[3], "lw": 0.8})
    ax.set_xlabel("商业权重倍率", labelpad=-2)
    ax.set_ylabel("惩罚权重倍率")
    cbar = fig.colorbar(filled, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_title("Z2", fontsize=8, pad=3)
    ticks = np.linspace(matrix.min(), matrix.max(), 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{value:.3f}" for value in ticks])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.tick_params(labelsize=7, pad=2)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.17, top=0.97)
    pu.save_fig(fig, _output("fig_q2_weight_sensitivity_contour"))


def fig_q3_advancement_shift(d):
    rows = sorted(d["q3"]["advancement_shift"], key=lambda row: row["updated"] - row["baseline"])
    y = np.arange(len(rows))
    before = np.asarray([row["baseline"] for row in rows], float)
    after = np.asarray([row["updated"] for row in rows], float)
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    changes = np.abs(after - before)
    threshold = np.sort(changes)[-10]
    for yi, left, right, change in zip(y, before, after, changes):
        color = pu.PALETTE[1] if right < left else pu.PALETTE[2]
        ax.plot([left, right], [yi, yi], color=color, lw=0.75 + 1.6 * change / max(changes.max(), 1e-12), alpha=0.65)
    ax.scatter(before, y, s=14, color=pu.PALETTE[0], alpha=0.70, edgecolor=pu.COLORS["light"], linewidth=0.3, label="更新前基准")
    ax.scatter(after, y, s=18, color=pu.PALETTE[1], alpha=0.78, edgecolor=pu.COLORS["light"], linewidth=0.3, label="前两轮更新后")
    ax.set_yticks(y, [row["team"] for row in rows], fontsize=5.6)
    ax.set_xlabel("晋级概率")
    ax.set_ylabel("参赛队")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.text(0.02, 0.02, f"变化最大的10队阈值：{threshold:.3f}", transform=ax.transAxes,
            fontsize=7, color=pu.COLORS["text"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": pu.COLORS["bg_box"], "edgecolor": pu.COLORS["grid"]})
    pu.save_fig(fig, _output("fig_q3_advancement_shift"))


def _sankey_panel(ax, transitions, label):
    sources = sorted({int(row["source"]) for row in transitions})
    targets = sorted({int(row["target"]) for row in transitions})
    source_totals = {level: sum(row["count"] for row in transitions if int(row["source"]) == level) for level in sources}
    target_totals = {level: sum(row["count"] for row in transitions if int(row["target"]) == level) for level in targets}
    total = sum(source_totals.values())
    gap = 0.035
    usable = 0.78

    def positions(levels, totals):
        current = 0.10
        result = {}
        scale = (usable - gap * max(len(levels) - 1, 0)) / max(total, 1)
        for level in levels:
            height = totals[level] * scale
            result[level] = [current, current + height]
            current += height + gap
        return result, scale

    left_pos, scale = positions(sources, source_totals)
    right_pos, _ = positions(targets, target_totals)
    left_cursor = {k: v[0] for k, v in left_pos.items()}
    right_cursor = {k: v[0] for k, v in right_pos.items()}
    for row in sorted(transitions, key=lambda r: (r["source"], r["target"])):
        source, target, count = int(row["source"]), int(row["target"]), int(row["count"])
        h = count * scale
        sy0, sy1 = left_cursor[source], left_cursor[source] + h
        ty0, ty1 = right_cursor[target], right_cursor[target] + h
        left_cursor[source] = sy1; right_cursor[target] = ty1
        vertices = [(0.17, sy0), (0.46, sy0), (0.54, ty0), (0.83, ty0),
                    (0.83, ty1), (0.54, ty1), (0.46, sy1), (0.17, sy1), (0.17, sy0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        color = pu.PALETTE[(source - 1) % len(pu.PALETTE)]
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", alpha=0.28))
    for level, (y0, y1) in left_pos.items():
        color = pu.PALETTE[(level - 1) % len(pu.PALETTE)]
        ax.add_patch(patches.Rectangle((0.12, y0), 0.05, y1 - y0, facecolor=color, edgecolor=color, alpha=0.78))
        ax.text(0.10, (y0 + y1) / 2, f"静态{level}\n{source_totals[level]}场", ha="right", va="center", fontsize=6.5)
    for level, (y0, y1) in right_pos.items():
        color = pu.PALETTE[(level - 1) % len(pu.PALETTE)]
        ax.add_patch(patches.Rectangle((0.83, y0), 0.05, y1 - y0, facecolor=color, edgecolor=color, alpha=0.78))
        ax.text(0.90, (y0 + y1) / 2, f"动态{level}\n{target_totals[level]}场", ha="left", va="center", fontsize=6.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    _panel_label(ax, label)


def fig_q3_resource_transition_sankey(d):
    frame = pd.DataFrame(d["q3"]["resource_transitions"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for i, resource in enumerate(["转播", "安保", "交通"]):
        _sankey_panel(axes[i], frame[frame["resource"] == resource].to_dict("records"), f"({chr(97 + i)}) {resource}")
    pu.save_fig(fig, _output("fig_q3_resource_transition_sankey"))


def fig_q3_dynamic_value_dumbbell(d):
    rows = sorted(d["q3"]["rows"], key=lambda row: row["dynamic_net_value"] - row["static_net_value"])
    y = np.arange(len(rows))
    static = np.asarray([row["static_net_value"] for row in rows], float)
    dynamic = np.asarray([row["dynamic_net_value"] for row in rows], float)
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for yi, left, right in zip(y, static, dynamic):
        color = pu.PALETTE[2] if right >= left else pu.PALETTE[1]
        ax.plot([left, right], [yi, yi], color=color, lw=1.6, alpha=0.65)
        ax.text(max(left, right) + (dynamic.max() - static.min()) * 0.025, yi, f"{right - left:+.4f}",
                va="center", fontsize=6.2, color=pu.COLORS["text"])
    ax.scatter(static, y, s=26, color=pu.PALETTE[0], edgecolor=pu.COLORS["light"], linewidth=0.5, label="静态策略重评")
    ax.scatter(dynamic, y, s=30, color=pu.PALETTE[1], edgecolor=pu.COLORS["light"], linewidth=0.5, label="动态重优化")
    ax.set_yticks(y, [row["match_id"] for row in rows], fontsize=7)
    ax.set_xlabel("逐场无量纲净效益")
    ax.set_ylabel("第三轮比赛")
    ax.legend(loc="best")
    span = dynamic.max() - static.min()
    ax.set_xlim(static.min() - 0.05 * span, dynamic.max() + 0.20 * span)
    pu.save_fig(fig, _output("fig_q3_dynamic_value_dumbbell"))


def fig_q3_risk_value_bubble_kde(d):
    frame = pd.DataFrame(d["q3"]["risk_value"])
    x = frame["risk"].to_numpy(float); y = frame["commercial_musd"].to_numpy(float)
    importance = frame["importance"].to_numpy(float); cost = frame["resource_cost"].to_numpy(float)
    bins = np.quantile(cost, [0, 0.34, 0.67, 1.0])
    group = np.clip(np.digitize(cost, bins[1:-1]), 0, 2)
    sizes = 55 + 420 * (importance - importance.min()) / max(importance.max() - importance.min(), 1e-12)
    fig = plt.figure(figsize=(8.7, 7.2))
    grid = fig.add_gridspec(4, 4, hspace=0.08, wspace=0.08)
    ax_top = fig.add_subplot(grid[0, :3])
    ax_main = fig.add_subplot(grid[1:, :3], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1:, 3], sharey=ax_main)
    for gidx in range(3):
        mask = group == gidx
        ax_main.scatter(x[mask], y[mask], s=sizes[mask], color=pu.PALETTE[gidx], alpha=0.50,
                        edgecolor=pu.COLORS["light"], linewidth=0.7, label=f"资源成本层级{gidx + 1}")
    xgrid = np.linspace(x.min() - 0.02, x.max() + 0.02, 180)
    ygrid = np.linspace(y.min() - 0.2, y.max() + 0.2, 180)
    xdensity = gaussian_kde(x)(xgrid); ydensity = gaussian_kde(y)(ygrid)
    ax_top.fill_between(xgrid, xdensity, color=pu.PALETTE[0], alpha=0.24)
    ax_top.plot(xgrid, xdensity, color=pu.PALETTE[0], lw=1.2)
    ax_right.fill_betweenx(ygrid, 0, ydensity, color=pu.PALETTE[1], alpha=0.24)
    ax_right.plot(ydensity, ygrid, color=pu.PALETTE[1], lw=1.2)
    score = x * y
    offsets = [(7, 8), (7, -13), (-32, 8), (-32, -13)]
    for k, index in enumerate(np.argsort(score)[-4:]):
        ax_main.annotate(frame.iloc[index]["match_id"], (x[index], y[index]), xytext=offsets[k],
                         textcoords="offset points", fontsize=7, color=pu.COLORS["text"],
                         arrowprops={"arrowstyle": "-", "color": pu.COLORS["ref_line"], "lw": 0.6})
    ax_main.set_xlabel("风险暴露指数")
    ax_main.set_ylabel("商业价值（百万美元）")
    ax_main.legend(loc="best", fontsize=7)
    ax_top.tick_params(labelbottom=False); ax_top.set_yticks([])
    ax_right.tick_params(labelleft=False); ax_right.set_xticks([])
    pu.save_fig(fig, _output("fig_q3_risk_value_bubble_kde"))


def fig_q3_sensitivity_surface(d):
    frame = pd.DataFrame(d["q3"]["sensitivity_surface"])
    xs = sorted(frame["state_coefficient_multiplier"].unique())
    ys = sorted(frame["injury_coefficient_multiplier"].unique())
    matrix = frame.pivot(index="injury_coefficient_multiplier", columns="state_coefficient_multiplier", values="Z3").loc[ys, xs].to_numpy(float)
    xx, yy = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(xx, yy, matrix, cmap="YlOrRd", alpha=0.88, edgecolor=pu.COLORS["light"], linewidth=0.5)
    ax.contour(xx, yy, matrix, zdir="z", offset=matrix.min() - 0.015, cmap="YlOrRd", levels=8)
    index = np.unravel_index(np.argmax(matrix), matrix.shape)
    ax.scatter([xx[index]], [yy[index]], [matrix[index]], s=70, marker="*", color=pu.PALETTE[3], edgecolor=pu.COLORS["light"])
    ax.set_xlabel("状态差系数倍率")
    ax.set_ylabel("伤病系数倍率")
    ax.set_zlabel("动态目标值 Z3")
    ax.set_zlim(matrix.min() - 0.015, matrix.max() + 0.015)
    ax.view_init(elev=28, azim=-52)
    cbar = fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("Z3")
    pu.save_fig(fig, _output("fig_q3_sensitivity_surface"))


def fig_q4_schedule_difference(d):
    rows = sorted(d["q4"]["differences"], key=lambda row: row["preferred_change"])
    labels = [METRIC_ZH[row["indicator_name"]] for row in rows]
    values = np.asarray([row["preferred_change"] for row in rows], float)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    limit = max(abs(values.min()), abs(values.max())) * 1.18
    ax.axvspan(-limit, 0, color=pu.PALETTE[1], alpha=0.07)
    ax.axvspan(0, limit, color=pu.PALETTE[2], alpha=0.07)
    colors = [pu.PALETTE[2] if value >= 0 else pu.PALETTE[1] for value in values]
    bars = ax.barh(y, values, color=colors, alpha=0.32, edgecolor=colors, linewidth=1.2)
    for bar, value in zip(bars, values):
        ax.text(value + (0.025 * limit if value >= 0 else -0.025 * limit),
                bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=7.5, color=pu.COLORS["text"])
    ax.axvline(0, color=pu.COLORS["ref_line"], lw=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("按偏好方向统一的相对变化（正值表示优化方案更优）")
    ax.set_ylabel("评价指标")
    pu.save_fig(fig, _output("fig_q4_schedule_difference"))


def fig_q4_metric_radar(d):
    rows = d["q4"]["radar"]
    labels = [METRIC_ZH[row["indicator"]] for row in rows]
    actual = np.asarray([row["actual"] for row in rows], float)
    optimized = np.asarray([row["optimized"] for row in rows], float)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(8.4, 6.8), subplot_kw={"projection": "polar"})
    for radius in np.linspace(0.2, 1.0, 5):
        ax.fill(closed_angles, np.full(len(closed_angles), radius), color=pu.PALETTE[4], alpha=0.018)
    ax.plot(closed_angles, np.r_[actual, actual[0]], color=pu.PALETTE[0], lw=1.7, marker="o", markersize=4, label="2022实际赛程")
    ax.fill(closed_angles, np.r_[actual, actual[0]], color=pu.PALETTE[0], alpha=0.12)
    ax.plot(closed_angles, np.r_[optimized, optimized[0]], color=pu.PALETTE[1], lw=1.9, marker="s", markersize=4, label="2026优化赛程")
    ax.fill(closed_angles, np.r_[optimized, optimized[0]], color=pu.PALETTE[1], alpha=0.13)
    ax.set_xticks(angles, labels, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.12), fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.08, top=0.94)
    pu.save_fig(fig, _output("fig_q4_metric_radar"))


DISPATCH = {
    "fig_q1_target_raincloud": fig_q1_target_raincloud,
    "fig_q1_feature_pairplot": fig_q1_feature_pairplot,
    "fig_q1_stage_timezone_violin": fig_q1_stage_timezone_violin,
    "fig_q1_model_accuracy_heatmap": fig_q1_model_accuracy_heatmap,
    "fig_q1_prediction_fit": fig_q1_prediction_fit,
    "fig_q1_residual_diagnostics": fig_q1_residual_diagnostics,
    "fig_q1_shap_summary": fig_q1_shap_summary,
    "fig_q2_schedule_gantt": fig_q2_schedule_gantt,
    "fig_q2_venue_travel_map": fig_q2_venue_travel_map,
    "fig_q2_venue_slot_matrix": fig_q2_venue_slot_matrix,
    "fig_q2_objective_waterfall": fig_q2_objective_waterfall,
    "fig_q2_constraint_margins_lollipop": fig_q2_constraint_margins_lollipop,
    "fig_q2_weight_sensitivity_contour": fig_q2_weight_sensitivity_contour,
    "fig_q3_advancement_shift": fig_q3_advancement_shift,
    "fig_q3_resource_transition_sankey": fig_q3_resource_transition_sankey,
    "fig_q3_dynamic_value_dumbbell": fig_q3_dynamic_value_dumbbell,
    "fig_q3_risk_value_bubble_kde": fig_q3_risk_value_bubble_kde,
    "fig_q3_sensitivity_surface": fig_q3_sensitivity_surface,
    "fig_q4_schedule_difference": fig_q4_schedule_difference,
    "fig_q4_metric_radar": fig_q4_metric_radar,
}


def generate(name: str):
    if name not in DISPATCH:
        raise KeyError(f"Unknown figure: {name}")
    DISPATCH[name](_data())
