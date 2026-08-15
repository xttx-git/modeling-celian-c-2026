"""Generate the planned LaTeX tables and all figure/table include snippets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def read_json(name):
    return json.loads((FIGURES / name).read_text(encoding="utf-8"))


def esc(value):
    if value is None:
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def soft_esc(value):
    """Escape LaTeX and add legal breakpoints to long technical identifiers."""
    return (esc(value)
            .replace(r"\_", r"\_\allowbreak{}")
            .replace("/", r"/\allowbreak{}")
            .replace(";", r";\allowbreak{}"))


def normal_table(filename, caption, label, headers, rows, align=None, resize=False, note=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    body = [r"\begin{table}[H]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small"]
    if resize:
        body.append(r"\resizebox{\textwidth}{!}{%")
    body.extend([f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"])
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        body.append("}")
    if note:
        body.append(r"\vspace{2pt}")
        body.append(r"\noindent\begin{minipage}{0.96\textwidth}\footnotesize 注：" + note + r"\end{minipage}")
    body.append(r"\end{table}")
    (FIGURES / filename).write_text("\n".join(body) + "\n", encoding="utf-8")


def long_table(filename, caption, label, headers, rows, align=None, note=None):
    align = align or ("l" * len(headers))
    header = " & ".join(headers) + r" \\"
    body = ["{", r"\footnotesize", r"\hfuzz=0.2pt", r"\setlength{\tabcolsep}{3pt}",
        f"\\begin{{longtable}}{{{align}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        f"\\multicolumn{{{len(headers)}}}{{c}}{{续表}} " + r"\\",
        r"\toprule", header, r"\midrule", r"\endhead",
        f"\\midrule \\multicolumn{{{len(headers)}}}{{r}}{{续下页}} " + r"\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.append(r"\end{longtable}")
    if note:
        body.append(r"\noindent\begin{minipage}{0.96\textwidth}\footnotesize 注：" + note + r"\end{minipage}")
    body.append(r"\par")
    body.append("}")
    (FIGURES / filename).write_text("\n".join(body) + "\n", encoding="utf-8")


def main():
    p1 = read_json("problem_1_results.json")
    p2 = read_json("problem_2_results.json")
    p3 = read_json("problem_3_results.json")
    p4 = read_json("problem_4_results.json")
    plot = read_json("_plot_data.json")

    lineage_rows = [
        [soft_esc(row["feature_name"]), soft_esc(row["source_table"]), soft_esc(row["source_column"]),
         soft_esc(row["source_time_rule"]), soft_esc(row["fitted_on_ids"])]
        for row in p1["feature_lineage"]
    ]
    long_table(
        "TABLE_q1_data_dictionary.tex", "问题一特征字典与时间血缘", "tab:q1-data-dictionary",
        ["特征", "来源表", "来源列", "时间规则", "拟合样本"], lineage_rows,
        align=r"p{0.15\textwidth}p{0.16\textwidth}p{0.15\textwidth}p{0.18\textwidth}p{0.19\textwidth}",
    )

    fold_rows = []
    for row in p1["fold_metrics"]:
        fold_rows.append([
            esc(row["model"]), f"折{int(row['fold'])}", fmt(row["mse"], 3), fmt(row["rmse"], 3), fmt(row["mae"], 3),
            esc(str(row["train_end"])[:10]), esc(str(row["valid_start"])[:10]),
        ])
    for row in p1["model_summary"]:
        fold_rows.append([esc(row["model"]), "总体均值", fmt(row["mean_mse"], 3), fmt(row["mean_rmse"], 3),
                          fmt(row["mean_mae"], 3), "--", "--"])
    normal_table(
        "TABLE_q1_model_metrics.tex", "问题一时序交叉验证误差", "tab:q1-model-metrics",
        ["模型", "窗口", "MSE", "RMSE", "MAE", "训练截至", "验证起始"], fold_rows,
        align="llrrrrr", resize=True,
        note=f"误差单位为百万人口径；最终选择模型为 {esc(p1['selected_model'])}。",
    )

    pair = pd.DataFrame(plot["q1"]["pairplot"])
    desc_rows = []
    for column in pair.columns:
        series = pair[column].dropna().astype(float)
        desc_rows.append([
            esc(column), str(len(series)), fmt(series.mean(), 2), fmt(series.std(ddof=1), 2), fmt(series.min(), 2),
            fmt(series.quantile(0.25), 2), fmt(series.median(), 2), fmt(series.quantile(0.75), 2), fmt(series.max(), 2),
        ])
    normal_table(
        "TABLE_q1_descriptive_statistics.tex", "问题一核心变量描述性统计", "tab:q1-descriptive",
        ["变量", "样本数", "均值", "标准差", "最小值", "P25", "中位数", "P75", "最大值"],
        desc_rows, align="lrrrrrrrr", resize=True,
        note="观看人数单位为百万人；其余变量采用源数据或赛前构造口径。",
    )

    constraint_rows = []
    for row in plot["q2"]["constraint_margins"]:
        constraint_rows.append([
            esc(row["name"]), fmt(row["actual"], 3), fmt(row["limit"], 3), esc(row["unit"]),
            fmt(row["margin_pct"], 1) + r"\%", "通过" if row["margin_pct"] >= -1e-9 else "未通过",
        ])
    normal_table(
        "TABLE_q2_constraint_audit.tex", "问题二关键约束复算与裕度", "tab:q2-constraint-audit",
        ["约束族", "实际值", "界值", "单位", "最小裕度", "状态"], constraint_rows,
        align="lrrrrc", resize=False,
        note="裕度为按各约束自身上限归一化后的最小剩余比例；零表示约束在最紧样本上取等。",
    )

    schedule = pd.DataFrame(p2["schedule"])
    schedule_rows = []
    for venue, group in schedule.groupby("venue_id", sort=True):
        schedule_rows.append([
            esc(venue), esc(group["city"].iloc[0]), str(len(group)), str(int(group["is_prime"].sum())),
            fmt(group["travel_cost_index"].mean(), 3), fmt(group["expected_attendance"].sum() / 10_000, 1),
        ])
    normal_table(
        "TABLE_q2_schedule_summary.tex", "问题二场馆负载与赛程汇总", "tab:q2-schedule-summary",
        ["场馆", "城市", "场次", "黄金时段场次", "平均旅行指数", "预计观众合计（万人）"], schedule_rows,
        align="llrrrr", resize=True,
        note=f"主方案 Z2={p2['metrics']['Z2']:.6f}，最小轮间休息 {p2['constraint_audit']['min_rest_hours']:.0f} 小时。",
    )

    cond = {row["match_id"]: row for row in p3["conditional_probabilities"]}
    probability_rows = []
    for row in p3["rows"]:
        c = cond[row["match_id"]]
        probability_rows.append([
            esc(row["match_id"]), esc(row["team_a"]), fmt(row["updated_p_team_a_advance"], 3),
            esc(row["team_b"]), fmt(row["updated_p_team_b_advance"], 3),
            fmt(c["p_a_cond_win"], 3), fmt(c["p_b_cond_win"], 3),
            fmt(row["stakeless_risk"], 3), fmt(row["collusion_risk"], 3),
        ])
    normal_table(
        "TABLE_q3_probability_risk.tex", "问题三晋级概率、条件概率与风险", "tab:q3-probability-risk",
        ["比赛", "A队", "A队晋级", "B队", "B队晋级", "A胜条件晋级", "B胜条件晋级", "无悬念风险", "默契风险"],
        probability_rows, align="llrlrrrrr", resize=True,
        note="条件晋级概率来自同一 20000 次联合蒙特卡洛样本，晋级名额质量守恒为每次 32 队。",
    )

    budget_rows = []
    for row in p3["daily_dynamic_audit"]:
        caps = row["caps"]
        budget_rows.append([
            esc(row["date"]), str(int(row["broadcast3"])), str(int(caps["high_broadcast_capacity"])),
            str(int(row["security3plus"])), str(int(caps["high_security_capacity"])),
            str(int(row["transport3"])), str(int(caps["enhanced_transport_capacity"])),
            fmt(row["budget"], 1), fmt(caps["daily_resource_budget_index"], 1), "通过" if row["pass"] else "未通过",
        ])
    normal_table(
        "TABLE_q3_resource_budget.tex", "问题三逐日动态资源使用与容量", "tab:q3-resource-budget",
        ["日期", "高转播", "转播上限", "高安保", "安保上限", "强化交通", "交通上限", "预算使用", "预算上限", "状态"],
        budget_rows, align="lrrrrrrrrc", resize=True,
    )

    action_lookup = {
        (row["match_id"], int(row["b"]), int(row["q"]), int(row["l"])): row
        for row in p3["action_table"]
    }
    static_actions = []
    dynamic_actions = []
    for row in p3["rows"]:
        s = row["static_decision"]; q = row["dynamic_decision"]
        static_actions.append(action_lookup[(row["match_id"], int(s["b"]), int(s["q"]), int(s["l"]))])
        dynamic_actions.append(action_lookup[(row["match_id"], int(q["b"]), int(q["q"]), int(q["l"]))])
    static_dynamic_rows = []
    definitions = [
        ("票务价值 TV", "TV", "sum", "百万美元", 1_000_000),
        ("转播价值 BV", "BV", "sum", "百万美元", 1_000_000),
        ("吸引力 A", "A", "mean", "无量纲", 1),
        ("资源成本 C", "C", "sum", "成本指数", 1),
        ("风险 R", "R", "mean", "无量纲", 1),
    ]
    for label, key, agg, unit, divisor in definitions:
        svals = np.asarray([row[key] for row in static_actions], float)
        dvals = np.asarray([row[key] for row in dynamic_actions], float)
        sval = (svals.sum() if agg == "sum" else svals.mean()) / divisor
        dval = (dvals.sum() if agg == "sum" else dvals.mean()) / divisor
        direction = dval - sval if key in {"TV", "BV", "A"} else sval - dval
        static_dynamic_rows.append([esc(label), fmt(sval, 3), fmt(dval, 3), fmt(direction, 3), esc(unit)])
    static_dynamic_rows.append(["综合目标 Z3", fmt(p3["Z3_static_reeval"], 6), fmt(p3["Z3_dynamic"], 6),
                                fmt(p3["Z3_dynamic"] - p3["Z3_static_reeval"], 6), "无量纲"])
    normal_table(
        "TABLE_q3_static_dynamic.tex", "问题三静态策略与动态重优化同界比较", "tab:q3-static-dynamic",
        ["指标", "静态策略重评", "动态重优化", "偏好方向改善", "单位"], static_dynamic_rows,
        align="lrrrr", resize=False,
        note="正的偏好方向改善表示动态方案更优；成本与风险按越低越优转换。",
    )

    structural_labels = {
        "rest_min_hours": "最小休息", "rest_mean_hours": "平均休息",
        "timezone_crossings": "跨时区次数", "timezone_crossing_rate": "跨时区率",
        "travel_mean_km": "场馆旅行", "venue_change_rate": "换馆率",
        "venue_utilization_rate": "场馆日利用率", "venue_daily_peak": "场馆日峰值",
        "capacity_occupancy_rate": "容量匹配", "prime_coverage_rate": "黄金覆盖",
        "expected_attendance_per_match": "场均预计观众",
    }
    percent_metrics = {"timezone_crossing_rate", "venue_change_rate", "venue_utilization_rate",
                       "capacity_occupancy_rate", "prime_coverage_rate"}
    integer_metrics = {"timezone_crossings", "venue_daily_peak"}
    structural_rows = []
    for row in p4["structural_comparison"]:
        name = row["indicator_name"]
        spec = p4["structural_definitions"][name]
        actual = row["actual_schedule_value"]
        optimized = row["optimized_schedule_value"]
        if name in percent_metrics:
            actual_text, optimized_text = fmt(100 * actual, 4), fmt(100 * optimized, 4)
        elif name in integer_metrics:
            actual_text, optimized_text = fmt(int(round(actual))), fmt(int(round(optimized)))
        elif name == "expected_attendance_per_match":
            actual_text, optimized_text = fmt(actual, 2), fmt(optimized, 2)
        else:
            actual_text, optimized_text = fmt(actual, 4), fmt(optimized, 4)
        structural_rows.append([
            structural_labels[name], esc(spec["definition"]), esc(spec["unit"]),
            actual_text, optimized_text,
            "越高越优" if spec["preferred_direction"] == "higher" else "越低越优",
        ])
    normal_table(
        "TABLE_q4_structural_comparison.tex", "两套赛程的可观测结构与规模强度比较",
        "tab:q4-structural-comparison",
        ["指标", "定义与分母", "单位", "2022实际", "2026优化", "偏好方向"], structural_rows,
        align="lllrrl", resize=True,
        note="休息、时区、场馆路径与场馆日由逐场赛程直接复算；黄金覆盖来自唯一时段映射，容量匹配和预计观众的需求端为题内赛前代理。实际赛程有64次球队转场，优化赛程有96次。",
    )

    formula = {
        "T": r"$T=\mathrm{norm}(\sum \mathrm{ticket})$", "B": r"$B=\mathrm{norm}(\sum \mathrm{broadcast})$",
        "U": r"$U=\overline{\mathrm{uncertainty}}$", "H": r"$H=\overline{\mathrm{attractiveness}}/100$",
        "C": r"$C=\mathrm{norm}(\mathrm{setup+operation+security})$", "D": r"$D=\overline{\mathrm{travel}}$",
        "F": r"$F=(\Delta prime+\Delta large)/6$", "R": r"$R=\overline{\mathrm{risk}}$",
        "Z4": r"$Z4=0.25T+0.25B+0.15U+0.10H-0.08C-0.07D-0.06F-0.04R$",
    }
    definition_rows = []
    for row in p4["comparison"]:
        name = row["indicator_name"]
        direction = "越高越优" if row["preferred_direction"] == "higher" else "越低越优"
        if name in p4["structural_definitions"]:
            spec = p4["structural_definitions"][name]
            evidence = {"direct":"现实赛程直接复算", "proxy_demand":"需求端为题内代理",
                        "time_mapping":"赛事进度与UTC时刻映射"}[spec["evidence"]]
            definition_rows.append([soft_esc(name), esc(spec["definition"]), esc(spec["unit"]), direction,
                                    "统一结构评价函数", evidence])
        else:
            note = "共同包络Min--Max" if name in {"T", "B", "C"} else ("条件性加权总分" if name == "Z4" else "题内代理评价")
            definition_rows.append([soft_esc(name), formula[name], "无量纲", direction, "统一代理评价函数", note])
    long_table(
        "TABLE_q4_metric_definition.tex", "问题四评价指标定义与可比口径", "tab:q4-metric-definition",
        ["指标", "定义", "单位", "偏好方向", "数据源", "口径说明"], definition_rows,
        align=r"@{}p{0.08\textwidth}p{0.26\textwidth}p{0.08\textwidth}p{0.11\textwidth}p{0.18\textwidth}p{0.209\textwidth}@{}",
        note=esc(p4["proxy_note"]),
    )

    captions = {
        "fig_q1_target_raincloud": ("训练集观看人数分布与长尾特征", 0.82),
        "fig_q1_feature_pairplot": ("问题一核心赛前特征成对关系", 0.88),
        "fig_q1_stage_timezone_violin": ("赛事阶段与时区组合下观看人数异质性", 0.96),
        "fig_q1_model_accuracy_heatmap": ("候选模型时序验证误差比较", 0.90),
        "fig_q1_prediction_fit": ("最终模型OOF预测拟合与残差分布", 0.94),
        "fig_q1_residual_diagnostics": ("最终模型残差诊断四联图", 0.92),
        "fig_q1_shap_summary": ("ElasticNet置换重要性与单样本预测影响", 0.96),
        "fig_q2_schedule_gantt": ("72场小组赛UTC排程甘特图", 0.98),
        "fig_q2_venue_travel_map": ("场馆负载与球队跨轮旅行网络", 0.92),
        "fig_q2_venue_slot_matrix": ("场馆与参考日期占用矩阵", 0.98),
        "fig_q2_objective_waterfall": ("问题二目标函数八项贡献分解", 0.90),
        "fig_q2_constraint_margins_lollipop": ("问题二最紧约束归一化裕度", 0.86),
        "fig_q2_weight_sensitivity_contour": ("商业与惩罚权重联合灵敏度", 0.78),
        "fig_q3_advancement_shift": ("更新前后48队晋级概率变化", 0.82),
        "fig_q3_resource_transition_sankey": ("静态策略到动态策略的资源等级流转", 0.96),
        "fig_q3_dynamic_value_dumbbell": ("24场静态与动态净效益配对比较", 0.86),
        "fig_q3_risk_value_bubble_kde": ("风险暴露与商业价值联合分布", 0.82),
        "fig_q3_sensitivity_surface": ("状态差与伤病系数下的动态目标曲面", 0.76),
        "fig_q4_schedule_difference": ("按偏好方向统一的两方案指标差异", 0.88),
        "fig_q4_metric_radar": ("实际赛程与优化赛程多维偏好得分", 0.72),
    }
    includes = []
    for name, (caption, width) in captions.items():
        label = "fig:" + name.removeprefix("fig_").replace("_", "-")
        includes.extend([
            r"\begin{figure}[H]", r"\centering",
            f"\\includegraphics[width={width:.2f}\\textwidth]{{figures/{name}.pdf}}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{figure}", "",
        ])
    for filename in [
        "TABLE_q1_data_dictionary.tex", "TABLE_q1_model_metrics.tex", "TABLE_q1_descriptive_statistics.tex",
        "TABLE_q2_constraint_audit.tex", "TABLE_q2_schedule_summary.tex", "TABLE_q3_probability_risk.tex",
        "TABLE_q3_resource_budget.tex", "TABLE_q3_static_dynamic.tex", "TABLE_q4_structural_comparison.tex",
        "TABLE_q4_metric_definition.tex",
    ]:
        includes.append(f"\\input{{figures/{filename}}}")
    (FIGURES / "latex_includes.tex").write_text("\n".join(includes) + "\n", encoding="utf-8")
    print("Generated 10 tables and figures/latex_includes.tex")


if __name__ == "__main__":
    main()
