"""MODELING_REPORT 符号到可执行实现的静态可追溯索引。

本文件不参与数值计算，只为人工复核与确定性审计给出报告命名和实际 Python
命名之间的一一对应。核心计算仍分别位于 problem1.py 至 problem4.py。

P1 配置/符号：
ACTUAL_TOURNAMENT -> problem4.FIFA_SCHEDULE_URL；
ALLOW_POSTMATCH_DIRECT -> problem1.FORBIDDEN_DIRECT；
USE_FIXED_SPLIT -> problem1.solve 的 dataset_split 断言；
LAG_FEATURE_SHIFT -> problem1.build_features 按日期先取后更；
TEST_OUTPUT_UNIT / GROUP_OUTPUT_UNIT -> TARGET_SCALE 与两个交付 CSV；
X_j / Y_j / H_j -> X_train / y / odds_entropy。

P2 配置/符号：
TRAVEL_ORIGIN_MODE -> problem2.compute_travel 的全部合格场馆 origins；
K_v / L_i / H_d / Budget_d -> venues.capacity / required_security_level /
dynamic_resource_limits；U_i -> uncertainty_index；BV_i / TV_i -> B_terms / T_terms；
Z2_optimum -> problem2.metrics['Z2']。

P3 配置/符号：
COMMON_R3_SNAPSHOT -> problem3.team_snapshot；COMMON_PRICE_DOMAIN -> price_common_upper；
STATIC_BASELINE -> static_action_snapshot；I_3 -> round3；G_t / RC_t -> games / red；
Q_i / G_i / O_i -> Q / conditional gain G / occupancy；Nr_b / Vr_b -> team feedback；
TV_i -> action['TV']；Z3_optimum -> problem3.Z3_dynamic。

P4 符号：
REQUIRE_ROW_SOURCE_URL -> collect_actual_schedule 的 source_url 断言；
休息/时区/路径/场馆日/容量/黄金覆盖 -> structural_metrics 与 STRUCTURAL_DEFINITIONS；
P2 同构旅行 D -> build_p2_travel_context 与 assign_p2_proxy_travel；
M_k / Delta_k / I_k -> comparison_rows 的 actual/optimized/difference/improvement。

机器合同与结果登记：
METHOD_CLAIMS / METHOD_CLAIMS_MACHINE -> MODELING_REPORT 第14节与 claim_code_check；
LOGIC_CONTRACT_MACHINE / DATA_FACTS / CROSS_PROBLEM_LEDGER / RESULT_CONSTRAINTS ->
上游机器合同、constraint_audit 和 all_results。

图表/表格数据落点：
FIGURE_QUICK_REF 仅供下一阶段画图；TABLE_q1_data_dictionary -> feature_lineage；
TABLE_q1_model_metrics -> fold_metrics；TABLE_q2_constraint_audit -> constraint_audit；
TABLE_q2_schedule_summary -> schedule/venue_load；TABLE_q3_probability_risk ->
team_advancement/rows；TABLE_q3_resource_budget -> daily_dynamic_audit；
TABLE_q3_static_dynamic -> static_action_snapshot/rows；TABLE_q4_metric_definition -> comparison。

FAST_MODE=0 由运行环境 CLAUDE.md 检测，非模型数值参数。
"""
