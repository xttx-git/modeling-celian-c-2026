# 2026 年“策联杯”C 题复现说明

本目录对应论文《基于时序学习与整数规划的世界杯观赛需求及资源协同优化》。程序按问题一至问题四、灵敏度分析、约束复核、图表和论文表格的顺序运行。所有正式随机过程使用种子 `20260813`；问题三另用连续五个种子检查稳定性。

## 1. 环境

- Windows 11
- Python 3.12.13
- 依赖版本见 `requirements-lock.txt`
- 论文编译需要 XeLaTeX、BibTeX 和项目内 `paper/cumcmthesis.cls`、`paper/simkai.ttf`

建议从空环境安装：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## 2. 数据位置

官方附件不放入支撑材料压缩包。请将赛题提供的唯一 Excel 文件复制为：

```text
user_data/C题_数据附件.xlsx
```

问题四自主查阅并随支撑材料提供的数据为：

- `user_data/worldcup_2022_openfootball.json`
- `user_data/worldcup_2022_pre_tournament_elo.csv`
- `actual_schedule_2022.csv`（逐场整理表，含来源 URL 和访问日期）

## 3. 一键完整复现

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe reproduce.py
```

该命令会重新训练问题一、重新求解问题二和问题三、重算问题四与灵敏度分析，随后运行独立于求解器模型的交付约束复核，并重新生成全部论文图表。问题二采用 1800 秒上限和 0.5% 相对 gap 停止准则；实际运行时间取决于硬件。

若只需核验已经持久化的正式结果并重画图表：

```powershell
.\.venv\Scripts\python.exe reproduce.py --reuse
```

## 4. 正式输出

题面规定的核心文件位于项目根目录，同时在 `output/` 保留同内容副本：

- `result_1_test_prediction.csv`
- `result_1_match_prediction.csv`
- `result_2_group_schedule.csv`
- `result_3_dynamic_strategy.csv`

问题四的可选复核材料为：

- `actual_schedule_2022.csv`
- `result_4_schedule_comparison.csv`

模型完整机器结果保存在 `figures/problem_1_results.json` 至 `figures/problem_4_results.json`、`figures/sensitivity_results.json` 和 `figures/all_results.json`。

## 5. 论文编译

模型与图表复现完成后，在 `paper/` 中执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

AI 详情 PDF 使用同目录下 `ai_detail_main.tex` 按相同的 XeLaTeX 命令单独编译。竞赛最终文件名中的 `XXX` 必须在提交前替换为三位参赛队号。

## 6. 关键复核口径

- 问题一严格保留官方 `train/test` 划分；测试集观看标签不参与训练、预处理或选模。
- 问题二的 `T/B/C` Min--Max 上下界来自全部合法比赛—场馆—时段候选的确定性包络；旅行负担的第二、三轮起点是上一轮全部安保合格候选场馆。
- 问题三的 24 场第三轮使用同一第二轮结束信息截面；静态方案先按赛前信息独立优化并锁定，再与动态方案在共同更新环境和共同上下界下评价。
- 问题四将可直接观测的结构指标与题内代理指标分开解释，不把代理综合分外推为 FIFA 现实排程的总体优劣。
