# 编程实现说明

本文档对应 2026-08-16 完整修订后的程序。计算代码生成 CSV，验收代码独立复算关键公式；不得手工修改结果文件。

## 1. 环境与运行

- 固定 Python 版本：`3.12.3`
- 固定依赖版本：见 `requirements.txt`
- 附件：`C题_数据附件.xlsx`

以下命令均从 `C题` 目录执行。Linux、macOS 或 WSL：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python code/run_all.py --output-dir output_v2
python code/check_outputs.py --output-dir output_v2
```

Windows PowerShell 的激活命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

隔离结果验收通过后生成正式结果：

```bash
python code/run_all.py --output-dir output
python code/check_outputs.py --output-dir output
```

论文图表与额外分析使用独立依赖文件，不改写正式 `output/`：

```bash
python -m pip install -r requirements-paper.txt
python code/paper_analysis.py --output-dir output --generated-dir paper/generated
```

本机复算使用的解释器为 `/home/xttx/.venvs/modeling-celian-c-2026/bin/python`，该绝对路径仅是环境记录，不是运行前提。

固定版本为：

| 组件 | 版本 |
|---|---:|
| Python | 3.12.3 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| openpyxl | 3.1.5 |
| joblib | 1.5.3 |
| threadpoolctl | 3.6.0 |

## 2. 文件职责

- `code/run_all.py`：四问流水线入口和输出写入，不含手工结果。
- `code/pipeline_core.py`：特征、模型、MILP、模拟、指标及代理映射的计算实现。
- `code/check_outputs.py`：模板、公式、约束和可复现性验收；会重新训练问题一并重跑问题三主种子的 20000 次模拟。
- `code/paper_analysis.py`：论文专用扩窗逐折、系数、残差、问题二权重重求解、问题三收敛及问题四映射质量分析。
- `requirements.txt`：固定 Python 依赖版本，解决 Extra Trees 在不同库版本下的轻微指标差异。
- `requirements-paper.txt`：绘图、图像和 PDF 质检依赖，不改变正式模型锁定环境的哈希。
- `material/user_data/worldcup_2022_openfootball.json`：普通项目文件形式保存的 2022 原始赛程。
- `material/user_data/worldcup_2022_pre_tournament_elo.csv`：2022 赛前 Elo 映射输入。
- `material/user_data/entity_mappings.json`：球队、场馆别名，场馆元数据和固定来源链接。
- `output/diagnostics.json`：特征范围、模型指标、归一化边界、求解状态、晋级概率和稳健性结果。
- `paper/generated/`：由论文分析脚本生成的 PDF/SVG/PNG 图、CSV/LaTeX 表和 `paper_analysis_results.json`。

## 3. 问题一

### 3.1 统一对称特征

训练、测试和 72 场预测全部调用 `build_prediction_features()`。数值特征和列结构使用同一生成函数，特征为：

- 排名、Elo 的 mean/min/max/absdiff。
- 市值、年龄、明星指数、球迷基础、攻防风格、东道主标记、综合实力的 mean/min/max/absdiff。
- 平局概率、双方胜率 min/max/absdiff、概率熵。
- neutral、competition、归并后的 stage、month、weekday。

不使用 `expected_attendance_base`、`attractiveness_index`、`commercial_value_index`、`uncertainty_index`，也不使用球队 A/B 有序类别、带符号差值或同场赛后字段。交换双方后最大预测差为 0。

赛事类别在统一特征函数内调用 `normalize_competition_name()`：清理大小写、年份和 FIFA 前缀，将 `World Cup`、`FIFA World Cup`、`FIFA World Cup 2026` 等同义写法统一为 `World Cup`。训练、扩窗验证、测试和 72 场预测全部使用同一函数。规范化后测试集和未来 72 场的未见类别样本数均为 0，详细的前后类别集与样本数已写入 `diagnostics.json`。

### 3.2 模型与结果

扩窗验证比较均值、Elastic Net 和 Extra Trees，指标单位为“百万人”的误差：

| 模型 | MSE | RMSE | MAE |
|---|---:|---:|---:|
| Elastic Net | 188.978683 | 13.726811 | 10.634511 |
| Extra Trees | 227.022897 | 15.033564 | 12.094744 |
| 均值基线 | 531.378075 | 22.996821 | 18.398928 |

最终选择 Elastic Net。随机模型种子为 `20260816`。测试预测均值 144.558531 百万人；72 场预测均值 161.599758 百万人。

输出：

- `output/result_1_test_prediction.csv`：140 行，单位百万人。
- `output/result_1_match_prediction.csv`：72 行，单位人。

## 4. 问题二

### 4.1 正确公式

转播价值：

$$
B_{is}=\widehat V_i\times u_i\times q_s\times w_i.
$$

没有额外的 0.75。

旅行负担先分别归一化距离、时间和时区差：

$$
D_{iv}=0.5d_{iv}^{km}+0.3d_{iv}^{time}+0.2d_{iv}^{tz}.
$$

第一轮从球队代表地出发；第二、三轮对上一轮全部安保合格候选场馆到当前场馆的分项取平均；最后对两队平均。

成本以百万美元计：

$$
C=\sum_v setup_v y_v+\sum_{i,v,s}x_{ivs}
\left(operation_v+0.1L_i^{req}security\_cost_v\right).
$$

所有场馆 `min_total_matches>0`，因此启用成本在本数据中是常数，但仍保留在原始成本中。

黄金时段按 `global_prime_score` 降序、`slot_id` 升序稳定处理并列后严格取 20 个。大容量场馆严格取容量前 4 个。

### 4.2 求解方法

实现是分解 MILP，不是三维联合模型的全局最优证明：

1. 时段 MILP：比赛-时段指派、真实 UTC 转播容量、同队相邻比赛 60 小时休息、同组相邻轮次边界 60 小时、黄金时段公平、第三轮高最低安保需求日容量。
2. 场馆 MILP：固定时段后的比赛-场馆指派、安保、场馆-UTC、当地日容量、总场次上下限和大容量场馆公平。

HiGHS 原始状态为 `optimal` 时，统一解释为“在设定 MIP 容差内求得最优解”，并保留实际 gap；不把非零 gap 表述为精确全局最优证明。程序只声明“分解 MILP 可行优化方案”，更不声称三维联合模型全局最优。

本轮固定环境的实际 gap：

- 时段子问题：0.000478443139
- 固定时段后的场馆子问题：0.000357088426

同组相邻轮次边界独立复算的最小值为 63.0 小时，违规数量为 0，`all_passed=True`。

### 4.3 指标、边界与结果

原始指标：

| 指标 | 原始值 | Min-Max 边界 | 标准化值 |
|---|---:|---:|---:|
| T | 285910470.000 | [180224640.000, 462345840.000] | 0.374611 |
| B | 326546865.536 | [86445484.184, 613637147.351] | 0.455435 |
| U | 0.461436 | [0, 1] | 0.461436 |
| H | 0.639699 | [0, 1] | 0.639699 |
| C | 53.055 | [38.470, 71.590] | 0.440368 |
| D | 0.189247 | [0, 1] | 0.189247 |
| F | 0.000000 | [0, 1] | 0.000000 |
| R | 0.410364 | [0.224709, 0.600402] | 0.494167 |

最终：

$$
Z_2=0.2724533543.
$$

输出为 `output/result_2_group_schedule.csv`。

论文额外分析对 `T、B、C、D、F、R` 的权重幅值分别做 ±20% 扰动，每个情景重新归一化权重、重求解两个 MILP 并复核全部硬约束。`U、H` 由固定对阵决定，不影响排程选择。该分析只写入 `paper/generated/`，不覆盖正式赛程。

## 5. 问题三

### 5.1 联合模拟与同分

- 第三轮 24 场共享前两轮结束后的同一信息截面。
- 每个种子同时模拟 24 场 20000 次，再统一计算 12 个小组和 8 个最佳第三名。
- 排名只用积分、净胜球和总进球。
- 完全同分时，组内前二、第三名身份和最佳第三名第 8 名截止均使用精确等比例浮点权重。
- 每次模拟晋级权重严格等于 32，平均晋级概率和严格等于 32。

### 5.2 动态需求、票价和风险

动态安保占用率使用资源配置前需求：

$$
O_i=\mathrm{clip}(\widetilde N_i/K_i,0,1).
$$

它不依赖最终票价或交通等级。票价候选包括折扣、0、涨价、容量切换和需求保持边界；每个候选都用容量截断后的完整需求重新验证：

$$
N_i(\delta)\ge0.88N_i(0).
$$

动作选择使用多重选择 MILP，约束每日高转播、高安保、强化交通和预算。

### 5.3 静态方案

静态方案在赛前使用：

- 问题二预计现场观众；
- 问题一预计转播观看人数；
- `attractiveness_index/100`；
- 无激励风险 `1-uncertainty_index`；
- 默契风险 0。

它在赛前候选边界下优化并固定转播、安保、交通和票价四项动作。前两轮结束后，程序按固定动作重新计算上座、票务、转播、安保需求、风险、成本和净值，不从动态候选中寻找“相近票价”。静态和动态最终均使用更新后的同一边界。

更新边界：

| 指标 | 下界 | 上界 |
|---|---:|---:|
| ticket_value | 2981226.448 | 6558212.829 |
| broadcast_value | 4726317.999 | 8519872.166 |
| attractiveness | 0.366896 | 0.686617 |
| resource_cost | 3 | 23 |
| risk | 0.079917 | 0.745848 |

主结果动态总值 6.5838485986，静态总值 6.5783676577，总体改善率约 0.083318%，不能表述为显著提升。

主模拟和模拟次数检查种子为 `20260816`，五种子检验为 `20260816`--`20260820`。不同模拟次数使用同一种子分别重置随机数生成器；由于主客队泊松数组分两次生成，不构成严格共享同一随机序列前缀的嵌套收敛实验。

五个种子稳定性：

- 动态总值范围：[6.572417, 6.644567]
- 静态总值范围：[6.566840, 6.639229]
- 改善率范围：[0.080406%, 0.084923%]
- 相对主种子的 24 场动态动作一致率均为 100%

输出为 `output/result_3_dynamic_strategy.csv`。

## 6. 问题四

2022 小组赛原始数据来自固定 commit：

`https://github.com/openfootball/worldcup.json/blob/516d3825c3bd23fdc298c4014e84bde78f2d4965/2022/worldcup.json`

并使用 FIFA 2022 Match Centre 核验。采集日期记录为 2026-08-16。程序直接读取项目内 JSON，不依赖 Git 历史。

材料校验结果：

- `worldcup_2022_openfootball.json`：SHA-256 为 `f4f0499c30076bc8e3702f400765900797fb22212a32aaa8137e1dfc49449a68`，含 64 场比赛和 48 场小组赛。
- `worldcup_2022_pre_tournament_elo.csv`：32 支球队，无缺失和重复。
- `entity_mappings.json`：覆盖原始赛程全部 32 个球队名称和 8 个场馆名称；`USA -> United States`；每个规范场馆均有名称、城市、容量、经纬度、时区、来源链接和采集日期。

结构层比较赛程跨度、总/场均/队均旅行、休息均值/最小值/标准差、跨时区、场馆利用、容量匹配、黄金时段和公平性。

代理层按同轮次 Elo 均值与差值映射比赛，按安保可行后的容量距离映射场馆，按循环 UTC 小时映射时段，再用问题二相同的 T、B、U、H、C、D、F、R、边界和权重。最近邻代理分为 0.196995，第二近邻为 0.210331，优化赛程为 0.272453；两种映射方向未反转，但结论仍只称“题内代理口径”。2000 组权重扰动使用固定种子 `20260816`，其作用是对固定赛程、固定映射和固定指标回评，不会重新排程。

输出：

- `output/actual_schedule_2022.csv`
- `output/result_4_schedule_comparison.csv`

## 7. 自动验收

`check_outputs.py` 已验证：

- 问题一 ID、单位、统一特征、禁用字段、交换不变性，并重新训练后逐值比对 CSV。
- 问题二身份、时区、容量、休息、24 个同组轮次边界、场馆负荷、安保、公平和 T/B/C/D/F/R/Z2 复算。
- 问题三等比例同分、晋级和为 32、动作合法、票价截断约束、动态公式、赛前静态选择、固定动作更新复算、总值和五种子稳健性。
- 问题四三个材料文件、原始文件 SHA-256、Elo 与实体覆盖、固定来源、48 场结构、UTC、第三轮同时开球、30 项指标和相对改善方向。

当前增强验收结果为 `CHECK PASSED`。
