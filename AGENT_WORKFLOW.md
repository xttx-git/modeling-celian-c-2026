# 队友 Agent 操作步骤

本仓库是 2026 年“策联杯”数学建模精英联赛 C 题当前工作快照。默认分支为 `main`，远程仓库为：

```bash
https://github.com/xttx-git/modeling-celian-c-2026.git
```

## 1. 首次获取项目

```bash
git clone https://github.com/xttx-git/modeling-celian-c-2026.git
cd modeling-celian-c-2026
git status -sb
```

如果需要推送，先确认 GitHub 账号已经被仓库所有者加入 collaborator，并配置本机 Git 身份：

```bash
git config user.name "你的 GitHub 用户名或姓名"
git config user.email "你的邮箱"
```

## 2. 每次开始工作前

先同步远程最新状态：

```bash
git switch main
git pull --rebase origin main
git status -sb
```

若已有本地未提交修改，先不要强行 pull；应先 `git status -sb` 查看改动，再决定提交、暂存或请人处理冲突。

## 3. 目录和关键文件

- `C题/2026年度“策联杯”数学建模精英联赛-C题.pdf`：赛题原文。
- `C题/2026年度“策联杯”数学建模精英联赛-C题-附件/`：题目附件和格式规范。
- `C题/XXX_参赛论文.pdf`：当前生成的参赛论文，提交前必须把 `XXX` 替换为正式三位参赛队号。
- `C题/material/`：支撑材料、代码、结果、论文 LaTeX 源文件。
- `C题/material/README.md`：复现说明。
- `C题/material/REPRODUCIBILITY_VERIFICATION.md`：队友 Agent 留下的复现验证记录。

注意：代码默认从 `C题/material/user_data/C题_数据附件.xlsx` 读取官方 Excel；当前仓库里的官方 Excel 位于原题附件目录。复现前如该路径缺失，需要把官方 Excel 复制到 `C题/material/user_data/C题_数据附件.xlsx`。

## 4. Python 复现

建议在 `C题/material/` 下创建独立虚拟环境：

```bash
cd C题/material
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

完整复现：

```bash
python reproduce.py
```

只核验已有结果并重新生成图表：

```bash
python reproduce.py --reuse
```

若依赖安装失败，先记录 Python 版本、系统、完整报错，不要直接改模型代码。原复现记录使用 Python 3.12.13。

## 5. 论文编译

进入论文目录：

```bash
cd C题/material/paper
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

AI 使用详情 PDF 使用：

```bash
xelatex -interaction=nonstopmode -halt-on-error ai_detail_main.tex
```

## 6. 修改与提交

修改前先确认工作区：

```bash
git status -sb
```

完成一组逻辑相关修改后：

```bash
git add <修改的文件>
git commit -m "简短说明本次修改"
git pull --rebase origin main
git push origin main
```

提交信息应说明实际内容，例如：

```bash
git commit -m "Fix result data path in reproduction docs"
git commit -m "Update AI disclosure with verified usage"
git commit -m "Rename final paper with team id"
```

## 7. 冲突处理原则

不要使用 `git reset --hard`、`git checkout -- <file>` 或删除文件来“快速解决”冲突，除非明确知道这些改动不需要保留。

遇到冲突时：

```bash
git status -sb
```

逐个打开冲突文件，保留正确版本，删除冲突标记：

```text
<<<<<<<
=======
>>>>>>>
```

然后：

```bash
git add <已解决的文件>
git rebase --continue
git push origin main
```

如果冲突涉及论文核心结论、正式 CSV、AI 使用详情或提交命名，先暂停并让队员人工确认。

## 8. 提交比赛前检查

- 把 `XXX_参赛论文.pdf` 改为官方要求的三位队号文件名。
- 核对 `AI 工具使用详情.pdf` 与真实使用记录一致。
- 确认正式 CSV 行数和字段符合题面模板。
- 确认支撑材料是否允许包含官方附件、字体文件和原题数据。
- 重新运行 `python reproduce.py --reuse` 或记录无法运行的环境原因。
- 最后执行 `git status -sb`，确保无遗漏改动。
