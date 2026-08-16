# LaTeX 论文编译

正式论文源文件为：

- `main.tex`：摘要专页、格式、AI 声明、参考文献和代码附录。
- `body.tex`：正文第 1 至第 8 节。
- `generated/`：由 `../code/paper_analysis.py` 自动生成的图、表和追溯数据。

先在 `C题` 目录创建固定环境、生成正式输出与论文分析资产：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-paper.txt
python code/run_all.py --output-dir output
python code/check_outputs.py --output-dir output
python code/paper_analysis.py --output-dir output --generated-dir paper/generated
```

Windows PowerShell 的激活命令为 `.venv\Scripts\Activate.ps1`。

在 Windows PowerShell 中从 `C题/paper` 目录编译：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

清理辅助文件但保留 PDF：

```powershell
latexmk -c
```

当前输出为 `main.pdf`。正式提交前，将其按实际三位参赛队号改名为 `XXX_参赛论文.pdf`，并用真实队号替换 `XXX`。

格式依据：摘要为第一页且不超过一页；第二页开始正文；不生成目录；正文、AI 声明和参考文献控制在 30 页内；参考文献后为不限页数附录。格式规范第三条要求附录包含全部完整可运行源程序，因此 `main.tex` 会在参考文献后引入四个 Python 文件；支撑材料中仍需另行提供这些源文件。
