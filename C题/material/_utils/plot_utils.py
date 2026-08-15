"""项目绘图风格与输出适配；渲染能力直接由 Matplotlib 提供。"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


PALETTE = ["#1F5A7A", "#D97841", "#4C956C", "#A44A3F", "#7B6D8D"]
COLORS = {
    "text": "#24313A",
    "ref_line": "#6F7A80",
    "grid": "#D8DEE2",
    "light": "#FFFFFF",
    "bg_box": "#F5F7F8",
}


def _register_chinese_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(__file__).resolve().parents[1] / "paper" / "simsun.ttc",
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    return "DejaVu Sans"


def setup_style(style: str = "nature") -> None:
    """配置统一出版风格；参数名保留原入口契约。"""
    font_name = _register_chinese_font()
    mpl.rcParams.update({
        "font.family": [font_name, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": COLORS["ref_line"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.55,
        "grid.alpha": 0.55,
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, output: str | Path) -> None:
    """通过 Matplotlib 的正式 PDF 后端保存并释放图对象。"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
