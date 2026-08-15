"""四个子问题共用的可复现、数据摄入和序列化工具。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(_HERE).parent
DATA_FILE = ROOT / "user_data" / "C题_数据附件.xlsx"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output"


def set_all_seeds(seed: int) -> None:
    """固定 Python 与 NumPy 随机源；各模型另显式接收同一 seed。"""
    random.seed(seed)
    np.random.seed(seed)


def load_all_sheets() -> dict[str, pd.DataFrame]:
    """完整读取 Excel；禁止依赖 pandas 默认首表行为。"""
    sheets = pd.read_excel(DATA_FILE, sheet_name=None)
    profile = json.loads((ROOT / "DATA_PROFILE.json").read_text(encoding="utf-8"))
    expected = profile["files"][DATA_FILE.name]
    actual_rows = int(sum(len(df) for df in sheets.values()))
    assert len(sheets) == int(expected["n_sheets"]), (
        f"[摄入不全] 实读 {len(sheets)} 张表，建档为 {expected['n_sheets']} 张"
    )
    assert actual_rows == int(expected["total_rows"]), (
        f"[摄入不全] 实读 {actual_rows} 行，建档为 {expected['total_rows']} 行"
    )
    expected_names = set(expected["sheets"])
    assert set(sheets) == expected_names, (
        f"[摄入不全] sheet 集合差异: {sorted(set(sheets) ^ expected_names)}"
    )
    for name, df in sheets.items():
        prof_rows = int(expected["sheets"][name]["rows"])
        assert len(df) == prof_rows, f"[摄入不全] {name}: {len(df)} != {prof_rows}"
    return sheets


def assert_key_equal(*dfs_and_cols: tuple[pd.DataFrame, str]) -> None:
    """断言多张表的业务主键集合完全相同且各自唯一。"""
    sets = []
    for df, col in dfs_and_cols:
        assert df[col].is_unique, f"主键 {col} 不唯一"
        sets.append(set(df[col].astype(str)))
    assert all(s == sets[0] for s in sets[1:]), "跨表主键集合不一致"


def write_json(path: Path, payload: dict) -> None:
    """以 UTF-8 和有限浮点数写出机器可读结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def runtime_metadata(seed: int) -> dict:
    """记录复现实验所需的核心版本与运行环境。"""
    def installed(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "seed": int(seed),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": installed("scipy"),
        "scikit_learn": installed("scikit-learn"),
        "catboost": installed("catboost"),
        "ortools": installed("ortools"),
        "openpyxl": installed("openpyxl"),
        "run_id": f"cumcm-c-{seed}",
        "data_file": DATA_FILE.name,
    }


def minmax(value, lower: float, upper: float):
    """题定零极差规则的 Min-Max 归一化。"""
    if abs(float(upper) - float(lower)) <= 1e-12:
        return np.zeros_like(value, dtype=float) if hasattr(value, "__len__") else 0.0
    return (value - lower) / (upper - lower)
