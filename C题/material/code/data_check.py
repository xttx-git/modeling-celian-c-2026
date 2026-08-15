"""附件数据摄入验证：必须先于四问求解运行。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
from pathlib import Path

import pandas as pd

from utils import DATA_FILE, ROOT, assert_key_equal, load_all_sheets


def main() -> None:
    sheets = load_all_sheets()
    print(f"[data_check] 文件={DATA_FILE.name} sheets={len(sheets)} total_rows={sum(map(len, sheets.values()))}")
    for name, df in sheets.items():
        missing = int(df.isna().sum().sum())
        print(f"  {name}: rows={len(df)} cols={len(df.columns)} missing_cells={missing}")
        print("    columns=" + ",".join(map(str, df.columns)))

    hist = sheets["historical_matches"].copy()
    split_counts = hist["dataset_split"].value_counts(dropna=False).to_dict()
    assert split_counts == {"train": 560, "test": 140}, f"固定 split 异常: {split_counts}"
    train = hist.loc[hist["dataset_split"].eq("train")]
    test = hist.loc[hist["dataset_split"].eq("test")]
    assert train["match_id"].is_unique and test["match_id"].is_unique
    assert set(train["match_id"]).isdisjoint(set(test["match_id"]))
    assert train["tv_viewers"].notna().all(), "train 标签存在缺失"
    assert test["tv_viewers"].isna().all(), "test 标签不应可见"

    assert_key_equal(
        (sheets["groups_matches"], "match_id"),
        (sheets["base_predictions"], "match_id"),
        (sheets["security_requirements"], "match_id"),
    )
    assert len(sheets["live_group_results"]) == 48
    assert set(sheets["live_group_results"]["round_in_group"]) == {1, 2}
    print("[data_check] PASS 固定560/140 split、72场跨表键、48场前两轮和全表摄入均一致")


if __name__ == "__main__":
    main()
