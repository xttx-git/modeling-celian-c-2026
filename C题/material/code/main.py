"""竞赛编程实现总入口：按 P1→P2→P3→P4→灵敏度分析串联并汇总结果。

默认执行完整计算；使用 ``--reuse`` 时仅验证并汇总工作区已经完成的四问结果，
适合交付前快速检查。任一子问题失败都会返回非零退出码，不会把残缺结果冒充成功。
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import argparse
import json
import shutil
from pathlib import Path

from utils import FIGURES_DIR, OUTPUT_DIR, write_json


def run_all(force: bool) -> dict:
    modules=[]
    if force:
        from problem1 import solve as solve1
        from problem2 import solve as solve2
        from problem3 import solve as solve3
        from problem4 import solve as solve4
        for n,func in enumerate((solve1,solve2,solve3,solve4),1):
            print(f"[main] running problem {n}",flush=True)
            func()
        # 灵敏度结果被正文与图表直接引用，必须属于正式复现链，不能依赖手工补跑。
        from sensitivity_analysis import main as solve_sensitivity
        print("[main] running sensitivity analysis", flush=True)
        solve_sensitivity()
    for n in range(1,5):
        path=FIGURES_DIR/f"problem_{n}_results.json"
        if not path.exists() or path.stat().st_size==0:
            raise FileNotFoundError(f"缺少非空结果: {path}")
        modules.append(json.loads(path.read_text(encoding="utf-8")))
    all_results={f"problem_{i+1}":value for i,value in enumerate(modules)}
    sensitivity=FIGURES_DIR/"sensitivity_results.json"
    if sensitivity.exists():
        all_results["sensitivity"]=json.loads(sensitivity.read_text(encoding="utf-8"))
    write_json(FIGURES_DIR/"all_results.json",all_results)
    # 题目模板要求的交付文件位于工作区根目录；output/ 保留同一份过程副本。
    for name in [
        "result_1_test_prediction.csv", "result_1_match_prediction.csv",
        "result_2_group_schedule.csv", "result_3_dynamic_strategy.csv",
        "result_4_schedule_comparison.csv", "actual_schedule_2022.csv",
    ]:
        source=Path(OUTPUT_DIR)/name
        if not source.exists() or source.stat().st_size==0:
            raise FileNotFoundError(f"缺少交付文件: {source}")
        shutil.copyfile(source, Path(FIGURES_DIR).parent/name)
    print(f"[main] PASS problems={len(modules)} all_results={FIGURES_DIR/'all_results.json'}",flush=True)
    return all_results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse",action="store_true",help="复用已完成结果，只做汇总")
    args=parser.parse_args()
    run_all(force=not args.reuse)
