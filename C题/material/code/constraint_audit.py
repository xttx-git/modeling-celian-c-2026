"""从最终 JSON/CSV 独立重算四问硬约束；仅输出结论和最多5个定位。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from params import ATTENDANCE_FLOOR_RATIO, GOAL_MEAN_CLIP, MINIMUM_REST_HOURS
from utils import FIGURES_DIR, OUTPUT_DIR, load_all_sheets

_BASE_KW = re.compile(r"baseline|naive|greedy|就近|等权|均分|不调整|lower.?bound|下界", re.I)


def _discover_baselines(obj, prefix=""):
    found=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            path=f"{prefix}.{k}" if prefix else k
            if _BASE_KW.search(k): found.append(path)
            if isinstance(v,dict): found.extend(_discover_baselines(v,path))
    return found


def main() -> None:
    sheets=load_all_sheets(); audited=set(); failures=[]; n_constraints=0
    p1=json.loads((FIGURES_DIR/"problem_1_results.json").read_text(encoding="utf-8"))
    p2=json.loads((FIGURES_DIR/"problem_2_results.json").read_text(encoding="utf-8"))
    p3=json.loads((FIGURES_DIR/"problem_3_results.json").read_text(encoding="utf-8"))
    p4=json.loads((FIGURES_DIR/"problem_4_results.json").read_text(encoding="utf-8"))
    discovered=_discover_baselines({"p1":p1,"p2":p2,"p3":p3,"p4":p4})

    # P1：切分、交付和正值。
    test=pd.read_csv(OUTPUT_DIR/"result_1_test_prediction.csv"); group=pd.read_csv(OUTPUT_DIR/"result_1_match_prediction.csv")
    validation_policy=str(p1.get("validation_target_history_policy", ""))
    ok=(len(test)==140 and test.match_id_test.is_unique and (test.predicted_test_tv_viewers>0).all()
        and len(group)==72 and group.match_id.is_unique and (group.predicted_tv_viewers>0).all()
        and "mask" in validation_policy.lower())
    n_constraints+=6
    if not ok: failures.append("P1 split/delivery/range")
    print(f"[P1] PASS={ok} n_violations={0 if ok else 1} rows_test={len(test)} rows_group={len(group)}")

    # P2 最优解与自动发现的可行基线使用同一审计函数。
    from problem2 import prepare_data, validate_schedule, evaluate_rows
    _,groups,venues,slots,limits,_=prepare_data(); bounds=p2["normalization_bounds"]
    for name,rows in [("p2.schedule",p2["schedule"]),("p2.feasible_baseline",p2["feasible_baseline"]["schedule"])]:
        try:
            a=validate_schedule(rows,groups,venues,slots,limits); m=evaluate_rows(rows,groups,venues,bounds)
            ok=a["pass"] and all(-1e-9<=m[k]<=1+1e-9 for k in ["T","B","U","H","C","D","F","R"])
        except Exception as e:
            ok=False; failures.append(f"{name}: {e}")
        n_constraints+=10; audited.add(name)
        print(f"[{name}] PASS={ok} n_violations={0 if ok else 1}")

    # P3：守恒、概率/lambda、固定赛程、安保、票价需求和日容量。
    mass=np.asarray(p3["qualifier_mass_by_draw"],float); probs=np.asarray([x["probability"] for x in p3["team_advancement"]]);
    lamb=np.asarray([y for x in p3["lambdas"] for y in (x["lambda_a"],x["lambda_b"])]); rows=p3["rows"]
    sched=pd.read_csv(OUTPUT_DIR/"result_2_group_schedule.csv").set_index("match_id")
    seed_runs=p3.get("five_seed_decision_stability", [])
    checks=[np.max(np.abs(mass-32))<=1e-6, np.all((0<=probs)&(probs<=1)), abs(probs.sum()-32)<=1e-5,
            np.all((GOAL_MEAN_CLIP[0]<=lamb)&(lamb<=GOAL_MEAN_CLIP[1])),
            all((r["venue_id"],r["slot_id"])==(sched.loc[r["match_id"],"venue_id"],sched.loc[r["match_id"],"slot_id"]) for r in rows),
            all(r["required_security_level"]<=r["recommended_security_level"]<=r["venue_security_level"] for r in rows),
            all(r["updated_expected_attendance"]+1e-8>=ATTENDANCE_FLOOR_RATIO*r["attendance_at_zero"] for r in rows),
            all(x["pass"] for x in p3["daily_static_audit"]),all(x["pass"] for x in p3["daily_dynamic_audit"]),
            p3["Z3_dynamic"]+1e-4>=p3["Z3_static_reeval"],
            len(seed_runs)==5 and all(x["solver_status"]=="OPTIMAL" and x["improvement"]>=-1e-8 for x in seed_runs)]
    ok=all(checks); n_constraints+=len(checks)
    if not ok: failures.append("P3 hard checks")
    print(f"[P3] PASS={ok} n_violations={sum(not x for x in checks)} max_mass_error={np.max(np.abs(mass-32)):.3e}")

    # P4：真实行、来源、UTC、共同归一化边界及代理敏感性。
    ar=p4["actual_schedule_rows"]; utc_ok=True
    for r in ar:
        local=pd.Timestamp(r["local_datetime"]); utc=pd.Timestamp(r["utc_datetime"])
        utc_ok &= local.tz_convert("UTC")==utc
    normalized_names=["T","B","U","H","C","D","F","R"]
    normalized=[p4[side][name] for side in ("actual_metrics","optimized_metrics") for name in normalized_names]
    common_bounds=p4["common_normalization_bounds"]
    shared_bounds=all(p4[side]["normalization_bounds"]==common_bounds for side in ("actual_metrics","optimized_metrics"))
    checks=[len(ar)==48,
            all("openfootball" in r["source_url"].lower() for r in ar),
            all(bool(r["official_schedule_url"]) and bool(r["stadium_source_url"]) for r in ar),
            utc_ok,
            all(np.isfinite(x["actual_schedule_value"]) and np.isfinite(x["optimized_schedule_value"]) for x in p4["comparison"]),
            all(-1e-9<=x<=1+1e-9 for x in normalized),
            shared_bounds,
            p4["proxy_diagnostics"]["actual_team_elo_coverage"]=="32/32",
            p4["direction_robustness_rate"]>=0.9 and p4["second_neighbor_sensitivity"]["direction_preserved"]]
    ok=all(checks); n_constraints+=len(checks)
    if not ok: failures.append("P4 row/source/UTC/finite")
    print(f"[P4] PASS={ok} n_violations={sum(not x for x in checks)} actual_rows={len(ar)}")

    # 自动发现的 baseline 必须有明确审计对象；只把叶级/方案级路径纳入登记断言。
    required=[x for x in discovered if x.endswith("feasible_baseline")]
    missing=[x for x in required if x not in audited]
    if missing: failures.append(f"基线漏审: {missing}")
    print(f"[baseline_registry] discovered={required} audited={sorted(audited & set(required))} missing={missing}")
    if failures:
        for x in failures[:5]: print(f"  VIOLATION {x}")
        raise SystemExit(1)
    print(f"[constraint_audit] PASS n_constraints={n_constraints} n_violations=0")


if __name__=="__main__":
    main()
