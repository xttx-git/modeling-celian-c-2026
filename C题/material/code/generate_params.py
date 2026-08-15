"""从 PROBLEM_FACTS.json 自动展开跨问共享命名参数。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
from pathlib import Path

ROOT = Path(_HERE).parent
facts = json.loads((ROOT / "PROBLEM_FACTS.json").read_text(encoding="utf-8"))
p2 = facts["problem_2"]
p3 = facts["problem_3"]
tour = facts["tournament"]
schedule = facts["schedule_constants"]

values = {
    "MASTER_SEED": 20260813,
    "N_TIME_SPLITS": 5,
    "TARGET_SCALE": 1_000_000,
    "N_TEAMS": tour["teams"],
    "N_GROUPS": tour["groups"],
    "N_GROUP_MATCHES": tour["group_matches"],
    "N_THIRD_ROUND_MATCHES": tour["third_round_matches"],
    "MC_DRAWS": p3["simulation_count"],
    "MINIMUM_REST_HOURS": schedule["minimum_rest_hours"],
    "GLOBAL_PRIME_FRACTION": schedule["global_prime_fraction"],
    "MAX_PRIME_COUNT_RANGE": schedule["maximum_team_prime_count_range"],
    "LARGE_VENUE_FRACTION": schedule["large_venue_fraction"],
    "HIGH_SECURITY_THRESHOLD": schedule["high_security_requirement_threshold"],
    "SECURITY_BASE_COST_USD": p2["security_base_cost_amount"] * 10_000,
    "P2_WEIGHTS": p2["objective_weights"],
    "TRAVEL_WEIGHTS": p2["travel_weights"],
    "P2_FAIRNESS_WEIGHTS": p2["fairness_weights"],
    "P2_RISK_WEIGHTS": p2["risk_weights"],
    "STATE_WEIGHTS": p3["state_weights"],
    "STATE_XG_SCALE": p3["state_xg_scale"],
    "RED_CARD_DECAY": p3["red_card_decay"],
    "DELTA_STATE_COEF": p3["delta_state_coefficient"],
    "DELTA_INJURY_COEF": p3["delta_injury_coefficient"],
    "DIRECT_INJURY_COEF": p3["direct_injury_coefficient"],
    "GOAL_MEAN_CLIP": p3["goal_mean_clip"],
    "RAW_FEEDBACK_CLIP": p3["raw_feedback_clip"],
    "TEAM_FEEDBACK_CLIP": p3["team_feedback_clip"],
    "ATTENDANCE_FLOOR_RATIO": p3["attendance_floor_ratio"],
    "P3_WEIGHTS": p3["objective_weights"],
}

header = '''"""自动生成：唯一数据源为 ../PROBLEM_FACTS.json；请勿手改常数。"""\nimport json\nfrom pathlib import Path\n_FACTS = json.loads((Path(__file__).resolve().parent.parent / "PROBLEM_FACTS.json").read_text(encoding="utf-8"))\n'''
body = []
for name, value in values.items():
    body.append(f"{name} = {value!r}")
body.extend([
    "CP_SAT_SCALE = 1_000_000  # dimensionless integer objective scale",
    "P2_TIME_LIMIT_SECONDS = 1800  # second, solver hard limit",
    "P3_TIME_LIMIT_SECONDS = 300  # second, solver hard limit",
    "assert MC_DRAWS == _FACTS['problem_3']['simulation_count']",
    "assert MINIMUM_REST_HOURS == _FACTS['schedule_constants']['minimum_rest_hours']",
    "assert abs(sum(abs(float(v if not isinstance(v, dict) else v['magnitude'])) for v in P2_WEIGHTS.values()) - 1.0) < 1e-12",
    "print('[params] 题面参数与跨问口径一致性 OK')",
])
(Path(_HERE) / "params.py").write_text(header + "\n".join(body) + "\n", encoding="utf-8")
print(f"[generate_params] wrote {Path(_HERE) / 'params.py'}")

