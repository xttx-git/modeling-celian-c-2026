"""自动生成：唯一数据源为 ../PROBLEM_FACTS.json；请勿手改常数。"""
import json
from pathlib import Path
_FACTS = json.loads((Path(__file__).resolve().parent.parent / "PROBLEM_FACTS.json").read_text(encoding="utf-8"))
MASTER_SEED = 20260813
N_TIME_SPLITS = 5
TARGET_SCALE = 1000000
N_TEAMS = 48
N_GROUPS = 12
N_GROUP_MATCHES = 72
N_THIRD_ROUND_MATCHES = 24
MC_DRAWS = 20000
MINIMUM_REST_HOURS = 60
GLOBAL_PRIME_FRACTION = 0.25
MAX_PRIME_COUNT_RANGE = 2
LARGE_VENUE_FRACTION = 0.25
HIGH_SECURITY_THRESHOLD = 3
SECURITY_BASE_COST_USD = 100000
P2_WEIGHTS = {'ticket_revenue': 0.25, 'broadcast_value': 0.25, 'uncertainty': 0.15, 'attractiveness': 0.1, 'organization_cost': {'magnitude': 0.08, 'direction': 'subtract'}, 'travel_burden': {'magnitude': 0.07, 'direction': 'subtract'}, 'fairness_penalty': {'magnitude': 0.06, 'direction': 'subtract'}, 'execution_risk': {'magnitude': 0.04, 'direction': 'subtract'}}
TRAVEL_WEIGHTS = {'distance': 0.5, 'travel_time': 0.3, 'timezone_difference': 0.2}
P2_FAIRNESS_WEIGHTS = {'prime_time_range': 0.5, 'large_venue_range': 0.5, 'divisor': 3}
P2_RISK_WEIGHTS = {'climate': 0.5, 'occupancy': 0.3, 'security_ratio': 0.2}
STATE_WEIGHTS = {'points': 0.45, 'xg_form': 0.35, 'discipline': 0.2}
STATE_XG_SCALE = 1.25
RED_CARD_DECAY = 0.55
DELTA_STATE_COEF = 0.32
DELTA_INJURY_COEF = 0.055
DIRECT_INJURY_COEF = 0.04
GOAL_MEAN_CLIP = [0.15, 4.5]
RAW_FEEDBACK_CLIP = [0.6, 1.5]
TEAM_FEEDBACK_CLIP = [0.75, 1.25]
ATTENDANCE_FLOOR_RATIO = 0.88
P3_WEIGHTS = {'ticket_value': 0.35, 'broadcast_value': 0.35, 'attractiveness': 0.1, 'resource_cost': {'magnitude': 0.1, 'direction': 'subtract'}, 'risk': {'magnitude': 0.1, 'direction': 'subtract'}}
CP_SAT_SCALE = 1_000_000  # dimensionless integer objective scale
P2_TIME_LIMIT_SECONDS = 1800  # second, solver hard limit
P3_TIME_LIMIT_SECONDS = 300  # second, solver hard limit
assert MC_DRAWS == _FACTS['problem_3']['simulation_count']
assert MINIMUM_REST_HOURS == _FACTS['schedule_constants']['minimum_rest_hours']
assert abs(sum(abs(float(v if not isinstance(v, dict) else v['magnitude'])) for v in P2_WEIGHTS.values()) - 1.0) < 1e-12
print('[params] 题面参数与跨问口径一致性 OK')
