"""模型检验与灵敏度：P2权重响应面、P3状态/伤病参数3×3重算。"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import json
import numpy as np
import pandas as pd

import problem3 as p3
from params import MASTER_SEED
from utils import FIGURES_DIR, OUTPUT_DIR, load_all_sheets, write_json


def p2_fixed_schedule_surface(metrics):
    """固定最优赛程的权重响应，不冒充每格重新优化。"""
    commercial=np.linspace(0.9,1.1,5); penalty=np.linspace(0.9,1.1,5); rows=[]
    for c in commercial:
        for p in penalty:
            raw=np.array([.25*c,.25*c,.15,.10,.08*p,.07*p,.06*p,.04*p]); raw=raw/raw.sum()
            z=raw[0]*metrics["T"]+raw[1]*metrics["B"]+raw[2]*metrics["U"]+raw[3]*metrics["H"]-raw[4]*metrics["C"]-raw[5]*metrics["D"]-raw[6]*metrics["F"]-raw[7]*metrics["R"]
            rows.append({"commercial_multiplier":float(c),"penalty_multiplier":float(p),"fixed_schedule_Z2":float(z)})
    return rows


def p3_surface():
    sheets=load_all_sheets(); groups=sheets["groups_matches"]
    base=groups.merge(sheets["base_predictions"],on=["match_id","group_id","round_in_group","team_a","team_b"],validate="one_to_one")
    base=base.merge(sheets["security_requirements"][["match_id","security_demand_score","required_security_level"]],on="match_id",validate="one_to_one")
    p1=pd.read_csv(OUTPUT_DIR/"result_1_match_prediction.csv");base=base.merge(p1[["match_id","predicted_tv_viewers"]],on="match_id",validate="one_to_one")
    round3=base[base.round_in_group.eq(3)].reset_index(drop=True);teams=sheets["group_membership"].team_name.astype(str).tolist()
    snap=p3.team_snapshot(sheets["live_group_results"],teams);feedback=p3.feedback_by_team(sheets["live_group_results"],sheets["base_predictions"],p1)
    schedule=pd.read_csv(OUTPUT_DIR/"result_2_group_schedule.csv");rows=[]
    state0,inj0=p3.DELTA_STATE_COEF,p3.DELTA_INJURY_COEF
    for sf in (0.9,1.0,1.1):
        for hf in (0.9,1.0,1.1):
            p3.DELTA_STATE_COEF=state0*sf;p3.DELTA_INJURY_COEF=inj0*hf
            la,lb=p3.updated_lambdas(round3,snap);sim=p3.simulate_qualification(round3,sheets["group_membership"],snap,la,lb,MASTER_SEED,keep_scores=True)
            probs={t:float(sim["probability"][i]) for i,t in enumerate(sim["teams"])};cond=p3.conditional_probabilities(sim,round3)
            env=p3.derive_match_environment(round3,schedule,sheets["venues"],sheets["ticket_broadcast"],sheets["dynamic_resource_limits"],snap,feedback,probs,cond)
            actions=p3.make_action_table(env,sheets["dynamic_resource_costs"],sheets["dynamic_resource_limits"],True);bounds=p3.action_bounds(actions)
            chosen,_=p3.optimize_actions(actions,sheets["dynamic_resource_limits"],bounds,f"sens_{sf}_{hf}")
            z=sum(p3.action_value(a,bounds) for a in chosen)
            rows.append({"state_coefficient_multiplier":sf,"injury_coefficient_multiplier":hf,"Z3":float(z),"mean_advancement_probability":float(np.mean(list(probs.values())))})
            print(f"[sensitivity] state={sf:.1f} injury={hf:.1f} Z3={z:.6f}",flush=True)
    p3.DELTA_STATE_COEF,p3.DELTA_INJURY_COEF=state0,inj0
    return rows


def main():
    p2=json.loads((FIGURES_DIR/"problem_2_results.json").read_text(encoding="utf-8"))
    p4=json.loads((FIGURES_DIR/"problem_4_results.json").read_text(encoding="utf-8"))
    result={"p2_weight_surface":p2_fixed_schedule_surface(p2["metrics"]),"p2_note":"固定主赛程重评，不冒充逐格重优化",
            "p3_state_injury_surface":p3_surface(),"p4_weight_robustness":p4["weight_robustness"],"seed":MASTER_SEED}
    write_json(FIGURES_DIR/"sensitivity_results.json",result)
    print("[sensitivity] PASS")


if __name__=="__main__":
    main()

