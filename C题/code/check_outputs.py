#!/usr/bin/env python3
"""Formula-level acceptance checks for the revised Problem C pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline_core import (
    MATERIAL_DIR,
    P1_NUMERIC_FEATURES,
    RANDOM_SEEDS,
    ROOT,
    TEMPLATE_DIR,
    actual_schedule_2022,
    build_p2_context,
    build_p3_actions,
    build_prediction_features,
    competition_normalization_diagnostics,
    dynamic_p3_environment,
    evaluate_p3_action,
    exact_top_quartile_ids,
    p2_hard_constraint_audit,
    p2_metrics,
    p3_constraints_audit,
    p3_normalize,
    read_data,
    normalize_competition_name,
    simulate_advancement,
    solve_p3_actions,
    solve_problem1,
    standings_from_live,
    static_p3_environment,
    swap_match_sides,
    utc_timestamp,
)


TEMPLATE_NAMES = {
    "result_1_test_prediction.csv": "result_1_test_prediction_template.csv",
    "result_1_match_prediction.csv": "result_1_match_prediction_template.csv",
    "result_2_group_schedule.csv": "result_2_template.csv",
    "result_3_dynamic_strategy.csv": "result_3_template.csv",
    "actual_schedule_2022.csv": "actual_schedule_template.csv",
    "result_4_schedule_comparison.csv": "result_4_template.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output_v2")
    return parser.parse_args()


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def assert_close(actual: float, expected: float, message: str, atol=1e-6, rtol=1e-8) -> None:
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=rtol):
        fail(f"{message}: actual={actual}, expected={expected}")


def finite_nonnegative(series: pd.Series, name: str) -> None:
    values = pd.to_numeric(series, errors="coerce")
    assert_true(values.notna().all() and np.isfinite(values).all(), f"{name} contains non-finite values")
    assert_true((values >= 0).all(), f"{name} contains negative values")


class Acceptance:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.data = read_data()
        self.diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
        self.output: dict[str, pd.DataFrame] = {}

    def check_environment(self) -> None:
        requirements = ROOT / "requirements.txt"
        assert_true(requirements.is_file(), "requirements.txt is missing")
        expected_packages = {
            "numpy": "2.5.2",
            "pandas": "3.0.5",
            "scipy": "1.18.0",
            "scikit-learn": "1.9.0",
            "openpyxl": "3.1.5",
            "joblib": "1.5.3",
            "threadpoolctl": "3.6.0",
        }
        actual_packages = {
            name: importlib.metadata.version(name) for name in expected_packages
        }
        assert_true(actual_packages == expected_packages, f"Runtime package versions differ: {actual_packages}")
        environment = self.diagnostics["environment"]
        assert_true(environment["python"] == platform.python_version(), "Recorded Python version mismatch")
        assert_true(environment["packages"] == actual_packages, "Recorded dependency versions mismatch")
        assert_true(
            environment["requirements_sha256"] == hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "Recorded requirements.txt hash mismatch",
        )
        assert_true(self.diagnostics["random_seeds"] == RANDOM_SEEDS, "Recorded random seeds mismatch")

    def read_outputs(self) -> None:
        for output_name, template_name in TEMPLATE_NAMES.items():
            output = pd.read_csv(self.output_dir / output_name)
            template = pd.read_csv(TEMPLATE_DIR / template_name)
            assert_true(
                list(output.columns) == list(template.columns),
                f"{output_name} columns differ from official template",
            )
            self.output[output_name] = output

    def check_problem1(self) -> None:
        test = self.output["result_1_test_prediction.csv"]
        future = self.output["result_1_match_prediction.csv"]
        historical_test = self.data["historical_matches"].query("dataset_split == 'test'")
        groups = self.data["groups_matches"]
        assert_true(len(test) == 140 and test["match_id_test"].is_unique, "P1 test must have 140 unique IDs")
        assert_true(test["match_id_test"].tolist() == historical_test["match_id"].tolist(), "P1 test IDs/order mismatch")
        assert_true(len(future) == 72 and future["match_id"].is_unique, "P1 future must have 72 unique matches")
        assert_true(
            future[["match_id", "team_a", "team_b"]].reset_index(drop=True).equals(
                groups[["match_id", "team_a", "team_b"]].reset_index(drop=True)
            ),
            "P1 future match/team identity mismatch",
        )
        finite_nonnegative(test["predicted_test_tv_viewers"], "P1 test predictions (million)")
        finite_nonnegative(future["predicted_tv_viewers"], "P1 future predictions (persons)")
        assert_true(test["predicted_test_tv_viewers"].mean() < 1000, "P1 test output is not in millions")
        assert_true(future["predicted_tv_viewers"].mean() > 1_000_000, "P1 future output is not in persons")

        forbidden = {"attractiveness_index", "commercial_value_index", "expected_attendance_base", "uncertainty_index"}
        recorded = self.diagnostics["problem1_feature_columns"]
        assert_true(not forbidden.intersection(recorded["numeric"] + recorded["categorical"]), "P1 forbidden features remain")
        assert_true(recorded["numeric"] == P1_NUMERIC_FEATURES, "P1 numeric feature dictionary mismatch")
        assert_true(
            self.diagnostics["problem1_feature_builder"]
            == "build_prediction_features_v3_symmetric_competition_normalized",
            "P1 feature builder signature mismatch",
        )

        regenerated_test, regenerated_future, regenerated_diag = solve_problem1(self.data)
        assert_true(
            np.allclose(test["predicted_test_tv_viewers"], regenerated_test["predicted_test_tv_viewers"], atol=5e-7),
            "P1 test CSV differs from regenerated model output",
        )
        assert_true(
            np.allclose(future["predicted_tv_viewers"], regenerated_future["predicted_tv_viewers"], atol=0),
            "P1 future CSV differs from regenerated model output",
        )
        assert_true(regenerated_diag["problem1_swap_invariance_max_abs_error_million"] <= 1e-10, "P1 side-swap invariance failed")
        assert_close(
            self.diagnostics["problem1_swap_invariance_max_abs_error_million"],
            regenerated_diag["problem1_swap_invariance_max_abs_error_million"],
            "P1 recorded swap error mismatch", atol=1e-12,
        )
        recorded_metrics = {
            row["model"]: row for row in self.diagnostics["problem1_model_metrics"]
        }
        regenerated_metrics = {
            row["model"]: row for row in regenerated_diag["problem1_model_metrics"]
        }
        assert_true(recorded_metrics.keys() == regenerated_metrics.keys(), "P1 model metric names mismatch")
        for model_name in recorded_metrics:
            for metric in ("mse", "rmse", "mae"):
                assert_close(
                    recorded_metrics[model_name][metric],
                    regenerated_metrics[model_name][metric],
                    f"P1 {model_name} {metric} diagnostic mismatch",
                    atol=1e-10,
                )
        train = self.data["historical_matches"].query("dataset_split == 'train' and tv_viewers == tv_viewers")
        raw_future = groups.merge(
            self.data["base_predictions"].drop(columns=["group_id", "round_in_group", "team_a", "team_b"], errors="ignore"),
            on="match_id", how="left",
        )
        train_features = build_prediction_features(train, self.data["teams"])
        historical_test_raw = self.data["historical_matches"].query("dataset_split == 'test'").copy()
        category_diagnostics = competition_normalization_diagnostics(train, historical_test_raw, raw_future)
        assert_true(
            category_diagnostics == self.diagnostics["problem1_competition_normalization"],
            "P1 competition normalization diagnostics mismatch",
        )
        assert_true(
            {normalize_competition_name(value) for value in (
                "World Cup", "world cup", "FIFA World Cup", "FIFA World Cup 2026", "WORLD-CUP 2026",
            )} == {"World Cup"},
            "P1 World Cup aliases do not normalize to one category",
        )
        assert_true(
            normalize_competition_name("World Cup Qualifier") == "World Cup Qualifier",
            "P1 qualifier was incorrectly merged with World Cup",
        )
        assert_true(
            category_diagnostics["future72"]["after_categories"] == ["World Cup"],
            "P1 future World Cup category does not match training category",
        )
        assert_true(
            category_diagnostics["test"]["unseen_row_count"] == 0
            and category_diagnostics["future72"]["unseen_row_count"] == 0,
            "P1 normalized test/future contains unseen competition rows",
        )
        swapped = build_prediction_features(swap_match_sides(raw_future), self.data["teams"])
        normal = build_prediction_features(raw_future, self.data["teams"])
        assert_true(np.allclose(normal[P1_NUMERIC_FEATURES], swapped[P1_NUMERIC_FEATURES], atol=1e-12), "P1 symmetric features change after side swap")
        for column in P1_NUMERIC_FEATURES:
            stats = self.diagnostics["problem1_feature_distributions"][column]
            mean, std = train_features[column].mean(), train_features[column].std(ddof=1)
            assert_close(stats["train"]["mean"], mean, f"P1 {column} train mean diagnostic mismatch")
            assert_close(stats["train_mean_minus_4sd"], mean - 4 * std, f"P1 {column} lower 4sd bound mismatch")

    def check_problem2(self) -> None:
        output = self.output["result_2_group_schedule.csv"]
        groups = self.data["groups_matches"]
        venues = self.data["venues"].set_index("venue_id")
        slots = self.data["time_slots"].set_index("slot_id")
        assert_true(len(output) == 72 and output["match_id"].is_unique, "P2 must contain 72 unique matches")
        identity_columns = ["match_id", "group_id", "round_in_group", "team_a", "team_b"]
        assert_true(
            output[identity_columns].reset_index(drop=True).equals(groups[identity_columns].reset_index(drop=True)),
            "P2 fixed match identity/order differs from groups_matches",
        )
        p1 = self.output["result_1_match_prediction.csv"]
        context = build_p2_context(self.data, p1)
        for row in output.itertuples(index=False):
            venue, slot = venues.loc[row.venue_id], slots.loc[row.slot_id]
            assert_true(row.city == venue.city and row.country == venue.country, f"P2 venue city/country mismatch {row.match_id}")
            assert_true(str(row.reference_date) == pd.to_datetime(slot.date).date().isoformat(), f"P2 reference date mismatch {row.match_id}")
            assert_true(str(row.reference_kickoff_time)[:5] == str(slot.reference_kickoff_time)[:5], f"P2 kickoff mismatch {row.match_id}")
            utc = utc_timestamp(slot.reference_utc_time)
            local = utc.tz_convert(ZoneInfo(str(venue.timezone)))
            assert_true(str(row.utc_datetime) == utc.strftime("%Y-%m-%d %H:%M"), f"P2 UTC mismatch {row.match_id}")
            assert_true(str(row.local_datetime) == local.strftime("%Y-%m-%d %H:%M"), f"P2 local datetime mismatch {row.match_id}")
            vm = context.venue_candidates.set_index(["match_id", "venue_id"]).xs((row.match_id, row.venue_id))
            sm = context.slot_candidates.set_index(["match_id", "slot_id"]).xs((row.match_id, row.slot_id))
            assert_close(row.expected_attendance, vm.expected_attendance, f"P2 attendance formula mismatch {row.match_id}", atol=0.011)
            assert_close(row.ticket_revenue_usd, vm.ticket_revenue_usd, f"P2 T formula mismatch {row.match_id}", atol=0.011)
            assert_close(row.broadcast_value_usd, sm.broadcast_value_usd, f"P2 B formula mismatch {row.match_id}", atol=0.011)
            assert_close(row.travel_cost_index, vm.travel_burden, f"P2 D formula mismatch {row.match_id}", atol=5e-9)
            assert_close(row.risk_index, vm.risk_index, f"P2 R formula mismatch {row.match_id}", atol=5e-9)
        audit = p2_hard_constraint_audit(output, context, self.data)
        assert_true(audit["all_passed"], f"P2 hard constraints failed: {audit}")
        independent_group_gaps = {}
        schedule_times = output.assign(_utc=pd.to_datetime(output["utc_datetime"], utc=True))
        for group_id, group_matches in schedule_times.groupby("group_id"):
            for round_no in (1, 2):
                previous_max = group_matches.loc[
                    group_matches["round_in_group"].eq(round_no), "_utc"
                ].max()
                next_min = group_matches.loc[
                    group_matches["round_in_group"].eq(round_no + 1), "_utc"
                ].min()
                key = f"{group_id}:R{round_no}-R{round_no + 1}"
                independent_group_gaps[key] = float((next_min - previous_max).total_seconds() / 3600)
        group_violation_count = sum(gap < 60 - 1e-9 for gap in independent_group_gaps.values())
        group_minimum = min(independent_group_gaps.values())
        assert_true(group_violation_count == 0, f"P2 group round boundary violations: {group_violation_count}")
        assert_true(
            independent_group_gaps == audit["group_round_gaps_hours"],
            "P2 group round gaps differ from hard-constraint audit",
        )
        assert_close(
            audit["group_round_minimum_gap_hours"], group_minimum,
            "P2 group round minimum gap mismatch",
        )
        assert_true(
            audit["group_round_violation_count"] == group_violation_count
            and audit["group_round_boundary_passed"],
            "P2 group round boundary audit state mismatch",
        )
        self.group_round_summary = {
            "gaps": independent_group_gaps,
            "minimum": group_minimum,
            "violations": int(group_violation_count),
            "all_passed": bool(audit["all_passed"]),
        }
        assert_true(output.duplicated(["venue_id", "utc_datetime"]).sum() == 0, "P2 venue-UTC conflict")
        for utc_value, subset in output.groupby("utc_datetime"):
            capacities = [int(slots.loc[slot_id, "broadcast_capacity"]) for slot_id in subset["slot_id"]]
            assert_true(len(set(capacities)) == 1 and len(subset) <= capacities[0], f"P2 real UTC broadcast capacity exceeded {utc_value}")
        prime_slots = exact_top_quartile_ids(self.data["time_slots"], "global_prime_score", "slot_id")
        assert_true(len(prime_slots) == 20 and prime_slots == set(self.diagnostics["problem2_prime_slots"]), "P2 prime slots are not exact top 20")
        metrics = p2_metrics(output, context, self.data)
        recorded = self.diagnostics["problem2_indicator_details"]
        for section in ("raw", "normalized", "weighted_contributions"):
            for key, expected in metrics[section].items():
                assert_close(recorded[section][key], expected, f"P2 diagnostic {section}.{key} mismatch", atol=1e-9)
        objective_values = pd.to_numeric(output["total_objective_value"], errors="coerce").dropna()
        assert_true(len(objective_values) == 1, "P2 objective must appear only in the first row")
        assert_close(objective_values.iloc[0], metrics["Z2"], "P2 Z2 mismatch", atol=5e-10)
        solver = self.diagnostics["problem2_solver"]
        assert_true(solver["global_optimality_claim"] is False, "P2 incorrectly claims global 3D optimum")
        valid_statuses = {"optimal_within_mip_tolerance", "limit_reached"}
        assert_true(solver["slot_subproblem"]["status"] in valid_statuses, "P2 slot solver status invalid")
        assert_true(solver["venue_subproblem_given_slots"]["status"] in valid_statuses, "P2 venue solver status invalid")
        for subproblem in ("slot_subproblem", "venue_subproblem_given_slots"):
            detail = solver[subproblem]
            if detail["solver_status"] == "optimal":
                assert_true(
                    detail["interpretation"] == "在设定 MIP 容差内求得最优解",
                    f"P2 {subproblem} optimal status wording mismatch",
                )
        self.p2_context = context

    def check_problem3(self) -> None:
        output = self.output["result_3_dynamic_strategy.csv"]
        p2 = self.output["result_2_group_schedule.csv"]
        third_ids = set(p2.loc[p2["round_in_group"].eq(3), "match_id"])
        assert_true(len(output) == 24 and set(output["match_id"]) == third_ids, "P3 third-round match set mismatch")
        for column in [
            "updated_p_team_a_advance", "updated_p_team_b_advance", "stakeless_risk",
            "collusion_risk", "updated_attractiveness", "risk_exposure_index",
        ]:
            values = pd.to_numeric(output[column], errors="coerce")
            assert_true(values.notna().all() and values.between(0, 1).all(), f"P3 {column} outside [0,1]")
        source = (ROOT / "code" / "pipeline_core.py").read_text(encoding="utf-8")
        assert_true("-team_index" not in source and "team_index)" not in source, "P3 contains team-index tie break")
        assert_true(
            self.diagnostics["tie_method"] == "exact_equal_fractional_weights_without_team_id_tiebreak",
            "P3 tie method diagnostic mismatch",
        )
        assert_close(self.diagnostics["advancement_probability_sum"], 32, "P3 advancement probability sum", atol=1e-10)
        assert_close(self.diagnostics["per_simulation_weight_sum_min"], 32, "P3 per-simulation min weight", atol=1e-10)
        assert_close(self.diagnostics["per_simulation_weight_sum_max"], 32, "P3 per-simulation max weight", atol=1e-10)

        standings = standings_from_live(self.data)
        seed = int(self.diagnostics["seed"])
        simulation_diag, simulation = simulate_advancement(self.data, standings, 20000, seed)
        assert_close(sum(simulation_diag["advancement_probs"].values()), 32, "P3 regenerated advancement sum", atol=1e-10)
        environment = dynamic_p3_environment(self.data, p2, simulation, standings)
        environment_index = environment.set_index("match_id")
        bounds = self.diagnostics["problem3_normalization_bounds"]
        dynamic_rows = []
        for row in output.itertuples(index=False):
            env = environment_index.loc[row.match_id].copy(); env["match_id"] = row.match_id
            evaluated = evaluate_p3_action(
                self.data, env, int(row.recommended_broadcast_priority), int(row.recommended_security_level),
                int(row.recommended_transport_level), float(row.recommended_ticket_adjustment),
            )
            dynamic_rows.append(evaluated)
            assert_close(row.updated_p_team_a_advance, env.p_team_a_advance, f"P3 advancement A mismatch {row.match_id}", atol=5e-9)
            assert_close(row.updated_p_team_b_advance, env.p_team_b_advance, f"P3 advancement B mismatch {row.match_id}", atol=5e-9)
            assert_close(row.updated_expected_attendance, evaluated["attendance"], f"P3 attendance mismatch {row.match_id}", atol=0.011)
            assert_close(row.updated_expected_tv_viewers, evaluated["broadcast_viewers"], f"P3 viewers mismatch {row.match_id}", atol=0.011)
            assert_close(row.updated_ticket_revenue_usd, evaluated["ticket_value"], f"P3 ticket value mismatch {row.match_id}", atol=0.011)
            assert_close(row.updated_broadcast_value_usd, evaluated["broadcast_value"], f"P3 broadcast value mismatch {row.match_id}", atol=0.011)
            assert_close(row.resource_cost_index, evaluated["resource_cost"], f"P3 resource cost mismatch {row.match_id}")
            assert_close(row.risk_exposure_index, evaluated["risk"], f"P3 risk mismatch {row.match_id}", atol=5e-9)
        dynamic, _ = p3_normalize(pd.DataFrame(dynamic_rows), bounds)
        assert_true(p3_constraints_audit(dynamic, self.data, p2)["all_passed"], "P3 dynamic constraints failed")

        static_environment = static_p3_environment(self.data, p2)
        static_actions, static_bounds = p3_normalize(build_p3_actions(self.data, static_environment))
        static_selected, _ = solve_p3_actions(static_actions, self.data["dynamic_resource_limits"])
        recorded_static = self.diagnostics["problem3_static_decisions"]
        for row in static_selected.itertuples(index=False):
            recorded = recorded_static[row.match_id]
            assert_true(
                [int(row.broadcast), int(row.security), int(row.transport)]
                == [recorded["broadcast"], recorded["security"], recorded["transport"]],
                f"P3 static pre-tournament action mismatch {row.match_id}",
            )
            assert_close(row.delta, recorded["delta"], f"P3 static delta mismatch {row.match_id}", atol=1e-10)
            assert_true(recorded["selected_using"] == "pre_tournament_static_information_only", "P3 static information marker invalid")
        static_rows = []
        for match_id, decision in recorded_static.items():
            env = environment_index.loc[match_id].copy(); env["match_id"] = match_id
            static_rows.append(evaluate_p3_action(
                self.data, env, decision["broadcast"], decision["security"], decision["transport"], decision["delta"]
            ))
        static_eval, _ = p3_normalize(pd.DataFrame(static_rows), bounds)
        assert_true(p3_constraints_audit(static_eval, self.data, p2)["all_passed"], "P3 fixed static constraints failed")
        dynamic_index, static_index = dynamic.set_index("match_id"), static_eval.set_index("match_id")
        for row in output.itertuples(index=False):
            assert_close(row.dynamic_net_value, dynamic_index.loc[row.match_id, "net_value"], f"P3 dynamic net mismatch {row.match_id}", atol=5e-10)
            assert_close(row.static_net_value, static_index.loc[row.match_id, "net_value"], f"P3 static net mismatch {row.match_id}", atol=5e-10)
            static_net = float(row.static_net_value)
            if abs(static_net) < 1e-12:
                assert_true(pd.isna(row.improvement_rate), f"P3 zero static net must have blank improvement {row.match_id}")
            else:
                expected = (float(row.dynamic_net_value) - static_net) / abs(static_net)
                assert_close(row.improvement_rate, expected, f"P3 improvement mismatch {row.match_id}", atol=5e-9)
        assert_close(dynamic["net_value"].sum(), self.diagnostics["problem3_dynamic_total_net"], "P3 dynamic total mismatch", atol=1e-9)
        assert_close(static_eval["net_value"].sum(), self.diagnostics["problem3_static_total_net"], "P3 static total mismatch", atol=1e-9)
        robustness = self.diagnostics["problem3_robustness"]
        assert_true(
            len(robustness["seeds"]) == 5
            and len(set(robustness["seeds"])) == 5
            and len(robustness["runs"]) == 5,
            "P3 robustness must contain five unique seeds",
        )
        for run in robustness["runs"]:
            assert_close(
                sum(run["advancement_probs"].values()), 32,
                f"P3 robustness seed {run['seed']} advancement sum", atol=1e-9,
            )
            assert_true(
                set(run["decision_signature"]) == third_ids,
                f"P3 robustness seed {run['seed']} decision set mismatch",
            )
        dynamic_totals = [run["dynamic_total_net"] for run in robustness["runs"]]
        static_totals = [run["static_total_net"] for run in robustness["runs"]]
        improvement_rates = [run["overall_improvement_rate"] for run in robustness["runs"]]
        assert_true(
            np.allclose(robustness["dynamic_total_net_range"], [min(dynamic_totals), max(dynamic_totals)])
            and np.allclose(robustness["static_total_net_range"], [min(static_totals), max(static_totals)])
            and np.allclose(
                robustness["overall_improvement_rate_range"],
                [min(improvement_rates), max(improvement_rates)],
            ),
            "P3 robustness ranges do not match five recorded runs",
        )
        primary_signature = robustness["runs"][0]["decision_signature"]
        expected_stability = [
            float(np.mean([
                run["decision_signature"][match_id] == primary_signature[match_id]
                for match_id in primary_signature
            ]))
            for run in robustness["runs"]
        ]
        assert_true(
            np.allclose(
                robustness["dynamic_decision_match_fraction_vs_primary"],
                expected_stability,
            ),
            "P3 robustness decision stability mismatch",
        )
        assert_true(static_bounds == self.diagnostics["problem3_static_pre_normalization_bounds"], "P3 static bounds mismatch")

    def check_problem4(self) -> None:
        material_paths = {
            "schedule": MATERIAL_DIR / "worldcup_2022_openfootball.json",
            "elo": MATERIAL_DIR / "worldcup_2022_pre_tournament_elo.csv",
            "mappings": MATERIAL_DIR / "entity_mappings.json",
        }
        assert_true(all(path.is_file() for path in material_paths.values()), "P4 one or more material files are missing")
        raw_source = json.loads(material_paths["schedule"].read_text(encoding="utf-8"))
        raw_matches = raw_source["matches"]
        assert_true(len(raw_matches) == 64, "P4 raw JSON must contain 64 matches")
        assert_true(sum("group" in match for match in raw_matches) == 48, "P4 raw JSON must contain 48 group matches")
        elo = pd.read_csv(material_paths["elo"])
        assert_true(
            len(elo) == 32 and elo["team_name"].nunique() == 32
            and not elo["team_name"].duplicated().any() and not elo.isna().any().any(),
            "P4 Elo table must contain 32 unique complete teams",
        )
        mappings = json.loads(material_paths["mappings"].read_text(encoding="utf-8"))
        raw_teams = {match["team1"] for match in raw_matches if "group" in match} | {
            match["team2"] for match in raw_matches if "group" in match
        }
        assert_true(raw_teams == set(mappings["team_aliases"]), "P4 team alias coverage is incomplete")
        assert_true(mappings["team_aliases"].get("USA") == "United States", "P4 USA alias is incorrect")
        raw_stadiums = {match["ground"].split(",")[0].strip() for match in raw_matches if "group" in match}
        assert_true(raw_stadiums == set(mappings["stadium_aliases"]), "P4 stadium alias coverage is incomplete")
        mapped_stadiums = set(mappings["stadium_aliases"].values())
        assert_true(len(mapped_stadiums) == 8 and mapped_stadiums == set(mappings["stadiums"]), "P4 must map exactly eight stadiums")
        required_stadium_fields = {
            "name", "city", "capacity", "latitude", "longitude", "timezone",
            "capacity_source_url", "geodata_source_url", "retrieval_date",
        }
        for stadium_name, metadata in mappings["stadiums"].items():
            assert_true(required_stadium_fields.issubset(metadata), f"P4 stadium metadata incomplete: {stadium_name}")
            assert_true(metadata["name"] == stadium_name, f"P4 stadium canonical name mismatch: {stadium_name}")
            assert_true(
                int(metadata["capacity"]) > 0
                and np.isfinite(float(metadata["latitude"]))
                and np.isfinite(float(metadata["longitude"])),
                f"P4 stadium numeric metadata invalid: {stadium_name}",
            )
            assert_true(
                str(metadata["capacity_source_url"]).startswith("https://")
                and str(metadata["geodata_source_url"]).startswith("https://")
                and bool(str(metadata["retrieval_date"])),
                f"P4 stadium source metadata invalid: {stadium_name}",
            )
        actual = self.output["actual_schedule_2022.csv"]
        comparison = self.output["result_4_schedule_comparison.csv"]
        assert_true(len(actual) == 48 and actual["match_id"].is_unique, "P4 actual schedule must have 48 unique matches")
        teams = sorted(set(actual["team_a"]) | set(actual["team_b"]))
        assert_true(len(teams) == 32, "P4 actual schedule must have 32 teams")
        for team in teams:
            assert_true(((actual["team_a"].eq(team)) | (actual["team_b"].eq(team))).sum() == 3, f"P4 {team} match count")
        assert_true((actual.groupby("group_id").size() == 6).all(), "P4 each group must have six matches")
        assert_true((actual.groupby("round_in_group").size() == 16).all(), "P4 each round must have 16 matches")
        expected_url = mappings["sources"]["openfootball_file"]
        assert_true(actual["source_url"].eq(expected_url).all(), "P4 source URL is not the pinned concrete file")
        assert_true(bool(re.search(r"/blob/[0-9a-f]{40}/2022/worldcup\.json$", expected_url)), "P4 source URL is not commit-pinned")
        regenerated = actual_schedule_2022()
        assert_true(
            actual.drop(columns=["retrieval_date"]).equals(regenerated.drop(columns=["retrieval_date"])),
            "P4 actual CSV differs from ordinary project source file",
        )
        for row in actual.itertuples(index=False):
            local = pd.Timestamp(f"{row.date} {row.local_kickoff_time}", tz="Asia/Qatar")
            assert_true(row.utc_datetime == local.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"), f"P4 UTC conversion mismatch {row.match_id}")
        third = actual[actual["round_in_group"].eq(3)]
        assert_true((third.groupby("group_id")["utc_datetime"].nunique() == 1).all(), "P4 third-round group matches are not simultaneous")
        source_hash = hashlib.sha256((MATERIAL_DIR / "worldcup_2022_openfootball.json").read_bytes()).hexdigest()
        assert_true(source_hash == "f4f0499c30076bc8e3702f400765900797fb22212a32aaa8137e1dfc49449a68", "P4 raw source hash changed")

        required = {
            "schedule_span_days", "total_travel_km", "avg_travel_per_team_km", "avg_travel_per_match_km",
            "cross_timezone_transitions", "average_rest_hours", "minimum_rest_hours", "rest_hours_std",
            "matches_per_venue", "matches_per_active_venue_day", "venue_active_day_utilization",
            "capacity_match_ratio", "prime_time_share", "prime_matches_per_team", "prime_count_range",
            "expected_attendance_per_match", "proxy_ticket_value_per_match_usd",
            "proxy_broadcast_value_per_match_usd", "proxy_uncertainty_mean",
            "proxy_attractiveness_mean", "proxy_cost_per_match_musd", "proxy_travel_burden",
            "proxy_fairness_penalty", "proxy_execution_risk", "proxy_Z2",
            "proxy_Z2_second_nearest", "prime_time_share_second_nearest",
        }
        assert_true(required.issubset(set(comparison["indicator_name"])), "P4 comparison indicators incomplete")
        assert_true(len(comparison) == 30 and comparison["indicator_name"].is_unique, "P4 must contain 30 unique indicators")
        assert_true({"structure", "proxy_nearest", "proxy_sensitivity"}.issubset(set(comparison["indicator_category"])), "P4 layers not separated")
        for row in comparison.itertuples(index=False):
            actual_value, optimized_value = float(row.actual_schedule_value), float(row.optimized_schedule_value)
            assert_close(row.absolute_difference, optimized_value - actual_value, f"P4 absolute difference {row.indicator_name}")
            if row.preferred_direction == "neutral" or abs(actual_value) < 1e-12:
                assert_true(pd.isna(row.relative_improvement), f"P4 relative improvement should be blank {row.indicator_name}")
            else:
                sign = 1 if row.preferred_direction == "larger_better" else -1
                expected = sign * (optimized_value - actual_value) / abs(actual_value)
                assert_close(row.relative_improvement, expected, f"P4 relative improvement {row.indicator_name}")
        assert_true(self.diagnostics["problem4_scale_basis"]["T_B_C"].startswith("per_match"), "P4 cross-size normalization basis missing")

    def run(self) -> list[str]:
        self.check_environment()
        self.read_outputs()
        self.check_problem1()
        self.check_problem2()
        self.check_problem3()
        self.check_problem4()
        return [
            "fixed Python/dependency environment checks passed",
            "problem1 formula/reproducibility checks passed",
            "problem2 identity/formula/hard-constraint/group-boundary checks passed",
            "problem3 simulation/static-dynamic recomputation checks passed",
            "problem4 three-material/source/structure/proxy checks passed",
        ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    acceptance = Acceptance(output_dir)
    messages = acceptance.run()
    print("CHECK PASSED")
    for message in messages:
        print("-", message)
    print("Problem 2 group adjacent-round boundary gaps (hours):")
    for key, gap in acceptance.group_round_summary["gaps"].items():
        print(f"- {key}: {gap:.1f}")
    print(
        "Problem 2 group boundary summary: "
        f"minimum={acceptance.group_round_summary['minimum']:.1f} hours, "
        f"violations={acceptance.group_round_summary['violations']}, "
        f"all_passed={acceptance.group_round_summary['all_passed']}"
    )


if __name__ == "__main__":
    main()
