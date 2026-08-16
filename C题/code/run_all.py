#!/usr/bin/env python3
"""Run all four revised models and write a complete, reproducible result set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from pipeline_core import (
    RANDOM_SEEDS,
    ROOT,
    read_data,
    solve_problem1,
    solve_problem2,
    solve_problem3,
    solve_problem4,
)


OUTPUT_FILES = [
    "result_1_test_prediction.csv",
    "result_1_match_prediction.csv",
    "result_2_group_schedule.csv",
    "result_3_dynamic_strategy.csv",
    "actual_schedule_2022.csv",
    "result_4_schedule_comparison.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="output_v2",
        help="Output directory relative to C题, or an absolute path (default: output_v2).",
    )
    return parser.parse_args()


def resolve_output_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_csv(frame, output_dir: Path, name: str) -> None:
    frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")


def environment_diagnostics() -> dict:
    requirements = ROOT / "requirements.txt"
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn",
        "openpyxl", "joblib", "threadpoolctl",
    ]
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_data()

    p1_test, p1_match, diagnostics1 = solve_problem1(data)
    write_csv(p1_test, output_dir, OUTPUT_FILES[0])
    write_csv(p1_match, output_dir, OUTPUT_FILES[1])

    p2, diagnostics2, p2_context = solve_problem2(data, p1_match)
    write_csv(p2, output_dir, OUTPUT_FILES[2])

    p3, diagnostics3 = solve_problem3(data, p2)
    write_csv(p3, output_dir, OUTPUT_FILES[3])

    actual, p4, diagnostics4 = solve_problem4(data, p1_match, p2, p2_context)
    write_csv(actual, output_dir, OUTPUT_FILES[4])
    write_csv(p4, output_dir, OUTPUT_FILES[5])

    diagnostics = {
        "pipeline_version": "2026-08-16-v3-competition-normalized",
        "output_directory": str(output_dir),
        "environment": environment_diagnostics(),
        "random_seeds": RANDOM_SEEDS,
        **diagnostics1, **diagnostics2, **diagnostics3, **diagnostics4,
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote revised outputs to {output_dir}")
    print(f"P1 model: {diagnostics1['problem1_selected_model']}")
    print(f"P2 Z2: {diagnostics2['problem2_total_objective']:.10f}")
    print(
        "P3 dynamic/static: "
        f"{diagnostics3['problem3_dynamic_total_net']:.10f}/"
        f"{diagnostics3['problem3_static_total_net']:.10f}"
    )


if __name__ == "__main__":
    main()
