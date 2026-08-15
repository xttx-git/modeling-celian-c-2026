"""One-command reproduction of model outputs, audits, figures and tables."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*parts: str) -> None:
    command = [sys.executable, *parts]
    print(f"[reproduce] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="validate persisted model outputs and regenerate figures without re-solving",
    )
    args = parser.parse_args()

    run("code/data_check.py")
    main_args = ["code/main.py"]
    if args.reuse:
        main_args.append("--reuse")
    run(*main_args)
    run("code/constraint_audit.py")
    run("figures/generate_all.py")
    print("[reproduce] PASS", flush=True)


if __name__ == "__main__":
    main()
