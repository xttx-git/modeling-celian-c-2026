"""Regenerate every publication figure and LaTeX table from persisted results."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def run(path: Path) -> None:
    print(f"[figures] running {path.relative_to(ROOT)}", flush=True)
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


def main() -> None:
    run(FIGURES / "prep_plot_data.py")
    for path in sorted(FIGURES.glob("gen_fig_*.py")):
        run(path)
    run(FIGURES / "generate_tables.py")
    print("[figures] PASS", flush=True)


if __name__ == "__main__":
    main()
