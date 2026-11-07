"""Run the whole research pipeline in order.

    python scripts/run_all.py              # everything (data download is cached)
    python scripts/run_all.py --skip-data  # reuse data/processed, rerun the research only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STEPS = [
    ("data", "build_dataset.py"),
    ("signals", "build_signals.py"),
    ("evaluation", "evaluate_signals.py"),
    ("combination", "combine_signals.py"),
    ("backtest", "backtest.py"),
    ("robustness", "robustness.py"),
    ("improvements", "improvements.py"),
    ("readme", "update_readme.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    for name, script in STEPS:
        if args.skip_data and name == "data":
            continue
        t0 = time.time()
        print(f"\n=== {name} ({script}) ===", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)
        print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
