from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.stats_ceb.plots import generate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis plots from saved evaluation CSV artifacts")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    paths = generate_all_plots(args.run_dir)
    print(f"Generated {len(paths)} plot(s) in {Path(args.run_dir).resolve() / 'plots'}")


if __name__ == "__main__":
    main()
