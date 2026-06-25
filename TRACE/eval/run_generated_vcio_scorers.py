from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TRACE.eval.generated_vcio_runner import run_generated_vcio_scorers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--scorers-dir", default="TRACE/generated_vcio")
    parser.add_argument("--value-breakdown", default="value_breakdown.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = run_generated_vcio_scorers(
        log_path=args.log,
        scorers_dir=args.scorers_dir,
    )

    output["value_breakdown"] = args.value_breakdown

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
