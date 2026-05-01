"""
Train models from Football_Dataset_2015_2025.csv and save artifacts/bundle.joblib.

Usage (from repo root):
  python scripts/train_model.py
  python scripts/train_model.py --csv path/to/Football_Dataset_2015_2025.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(ROOT / "Football_Dataset_2015_2025.csv"),
        help="Path to the football CSV dataset",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output path for trained bundle (default: EPL_ARTIFACT_PATH or artifacts/bundle.joblib)",
    )
    args = parser.parse_args()
    from app.config import get_settings

    out_default = str(get_settings().artifact_path)
    out_path = Path(args.out) if args.out else Path(out_default)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(
            f"CSV not found: {csv_path}\n"
            "Place Football_Dataset_2015_2025.csv in the project root or pass --csv."
        )

    from app.ml_core import train_bundle

    bundle = train_bundle(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    metrics_path = out_path.parent / "bundle_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    print(f"Saved bundle to {out_path}")
    print("Test-set metrics:", json.dumps(bundle["metrics"], indent=2))


if __name__ == "__main__":
    main()
