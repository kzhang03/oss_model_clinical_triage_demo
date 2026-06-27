import argparse
import json
from pathlib import Path

import pandas as pd


PUBLIC_BASE_URL = "https://physionet.org/files/mimic-iv-ed-demo/2.2/ed"
TABLES = [
    "edstays.csv",
    "triage.csv",
    "vitalsign.csv",
    "diagnosis.csv",
    "medrecon.csv",
    "pyxis.csv",
]


def public_url(filename):
    return f"{PUBLIC_BASE_URL}/{filename}.gz"


def table_summary(frame):
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_fraction": {
            column: round(float(value), 4)
            for column, value in frame.isna().mean().items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the public MIMIC-IV-ED demo and validate generated table schemas."
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("data/synthetic_mimic_iv_ed/ed"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/mimic_schema_report.json"),
    )
    args = parser.parse_args()

    report = {"public_demo": {}, "generated": {}, "valid": True}

    for filename in TABLES:
        public = pd.read_csv(public_url(filename))
        report["public_demo"][filename] = table_summary(public)
        print(f"{filename}: public demo has {len(public):,} rows")
        print(f"  columns: {', '.join(public.columns)}")

        generated_path = args.generated_dir / filename
        if not generated_path.exists():
            report["generated"][filename] = {"exists": False}
            report["valid"] = False
            print(f"  generated file missing: {generated_path}")
            continue

        generated = pd.read_csv(generated_path)
        columns_match = list(generated.columns) == list(public.columns)
        report["generated"][filename] = {
            "exists": True,
            **table_summary(generated),
            "columns_match": columns_match,
        }
        report["valid"] = report["valid"] and columns_match
        print(f"  generated rows: {len(generated):,}; columns match: {columns_match}")

    if all((args.generated_dir / name).exists() for name in TABLES):
        edstays = pd.read_csv(args.generated_dir / "edstays.csv")
        triage = pd.read_csv(args.generated_dir / "triage.csv")
        vitalsign = pd.read_csv(args.generated_dir / "vitalsign.csv")

        stay_ids = set(edstays["stay_id"])
        relationship_checks = {
            "edstays_stay_id_unique": bool(edstays["stay_id"].is_unique),
            "triage_stay_id_unique": bool(triage["stay_id"].is_unique),
            "triage_stays_exist": set(triage["stay_id"]).issubset(stay_ids),
            "vitalsign_stays_exist": set(vitalsign["stay_id"]).issubset(stay_ids),
            "acuity_in_1_to_5": bool(triage["acuity"].dropna().isin([1, 2, 3, 4, 5]).all()),
        }
        report["relationship_checks"] = relationship_checks
        report["valid"] = report["valid"] and all(relationship_checks.values())

        print("Relationship checks:")
        for name, passed in relationship_checks.items():
            print(f"  {name}: {passed}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {args.report.resolve()}")

    if not report["valid"]:
        raise SystemExit("Schema validation failed.")

    print("Schema validation passed.")


if __name__ == "__main__":
    main()
