import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_COLUMNS = {
    "edstays.csv": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "gender",
        "race",
        "arrival_transport",
        "disposition",
    ],
    "triage.csv": [
        "subject_id",
        "stay_id",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
        "acuity",
        "chiefcomplaint",
    ],
    "vitalsign.csv": [
        "subject_id",
        "stay_id",
        "charttime",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "rhythm",
        "pain",
    ],
    "diagnosis.csv": [
        "subject_id",
        "stay_id",
        "seq_num",
        "icd_code",
        "icd_version",
        "icd_title",
    ],
    "medrecon.csv": [
        "subject_id",
        "stay_id",
        "charttime",
        "name",
        "gsn",
        "ndc",
        "etc_rn",
        "etccode",
        "etcdescription",
    ],
    "pyxis.csv": [
        "subject_id",
        "stay_id",
        "charttime",
        "med_rn",
        "name",
        "gsn_rn",
        "gsn",
    ],
}

REQUIRED_SOURCE_COLUMNS = [
    "site_id",
    "patient_id",
    "encounter_id",
    "arrival_timestamp",
    "sex",
    "chief_complaint",
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "pain_score",
    "esi_level",
]


def pain_as_text(value):
    if pd.isna(value):
        return None
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Convert the synthetic FedMML CSV into MIMIC-IV-ED-shaped tables."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("fedmml_ed_triage_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_mimic_iv_ed/ed"),
    )
    args = parser.parse_args()

    source = pd.read_csv(args.source)
    missing = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {missing}")

    subject_key = (
        source["site_id"].astype("string")
        + "::"
        + source["patient_id"].astype("string")
    )
    subject_codes, _ = pd.factorize(subject_key, sort=True)
    subject_id = pd.Series(subject_codes + 10_000_000, dtype="int64")
    stay_id = pd.Series(np.arange(len(source)) + 30_000_000, dtype="int64")
    intime = pd.to_datetime(source["arrival_timestamp"], errors="raise")
    pain = source["pain_score"].map(pain_as_text)

    edstays = pd.DataFrame(
        {
            "subject_id": subject_id,
            "hadm_id": pd.Series(pd.NA, index=source.index, dtype="Int64"),
            "stay_id": stay_id,
            "intime": intime,
            "outtime": intime + pd.Timedelta(hours=4),
            "gender": source["sex"].astype("string").str.upper(),
            "race": "UNKNOWN",
            "arrival_transport": "UNKNOWN",
            "disposition": "UNKNOWN",
        }
    )

    triage = pd.DataFrame(
        {
            "subject_id": subject_id,
            "stay_id": stay_id,
            "temperature": source["temperature"],
            "heartrate": source["heart_rate"],
            "resprate": source["respiratory_rate"],
            "o2sat": source["spo2"],
            "sbp": source["systolic_bp"],
            "dbp": source["diastolic_bp"],
            "pain": pain,
            "acuity": source["esi_level"],
            "chiefcomplaint": source["chief_complaint"],
        }
    )

    vitalsign = pd.DataFrame(
        {
            "subject_id": subject_id,
            "stay_id": stay_id,
            "charttime": intime,
            "temperature": source["temperature"],
            "heartrate": source["heart_rate"],
            "resprate": source["respiratory_rate"],
            "o2sat": source["spo2"],
            "sbp": source["systolic_bp"],
            "dbp": source["diastolic_bp"],
            "rhythm": pd.NA,
            "pain": pain,
        }
    )

    tables = {
        "edstays.csv": edstays,
        "triage.csv": triage,
        "vitalsign.csv": vitalsign,
        "diagnosis.csv": pd.DataFrame(columns=TABLE_COLUMNS["diagnosis.csv"]),
        "medrecon.csv": pd.DataFrame(columns=TABLE_COLUMNS["medrecon.csv"]),
        "pyxis.csv": pd.DataFrame(columns=TABLE_COLUMNS["pyxis.csv"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table = table[TABLE_COLUMNS[filename]]
        table.to_csv(args.output_dir / filename, index=False)
        print(f"{filename}: {len(table):,} rows")

    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
