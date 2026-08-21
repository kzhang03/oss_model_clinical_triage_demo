import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "chiefcomplaint",
]


def read_triage_table(input_dir):
    for filename in ("triage.csv.gz", "triage.csv"):
        path = input_dir / filename
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No triage.csv.gz or triage.csv found in {input_dir}")


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def split_subjects(subject_ids, seed):
    subject_ids = np.array(sorted(subject_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    validation_size = round(len(subject_ids) * 0.10)
    test_size = round(len(subject_ids) * 0.10)
    train_end = len(subject_ids) - validation_size - test_size
    validation_end = train_end + validation_size

    return {
        "train": set(subject_ids[:train_end]),
        "validation": set(subject_ids[train_end:validation_end]),
        "test": set(subject_ids[validation_end:]),
    }


def limit_rows(frame, maximum, balanced, seed):
    if maximum is None or len(frame) <= maximum:
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)

    if not balanced:
        return frame.sample(n=maximum, random_state=seed).reset_index(drop=True)

    per_class = max(1, int(np.ceil(maximum / 5)))
    sampled = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby("acuity")
    ]
    return (
        pd.concat(sampled)
        .sample(frac=1, random_state=seed)
        .head(maximum)
        .reset_index(drop=True)
    )


def write_jsonl(frame, path):
    with path.open("w", encoding="utf-8") as output:
        for _, row in frame.iterrows():
            patient = {
                column: json_value(row[column])
                for column in FEATURE_COLUMNS
            }
            record = {
                "subject_id": int(row["subject_id"]),
                "stay_id": int(row["stay_id"]),
                "label": int(row["acuity"]),
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Estimate the Emergency Severity Index (ESI) level "
                            "from this triage record. Return only one ESI level: "
                            "1, 2, 3, 4, or 5.\n\n"
                            f"Patient data:\n{json.dumps(patient, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": str(int(row["acuity"])),
                    },
                ],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare patient-separated JSONL splits from MIMIC-IV-ED triage data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/mimic-iv-ed-2.2/ed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finetune_mimic_smoke"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument(
        "--balanced",
        "--balanced-smoke",
        dest="balanced",
        action="store_true",
        help="Sample approximately equal numbers of each ESI level.",
    )
    args = parser.parse_args()

    triage = read_triage_table(args.input_dir)

    required_columns = {"subject_id", "stay_id", "acuity", *FEATURE_COLUMNS}
    if missing := sorted(required_columns - set(triage.columns)):
        raise ValueError(f"triage table is missing columns: {missing}")
    if triage["stay_id"].duplicated().any():
        raise ValueError("triage table contains duplicate stay_id values")

    triage["acuity"] = pd.to_numeric(triage["acuity"], errors="coerce")
    triage = triage[triage["acuity"].isin([1, 2, 3, 4, 5])].copy()
    triage["acuity"] = triage["acuity"].astype(int)
    triage["chiefcomplaint"] = triage["chiefcomplaint"].astype("string").str.strip()
    triage = triage[["subject_id", "stay_id", *FEATURE_COLUMNS, "acuity"]]

    subject_splits = split_subjects(triage["subject_id"].unique(), args.seed)
    split_frames = {
        name: triage[triage["subject_id"].isin(subjects)].copy()
        for name, subjects in subject_splits.items()
    }

    subject_sets = {
        name: set(frame["subject_id"])
        for name, frame in split_frames.items()
    }
    assert subject_sets["train"].isdisjoint(subject_sets["validation"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["validation"].isdisjoint(subject_sets["test"])

    limits = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
    }
    for index, name in enumerate(("train", "validation", "test")):
        split_frames[name] = limit_rows(
            split_frames[name],
            limits[name],
            args.balanced,
            args.seed + index,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_rows": int(len(triage)),
        "total_subjects": int(triage["subject_id"].nunique()),
        "features": FEATURE_COLUMNS,
        "balanced": args.balanced,
        "seed": args.seed,
        "splits": {},
        "subject_overlap": False,
    }
    for name, frame in split_frames.items():
        write_jsonl(frame, args.output_dir / f"{name}.jsonl")
        summary["splits"][name] = {
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "label_counts": {
                str(label): int(count)
                for label, count in frame["acuity"].value_counts().sort_index().items()
            },
        }
        print(
            f"{name}: {len(frame):,} rows, "
            f"{frame['subject_id'].nunique():,} subjects"
        )

    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("No subject overlap.")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
