import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "gender",
    "race",
    "arrival_transport",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "chiefcomplaint",
]


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
                            "from this triage record. Return only JSON with the "
                            "key predicted_esi_level.\n\n"
                            f"Patient data:\n{json.dumps(patient, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"predicted_esi_level": int(row["acuity"])}
                        ),
                    },
                ],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Join MIMIC-style triage data and create subject-separated JSONL splits."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/synthetic_mimic_iv_ed/ed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/finetune_smoke"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--balanced-smoke", action="store_true")
    args = parser.parse_args()

    edstays = pd.read_csv(args.input_dir / "edstays.csv")
    triage = pd.read_csv(args.input_dir / "triage.csv")

    required_edstays = {"subject_id", "stay_id", "gender", "race", "arrival_transport"}
    required_triage = {"subject_id", "stay_id", "acuity", *FEATURE_COLUMNS[3:]}
    if missing := sorted(required_edstays - set(edstays.columns)):
        raise ValueError(f"edstays.csv is missing columns: {missing}")
    if missing := sorted(required_triage - set(triage.columns)):
        raise ValueError(f"triage.csv is missing columns: {missing}")

    joined = triage.merge(
        edstays[["subject_id", "stay_id", "gender", "race", "arrival_transport"]],
        on=["subject_id", "stay_id"],
        how="inner",
        validate="one_to_one",
    )
    joined["acuity"] = pd.to_numeric(joined["acuity"], errors="coerce")
    joined = joined[joined["acuity"].isin([1, 2, 3, 4, 5])].copy()
    joined["acuity"] = joined["acuity"].astype(int)
    joined["gender"] = joined["gender"].astype("string").str.strip().str.upper()
    joined["race"] = joined["race"].astype("string").str.strip().str.upper()
    joined["arrival_transport"] = (
        joined["arrival_transport"].astype("string").str.strip().str.upper()
    )
    joined["chiefcomplaint"] = joined["chiefcomplaint"].astype("string").str.strip()
    joined = joined[["subject_id", "stay_id", *FEATURE_COLUMNS, "acuity"]]

    subject_splits = split_subjects(joined["subject_id"].unique(), args.seed)
    split_frames = {
        name: joined[joined["subject_id"].isin(subjects)].copy()
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
            args.balanced_smoke,
            args.seed + index,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_dir / "prepared_triage.csv", index=False)

    summary = {
        "total_rows": int(len(joined)),
        "total_subjects": int(joined["subject_id"].nunique()),
        "features": FEATURE_COLUMNS,
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
    print("Verified: no subject_id appears in more than one split.")
    print(f"Prepared files written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
