"""Small helpers used by the baseline comparison notebook."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


ESI_LEVELS = (1, 2, 3, 4, 5)
NUMERIC_FEATURES = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "pain_unscorable",
]
TEXT_FEATURE = "chiefcomplaint"


def load_split(path: str | Path) -> pd.DataFrame:
    """Load one prepared JSONL split into a flat table."""
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            content = record["messages"][0]["content"]
            patient = json.loads(content[content.index("{") :])
            rows.append(
                {
                    "subject_id": record["subject_id"],
                    "stay_id": record["stay_id"],
                    "label": int(record["label"]),
                    **patient,
                }
            )

    frame = pd.DataFrame(rows)
    pain = pd.to_numeric(frame["pain"], errors="coerce")
    frame["pain_unscorable"] = (pain.isna() & frame["pain"].notna()).astype(int)
    frame["pain"] = pain
    frame[TEXT_FEATURE] = frame[TEXT_FEATURE].fillna("").astype(str)
    return frame


def test_fingerprint(frame: pd.DataFrame) -> str:
    """Create the same test-set fingerprint used by the MedGemma notebook."""
    keys = [
        f"{row.subject_id}:{row.stay_id}:{row.label}"
        for row in frame.itertuples(index=False)
    ]
    return hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()


def serialize_record(row: pd.Series) -> str:
    """Convert one structured triage record into short text for BERT."""
    def value(column: str) -> str:
        raw = row[column]
        return "unknown" if pd.isna(raw) else f"{raw:g}"

    pain = "unscorable" if row["pain_unscorable"] else value("pain")
    complaint = row[TEXT_FEATURE] or "unknown"
    return (
        f"Chief complaint: {complaint}. Temperature: {value('temperature')}. "
        f"Heart rate: {value('heartrate')}. Respiratory rate: {value('resprate')}. "
        f"Oxygen saturation: {value('o2sat')}. Blood pressure: "
        f"{value('sbp')}/{value('dbp')}. Pain: {pain}."
    )


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    """Calculate the metrics shared by every model."""
    true = np.asarray(y_true, dtype=int)
    raw_pred = np.asarray(y_pred, dtype=object)
    valid = np.array(
        [isinstance(value, (int, np.integer)) and int(value) in ESI_LEVELS for value in raw_pred]
    )
    pred = np.array([int(value) if ok else 0 for value, ok in zip(raw_pred, valid)])
    error = pred[valid] - true[valid]

    recall = {}
    for level in ESI_LEVELS:
        mask = true == level
        recall[str(level)] = float((pred[mask] == level).mean()) if mask.any() else None

    return {
        "examples": int(len(true)),
        "accuracy": float((pred == true).mean()),
        "macro_f1": float(
            f1_score(true, pred, labels=ESI_LEVELS, average="macro", zero_division=0)
        ),
        "under_triage_rate": float((error > 0).sum() / len(true)),
        "over_triage_rate": float((error < 0).sum() / len(true)),
        "severe_under_triage_rate": float(
            (np.isin(true, [1, 2]) & np.isin(pred, [4, 5])).mean()
        ),
        "invalid_output_rate": float((~valid).mean()),
        "recall_by_esi": recall,
        "confusion_matrix_labels_1_to_5": confusion_matrix(
            true, pred, labels=ESI_LEVELS
        ).tolist(),
    }


def fit_transformer(
    model_name: str,
    train_texts: list[str],
    train_labels,
    test_texts: list[str],
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 96,
) -> dict[str, Any]:
    """Fine-tune one BERT classifier and return its test predictions."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ESI_LEVELS)
    ).to(device)

    encoded = tokenizer(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(np.asarray(train_labels, dtype=int) - 1)
    loader = DataLoader(
        TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for _ in range(epochs):
        for input_ids, attention_mask, targets in loader:
            optimizer.zero_grad()
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=targets.to(device),
            )
            output.loss.backward()
            optimizer.step()
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    encoded = tokenizer(
        test_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_texts), 64):
            logits = model(
                input_ids=encoded["input_ids"][start : start + 64].to(device),
                attention_mask=encoded["attention_mask"][start : start + 64].to(device),
            ).logits
            predictions.extend((logits.argmax(dim=1) + 1).cpu().tolist())
    predict_seconds = time.perf_counter() - predict_start

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "predictions": predictions,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "device": device.type,
    }


def summary_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build one concise model comparison table."""
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "model": result["model"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "under_triage": metrics["under_triage_rate"],
                "over_triage": metrics["over_triage_rate"],
                "severe_under": metrics["severe_under_triage_rate"],
                "invalid": metrics["invalid_output_rate"],
                "fit_s": result.get("fit_seconds"),
                "predict_s": result.get("predict_seconds"),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
