from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROMPT_FEATURE_COLUMNS = [
    "age",
    "sex",
    "chief_complaint",
    "clinical_notes",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "spo2",
    "pain_score",
]

COUNT_KEYS = ("correct", "incorrect", "uncertain", "invalid_output", "invalid_input")


def to_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def row_to_case_json(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {column: to_json_value(row[column]) for column in PROMPT_FEATURE_COLUMNS}


def build_esi_prompt(case_json: dict[str, Any], template_path: str | Path) -> str:
    case_block = json.dumps(case_json, indent=2, ensure_ascii=False)
    template = Path(template_path).read_text(encoding="utf-8")
    return re.sub(
        r"(Patient data:\s*)\{.*?\}(\s*Return)",
        lambda match: f"{match.group(1)}{case_block}{match.group(2)}",
        template,
        count=1,
        flags=re.DOTALL,
    ).strip()


def is_valid_model_input(row: pd.Series | dict[str, Any]) -> bool:
    keys = row.index if hasattr(row, "index") else row.keys()
    return all(column in keys and pd.notna(row[column]) for column in PROMPT_FEATURE_COLUMNS)


def empty_result_counter() -> Counter:
    return Counter(dict.fromkeys(COUNT_KEYS, 0))


def score_prediction(prediction: int | str, actual_label: int, counts: Counter) -> str:
    if prediction == "uncertain":
        counts["uncertain"] += 1
        return "uncertain"
    if prediction == "invalid_output":
        counts["invalid_output"] += 1
        return "invalid_output"
    if prediction == actual_label:
        counts["correct"] += 1
        return "correct"
    counts["incorrect"] += 1
    return "incorrect"


def first_model_device(model: Any) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for device in model.hf_device_map.values():
            if device not in {"cpu", "disk"}:
                return torch.device(device)
    return next(model.parameters()).device


def move_inputs_to_model(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = first_model_device(model)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def parse_esi_value(text: str) -> int | str:
    match = re.search(r"\b(uncertain|[1-5])\b", str(text), flags=re.IGNORECASE)
    if not match:
        return "invalid_output"
    value = match.group(1).lower()
    return "uncertain" if value == "uncertain" else int(value)


def generation_token_blocks(tokenizer: Any) -> list[list[int]]:
    blocks = []
    for token in ("<unused94>", "thought"):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            blocks.append(token_ids)
    return blocks


def generate_medgemma_4b_prediction(
    prompt: str,
    processor: Any,
    model: Any,
    max_new_tokens: int = 8,
) -> tuple[int | str, str]:
    prompt = f"{prompt}\n\nAnswer with only the value for predicted_esi_level."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    ) + "predicted_esi_level:"
    inputs = processor.tokenizer(chat_text, return_dict=True, return_tensors="pt")
    inputs = move_inputs_to_model(inputs, model)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            bad_words_ids=generation_token_blocks(processor.tokenizer),
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=[
                processor.tokenizer.eos_token_id,
                processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            ],
        )
    generated = processor.tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return parse_esi_value(generated), generated


def generate_text_model_prediction(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 8,
) -> tuple[int | str, str]:
    prompt = f"{prompt}\n\nAnswer with only the value for predicted_esi_level."
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    ) + "predicted_esi_level:"
    inputs = tokenizer(chat_text, return_tensors="pt", return_dict=True)
    inputs = move_inputs_to_model(inputs, model)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            bad_words_ids=generation_token_blocks(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return parse_esi_value(generated), generated


def case_json_to_retrieval_text(case_json: dict[str, Any]) -> str:
    return json.dumps(case_json, ensure_ascii=False, sort_keys=True)


def as_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unsupported embedding output type: {type(output)!r}")


def embed_texts_medsiglip(
    texts: list[str],
    processor: Any,
    model: Any,
    batch_size: int = 8,
    desc: str | None = None,
) -> np.ndarray:
    vectors = []
    model.eval()
    device = first_model_device(model)
    batch_starts = range(0, len(texts), batch_size)
    if desc:
        from tqdm.auto import tqdm

        batch_starts = tqdm(
            batch_starts,
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=desc,
        )
    with torch.no_grad():
        for start in batch_starts:
            batch = texts[start : start + batch_size]
            inputs = processor(
                text=batch,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            if hasattr(model, "get_text_features"):
                features = as_feature_tensor(model.get_text_features(**inputs))
            else:
                features = as_feature_tensor(model.text_model(**inputs))
            features = torch.nn.functional.normalize(features, dim=-1)
            vectors.append(features.cpu().numpy())
    return np.vstack(vectors)
