# MedGemma ESI Smoke Test Report

Date: 2026-06-27

## Purpose

This smoke test checks whether `google/medgemma-1.5-4b-it` can be fine-tuned with QLoRA to predict Emergency Severity Index (ESI) level from synthetic MIMIC-IV-ED-shaped triage data.

This is a technical feasibility test only. The dataset is synthetic, so the result should not be interpreted as clinical validation.


## Dataset Preparation and Modifications

The starting dataset was the synthetic FedMML emergency department triage CSV, `fedmml_ed_triage_dataset.csv`. It contained 87,234 encounter-level rows. To align the demo with the public MIMIC-IV-ED demo structure, the source CSV was converted into MIMIC-IV-ED-shaped tables under `data/synthetic_mimic_iv_ed/ed/`.

| Output Table | Rows | Modification |
| --- | ---: | --- |
| `edstays.csv` | 87,234 | Created synthetic `subject_id`, `stay_id`, `intime`, `outtime`, `gender`, and placeholder administrative fields. |
| `triage.csv` | 87,234 | Mapped triage vitals, pain score, chief complaint, and `esi_level` to MIMIC-style `acuity`. |
| `vitalsign.csv` | 87,234 | Duplicated arrival vital signs as charted vital-sign rows. |
| `diagnosis.csv` | 0 | Header-only table because the source did not provide compatible ICD diagnosis events. |
| `medrecon.csv` | 0 | Header-only table because the source did not provide medication reconciliation events. |
| `pyxis.csv` | 0 | Header-only table because the source did not provide medication dispensing events. |

Key modifications:

- `subject_id` was generated from the combination of `site_id` and `patient_id`, because patient IDs can repeat across sites.
- `stay_id` was generated sequentially, one per source encounter.
- `encounter_id`, `patient_id`, and `site_id` were not used as model inputs.
- `esi_level` was treated as the target label and mapped to `triage.acuity`.
- `sex` was mapped to `gender`.
- `heart_rate`, `respiratory_rate`, `spo2`, `systolic_bp`, `diastolic_bp`, `temperature`, `pain_score`, and `chief_complaint` were mapped to MIMIC-style triage fields.
- `race`, `arrival_transport`, and `disposition` were unavailable in the source data and were represented as `UNKNOWN` rather than invented.
- `hadm_id` was left null, because the synthetic source did not provide hospital admission identifiers.
- `outtime` was generated as four hours after `intime` only to satisfy the table shape; it was not used as a model input.

The model prompt used only arrival-time triage features:

```text
gender
race
arrival_transport
temperature
heartrate
resprate
o2sat
sbp
dbp
pain
chiefcomplaint
```

For fine-tuning, `edstays.csv` and `triage.csv` were joined into JSONL prompt/completion records. Splits were made by `subject_id` to prevent the same synthetic patient from appearing in more than one split. The smoke-test split was class-balanced:

| Split | Rows | Subjects | ESI Distribution |
| --- | ---: | ---: | --- |
| Train | 500 | 135 | 100 examples each for ESI 1-5 |
| Validation | 100 | 16 | 20 examples each for ESI 1-5 |
| Test | 100 | 20 | 20 examples each for ESI 1-5 |

Missing feature values were preserved as JSON `null`. The target label was not included in the prompt text; it appeared only in the assistant completion during training and in the evaluation label during testing.


## Setup

| Item | Value |
| --- | --- |
| Model | `google/medgemma-1.5-4b-it` |
| Profile | `larger_smoke_more_steps` |
| Method | 4-bit QLoRA |
| Train records | 500 |
| Validation records | 100 |
| Test records | 100 |
| Training steps | 60 |
| Epoch fraction | 0.96 |
| Trainable parameters | 14,901,248 |
| Final training loss | 0.1086 |
| Training time | 41.0 minutes |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Peak allocated VRAM | 7.59 GiB |

## Main Results

| Metric | Base Model | Fine-Tuned Adapter |
| --- | ---: | ---: |
| Accuracy | 20.0% | 97.0% |
| Macro F1 | 8.4% | 97.0% |
| Under Triage Rate | 40.0% | 0.0% |
| Over Triage Rate | 40.0% | 3.0% |
| Severe Under Triage Rate | 25.0% | 0.0% |
| Invalid Output Rate | 0.0% | 0.0% |
| Human Review Rate | 0.0% | 0.0% |

The base model performed poorly on this task, with 20.0% accuracy and a 25.0% severe under-triage rate. After QLoRA fine-tuning, accuracy improved to 97.0%, macro F1 improved to 97.0%, and severe under-triage fell to 0.0% on the 100-case synthetic test set.

## Recall By ESI Level

| Label | Base Model Recall | Fine-Tuned Recall |
| --- | ---: | ---: |
| ESI 1 | 0.0% | 100.0% |
| ESI 2 | 0.0% | 100.0% |
| ESI 3 | 100.0% | 95.0% |
| ESI 4 | 0.0% | 100.0% |
| ESI 5 | 0.0% | 90.0% |

The fine-tuned adapter recovered all ESI 1, ESI 2, and ESI 4 cases in this test set. The remaining errors were near-neighbor over-triage mistakes.

## Fine-Tuned Confusion Matrix

Rows are actual labels. Columns are predicted labels.

| Actual \ Predicted | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual 1 | 20 | 0 | 0 | 0 | 0 |
| Actual 2 | 0 | 20 | 0 | 0 | 0 |
| Actual 3 | 0 | 1 | 19 | 0 | 0 |
| Actual 4 | 0 | 0 | 0 | 20 | 0 |
| Actual 5 | 0 | 0 | 0 | 2 | 18 |

## Wrong Predictions

| Subject ID | Stay ID | Actual ESI | Predicted ESI | First-Token Confidence |
| --- | --- | ---: | ---: | ---: |
| 10000006 | 30005851 | 5 | 4 | 0.9996 |
| 10000006 | 30004203 | 5 | 4 | 0.9996 |
| 10000287 | 30053360 | 3 | 2 | 0.9993 |

There were 3 wrong predictions out of 100 test cases. All 3 were over-triage errors, meaning the model predicted a more urgent category than the label. There were no under-triage or severe under-triage errors in this run.

## Safety Interpretation

The result is promising for a smoke test because the fine-tuned model substantially improved over the base model and avoided under-triage on the synthetic test split. However, the model also produced very high confidence scores, with an average prediction confidence of 0.9992. The human-review rule did not flag any cases:

```text
human_review_rate: 0.0%
low_confidence_review_rate: 0.0%
```

This means the current confidence threshold is not useful as a reliable uncertainty mechanism. Future work should test out-of-distribution cases, missing data, degraded notes, and calibration methods before relying on confidence for human delegation.

## Limitations

- The dataset is synthetic and only shaped like MIMIC-IV-ED. Some MIMIC-style fields were placeholders because the source did not contain them.
- The test set contains only 100 cases.
- The model may be learning synthetic patterns that do not transfer to real emergency department data.
- Confidence scores are not calibrated clinical probabilities.
- This does not validate real-world clinical triage performance.

## Conclusion

The smoke test confirms that MedGemma 4B can be fine-tuned with QLoRA on the available laptop GPU for a synthetic ESI prediction task. The fine-tuned adapter strongly outperformed the base model on this small synthetic test set, improving accuracy from 20.0% to 97.0% and reducing severe under-triage from 25.0% to 0.0%.

The result supports technical feasibility, but not clinical readiness. The next step should be evaluation on authorized real MIMIC-IV-ED data or a stronger synthetic stress-test suite with missing values, noisy notes, outliers, and explicit human-review criteria.
