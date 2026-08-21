# Synthetic MedGemma ESI Smoke Test

> Historical result: this report describes the earlier generation-based
> evaluation. The current fine-tuning notebook uses single-label training and
> direct five-label scoring, so new results should be reported separately.

Date: 2026-06-27

## Dataset

The test used 87,234 synthetic encounters from
`fedmml_ed_triage_dataset.csv`. The source columns were mapped to MIMIC-IV-ED
table names for an early pipeline test. This was not real MIMIC-IV-ED data.

The model input contained sex, placeholders for race and arrival transport,
triage vital signs, pain, and chief complaint. ESI level was used only as the
target. Splits were balanced by ESI level and separated by synthetic patient ID.

| Split | Rows | Subjects | ESI distribution |
| --- | ---: | ---: | --- |
| Train | 500 | 135 | 100 per level |
| Validation | 100 | 16 | 20 per level |
| Test | 100 | 20 | 20 per level |

## Configuration

| Item | Value |
| --- | --- |
| Model | `google/medgemma-1.5-4b-it` |
| Method | 4-bit QLoRA |
| Training steps | 60 |
| Trainable parameters | 14,901,248 |
| Training loss | 0.1086 |
| Training time | 1 hour, 41 minutes |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Peak allocated VRAM | 7.59 GiB |

## Results

| Metric | Base model | Fine-tuned adapter |
| --- | ---: | ---: |
| Accuracy | 20.0% | 97.0% |
| Macro F1 | 8.4% | 97.0% |
| Under-triage | 40.0% | 0.0% |
| Over-triage | 40.0% | 3.0% |
| Severe under-triage | 25.0% | 0.0% |
| Invalid output | 0.0% | 0.0% |

Fine-tuned recall was 100% for ESI 1, 2, and 4; 95% for ESI 3; and 90% for
ESI 5. The three errors were over-triage by one level.

The adapter's mean first-token probability was 0.9992, including on incorrect
predictions. This value was not calibrated and did not provide a useful review
threshold.

## Limitations

- The records were synthetic and included placeholder fields.
- The test set contained only 100 encounters.
- Synthetic patterns may not transfer to real emergency department records.
- Token probabilities were not calibrated.

This run established that the QLoRA pipeline worked on the laptop GPU. It did
not establish clinical performance. Real MIMIC-IV-ED results should be reported
separately.
