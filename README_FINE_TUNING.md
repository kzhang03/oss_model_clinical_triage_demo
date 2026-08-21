# MedGemma ESI Fine-Tuning

This project fine-tunes `google/medgemma-1.5-4b-it` to predict
MIMIC-IV-ED `triage.acuity` from information recorded at triage.

The MIMIC-IV-ED archive, extracted tables, prepared splits, checkpoints, and
evaluation outputs are local-only and ignored by Git.

## Project layout

```text
notebooks/   Experiment notebooks
src/         Shared Python helpers
scripts/     Data preparation and validation commands
data/        Local datasets and prepared splits
outputs/     Local model outputs and checkpoints
```

## Model input

The prompt contains these fields from `triage.csv.gz`:

```text
temperature
heartrate
resprate
o2sat
sbp
dbp
pain
chiefcomplaint
```

`acuity` is the target. Identifiers and post-triage information are excluded.
Missing feature values are represented as JSON `null`.

## Setup

Activate the environment and install the pinned dependencies:

```powershell
conda activate meddemo
python -m pip install -r requirements-finetune.txt
```

QLoRA training requires a CUDA GPU. Base-model evaluation can run on CPU, but
it is slow.

## Prepare MIMIC-IV-ED

Place `mimic-iv-ed-2.2.zip` in the folder `data` and extract it:

```powershell
tar -xf mimic-iv-ed-2.2.zip -C data
```

Create a balanced smoke-test split:

```powershell
python scripts/prepare_training_data.py `
  --input-dir data/mimic-iv-ed-2.2/ed `
  --output-dir data/finetune_mimic_smoke `
  --max-train 500 `
  --max-validation 100 `
  --max-test 100 `
  --balanced-smoke
```

Create the larger balanced split used by the fast baselines:

```powershell
python scripts/prepare_training_data.py `
  --input-dir data/mimic-iv-ed-2.2/ed `
  --output-dir data/finetune_mimic_balanced_large `
  --max-train 4000 `
  --max-validation 500 `
  --max-test 500 `
  --balanced
```

Create unrestricted patient-separated splits for the full profile:

```powershell
python scripts/prepare_training_data.py `
  --input-dir data/mimic-iv-ed-2.2/ed `
  --output-dir data/finetune_mimic
```

Rows without a valid acuity label are removed. Patients are assigned to one
split only, so repeated visits cannot cross train, validation, and test sets.

## Run the notebook

Open `notebooks/medgemma_finetuning_pipeline.ipynb` with the `meddemo` kernel
and select a profile:

```python
PROFILE = "smoke"         # 20/10/10 records, 1 update
PROFILE = "larger_smoke"  # 500/100/100 records, 20 updates
PROFILE = "mimic_smoke"   # 500/100/100 records, 60 updates
PROFILE = "balanced_large" # 4000/500/500 records, 1 epoch
PROFILE = "full"          # full training split, 1 epoch
```

Expensive stages are disabled by default:

```python
RUN_BASE_EVALUATION = False
RUN_TRAINING = False
RUN_ADAPTER_EVALUATION = False
```

Enable and run each stage separately. Results are written under `outputs/`.
The notebook reports accuracy, macro F1, per-class recall, under-triage,
over-triage, severe under-triage, automatic coverage, and a confusion matrix.

Training and evaluation both use a single ESI label. Evaluation directly
compares the next-token scores for labels 1-5 instead of generating and parsing
text. Candidate probabilities are temperature-scaled on validation data. The
review threshold is still an experimental policy setting, not clinically
validated confidence.

The revised profiles write to directories ending in `_label_scoring`, so older
generation-based results and adapters are not reused accidentally. Model
results are experimental and must not be used for patient care.

### Sampling

The smoke and `balanced_large` profiles downsample the common ESI classes to
give every level equal representation. This is useful for training and for
measuring macro-F1 and per-class recall, especially for rare ESI 5 cases. It
does not estimate real-world accuracy because the real ESI distribution is
uneven. Final evaluation should therefore also use a larger untouched test set
with the natural class distribution. Do not rebalance that deployment-style
test set.

## Baselines

`notebooks/baseline_comparison.ipynb` compares these models on the larger
balanced split:

| Model | Inputs |
| --- | --- |
| Always ESI 3 | none |
| Histogram gradient boosting | vitals and pain |
| TF-IDF + logistic regression | chief complaint + vitals |
| TF-IDF + linear SVM | chief complaint + vitals |
| DistilBERT | serialized triage record |
| BioClinicalBERT | serialized triage record |

The four classical models and two transformer models now train on 4,000
balanced encounters and are evaluated on a patient-separated balanced test set
of 500 encounters. Matching MedGemma results are added only after the optional
`balanced_large` profile has been run on the same fingerprint.

```powershell
conda activate meddemo
jupyter nbconvert --to notebook --execute --inplace notebooks/baseline_comparison.ipynb
```

Set `RUN_TRANSFORMERS = False` to run only the classical baselines. The
transformer checkpoints are downloaded automatically if they are not cached.
Results are written to
`outputs/baselines/baseline_results_large_balanced.json`.

## MedSigLIP text test

`notebooks/medsiglip_triage_test.ipynb` tests `google/medsiglip-448` as a frozen
text encoder. A logistic-regression classifier is trained on its embeddings
because MedSigLIP produces embeddings rather than generated ESI labels. The
notebook uses the same balanced 4,000/500 split and writes results to
`outputs/medsiglip/medsiglip_text_probe_large_balanced.json`.
