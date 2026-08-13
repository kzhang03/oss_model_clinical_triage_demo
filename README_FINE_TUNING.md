# MedGemma ESI Fine-Tuning

This project fine-tunes `google/medgemma-1.5-4b-it` to predict
MIMIC-IV-ED `triage.acuity` from information recorded at triage.

The MIMIC-IV-ED archive, extracted tables, prepared splits, checkpoints, and
evaluation outputs are local-only and ignored by Git.

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

Place `mimic-iv-ed-2.2.zip` in the repository root and extract it:

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

Create unrestricted patient-separated splits for the full profile:

```powershell
python scripts/prepare_training_data.py `
  --input-dir data/mimic-iv-ed-2.2/ed `
  --output-dir data/finetune_mimic
```

Rows without a valid acuity label are removed. Patients are assigned to one
split only, so repeated visits cannot cross train, validation, and test sets.

## Run the notebook

Open `medgemma_finetuning_pipeline.ipynb` with the `meddemo` kernel and select a
profile:

```python
PROFILE = "smoke"         # 20/10/10 records, 1 update
PROFILE = "larger_smoke"  # 500/100/100 records, 20 updates
PROFILE = "mimic_smoke"   # 500/100/100 records, 60 updates
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
over-triage, severe under-triage, invalid outputs, and a confusion matrix.

The reported token probability is not calibrated clinical confidence. Model
results are experimental and must not be used for patient care.
