# MedGemma ESI QLoRA Smoke Test

This pipeline tests whether `google/medgemma-1.5-4b-it` can be adapted to
predict MIMIC-IV-ED `triage.acuity` from arrival-time ED features.

The source FedMML CSV and the generated MIMIC-style tables are synthetic. This
tests technical feasibility only; it is not clinical validation.

## Scope

- Base model: `google/medgemma-1.5-4b-it`
- Method: text-only 4-bit QLoRA
- Target: `triage.acuity` values 1-5
- Split unit: `subject_id`
- MedSigLIP: not fine-tuned because MIMIC-IV-ED contains no images
- MedGemma 27B: excluded because its memory requirements are inappropriate for this smoke test

## MIMIC-IV-ED Features

The model receives only fields available at initial triage:

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

The following are never model inputs because they are identifiers or
post-triage information:

```text
subject_id
stay_id
hadm_id
outtime
disposition
diagnoses
medications
```

Missing feature values are preserved as JSON `null`.

## Environment

Create or activate the environment. If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build appropriate for that machine. If not, CPU evaluation can still be attempted. Then install the remaining packages:

```powershell
conda activate meddemo
python -m pip install -r requirements-finetune.txt
```

The tested local package versions are recorded in `requirements-finetune.txt`.

## 1. Generate MIMIC-Compatible Synthetic Tables

Full conversion:

```powershell
python scripts/generate_synthetic_mimic_ed.py
python scripts/validate_mimic_schema.py
```

This creates:

```text
data/synthetic_mimic_iv_ed/ed/
  edstays.csv
  triage.csv
  vitalsign.csv
  diagnosis.csv
  medrecon.csv
  pyxis.csv
```

`diagnosis`, `medrecon`, and `pyxis` contain headers only because the FedMML
source does not provide directly compatible events.

The source fields map directly to the MIMIC triage vital-sign columns and
`esi_level` maps to `acuity`. Synthetic `subject_id` values are based on both
`site_id` and `patient_id`, because patient IDs repeat across sites. Fields
that are unavailable in the source are not invented: `race`,
`arrival_transport`, and `disposition` are `UNKNOWN`, while `hadm_id` is null.

## 2. Prepare Subject-Separated Smoke-Test Data

```powershell
python scripts/prepare_training_data.py `
  --output-dir data/finetune_smoke `
  --max-train 500 `
  --max-validation 100 `
  --max-test 100 `
  --balanced-smoke
```

The resulting JSONL files use conversational prompt/completion records. No
`subject_id` appears in more than one split.

## 3. Notebook Pipeline for Fine-Tuning and Evaluation

Steps 5-10 are implemented in one notebook:

```text
medgemma_finetuning_pipeline.ipynb
```

Open it with the `meddemo` kernel. At the top, choose one profile:

```python
PROFILE = "smoke"                     # 20 train, 10 validation, 10 test, one update
PROFILE = "larger_smoke"              # 500 train, 100 validation, 100 test, 20 updates
PROFILE = "larger_smoke_more_steps"   # same rows as larger_smoke, 60 updates
PROFILE = "full"                      # full prepared training split, one epoch
```

The notebook is not tied to a specific GPU model. It auto-detects the available device:

- If CUDA is available, model loading and evaluation use the GPU.
- QLoRA training uses whatever CUDA GPU PyTorch detects.
- If no CUDA GPU is available, base evaluation can run on CPU, but it will be slow.
- QLoRA training without CUDA is not supported by this simple demo pipeline.

If a profile runs out of VRAM, use a smaller profile or reduce `max_length`,
`lora_r`, or the row limits in the profile configuration.

## 4. Running the Notebook

Keep the expensive stages disabled until you are ready:

```python
RUN_BASE_EVALUATION = False
RUN_TRAINING = False
RUN_ADAPTER_EVALUATION = False
```

Recommended order:

1. Run the notebook setup and data-loading cells.
2. Set `RUN_BASE_EVALUATION = True` and run the base evaluation cell.
3. Set `RUN_TRAINING = True` and run the QLoRA cell.
4. After the adapter is saved, set `RUN_ADAPTER_EVALUATION = True` and run the adapter evaluation cell.
5. Run the final hardware-report cell.

Outputs are written under the selected profile directory:

```text
outputs/notebook_smoke/
outputs/notebook_larger_smoke/
outputs/notebook_larger_smoke_more_steps/
outputs/notebook_full/
```

Reported metrics include accuracy, macro F1, invalid-output rate,
under-triage rate, over-triage rate, human-review rate, per-class recall, and
a 1-5 confusion matrix.

The notebook also includes an error-analysis section. It displays wrong cases,
labels under-triage and severe under-triage, and applies a simple human-review
flag for invalid output, explicit `uncertain`, or low first-token confidence.
The confidence value is useful for triage-demo analysis but should not be treated
as calibrated clinical probability.

## Interpretation

A successful smoke test proves that the model and data pipeline can be
fine-tuned on the target hardware. It does not establish clinical performance.
The next stage should replace synthetic tables with properly authorized
MIMIC-IV-ED data while retaining the same schema and subject-level split.

