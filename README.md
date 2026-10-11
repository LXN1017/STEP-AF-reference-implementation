# STEP-AF reference implementation

This repository contains **new code reconstructed from the STEP-AF manuscript**:
*STEP-AF: Identifying Atrial Fibrillation Burden Changes from the Temporal Evolution of PPG Snapshots*.
It is not the original code used to produce the manuscript's reported experiments.
Clinical data and original trained weights are not distributed. No reported clinical
performance has been reproduced or independently verified with this code.

## Implemented pipeline

1. Review all available 60-s, 200-Hz fingertip PPG recordings and select the **first
   recording that passes quality control each day**. Recruitment requested three
   measurements per day; the analysis uses one valid recording per day.
2. Extract 10 PPI-derived rhythm features with explicit preservation of rejected
   interval gaps. Undefined individual features remain missing until training-only
   median imputation and standardization.
3. Partition 96 days into 12 non-overlapping 8-day stages. For the 2/4/8-day branches,
   compute 4/2/1 ordered segment means, flatten and project 40/20/10 inputs to 32 dimensions.
4. Apply pairwise cosine agreement, Softmax scale weights, a positive-slope Sigmoid
   stage gate, one Transformer block (two heads), global average pooling, and a
   32–16–2 classifier.
5. Train and evaluate with five repetitions of patient-level three-fold outer CV,
   three-fold inner model selection, training-only preprocessing, early stopping,
   and a probability threshold selected inside the outer training partition.
6. Summarize AUROC and average precision per repetition, then average. Pool confusion
   counts for discrete metrics. Use class-stratified patient-cluster bootstrap for
   paired comparisons, carrying all repeat predictions with each sampled patient.

The main STEP-AF model, scale/feature removal, alternative fusion, stage removal,
monitoring-history truncation and stage-permutation controls are supported. This
repository does **not** reconstruct the manuscript's Mamba or other encoder-comparison
implementations; their architecture details and original code were not supplied.

## Installation

Use Python 3.11 or later. A CPU environment is sufficient.

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your operating system.
python -m pip install -e ".[test]"
python -m pytest -q
```

For CPU-only PyTorch, install `torch` from https://download.pytorch.org/whl/cpu
before installing this package. `requirements-tested.txt` records the versions used
to run the repository's tests; these are not claimed to be original experiment versions.

## Private data formats

No patient examples are included. Use pseudonymous identifiers locally.

**Recording manifest CSV**: `patient_id,day,started_at,recording_path`.
`started_at` is an ISO-8601 timestamp using a consistent timezone. Each recording
path points to a one-dimensional `.npy` array or headerless, single-column CSV.
Raw PPG amplitudes must use the same units and positive pulse polarity as the sensor.

```bash
step-af extract --manifest data/recordings.csv --output data/features.csv --audit data/qc_audit.csv
```

**Daily feature CSV**: `patient_id,day,RMSSD,SDSD,pNN70,SDNN,LF_power,HF_power,LF_HF,Shannon_entropy,sample_entropy,SD1_SD2`.
Blank cells or `nan` represent undefined individual features. Every included patient
must have exactly one row for each day 1–96. Missing days are rejected, **not imputed**.

**Label CSV**: `patient_id,baseline_burden_percent,followup_burden_percent`.
AF burden must be expressed as percentages from 0 to 100. The positive endpoint is
`followup − baseline > 10` percentage points. This is distinct from the classifier's
probability threshold.

```bash
step-af train --features data/features.csv --labels data/labels.csv --config configs/default.json --output results/full
step-af evaluate results/full/oof_predictions.npz
step-af predict --checkpoint results/full/repeat_1_fold_1.pt --features data/new_patients.npy --output results/predictions.npz
```

Each model fit writes a private checkpoint, preprocessing state, split indices,
selected hyperparameters, operating threshold and training metadata. Saved OOF arrays
contain one probability per patient per repetition. Never upload these patient-level
outputs or checkpoints without appropriate authorization. `.gitignore` excludes
common data, result and model-file formats, but is not a substitute for a privacy review.

## Controlled analyses

Copy `configs/default.json` and change only the relevant setting. The patient cohort
and base seed should stay identical for paired comparisons.

| Condition | Reference configuration |
|---|---|
| Remove one scale | `model.scales`: any two of `[2,4,8]` |
| +1-day | `model.scales`: `[1,2,4,8]`, keep `cycle_days=8` |
| +16-day | `cycle_days=16`, `scales=[2,4,8,16]`: six 16-day cycles |
| Direct concatenation | `fusion="concat"` |
| Simple attention | `fusion="attention"` |
| Remove one feature | `--remove-features INDEX`, zero-based in the feature order above |
| Remove one stage | `stage_indices`: chronological zero-based indices with one omitted |
| Shorten history | `stage_indices`: a suffix of `[0,...,11]`; start days are 1,9,...,89 |
| Permute stages | `--permutation-seed SEED`; retrain, repeat for five fixed seeds |
| Endpoint threshold | `--threshold-pp 5`, `10` or `15` |

Stage permutation moves only entire 8-day stages, preserves within-stage data, and
reassigns positional encodings to shuffled slots. It does not shuffle daily snapshots.
The current manuscript defines +16-day with six 16-day cycles; this reference code
does not reinterpret it as a preceding-context branch with twelve tokens.

```bash
step-af compare --reference results/full/oof_predictions.npz --alternatives results/ablation1/oof_predictions.npz results/ablation2/oof_predictions.npz --samples 10000
```

Invoke comparisons separately for prespecified families. Clinical threshold comparisons
require paired models trained with the **same threshold-specific labels**. Aligning
patient order alone is insufficient. Statistical inference is conditional on saved
OOF predictions and does not include variability from retraining bootstrap cohorts.
For the three-threshold comparison family, obtain the unadjusted paired p-value
within each threshold separately, then pass all three unadjusted values together
to `step_af.statistics.holm_adjust`. Do not treat three separate one-comparison
adjustments as a correction across the three thresholds.

## Provenance and boundaries

See [IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md) for implementation
choices not specified by the manuscript. The tests use artificial signals and arrays
only to check code behavior; they provide no evidence of clinical validity. The software
is a research implementation, not a diagnostic device or a validated clinical system.
No software reuse license has been assigned in this repository.

Official implementation references: [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html),
[SciPy Welch estimator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html),
[scikit-learn StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

中文说明：本仓库是依据论文新编写的参考实现，不是产生论文结果的原始实验代码。
不包含患者数据、原始权重或任何新临床实验结果。每日使用第一条通过质量检查的记录；
整日缺失和单个特征无法计算分别处理。具体实现中未由原稿明确给出的选择已单独列出。
