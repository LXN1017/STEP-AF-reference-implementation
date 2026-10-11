# Implementation provenance

This reference implementation was written on 11 October 2026 from the manuscript.
The original experiment code, private raw PPG, per-patient prediction arrays and trained
weights were not available. The reference implementation cannot authenticate the
manuscript's numerical results. Existing reported results have not been replaced by
outputs from this code.

## Directly described by the manuscript or author

- First QC-passing record of each patient-day, 60 s at 200 Hz; 96 complete days.
- Four-order Butterworth 0.5–4 Hz zero-phase filtering, morphology correlation ≥0.80,
  300–2000 ms PPIs; valid recording retains ≥80% of candidate PPIs and ≥15 intervals.
- Ten rhythm features, gap-preserving successive statistics, defined spectral and
  entropy parameters, and training-only feature-level imputation.
- Ordered segment means at 2/4/8 days, scale-specific linear/ReLU projections,
  cosine agreement, positive-slope gate, Transformer and two-layer classifier.
- Five repetitions of patient-level outer three-fold CV with inner three-fold selection;
  AdamW, learning rate 3e-4, weight decay 1e-3, batch 8, maximum 120 epochs, patience 15.
- Repetition-specific AUROC/AP averaging; pooled confusion counts; patient-cluster
  bootstrap with 10,000 draws, PCG64 seed 20260922 and Holm comparison families.

## Explicit choices made in this new implementation

These are **not evidence of what the original experiments used**. They must be checked
against the original acquisition and training system before claiming identical reproduction.

- Systolic peaks: SciPy `find_peaks` with a minimum distance of 0.30 s; assumes positive
  pulse polarity. Complete pulses only; outermost peaks are excluded from the QC
  candidate pool. Resample each cycle to 128 points, center and normalize its amplitude
  before constructing the median template.
- Sample SD for SDNN/SDSD/Poincaré quantities (`ddof=1`), population SD for feature
  standardization. Gap-contiguous PPI pairs determine Poincaré SD1 and SD2 directly.
- Deterministic sinusoidal positional encoding, post-norm Transformer, PyTorch default
  layer-normalization epsilon. Stage-removal/history analyses retain original position
  indices; stage permutations use destination-slot positions.
- Direct concatenation uses a linear projection plus ReLU; simple attention uses a
  shared linear scalar score and Softmax. These baseline details were unspecified.
- Base CV seed 20261011. Seeds are derived separately for outer splits, inner splits,
  model fitting and final validation splits and stored in run metadata.
- Default hyperparameter list contains only the manuscript's nominal learning rate.
  Supply `learning_rates` in the configuration to define a larger prespecified search.
  The source material did not supply the original grid or search budget.
- Probability threshold maximizes F1 over inner OOF probabilities; ties choose the
  largest threshold. Original operating-threshold criterion was unspecified.
- After selecting a candidate, an independent stratified 20% holdout inside the outer
  training partition supplies final-fit early stopping. Fit-patient preprocessing excludes
  both this validation set and outer test patients. Original final-refit procedure was unspecified.
- PyTorch deterministic algorithms are requested; identical CPU/GPU or cross-version
  numerical output is not guaranteed. No augmentation, class weighting, external data
  or unavailable original pretraining is added.

## Checks and limits

The repository checks mathematical shapes, chronology, positive gating, non-bridging
of invalid PPIs, first-valid daily selection, training-only preprocessing, strict >10-pp
labels, OOF aggregation, paired bootstrap behavior and a minimal nested-training run.
All test inputs are artificial and clearly identified as unit-test fixtures. Test output
must not be reported as clinical performance. There is no claim that the reference
peak detector or QC fully matches the device software.
