# Reference-code validation

Checked on 11 October 2026 with Python 3.12, Windows and CPU PyTorch.
`python -m pytest -q` completed with **13 passed**.

The checks covered:

- Ordered 2/4/8-day mean aggregation and end-of-window coverage.
- Forward and backward passes for full, scale-removal, +1-day and +16-day conditions.
- Positive gate slope, normalized scale weights and bounded stage gates.
- Whole-stage permutation, rejection of incomplete permutations and one-stage inputs.
- Training-only median/standardization fitting and state round-trip.
- Preservation of invalid PPI gaps, undefined spectral features and entropy edge cases.
- Morphology QC and selection of the first QC-passing daily recording.
- Strict >10-pp endpoint labels and rejection of missing patient-days.
- Repetition-specific AUROC/AP averaging, pooled confusion counts and paired clustering.
- A minimal nested cross-validation run, checkpoint restoration and patient split separation.

All test signals, patient identifiers and arrays were **artificial unit-test fixtures**.
The training smoke test used one epoch and one CV repetition for computational checking,
not the manuscript's experimental budget. No clinical performance was estimated or
verified. No private patient data or trained clinical model weights are present.
