"""Repeated OOF metrics and paired, class-stratified patient-cluster bootstrap."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def repeated_metrics(labels, probabilities, predictions):
    labels = np.asarray(labels, dtype=int)
    p, hard = np.asarray(probabilities), np.asarray(predictions)
    if p.ndim != 2 or p.shape != hard.shape or p.shape[0] != len(labels):
        raise ValueError("Expected patient x repetition probabilities and predictions.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between zero and one.")
    if set(np.unique(labels)) != {0, 1} or not np.isin(hard, [0, 1]).all():
        raise ValueError("Both endpoint classes and binary predictions are required.")
    per_repeat = [{"AUROC": float(roc_auc_score(labels, p[:, j])),
                   "AUPRC": float(average_precision_score(labels, p[:, j]))}
                  for j in range(p.shape[1])]
    repeated_labels = np.repeat(labels, p.shape[1])
    tn, fp, fn, tp = confusion_matrix(repeated_labels, hard.ravel(), labels=[0, 1]).ravel()
    return {"AUROC": float(np.mean([r["AUROC"] for r in per_repeat])),
            "AUPRC": float(np.mean([r["AUPRC"] for r in per_repeat])),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "F1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.,
            "confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
            "patients": len(labels), "repetitions": p.shape[1], "per_repeat": per_repeat}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or np.any((pvalues < 0) | (pvalues > 1)) or not np.isfinite(pvalues).all():
        raise ValueError("p-values must be a finite one-dimensional array in [0,1].")
    order = np.argsort(pvalues)
    result = np.empty_like(pvalues)
    result[order] = np.minimum(1., np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return result


def paired_cluster_bootstrap(labels, reference, alternatives, samples=10000, seed=20260922):
    """Alternative-minus-reference AUROC differences; all repetitions travel together.

    Each alternative must have the same patient order and endpoint labels. Call
    separately for different clinical thresholds. Holm adjustment covers ONLY
    the alternatives provided in a single explicitly declared comparison family.
    """
    y = np.asarray(labels, dtype=int)
    reference = np.asarray(reference, dtype=float)
    alternatives = np.asarray(alternatives, dtype=float)
    if reference.ndim != 2 or reference.shape[0] != len(y):
        raise ValueError("Reference must have shape patients x repetitions.")
    if alternatives.ndim != 3 or alternatives.shape[1:] != reference.shape:
        raise ValueError("Alternatives must have shape comparisons x patients x repetitions.")
    if not np.isfinite(reference).all() or not np.isfinite(alternatives).all() or samples < 1:
        raise ValueError("Finite predictions and a positive bootstrap count are required.")
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    if any(not len(group) for group in classes):
        raise ValueError("Both endpoint classes are required.")
    def differences(index):
        return np.asarray([np.mean([
            roc_auc_score(y[index], alt[index, rep]) - roc_auc_score(y[index], reference[index, rep])
            for rep in range(reference.shape[1])]) for alt in alternatives])
    observed = differences(np.arange(len(y)))
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = np.zeros(len(alternatives), dtype=int)
    for _ in range(samples):
        chosen = np.concatenate([rng.choice(group, len(group), replace=True) for group in classes])
        delta = differences(chosen)
        counts += np.abs(delta - observed) >= np.abs(observed)
    unadjusted = (1 + counts) / (samples + 1)
    return {"delta_auroc": observed.tolist(), "p_unadjusted": unadjusted.tolist(),
            "p_holm": holm_adjust(unadjusted).tolist(), "samples": samples, "seed": seed}
