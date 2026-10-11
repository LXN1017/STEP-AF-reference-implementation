"""Patient-level loading, first-valid daily selection and training-only transforms."""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from .features import FEATURES, quality_check, features_from_ppi


@dataclass
class Cohort:
    patient_ids: tuple
    x: np.ndarray
    baseline: np.ndarray
    followup: np.ndarray

    def labels(self, threshold_pp=10.0):
        return ((self.followup - self.baseline) > threshold_pp).astype(np.int64)


def load_cohort(feature_csv, label_csv, days=96):
    labels = {}
    with open(label_csv, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pid = row["patient_id"]
            if pid in labels:
                raise ValueError("Duplicate patient in labels.")
            burden = [float(row["baseline_burden_percent"]), float(row["followup_burden_percent"])]
            if not np.isfinite(burden).all() or not all(0 <= b <= 100 for b in burden):
                raise ValueError("Holter burdens must be finite percentages in [0,100].")
            labels[pid] = burden
    entries = {}
    with open(feature_csv, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["patient_id"], int(row["day"]))
            if key in entries:
                raise ValueError("Duplicate daily feature row.")
            if key[0] not in labels or not 1 <= key[1] <= days:
                raise ValueError("Feature patient/day is outside the declared cohort.")
            entries[key] = [float(row[name]) if row[name].strip() else np.nan for name in FEATURES]
    ids = tuple(sorted(labels))
    if not ids:
        raise ValueError("Empty cohort.")
    missing = [(pid, day) for pid in ids for day in range(1, days + 1)
               if (pid, day) not in entries]
    if missing:
        raise ValueError(f"{len(missing)} missing patient-days: day-level imputation is not allowed.")
    x = np.asarray([[entries[pid, day] for day in range(1, days + 1)] for pid in ids])
    x[~np.isfinite(x)] = np.nan
    burden = np.asarray([labels[pid] for pid in ids])
    return Cohort(ids, x, burden[:, 0], burden[:, 1])


class TrainingPreprocessor:
    """Fit medians and z-score parameters only on designated training patients."""
    def fit(self, x):
        x = np.asarray(x, dtype=float).copy()
        x[~np.isfinite(x)] = np.nan
        flat = x.reshape(-1, x.shape[-1])
        if np.isnan(flat).all(axis=0).any():
            raise ValueError("A feature is undefined in every training observation.")
        self.median = np.nanmedian(flat, axis=0)
        imputed = np.where(np.isnan(flat), self.median, flat)
        self.mean = imputed.mean(axis=0)
        self.std = imputed.std(axis=0)
        self.std[self.std == 0] = 1.
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != len(self.median):
            raise ValueError("Feature dimension differs from training.")
        imputed = np.where(np.isfinite(x), x, self.median)
        return ((imputed - self.mean) / self.std).astype(np.float32)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ("median", "mean", "std")}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for k in ("median", "mean", "std"):
            setattr(obj, k, np.asarray(state[k]))
        return obj


def extract_first_valid(manifest_csv, output_csv, audit_csv, fs=200):
    """Manifest: patient_id,day,started_at,recording_path.

    ISO-8601 started_at values must use a consistent timezone convention.
    Paths refer to one-dimensional .npy arrays or one-column headerless CSV files.
    All attempts are audited. Only the earliest QC-passing attempt is selected.
    """
    manifest_csv = Path(manifest_csv)
    with manifest_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        day = int(row["day"])
        if not 1 <= day <= 96:
            raise ValueError("Day must be in [1,96].")
        row["timestamp"] = datetime.fromisoformat(row["started_at"])
        groups.setdefault((row["patient_id"], day), []).append(row)
    outputs, audit = [], []
    for (pid, day), attempts in sorted(groups.items()):
        # Reject ambiguous duplicate acquisition times rather than silently pick a tie.
        times = [a["timestamp"] for a in attempts]
        if len(set(times)) != len(times):
            raise ValueError("Duplicate acquisition timestamps within a patient-day.")
        selected = False
        for row in sorted(attempts, key=lambda a: a["timestamp"]):
            path = Path(row["recording_path"])
            if not path.is_absolute():
                path = manifest_csv.parent / path
            waveform = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" \
                else np.loadtxt(path, delimiter=",")
            qc = quality_check(waveform, fs=fs)
            take = qc.valid and not selected
            if take:
                features = features_from_ppi(qc.values, qc.candidate_indices, qc.midpoints)
                outputs.append([pid, day, *features.tolist()])
                selected = True
            audit.append([pid, day, row["started_at"], qc.valid, take,
                          qc.retained_fraction, len(qc.values), qc.reason])
    for filename, header, content in (
        (output_csv, ["patient_id", "day", *FEATURES], outputs),
        (audit_csv, ["patient_id", "day", "started_at", "quality_passed", "selected",
                     "retained_ppi_fraction", "valid_ppi_count", "reason"], audit)):
        with open(filename, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(content)
    return {"selected_patient_days": len(outputs), "audited_attempts": len(audit),
            "days_without_valid_record": len(groups) - len(outputs)}
