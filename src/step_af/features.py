"""Daily rhythm features with explicit invalid-interval gap handling."""
from dataclasses import dataclass
import numpy as np
from scipy import signal, integrate

FEATURES = ("RMSSD", "SDSD", "pNN70", "SDNN", "LF_power", "HF_power",
            "LF_HF", "Shannon_entropy", "sample_entropy", "SD1_SD2")


@dataclass
class QualityResult:
    values: np.ndarray
    candidate_indices: np.ndarray
    midpoints: np.ndarray
    retained_fraction: float
    valid: bool
    reason: str


def quality_check(ppg, fs=200.0, minimum_peak_distance_s=0.30, cycle_points=128):
    """Morphology QC. Peak distance and resampling size are reference choices.

    Only complete pulse cycles between neighboring peak midpoints are scored;
    both end peaks are excluded because their full cycles are unavailable.
    Ratios use candidate PPIs between the remaining complete-cycle peaks.
    """
    ppg = np.asarray(ppg, dtype=float)
    empty = np.empty(0)
    if ppg.ndim != 1 or len(ppg) < int(fs * 60) or not np.isfinite(ppg).all():
        return QualityResult(empty, empty.astype(int), empty, 0., False,
                             "short_or_nonfinite_recording")
    if fs <= 8 or cycle_points < 3:
        raise ValueError("Sampling rate or cycle length is invalid.")
    detrended = signal.detrend(ppg, type="linear")
    sos = signal.butter(4, [0.5, 4.0], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, detrended)
    peaks, _ = signal.find_peaks(
        filtered, distance=max(1, int(round(fs * minimum_peak_distance_s))))
    if len(peaks) < 4:
        return QualityResult(empty, empty.astype(int), empty, 0., False, "too_few_peaks")
    rows = []
    for k in range(1, len(peaks) - 1):
        lo = int((peaks[k - 1] + peaks[k]) // 2)
        hi = int((peaks[k] + peaks[k + 1]) // 2)
        waveform = filtered[lo:hi + 1]
        row = np.interp(np.linspace(0, len(waveform) - 1, cycle_points),
                        np.arange(len(waveform)), waveform)
        # Amplitude normalization before template construction is a reference choice.
        scale = np.std(row)
        rows.append((row - np.mean(row)) / scale if scale > 1e-12 else np.zeros_like(row))
    rows = np.asarray(rows)
    template = np.median(rows, axis=0)
    centered = rows - rows.mean(axis=1, keepdims=True)
    template = template - template.mean()
    denom = np.linalg.norm(centered, axis=1) * np.linalg.norm(template)
    correlations = np.divide(centered @ template, denom,
                             out=np.zeros(len(rows)), where=denom > 1e-12)
    pulse_ok = correlations >= 0.80
    complete_peaks = peaks[1:-1]
    intervals = np.diff(complete_peaks) * (1000. / fs)
    midpoints = (complete_peaks[:-1] + complete_peaks[1:]) / (2. * fs)
    keep = pulse_ok[:-1] & pulse_ok[1:] & (intervals >= 300.) & (intervals <= 2000.)
    fraction = float(keep.mean()) if len(keep) else 0.
    valid = fraction >= 0.8 and int(keep.sum()) >= 15
    return QualityResult(intervals[keep], np.flatnonzero(keep), midpoints[keep],
                         fraction, valid, "passed" if valid else "failed_ppi_quality")


def _sample_entropy(values, indices):
    eligible = [j for j in range(len(values) - 2)
                if indices[j + 1] == indices[j] + 1
                and indices[j + 2] == indices[j] + 2]
    if len(eligible) < 2:
        return np.nan
    tolerance = 0.2 * np.std(values, ddof=1)
    a = b = 0
    for ii, start in enumerate(eligible[:-1]):
        for other in eligible[ii + 1:]:
            if np.max(np.abs(values[start:start + 2] - values[other:other + 2])) <= tolerance:
                b += 1
                a += int(abs(values[start + 2] - values[other + 2]) <= tolerance)
    return -np.log(a / b) if a and b else np.nan


def _band_integral(freq, density, low, high):
    inside = (freq > low) & (freq < high)
    f = np.r_[low, freq[inside], high]
    p = np.r_[np.interp(low, freq, density), density[inside],
              np.interp(high, freq, density)]
    return integrate.trapezoid(p, f)


def features_from_ppi(values, candidate_indices, midpoints):
    """Return 10 features. PPIs are ms; original midpoint timestamps are seconds.

    Rejected PPI gaps are NOT bridged for successive-interval features or entropy.
    Undefined features remain NaN, to be imputed from training patients only.
    """
    v = np.asarray(values, dtype=float)
    idx = np.asarray(candidate_indices, dtype=int)
    t = np.asarray(midpoints, dtype=float)
    if v.ndim != 1 or v.shape != idx.shape or v.shape != t.shape:
        raise ValueError("PPI values, candidate indices and midpoints must align.")
    if len(v) < 2 or not np.isfinite(v).all() or not np.isfinite(t).all():
        raise ValueError("At least two finite valid PPIs are required.")
    if np.any(np.diff(idx) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Original indices and timestamps must increase strictly.")
    if np.any((v < 300) | (v > 2000)):
        raise ValueError("PPIs must pass the documented physiological range check.")
    adjacent = np.diff(idx) == 1
    differences = np.diff(v)[adjacent]
    rmssd = np.sqrt(np.mean(differences ** 2)) if len(differences) else np.nan
    sdsd = np.std(differences, ddof=1) if len(differences) > 1 else np.nan
    pnn70 = 100. * np.mean(np.abs(differences) > 70) if len(differences) else np.nan
    sdnn = np.std(v, ddof=1)
    lf = hf = ratio = np.nan
    if t[-1] - t[0] >= 30 and np.max(np.diff(t)) <= 5:
        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * 4)) + 1) / 4
        interpolated = np.interp(grid, t, v)
        length = min(128, len(grid))
        freq, density = signal.welch(
            interpolated, fs=4, window=signal.windows.hann(length, sym=False),
            nperseg=length, noverlap=length // 2, nfft=1024,
            detrend="linear", return_onesided=True, scaling="density", average="mean")
        lf = _band_integral(freq, density, 0.04, 0.15)
        hf = _band_integral(freq, density, 0.15, 0.40)
        ratio = lf / hf if hf > 1e-12 else np.nan
    counts, _ = np.histogram(v, bins=np.arange(300, 2001, 50))
    probability = counts[counts > 0] / len(v)
    shannon = -np.sum(probability * np.log(probability))
    sampen = _sample_entropy(v, idx)
    poincare_ratio = np.nan
    if adjacent.sum() > 1:
        before, after = v[:-1][adjacent], v[1:][adjacent]
        # Divide after taking SD to avoid roundoff yielding a nonzero spread
        # for a perfectly constant rotated coordinate.
        sd1 = np.std(after - before, ddof=1) / np.sqrt(2)
        sd2 = np.std(after + before, ddof=1) / np.sqrt(2)
        poincare_ratio = sd1 / sd2 if sd2 > 0 else np.nan
    result = np.asarray([rmssd, sdsd, pnn70, sdnn, lf, hf, ratio, shannon,
                         sampen, poincare_ratio], dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result
