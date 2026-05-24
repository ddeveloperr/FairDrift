"""
Drift detection algorithms: KS test, ADWIN, Page-Hinkley.
Plus majority-vote ensemble combining all three.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from collections import deque

from . import config


class KSDriftDetector:
    """
    Kolmogorov-Smirnov two-sample test for drift detection.
    Compares new data window against reference (training) distribution.
    """

    def __init__(self, reference_data: np.ndarray, alpha: float = None):
        """
        Args:
            reference_data: Data from reference period (W1)
            alpha: Significance level for drift declaration
        """
        self.reference = reference_data.copy()
        self.alpha = alpha if alpha is not None else config.BONFERRONI_ALPHA_DRIFT

    def test(self, new_data: np.ndarray) -> dict:
        """
        Test whether new_data comes from the same distribution as reference.

        Returns:
            dict with keys: drift_detected, ks_statistic, p_value
        """
        stat, p_value = ks_2samp(self.reference, new_data)
        return {
            "drift_detected": p_value < self.alpha,
            "ks_statistic": stat,
            "p_value": p_value,
        }


class ADWINDetector:
    """
    ADaptive WINdowing (ADWIN) for concept drift detection.
    Maintains a variable-length window and detects distributional changes.

    Simplified implementation suitable for batch monitoring (not streaming).
    """

    def __init__(self, delta: float = None):
        """
        Args:
            delta: Confidence parameter. Smaller = fewer false alarms.
        """
        self.delta = delta if delta is not None else config.ADWIN_DELTA
        self.window = deque()
        self._sum = 0.0
        self._variance = 0.0
        self._width = 0

    def reset(self):
        """Reset the detector state."""
        self.window = deque()
        self._sum = 0.0
        self._width = 0

    def update(self, value: float) -> bool:
        """
        Add a new observation. Returns True if drift is detected.
        """
        self.window.append(value)
        self._sum += value
        self._width += 1

        if self._width < 10:
            return False

        return self._check_cuts()

    def _check_cuts(self) -> bool:
        """Check all possible cuts of the window for distributional change."""
        data = np.array(self.window)
        n = len(data)

        for split in range(max(5, n // 4), min(n - 5, 3 * n // 4)):
            left = data[:split]
            right = data[split:]

            n0 = len(left)
            n1 = len(right)
            mu0 = left.mean()
            mu1 = right.mean()

            # Hoeffding bound for the difference of means
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))

            if abs(mu0 - mu1) >= epsilon:
                # Drift detected: shrink window to right portion
                self.window = deque(right)
                self._sum = right.sum()
                self._width = len(right)
                return True

        return False

    def test_batch(self, reference: np.ndarray, new_data: np.ndarray) -> dict:
        """
        Batch mode: test if new_data differs from reference.
        Uses the same Hoeffding bound principle.
        """
        n0 = len(reference)
        n1 = len(new_data)
        mu0 = reference.mean()
        mu1 = new_data.mean()

        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))

        drift_detected = abs(mu0 - mu1) >= epsilon
        return {
            "drift_detected": drift_detected,
            "mean_diff": abs(mu0 - mu1),
            "threshold": epsilon,
        }


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting gradual mean shifts.
    Tracks cumulative sum of deviations from running mean.
    """

    def __init__(self, delta: float = None, threshold: float = None):
        """
        Args:
            delta: Minimum allowable change magnitude
            threshold: Decision boundary (lambda)
        """
        self.delta = delta if delta is not None else config.PAGE_HINKLEY_DELTA
        self.threshold = threshold if threshold is not None else config.PAGE_HINKLEY_THRESHOLD
        self.reset()

    def reset(self):
        """Reset detector state."""
        self._n = 0
        self._sum = 0.0
        self._x_mean = 0.0
        self._cum_sum = 0.0
        self._min_cum_sum = float("inf")

    def update(self, value: float) -> bool:
        """
        Add observation. Returns True if drift detected.
        """
        self._n += 1
        self._sum += value
        self._x_mean = self._sum / self._n

        self._cum_sum += value - self._x_mean - self.delta
        self._min_cum_sum = min(self._min_cum_sum, self._cum_sum)

        # Test statistic
        ph_stat = self._cum_sum - self._min_cum_sum

        return ph_stat > self.threshold

    @property
    def statistic(self) -> float:
        """Current test statistic value."""
        return self._cum_sum - self._min_cum_sum

    def test_batch(self, reference: np.ndarray, new_data: np.ndarray) -> dict:
        """
        Batch mode: feed reference then new_data sequentially.
        Returns drift status after processing all of new_data.
        """
        self.reset()

        # Feed reference (should not trigger)
        for val in reference:
            self.update(val)

        # Reset after establishing baseline
        baseline_mean = self._x_mean
        self.reset()

        # Feed new data relative to baseline
        drift_detected = False
        for val in new_data:
            if self.update(val - baseline_mean + self.delta):
                drift_detected = True
                break

        return {
            "drift_detected": drift_detected,
            "statistic": self.statistic,
            "threshold": self.threshold,
        }


class DriftEnsemble:
    """
    Ensemble of drift detectors with majority voting.
    Drift is declared when >= 2 of 3 detectors agree.
    """

    def __init__(self, reference_data: np.ndarray, alpha: float = None):
        """
        Initialize all three detectors with the reference data.

        Args:
            reference_data: Training period data for this feature/subgroup
            alpha: Significance level for KS test
        """
        self.reference = reference_data.copy()
        self.ks = KSDriftDetector(reference_data, alpha)
        self.adwin = ADWINDetector()
        self.ph = PageHinkleyDetector()

    def test(self, new_data: np.ndarray) -> dict:
        """
        Test new data against reference using all three detectors.
        Majority vote determines final decision.

        Returns:
            dict with: drift_detected, n_detectors_agree, per_detector_results
        """
        # KS test (batch)
        ks_result = self.ks.test(new_data)

        # ADWIN (batch mode)
        adwin_result = self.adwin.test_batch(self.reference, new_data)

        # Page-Hinkley (batch mode)
        ph_result = self.ph.test_batch(self.reference, new_data)

        # Majority vote
        votes = [
            ks_result["drift_detected"],
            adwin_result["drift_detected"],
            ph_result["drift_detected"],
        ]
        n_agree = sum(votes)
        drift_detected = n_agree >= 2  # Majority

        return {
            "drift_detected": drift_detected,
            "n_detectors_agree": n_agree,
            "ks": ks_result,
            "adwin": adwin_result,
            "page_hinkley": ph_result,
        }


def run_subgroup_drift_detection(
    windows: list,
    feature_cols: list,
    protected_col: str,
    subgroups: list,
) -> pd.DataFrame:
    """
    Run drift detection for each subgroup × feature combination.

    Args:
        windows: List of DataFrames [W1, W2, ..., W5]
        feature_cols: Features to monitor
        protected_col: Column identifying subgroups
        subgroups: List of subgroup values to monitor

    Returns:
        DataFrame with: window, subgroup, feature, drift_detected, n_detectors, ks_stat
    """
    w1 = windows[0]
    results = []

    for subgroup in subgroups:
        ref_mask = w1[protected_col] == subgroup
        if ref_mask.sum() < config.MIN_SUBGROUP_SIZE:
            continue

        for feat in feature_cols:
            ref_data = w1.loc[ref_mask, feat].dropna().values
            if len(ref_data) < 30:
                continue

            ensemble = DriftEnsemble(ref_data)

            for w_idx in range(1, len(windows)):
                test_mask = windows[w_idx][protected_col] == subgroup
                test_data = windows[w_idx].loc[test_mask, feat].dropna().values

                if len(test_data) < 30:
                    continue

                result = ensemble.test(test_data)
                results.append({
                    "window": w_idx + 1,
                    "subgroup": subgroup,
                    "feature": feat,
                    "drift_detected": result["drift_detected"],
                    "n_detectors": result["n_detectors_agree"],
                    "ks_statistic": result["ks"]["ks_statistic"],
                    "ks_p_value": result["ks"]["p_value"],
                })

    return pd.DataFrame(results)
