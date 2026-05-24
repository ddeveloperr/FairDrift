"""
CUSUM (Cumulative Sum) control charts for sequential fairness testing.
Includes Monte Carlo calibration of decision boundary h.
"""
import numpy as np
from typing import Optional

from . import config


class CUSUMChart:
    """
    Upper one-sided CUSUM chart for detecting sustained increases in a metric.
    S_t+ = max(0, S_{t-1}+ + (x_t - mu0 - k))
    Alarm when S_t+ > h.

    Used to detect sustained fairness metric degradation after drift is confirmed.
    """

    def __init__(
        self,
        mu0: float,
        delta: float,
        h: float = None,
        name: str = "",
    ):
        """
        Args:
            mu0: In-control mean (training-period baseline of the fairness metric)
            delta: Shift magnitude to detect (e.g., EOD threshold - baseline)
            h: Decision boundary. If None, must be calibrated via calibrate_h().
            name: Identifier for this chart (e.g., "EOD_AfricanAmerican")
        """
        self.mu0 = mu0
        self.delta = delta
        self.k = delta / 2  # Allowance: halfway between in-control and out-of-control
        self.h = h
        self.name = name

        # State
        self._s_plus = 0.0
        self._observations = []
        self._alarm_triggered = False
        self._alarm_point = None

    @property
    def statistic(self) -> float:
        """Current CUSUM statistic."""
        return self._s_plus

    @property
    def is_active(self) -> bool:
        """Whether the chart has been calibrated and is ready for monitoring."""
        return self.h is not None

    def reset(self):
        """Reset chart state (after alarm acknowledgement)."""
        self._s_plus = 0.0
        self._observations = []
        self._alarm_triggered = False
        self._alarm_point = None

    def update(self, observation: float) -> dict:
        """
        Add a new observation to the chart.

        Args:
            observation: New fairness metric value

        Returns:
            dict with: alarm, statistic, observation_index
        """
        if self.h is None:
            raise ValueError("Chart not calibrated. Call calibrate_h() first.")

        self._observations.append(observation)
        self._s_plus = max(0.0, self._s_plus + (observation - self.mu0 - self.k))

        alarm = self._s_plus > self.h
        if alarm and not self._alarm_triggered:
            self._alarm_triggered = True
            self._alarm_point = len(self._observations)

        return {
            "alarm": alarm,
            "statistic": self._s_plus,
            "observation_index": len(self._observations),
            "threshold": self.h,
        }

    def calibrate_h(
        self,
        stationary_data: np.ndarray,
        arl0_target: int = None,
        n_paths: int = None,
        seed: int = None,
    ) -> float:
        """
        Calibrate decision boundary h via Monte Carlo simulation.
        Generate n_paths synthetic paths from stationary data distribution
        and find h such that the average run length (ARL0) >= target.

        Args:
            stationary_data: Array of fairness metric values from the in-control period
            arl0_target: Target ARL0 (default from config)
            n_paths: Number of Monte Carlo paths (default from config)
            seed: Random seed

        Returns:
            Calibrated h value
        """
        if arl0_target is None:
            arl0_target = config.CUSUM_ARL0_TARGET
        if n_paths is None:
            n_paths = config.CUSUM_MC_PATHS
        if seed is None:
            seed = config.PRIMARY_SEED

        rng = np.random.default_rng(seed)

        # Estimate distribution parameters from stationary data
        mean = np.mean(stationary_data)
        std = np.std(stationary_data)

        # Binary search for h
        h_low = 0.1
        h_high = 20.0 * std  # Upper bound: very conservative
        h_mid = None

        for _ in range(50):  # Binary search iterations
            h_mid = (h_low + h_high) / 2
            arl = self._estimate_arl(mean, std, h_mid, arl0_target * 3, n_paths, rng)

            if arl >= arl0_target:
                h_high = h_mid  # h is too large or just right, try smaller
            else:
                h_low = h_mid  # h is too small, try larger

            # Convergence check
            if abs(h_high - h_low) < 0.001 * std:
                break

        # Use h_high to be conservative (fewer false alarms)
        self.h = h_high
        self.mu0 = mean  # Update mu0 to actual stationary mean

        return self.h

    def _estimate_arl(
        self,
        mean: float,
        std: float,
        h: float,
        max_length: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Estimate ARL for a given h via Monte Carlo.
        Generates paths from N(mean, std) and measures run lengths.
        """
        run_lengths = []

        for _ in range(n_paths):
            s_plus = 0.0
            for t in range(1, max_length + 1):
                x = rng.normal(mean, std)
                s_plus = max(0.0, s_plus + (x - mean - self.k))
                if s_plus > h:
                    run_lengths.append(t)
                    break
            else:
                run_lengths.append(max_length)  # Censored

        return np.mean(run_lengths)


class FairnessCUSUMMonitor:
    """
    Manages multiple CUSUM charts for all subgroup × metric combinations.
    Charts remain inactive until drift is confirmed by the drift detection ensemble.
    """

    def __init__(self):
        """Initialize empty monitor. Charts added via add_chart()."""
        self.charts = {}  # key: (subgroup, metric) -> CUSUMChart
        self.active_charts = set()  # Charts activated by drift detection

    def add_chart(self, subgroup: str, metric: str, mu0: float, delta: float):
        """
        Register a CUSUM chart for a subgroup × metric combination.

        Args:
            subgroup: e.g., "AfricanAmerican"
            metric: e.g., "eod"
            mu0: Baseline mean from training period
            delta: Shift to detect (typically EOD_THRESHOLD - mu0)
        """
        key = (subgroup, metric)
        chart = CUSUMChart(
            mu0=mu0,
            delta=delta,
            name=f"{metric}_{subgroup}",
        )
        self.charts[key] = chart

    def activate_chart(self, subgroup: str, metric: str):
        """Activate a chart after drift detection confirms drift in this subgroup."""
        key = (subgroup, metric)
        if key in self.charts:
            self.active_charts.add(key)

    def calibrate_all(self, stationary_metrics: dict, seed: int = None):
        """
        Calibrate all charts using stationary-period metric values.

        Args:
            stationary_metrics: dict of {(subgroup, metric): np.ndarray of values}
            seed: Random seed
        """
        for key, chart in self.charts.items():
            if key in stationary_metrics:
                chart.calibrate_h(stationary_metrics[key], seed=seed)

    def update(self, subgroup: str, metric: str, value: float) -> Optional[dict]:
        """
        Update a specific chart with a new observation.
        Only processes if the chart is active.

        Returns:
            Alert dict if alarm triggered, None otherwise
        """
        key = (subgroup, metric)

        # Only monitor active charts
        if key not in self.active_charts:
            return None

        chart = self.charts.get(key)
        if chart is None or not chart.is_active:
            return None

        result = chart.update(value)

        if result["alarm"]:
            return {
                "subgroup": subgroup,
                "metric": metric,
                "magnitude": value,
                "onset_point": result["observation_index"],
                "cusum_statistic": result["statistic"],
                "threshold": result["threshold"],
            }

        return None

    def update_all_active(self, metric_values: dict) -> list:
        """
        Update all active charts with current metric values.

        Args:
            metric_values: dict of {(subgroup, metric): float}

        Returns:
            List of alarm dicts for any triggered charts
        """
        alarms = []
        for key in self.active_charts:
            if key in metric_values:
                subgroup, metric = key
                result = self.update(subgroup, metric, metric_values[key])
                if result is not None:
                    alarms.append(result)
        return alarms

    def get_status(self) -> pd.DataFrame:
        """Return current status of all charts."""
        import pandas as pd

        records = []
        for key, chart in self.charts.items():
            subgroup, metric = key
            records.append({
                "subgroup": subgroup,
                "metric": metric,
                "active": key in self.active_charts,
                "calibrated": chart.is_active,
                "current_statistic": chart.statistic,
                "h": chart.h,
                "alarm_triggered": chart._alarm_triggered,
            })
        return pd.DataFrame(records)
