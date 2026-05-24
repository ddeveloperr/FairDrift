"""
Controlled drift injection for RQ3 experiments.
Injects covariate shift, label shift, and concept drift into specific subgroups.
"""
import numpy as np
import pandas as pd

from . import config


class DriftInjector:
    """
    Injects controlled distribution shifts into data for a target subgroup.
    Used to establish causal drift-type → fairness-metric mapping.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed if seed is not None else config.PRIMARY_SEED)

    def inject_covariate_shift(
        self,
        df: pd.DataFrame,
        subgroup_col: str,
        subgroup_value: str,
        features: list,
        magnitude_sd: float,
    ) -> pd.DataFrame:
        """
        Shift feature distributions for a specific subgroup by magnitude_sd standard deviations.

        Mechanism: For the target subgroup, add magnitude_sd * std(feature) to each feature.
        This simulates the subgroup's feature distribution drifting while others remain stable.

        Args:
            df: Input DataFrame
            subgroup_col: Column identifying subgroups (e.g., "race")
            subgroup_value: Target subgroup (e.g., "AfricanAmerican")
            features: Features to shift
            magnitude_sd: Shift magnitude in standard deviations

        Returns:
            Modified DataFrame with shifted features for target subgroup
        """
        df = df.copy()
        mask = df[subgroup_col] == subgroup_value

        for feat in features:
            if feat not in df.columns:
                continue
            if df[feat].dtype not in ["float64", "int64", "float32", "int32"]:
                continue

            std = df[feat].std()
            shift = magnitude_sd * std

            # Add shift + small noise to avoid identical values
            noise = self.rng.normal(0, std * 0.05, size=mask.sum())
            df.loc[mask, feat] = df.loc[mask, feat] + shift + noise

        return df

    def inject_label_shift(
        self,
        df: pd.DataFrame,
        subgroup_col: str,
        subgroup_value: str,
        target_col: str,
        new_rate: float,
    ) -> pd.DataFrame:
        """
        Alter the positive label rate for a specific subgroup.

        Mechanism: Randomly flip labels in the subgroup to achieve the desired rate.
        If new_rate > current rate: flip some 0→1.
        If new_rate < current rate: flip some 1→0.

        Args:
            df: Input DataFrame
            subgroup_col: Column identifying subgroups
            subgroup_value: Target subgroup
            target_col: Binary target column to modify
            new_rate: Desired positive rate for the subgroup

        Returns:
            Modified DataFrame with altered labels for target subgroup
        """
        df = df.copy()
        mask = df[subgroup_col] == subgroup_value
        subgroup_data = df.loc[mask, target_col].values.copy()

        current_rate = subgroup_data.mean()
        n = len(subgroup_data)
        target_positives = int(n * new_rate)
        current_positives = subgroup_data.sum()

        if target_positives > current_positives:
            # Need to flip some 0→1
            zero_indices = np.where(subgroup_data == 0)[0]
            n_flip = min(target_positives - int(current_positives), len(zero_indices))
            flip_idx = self.rng.choice(zero_indices, size=n_flip, replace=False)
            subgroup_data[flip_idx] = 1
        elif target_positives < current_positives:
            # Need to flip some 1→0
            one_indices = np.where(subgroup_data == 1)[0]
            n_flip = min(int(current_positives) - target_positives, len(one_indices))
            flip_idx = self.rng.choice(one_indices, size=n_flip, replace=False)
            subgroup_data[flip_idx] = 0

        df.loc[mask, target_col] = subgroup_data
        return df

    def inject_concept_drift(
        self,
        df: pd.DataFrame,
        subgroup_col: str,
        subgroup_value: str,
        target_col: str,
        feature_cols: list,
        multiplier: float,
    ) -> pd.DataFrame:
        """
        Alter the feature-outcome relationship for a specific subgroup.

        Mechanism: For the target subgroup, relabel samples based on an amplified
        logistic model: P(Y=1) becomes more strongly dependent on specified features.
        This simulates concept drift where the predictive relationship changes.

        Args:
            df: Input DataFrame
            subgroup_col: Column identifying subgroups
            subgroup_value: Target subgroup
            target_col: Binary target column
            feature_cols: Features whose relationship to the target is amplified
            multiplier: Factor by which to amplify the feature-outcome relationship

        Returns:
            Modified DataFrame with concept-drifted labels for target subgroup
        """
        df = df.copy()
        mask = df[subgroup_col] == subgroup_value
        subgroup_df = df.loc[mask].copy()

        if len(subgroup_df) == 0:
            return df

        # Compute feature-based score (z-scored, then amplified)
        scores = np.zeros(len(subgroup_df))
        for feat in feature_cols:
            if feat not in subgroup_df.columns:
                continue
            if subgroup_df[feat].dtype not in ["float64", "int64", "float32", "int32"]:
                continue

            feat_vals = subgroup_df[feat].values.astype(float)
            std = feat_vals.std()
            if std > 0:
                z = (feat_vals - feat_vals.mean()) / std
                scores += z * multiplier

        # Convert scores to probabilities via sigmoid
        probs = 1 / (1 + np.exp(-scores))

        # Maintain approximate base rate by adjusting the intercept
        current_rate = subgroup_df[target_col].mean()
        # Adjust: shift probs so mean(probs) ≈ new_rate (slightly elevated)
        # The multiplier naturally increases the rate, which is the concept drift effect
        adjusted_rate = min(current_rate * (1 + (multiplier - 1) * 0.3), 0.9)

        # Generate new labels
        new_labels = (self.rng.random(len(subgroup_df)) < probs).astype(int)

        # Scale to maintain approximately the target rate
        # (concept drift should change the relationship, not just the rate)
        target_n_pos = int(len(new_labels) * adjusted_rate)
        current_n_pos = new_labels.sum()

        if current_n_pos > target_n_pos:
            ones = np.where(new_labels == 1)[0]
            flip = self.rng.choice(ones, size=current_n_pos - target_n_pos, replace=False)
            new_labels[flip] = 0
        elif current_n_pos < target_n_pos:
            zeros = np.where(new_labels == 0)[0]
            n_flip = min(target_n_pos - current_n_pos, len(zeros))
            flip = self.rng.choice(zeros, size=n_flip, replace=False)
            new_labels[flip] = 1

        df.loc[mask, target_col] = new_labels
        return df


def run_single_injection_experiment(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    task: str,
    protected_col: str,
    target_subgroup: str,
    reference_group: str,
    drift_type: str,
    severity_index: int,
    seed: int,
) -> dict:
    """
    Run one injection experiment and compute fairness metrics.

    Args:
        df: Clean DataFrame (one window of data)
        model: Trained model
        feature_cols: Feature columns for prediction
        task: Target column name
        protected_col: Protected attribute column
        target_subgroup: Subgroup to inject drift into
        reference_group: Reference group for comparison
        drift_type: "covariate", "label", or "concept"
        severity_index: 0=low, 1=medium, 2=high
        seed: Random seed

    Returns:
        dict with drift_type, severity, eod, dpd, ece
    """
    from .fairness_metrics import (
        equalized_odds_difference,
        demographic_parity_difference,
        expected_calibration_error,
    )

    injector = DriftInjector(seed=seed)

    # Apply injection based on type and severity
    if drift_type == "covariate":
        magnitude = config.COVARIATE_SHIFT_MAGNITUDES[severity_index]
        # Shift key clinical features
        shift_features = ["num_medications", "number_inpatient", "num_lab_procedures",
                         "number_diagnoses", "number_emergency"]
        shift_features = [f for f in shift_features if f in df.columns]
        df_injected = injector.inject_covariate_shift(
            df, protected_col, target_subgroup, shift_features, magnitude
        )

    elif drift_type == "label":
        new_rate = config.LABEL_SHIFT_RATES[severity_index]
        df_injected = injector.inject_label_shift(
            df, protected_col, target_subgroup, task, new_rate
        )

    elif drift_type == "concept":
        multiplier = config.CONCEPT_DRIFT_MULTIPLIERS[severity_index]
        concept_features = ["num_medications", "number_inpatient"]
        concept_features = [f for f in concept_features if f in df.columns]
        df_injected = injector.inject_concept_drift(
            df, protected_col, target_subgroup, task, concept_features, multiplier
        )
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")

    # Compute predictions on injected data
    X = df_injected[feature_cols].values
    y_true = df_injected[task].values
    protected = df_injected[protected_col].values

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute fairness metrics
    group_mask = protected == target_subgroup
    ref_mask = protected == reference_group

    if group_mask.sum() < config.MIN_SUBGROUP_SIZE or ref_mask.sum() < config.MIN_SUBGROUP_SIZE:
        return {
            "drift_type": drift_type,
            "severity": severity_index,
            "eod": np.nan,
            "dpd": np.nan,
            "ece": np.nan,
        }

    eod = equalized_odds_difference(y_true, y_pred, group_mask, ref_mask)
    dpd = demographic_parity_difference(y_pred_proba, group_mask, ref_mask)
    ece_group = expected_calibration_error(y_true[group_mask], y_pred_proba[group_mask])

    return {
        "drift_type": drift_type,
        "severity": severity_index,
        "severity_value": [
            config.COVARIATE_SHIFT_MAGNITUDES,
            config.LABEL_SHIFT_RATES,
            config.CONCEPT_DRIFT_MULTIPLIERS,
        ][["covariate", "label", "concept"].index(drift_type)][severity_index],
        "eod": eod,
        "dpd": dpd,
        "ece": ece_group,
    }
