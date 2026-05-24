"""
Fairness metrics computation: EOD, DPD, per-subgroup ECE.
Single canonical implementation used throughout all experiments.
"""
import numpy as np
import pandas as pd
from typing import Optional

from . import config


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TPR = TP / (TP + FN). Returns 0 if no positives."""
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    return y_pred[positives].mean()


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN). Returns 0 if no negatives."""
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    return y_pred[negatives].mean()


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_a_mask: np.ndarray,
    group_b_mask: np.ndarray,
) -> float:
    """
    Equalized Odds Difference (EOD).
    EOD = max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Binary predictions (0/1)
        group_a_mask: Boolean mask for group A
        group_b_mask: Boolean mask for group B (reference)

    Returns:
        EOD value (0 = perfectly fair)
    """
    # Group A metrics
    tpr_a = true_positive_rate(y_true[group_a_mask], y_pred[group_a_mask])
    fpr_a = false_positive_rate(y_true[group_a_mask], y_pred[group_a_mask])

    # Group B metrics (reference)
    tpr_b = true_positive_rate(y_true[group_b_mask], y_pred[group_b_mask])
    fpr_b = false_positive_rate(y_true[group_b_mask], y_pred[group_b_mask])

    eod = max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))
    return eod


def demographic_parity_difference(
    y_pred_proba: np.ndarray,
    group_a_mask: np.ndarray,
    group_b_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Demographic Parity Difference (DPD).
    DPD = |P(Y_hat=1|A=a) - P(Y_hat=1|A=b)|

    Args:
        y_pred_proba: Predicted probabilities
        group_a_mask: Boolean mask for group A
        group_b_mask: Boolean mask for group B (reference)
        threshold: Classification threshold (default 0.5)

    Returns:
        DPD value (0 = perfectly fair)
    """
    pred_a = (y_pred_proba[group_a_mask] >= threshold).mean()
    pred_b = (y_pred_proba[group_b_mask] >= threshold).mean()

    return abs(pred_a - pred_b)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins.
    ECE = sum_k (n_k / N) * |acc_k - conf_k|

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of equal-width confidence bins

    Returns:
        ECE value (0 = perfectly calibrated)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        # Include the right edge for the last bin
        if i == n_bins - 1:
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = y_pred_proba[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_all_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_attr: np.ndarray,
    reference_group: str,
    threshold: float = 0.5,
    min_size: int = None,
) -> pd.DataFrame:
    """
    Compute all three fairness metrics for each subgroup vs reference.

    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        y_pred_proba: Predicted probabilities
        protected_attr: Array of group labels (e.g., race values)
        reference_group: The reference group to compare against
        threshold: Classification threshold
        min_size: Minimum subgroup size (default from config)

    Returns:
        DataFrame with columns: [group, n, eod, dpd, ece, ece_reference, violation_eod]
    """
    if min_size is None:
        min_size = config.MIN_SUBGROUP_SIZE

    ref_mask = protected_attr == reference_group
    groups = np.unique(protected_attr)

    # Reference group ECE
    if ref_mask.sum() >= min_size:
        ece_ref = expected_calibration_error(y_true[ref_mask], y_pred_proba[ref_mask])
    else:
        ece_ref = np.nan

    results = []
    for group in groups:
        if group == reference_group:
            continue

        group_mask = protected_attr == group
        n_group = group_mask.sum()

        if n_group < min_size:
            results.append({
                "group": group,
                "n": n_group,
                "eod": np.nan,
                "dpd": np.nan,
                "ece": np.nan,
                "ece_reference": ece_ref,
                "violation_eod": False,
                "insufficient_size": True,
            })
            continue

        eod = equalized_odds_difference(y_true, y_pred, group_mask, ref_mask)
        dpd = demographic_parity_difference(y_pred_proba, group_mask, ref_mask, threshold)
        ece = expected_calibration_error(y_true[group_mask], y_pred_proba[group_mask])

        results.append({
            "group": group,
            "n": n_group,
            "eod": eod,
            "dpd": dpd,
            "ece": ece,
            "ece_reference": ece_ref,
            "violation_eod": eod >= config.EOD_VIOLATION_THRESHOLD,
            "insufficient_size": False,
        })

    return pd.DataFrame(results)


def compute_fairness_over_windows(
    windows: list,
    model,
    feature_cols: list,
    task: str,
    protected_col: str,
    reference_group: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute fairness metrics for each temporal window (W2–W5).

    Args:
        windows: List of DataFrames [W1, W2, W3, W4, W5]
        model: Trained model with predict() and predict_proba()
        feature_cols: Feature column names
        task: Target column name
        protected_col: Protected attribute column name
        reference_group: Reference group value

    Returns:
        DataFrame with window, group, eod, dpd, ece columns
    """
    all_results = []

    # Evaluate on W2–W5 (skip W1 = training data)
    for w_idx in range(1, len(windows)):
        window = windows[w_idx]
        X = window[feature_cols].values
        y_true = window[task].values
        protected = window[protected_col].values

        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = compute_all_fairness_metrics(
            y_true, y_pred, y_pred_proba,
            protected, reference_group, threshold
        )
        metrics["window"] = w_idx + 1
        metrics["task"] = task
        metrics["protected_attr"] = protected_col

        all_results.append(metrics)

    return pd.concat(all_results, ignore_index=True)
