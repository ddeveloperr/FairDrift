"""
Temporal window construction and drift validation.
Splits data into 5 chronological windows using encounter_id ordering.
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from . import config


def construct_windows(df: pd.DataFrame, n_windows: int = None) -> list:
    """
    Sort by encounter_id (proxy for time) and split into n equal windows.

    Returns:
        List of DataFrames [W1, W2, W3, W4, W5]
    """
    if n_windows is None:
        n_windows = config.N_WINDOWS

    df_sorted = df.sort_values("encounter_id").reset_index(drop=True)
    n_per_window = len(df_sorted) // n_windows

    windows = []
    for w in range(n_windows):
        start = w * n_per_window
        end = start + n_per_window if w < n_windows - 1 else len(df_sorted)
        window_df = df_sorted.iloc[start:end].copy()
        window_df["window_id"] = w + 1
        windows.append(window_df)

    return windows


def validate_temporal_ordering(windows: list, features: list = None) -> pd.DataFrame:
    """
    Compute KS statistics between W1 and each subsequent window.
    Monotonic increase confirms drift accumulates over time.

    Returns:
        DataFrame with columns: feature, ks_w1w2, ks_w1w3, ks_w1w4, ks_w1w5
    """
    if features is None:
        features = [
            "num_medications", "num_lab_procedures", "number_diagnoses",
            "time_in_hospital", "number_inpatient", "number_emergency",
            "num_procedures",
        ]

    # Filter to features that exist in the data
    features = [f for f in features if f in windows[0].columns]

    results = []
    for feat in features:
        row = {"feature": feat}
        ref_data = windows[0][feat].dropna().values

        for w_idx in range(1, len(windows)):
            test_data = windows[w_idx][feat].dropna().values
            ks_stat, p_val = ks_2samp(ref_data, test_data)
            row[f"ks_w1_w{w_idx+1}"] = ks_stat
            row[f"p_w1_w{w_idx+1}"] = p_val

        results.append(row)

    return pd.DataFrame(results)


def compute_subgroup_sizes(windows: list) -> pd.DataFrame:
    """
    Compute sample size per subgroup per window.
    Flags subgroups below MIN_SUBGROUP_SIZE.
    """
    records = []

    for w_idx, window in enumerate(windows):
        # Race
        for race in config.RACE_GROUPS + ["?"]:
            n = (window["race"] == race).sum()
            records.append({
                "window": w_idx + 1,
                "attribute": "race",
                "group": race,
                "n": n,
                "sufficient": n >= config.MIN_SUBGROUP_SIZE,
            })

        # Gender
        for gender in config.GENDER_GROUPS:
            n = (window["gender"] == gender).sum()
            records.append({
                "window": w_idx + 1,
                "attribute": "gender",
                "group": gender,
                "n": n,
                "sufficient": n >= config.MIN_SUBGROUP_SIZE,
            })

        # Age group
        if "age_group" in window.columns:
            for age in config.AGE_GROUPS:
                n = (window["age_group"] == age).sum()
                records.append({
                    "window": w_idx + 1,
                    "attribute": "age_group",
                    "group": age,
                    "n": n,
                    "sufficient": n >= config.MIN_SUBGROUP_SIZE,
                })

    return pd.DataFrame(records)


def demographic_shift_report(windows: list) -> pd.DataFrame:
    """
    Report demographic composition changes across windows.
    Key validation: AA proportion should shift from ~21.5% (W1) to ~12.8% (W5).
    """
    records = []

    for w_idx, window in enumerate(windows):
        n_total = len(window)

        for race in config.RACE_GROUPS + ["?"]:
            n = (window["race"] == race).sum()
            records.append({
                "window": w_idx + 1,
                "race": race,
                "n": n,
                "pct": n / n_total * 100,
            })

    return pd.DataFrame(records)


def subgroup_drift_comparison(windows: list, features: list = None) -> pd.DataFrame:
    """
    Compare drift magnitude between subgroups (W1 vs W5).
    Key finding: AA subgroup experiences more drift than Caucasian for some features.

    Returns:
        DataFrame with KS stats per feature per subgroup.
    """
    if features is None:
        features = [
            "num_medications", "num_lab_procedures", "number_diagnoses",
            "time_in_hospital", "number_inpatient", "number_emergency",
            "num_procedures",
        ]

    features = [f for f in features if f in windows[0].columns]

    results = []
    w1 = windows[0]
    w5 = windows[-1]

    for feat in features:
        row = {"feature": feat}

        # Caucasian: W1 vs W5
        cau_w1 = w1[w1["race"] == "Caucasian"][feat].dropna().values
        cau_w5 = w5[w5["race"] == "Caucasian"][feat].dropna().values
        ks_cau, _ = ks_2samp(cau_w1, cau_w5)

        # African American: W1 vs W5
        aa_w1 = w1[w1["race"] == "AfricanAmerican"][feat].dropna().values
        aa_w5 = w5[w5["race"] == "AfricanAmerican"][feat].dropna().values
        ks_aa, _ = ks_2samp(aa_w1, aa_w5)

        row["ks_caucasian"] = ks_cau
        row["ks_african_american"] = ks_aa
        row["ratio_aa_cau"] = ks_aa / ks_cau if ks_cau > 0 else np.inf

        results.append(row)

    return pd.DataFrame(results)
