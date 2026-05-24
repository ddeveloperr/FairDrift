"""
Data loading and preprocessing pipeline for FairDrift.
Handles: loading, cleaning, encoding, target creation, feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pathlib import Path

from . import config


def load_data(path: str = None) -> pd.DataFrame:
    """Load raw diabetes dataset and perform initial validation."""
    if path is None:
        path = config.DATA_PATH

    df = pd.read_csv(path)

    # Validation
    assert df.shape[0] == config.N_TOTAL_ENCOUNTERS, (
        f"Expected {config.N_TOTAL_ENCOUNTERS} rows, got {df.shape[0]}"
    )
    assert "encounter_id" in df.columns
    assert "race" in df.columns
    assert "readmitted" in df.columns

    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create three binary prediction targets."""
    df = df.copy()

    # Task 1: 30-day readmission
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)

    # Task 2: Extended hospital stay (>5 days)
    df["extended_stay"] = (df["time_in_hospital"] > 5).astype(int)

    # Task 3: Medication change
    df["med_change"] = (df["change"] == "Ch").astype(int)

    return df


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values encoded as '?' in the dataset."""
    df = df.copy()

    # Replace '?' with NaN for proper handling
    df = df.replace("?", np.nan)

    # Drop weight column (96.9% missing, differential by race)
    if "weight" in df.columns:
        df = df.drop(columns=["weight"])

    # Drop payer_code (high missingness, not clinically relevant for prediction)
    if "payer_code" in df.columns:
        df = df.drop(columns=["payer_code"])

    return df


def encode_age_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Encode age brackets as ordinal integers."""
    df = df.copy()
    age_map = {
        "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
        "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
        "[80-90)": 8, "[90-100)": 9,
    }
    # Keep original age for subgroup analysis
    df["age_bracket"] = df["age"]
    df["age_ordinal"] = df["age"].map(age_map)
    return df


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create broader age groups for fairness analysis (matching proposal)."""
    df = df.copy()
    age_group_map = {
        "[0-10)": "[0-30)", "[10-20)": "[0-30)", "[20-30)": "[0-30)",
        "[30-40)": "[30-50)", "[40-50)": "[30-50)",
        "[50-60)": "[50-70)", "[60-70)": "[50-70)",
        "[70-80)": "[70-90)", "[80-90)": "[70-90)",
        "[90-100)": "[90-100)",
    }
    df["age_group"] = df["age"].map(age_group_map)
    return df


def group_icd9_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Group ICD-9 diagnosis codes into broad categories."""
    df = df.copy()

    def icd9_to_category(code):
        """Map ICD-9 code to broad category."""
        if pd.isna(code):
            return "Unknown"
        code = str(code)
        # Handle E and V codes
        if code.startswith("E"):
            return "External"
        if code.startswith("V"):
            return "Supplementary"
        # Numeric codes
        try:
            num = float(code)
            if num < 140:
                return "Infectious"
            elif num < 240:
                return "Neoplasms"
            elif num < 280:
                return "Endocrine"
            elif num < 290:
                return "Blood"
            elif num < 320:
                return "Mental"
            elif num < 390:
                return "Nervous"
            elif num < 460:
                return "Circulatory"
            elif num < 520:
                return "Respiratory"
            elif num < 580:
                return "Digestive"
            elif num < 630:
                return "Genitourinary"
            elif num < 680:
                return "Pregnancy"
            elif num < 710:
                return "Skin"
            elif num < 740:
                return "Musculoskeletal"
            elif num < 760:
                return "Congenital"
            elif num < 780:
                return "Perinatal"
            elif num < 800:
                return "Symptoms"
            else:
                return "Injury"
        except (ValueError, TypeError):
            return "Unknown"

    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[f"{diag_col}_cat"] = df[diag_col].apply(icd9_to_category)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features for modeling."""
    df = df.copy()

    # Columns to one-hot encode
    onehot_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

    # Diagnosis categories (from ICD-9 grouping)
    diag_cat_cols = [c for c in df.columns if c.endswith("_cat")]
    onehot_cols.extend(diag_cat_cols)

    # Medical specialty (group rare specialties)
    if "medical_specialty" in df.columns:
        # Keep top 10 specialties, group rest as "Other"
        top_specs = df["medical_specialty"].value_counts().head(10).index.tolist()
        df["medical_specialty_grouped"] = df["medical_specialty"].apply(
            lambda x: x if x in top_specs else "Other_Specialty"
        )
        onehot_cols.append("medical_specialty_grouped")

    # Medication columns are already categorical (Up/Down/Steady/No)
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "glipizide", "glyburide", "pioglitazone",
        "rosiglitazone", "insulin",
    ]
    # Only include medications that have variation
    for col in med_cols:
        if col in df.columns:
            if df[col].nunique() > 1:
                onehot_cols.append(col)

    # Convert numeric IDs to string for proper one-hot encoding
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encode
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False, dtype=int)

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute remaining missing values: median for numeric, mode for categorical."""
    df = df.copy()

    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns (excluding targets, IDs, protected attrs)."""
    exclude = set(
        config.DROP_COLUMNS
        + config.PROTECTED_COLUMNS
        + ["readmit_30", "extended_stay", "med_change"]
        + ["readmitted", "change", "time_in_hospital"]
        + ["age_bracket", "age_group", "age"]
        + ["diag_1", "diag_2", "diag_3"]
        + ["medical_specialty"]
        + ["encounter_id", "patient_nbr"]
    )

    feature_cols = [c for c in df.columns if c not in exclude]
    # Only keep numeric columns for modeling
    feature_cols = [c for c in feature_cols if df[c].dtype in ["int64", "float64", "int32", "float32", "uint8"]]

    return sorted(feature_cols)


def preprocess_full(path: str = None) -> tuple:
    """
    Full preprocessing pipeline.

    Returns:
        df: Full preprocessed DataFrame (with protected attrs for fairness analysis)
        feature_cols: List of feature column names for modeling
    """
    # Load
    df = load_data(path)

    # Create targets
    df = create_targets(df)

    # Clean missing
    df = clean_missing(df)

    # Encode age
    df = encode_age_ordinal(df)
    df = create_age_groups(df)

    # Group diagnoses
    df = group_icd9_codes(df)

    # Encode categoricals
    df = encode_categoricals(df)

    # Impute remaining
    df = impute_missing(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Report
    print(f"Preprocessing complete:")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Target base rates:")
    for task in config.TASKS:
        rate = df[task].mean()
        print(f"    {task}: {rate:.3f} ({rate*100:.1f}%)")

    return df, feature_cols
