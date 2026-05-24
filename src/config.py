"""
Global configuration for FairDrift experiments.
All constants, thresholds, and paths in one place.
"""
import os
from pathlib import Path

# === Paths ===
# Project root is the repo root (parent of src/)
_PROJECT_ROOT = Path(__file__).parent.parent

# Auto-detect environment: Colab vs local
if os.path.exists("/content"):
    # Google Colab
    DATA_PATH = "/content/drive/MyDrive/FairDrift/diabetic_data.csv"
    OUTPUT_DIR = "/content/FairDrift-outputs"
    RUNNING_ON_COLAB = True
else:
    # Local
    DATA_PATH = str(_PROJECT_ROOT / "data" / "diabetic_data.csv")
    OUTPUT_DIR = str(_PROJECT_ROOT / "outputs")
    RUNNING_ON_COLAB = False

# === Random Seeds (fixed for reproducibility) ===
RANDOM_SEEDS = [42, 123, 456, 789, 1024]
PRIMARY_SEED = 42

# === Dataset ===
N_TOTAL_ENCOUNTERS = 101_766
N_WINDOWS = 5
ENCOUNTERS_PER_WINDOW = N_TOTAL_ENCOUNTERS // N_WINDOWS  # ~20,353

# === Protected Attributes ===
RACE_GROUPS = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
GENDER_GROUPS = ["Female", "Male"]
AGE_GROUPS = ["[0-30)", "[30-50)", "[50-70)", "[70-90)", "[90-100)"]
REFERENCE_RACE = "Caucasian"
REFERENCE_GENDER = "Female"
REFERENCE_AGE = "[50-70)"

# === Prediction Tasks ===
TASKS = {
    "readmit_30": {
        "description": "30-day hospital readmission",
        "source_col": "readmitted",
        "positive_value": "<30",
        "expected_base_rate": 0.11,
    },
    "extended_stay": {
        "description": "Extended hospital stay (>5 days)",
        "source_col": "time_in_hospital",
        "threshold": 5,
        "expected_base_rate": 0.45,
    },
    "med_change": {
        "description": "Medication change during encounter",
        "source_col": "change",
        "positive_value": "Ch",
        "expected_base_rate": 0.46,
    },
}

# === Fairness Thresholds ===
EOD_VIOLATION_THRESHOLD = 0.05
MIN_SUBGROUP_SIZE = 200

# === Statistical Testing ===
ALPHA = 0.05
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 1_000
# Bonferroni for H1: 9 tests (3 metrics x 3 protected attributes)
BONFERRONI_ALPHA_H1 = ALPHA / 9  # 0.0056
# Bonferroni for drift tests: updated after preprocessing finalizes feature count
# Approximate: 2520 tests (12 subgroups x ~70 features x 3 algorithms)
BONFERRONI_N_DRIFT_TESTS = 2520
BONFERRONI_ALPHA_DRIFT = ALPHA / BONFERRONI_N_DRIFT_TESTS
# Benjamini-Hochberg FDR level (robustness check)
FDR_LEVEL = 0.05

# === CUSUM Parameters ===
CUSUM_ARL0_TARGET = 200
CUSUM_MC_PATHS = 10_000
CUSUM_MONITORING_FREQUENCY = 500  # One computation per 500 encounters

# === Drift Detection ===
ADWIN_DELTA = 0.002
PAGE_HINKLEY_DELTA = 0.005
PAGE_HINKLEY_THRESHOLD = 50
DRIFT_ENSEMBLE_VOTING = "majority"  # Drift if >= 2 of 3 detectors agree

# === Drift Injection (RQ3) ===
COVARIATE_SHIFT_MAGNITUDES = [0.5, 1.0, 1.5]  # Standard deviations
LABEL_SHIFT_RATES = [0.15, 0.20, 0.25]  # Target rates for minority
CONCEPT_DRIFT_MULTIPLIERS = [1.5, 2.0, 3.0]
N_REPLICATIONS = 30  # 5 seeds x 6 reps each

# === Model Hyperparameters ===
MODEL_TYPES = ["logistic_regression", "xgboost", "mlp"]
HYPERPARAM_SEARCH = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": 1000,
    },
    "xgboost": {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
    },
    "mlp": {
        "hidden_layer_sizes": (64, 32),
        "dropout": 0.3,
        "learning_rate_init": 0.001,
        "early_stopping": True,
        "n_iter_no_change": 10,
        "max_iter": 500,
    },
}

# === Baseline Strategies ===
AGGREGATE_AUROC_THRESHOLD = 0.03  # Alert when AUROC drops by this much
PERIODIC_CHECK_WINDOWS = [3, 5]  # Evaluate at W3 and W5 only

# === Alert Severity ===
SEVERITY_THRESHOLDS = {
    "minor": (0.05, 0.08),
    "moderate": (0.08, 0.12),
    "critical": (0.12, float("inf")),
}

# === H2 Decision Thresholds ===
H2_DELAY_REDUCTION_MIN = 0.30  # 30% minimum reduction
H2_FAR_MAX = 0.05  # 5% maximum false alarm rate

# === Features to drop ===
DROP_COLUMNS = [
    "encounter_id",  # Identifier only
    "patient_nbr",   # Identifier only
    "weight",        # 96.9% missing, differential by race
    "payer_code",    # High missingness
]

# Protected attributes: used for stratification, NOT as model features
PROTECTED_COLUMNS = ["race", "gender", "age"]

# === Reporting ===
FIGURE_DPI = 300
FIGURE_FORMAT = "png"  # Also save as PDF for thesis
