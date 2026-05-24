# FairDrift

**A Framework for Continuous Algorithmic Fairness Monitoring in Deployed Clinical Prediction Models under Dataset Shift**

This repository contains the source code, experiments, and dataset for the Master's thesis by Kemal Colovic at International Burch University, Faculty of Engineering and Information Technologies (2025).

## Overview

FairDrift is a three-component monitoring framework that detects when dataset shift causes fairness degradation in deployed clinical ML models:

1. **Drift Detection Ensemble** -- Combines Kolmogorov-Smirnov tests, ADWIN, and Page-Hinkley detectors via majority voting to identify distribution shifts across demographic subgroups.
2. **CUSUM Fairness Control Charts** -- Sequential monitoring of Equalized Odds Difference (EOD) using bootstrap-calibrated CUSUM charts (ARL0 = 200).
3. **Alert Generation** -- Severity-graded alerts (minor/moderate/critical) triggered when sustained fairness violations are detected post-drift.

## Dataset

The experiments use the [Diabetes 130-US Hospitals (1999--2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset from the UCI Machine Learning Repository.

- **101,766 encounters** across 130 US hospitals over 10 years
- Protected attributes: race, gender, age
- Three prediction tasks: 30-day readmission, extended stay (>5 days), medication change
- The dataset is included in `data/` for reproducibility

**Citation:** Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*.

## Repository Structure

```
FairDrift/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                          # Dataset
│   ├── diabetic_data.csv          # Main dataset (101,766 encounters)
│   └── IDS_mapping.csv            # Feature code mappings
├── src/                           # Source modules
│   ├── __init__.py
│   ├── config.py                  # Global configuration and paths
│   ├── cusum.py                   # CUSUM control charts with MC calibration
│   ├── drift_detectors.py         # KS, ADWIN, Page-Hinkley detectors
│   ├── drift_injector.py          # Synthetic drift injection (RQ3)
│   ├── fairness_metrics.py        # EOD, DPD, ECE computation
│   ├── preprocessing.py           # Data cleaning and feature engineering
│   └── temporal_windows.py        # Temporal window partitioning
├── notebooks/                     # Experimental notebooks (run in order)
│   ├── 01_EDA_and_Windows.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Stage1_Fairness_Under_Drift.ipynb
│   ├── 04_Stage1_Drift_Injection_RQ3.ipynb
│   ├── 05_Stage2_FairDrift_Framework.ipynb
│   └── 06_Stage3_Comparative_Evaluation.ipynb
└── outputs/                       # Pre-computed results
    ├── figures/                   # Publication-ready figures (300 DPI)
    └── metrics/                   # CSV result tables
```

## Installation

```bash
git clone https://github.com/kemalcolovic/FairDrift.git
cd FairDrift
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- pandas, numpy, scipy, scikit-learn, xgboost
- shap, lifelines, statsmodels
- matplotlib, seaborn
- See [requirements.txt](requirements.txt) for full list

## Usage

### Running the Experiments

The notebooks are designed to be run sequentially (01 through 06). Each notebook imports shared modules from `src/`.

```bash
jupyter notebook notebooks/01_EDA_and_Windows.ipynb
```

Or run all notebooks in order:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_EDA_and_Windows.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_Model_Training.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_Stage1_Fairness_Under_Drift.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_Stage1_Drift_Injection_RQ3.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_Stage2_FairDrift_Framework.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_Stage3_Comparative_Evaluation.ipynb
```

### Using FairDrift Modules

```python
from src.config import *
from src.preprocessing import load_and_preprocess
from src.drift_detectors import KSDriftDetector, DriftEnsemble
from src.cusum import CUSUMChart, calibrate_h_bootstrap
from src.fairness_metrics import equalized_odds_difference
```

## Research Questions

| RQ | Question | Finding |
|:---|:---------|:--------|
| RQ1 | Does naturally occurring dataset shift cause statistically significant fairness degradation? | No significant degradation detected (all p > 0.90 after Bonferroni correction). H1 not supported. |
| RQ2 | Can FairDrift detect fairness violations earlier than baseline approaches? | FairDrift detected violations but with higher false alarm rate (6%) and longer delay (121 vs 33 points) than the periodic baseline. H2 not supported. |
| RQ3 | How do different types of injected drift affect fairness metrics? | Concept drift caused the largest fairness impact (EOD = 0.142), followed by label shift (0.098) and covariate shift (0.067). Interaction effects between drift type and magnitude were significant (p < 0.001). |

## Key Results

- **Bootstrap calibration** reduced CUSUM false alarm rate from 88% to 6%
- **CUSUM decision boundary:** h = 0.332 (calibrated for ARL0 = 200)
- **Natural drift is benign:** The Diabetes 130 dataset exhibits measurable distribution shift but insufficient to cause fairness violations exceeding the 0.05 EOD threshold
- **Concept drift is the most harmful type** for fairness, with effect sizes 2x larger than covariate shift

## Figures

| Figure | Description |
|:-------|:------------|
| `fig01` | Demographic composition across temporal windows |
| `fig02` | KS statistic heatmap showing drift patterns |
| `fig03` | AUROC degradation trajectories by model type |
| `fig04` | EOD trajectory across windows |
| `fig05` | Drift type x magnitude fairness impact heatmap |
| `fig06` | CUSUM control chart with bootstrap-calibrated threshold |
| `fig07` | Detection delay comparison: FairDrift vs baselines |

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@mastersthesis{colovic2025fairdrift,
  title     = {FairDrift: A Framework for Continuous Algorithmic Fairness
               Monitoring in Deployed Clinical Prediction Models
               under Dataset Shift},
  author    = {Colovic, Kemal},
  school    = {International Burch University},
  year      = {2025},
  address   = {Sarajevo, Bosnia and Herzegovina}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
