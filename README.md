# SpineIDS Clinical ML

**Part 1 of 3** · AI-Based Differential Diagnosis of Spinal Infections  

---

##Overview

This repository contains the clinical-feature machine learning component of a
three-part study on automated differential diagnosis of spinal infections.  The
pipeline distinguishes three conditions — spinal tuberculosis (**STB**),
brucella spondylitis (**BS**), and pyogenic spondylitis (**PS**) — using a
10-feature clinical panel extracted from two-centre data.

| Dataset | Cohort | Samples |
|---------|--------|---------|
| Internal | Training / 5-fold CV | 706 |
| External | Independent validation | 354 |

---

## Repository Structure

```
SpineIDS_Clinical/
├── src_core/                   Core library modules
│   ├── config.py               Global configuration and hyperparameter spaces
│   ├── preprocessing.py        Data loading and ID normalisation
│   ├── feature_selection.py    Three-stage feature selection pipeline
│   ├── models.py               Model factory (LR / RF / XGB / LGBM / CB)
│   ├── trainer.py              Nested CV training with Optuna
│   ├── inference.py            Ensemble inference and fusion-feature export
│   ├── metrics.py              Evaluation metrics (AUC, ECE, DCA, …)
│   └── statistics.py           DeLong test, Wilcoxon test, bootstrap CI
│
├── scripts/
│   ├── run_pipeline.py         Full end-to-end pipeline
│   ├── run_single_model.py     Single-model convenience wrapper
│   ├── run_feature_selection.py  Standalone feature selection
│   ├── run_external_eval.py    Post-hoc external validation
│   └── run_all_models.sh       Batch shell script
│
├── requirements.txt
├── .gitignore
└── QUICK_START.md
```

---

## Feature Selection

Three sequential stages reduce the raw clinical feature pool to a final panel:

1. **Statistical filtering** – Kruskal-Wallis H-test + Benjamini-Hochberg FDR  
   (`FDR_ALPHA = 0.05`)
2. **Collinearity reduction** – Spearman |ρ| > 0.70, then iterative VIF  
   (`VIF_THRESHOLD = 5.0`)
3. **Stability selection** – ElasticNet + LightGBM over 15 repeated CV splits;  
   features with selection frequency ≥ 0.90 are retained

All intermediate artefacts (correlation matrix, VIF history, LASSO path, etc.)
are saved for visualisation compatibility with the companion CNN and multimodal
fusion pipelines.

---

## Models

| Identifier | Algorithm |
|---|---|
| `LogisticRegression` | ElasticNet-regularised logistic regression |
| `RandomForest` | Random forest (balanced class weights) |
| `XGBoost` | Gradient boosted trees (XGBClassifier) |
| `LightGBM` | Gradient boosted trees (LGBMClassifier) |
| `CatBoost` | Gradient boosted trees (CatBoostClassifier) |

Hyperparameters are optimised per outer fold via **Optuna** (TPE sampler,
50 trials, 3-fold inner CV; metric: macro-averaged OvR AUC).

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.8 is required.  XGBoost, LightGBM, and CatBoost are optional but
strongly recommended; the factory raises an informative error if a requested
model is unavailable.

---

## Quick Start

```bash
# 1. Feature selection (run once)
python scripts/run_feature_selection.py

# 2. Train a single model
python scripts/run_pipeline.py --model XGBoost

# 3. Train all models
python scripts/run_pipeline.py --model all

# 4. External validation
python scripts/run_external_eval.py --model all --bootstrap-ci

# 5. Batch training (shell)
bash scripts/run_all_models.sh
```

See `QUICK_START.md` for extended examples and configuration options.

---

## Output Schema

Prediction CSV files follow a fixed schema shared with the CNN and multimodal
fusion pipelines:

| Column |描述|
|--------|-------------|
| `RealPID` | Normalised patient ID (`{STB\|BS\|PS}_{0000}`) |
| `GT` | Ground-truth class index (0 = STB, 1 = BS, 2 = PS) |
| `Prob_STB` | Predicted probability for STB |
| `Prob_BS` | Predicted probability for BS |
| `Prob_PS` | Predicted probability for PS |
| `Pred` | Predicted class index |
| `GT_Name` / `Pred_Name` | String label equivalents |
| `Fold` | CV fold index (training split only) |

---

## Reproducibility

All experiments use `RANDOM_SEED = 42`.  A `config_snapshot.json` is written
to each model output directory recording the exact hyperparameter configuration
and timestamp.

---

## Citation

> [Citation will be added upon publication]

---

## License

This code is released for academic reproducibility purposes.  
All patient data are de-identified and held at the originating institutions.
