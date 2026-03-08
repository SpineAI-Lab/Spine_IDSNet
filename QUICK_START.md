# Quick Start Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure data files are placed at the paths configured in `src_core/config.py`:

```
BASE_DIR/Dataset/Train_Spinal_Infections.csv
BASE_DIR/Dataset/Test_Spinal_Infections.csv
BASE_DIR/Dataset/706_make_folds_region4_with_folds5_joint.csv
```

---

## Step 1 – Feature Selection

Run once before any model training:

```bash
python scripts/run_feature_selection.py
```

Outputs are written to `outputs/feature_selection/` and include:

| File | Contents |
|------|----------|
| `selected_features.csv` | Final feature panel |
| `step1_statistical_tests.csv` | Kruskal-Wallis + FDR results |
| `step3_feature_importance.csv` | Stability selection importance scores |
| `correlation_matrix.csv` | Spearman correlation matrix (selected features) |
| `lasso_path.csv` | LASSO regularisation path |
| `lasso_cv.csv` | 5-fold CV error curve |
| `vif_history.csv` | VIF iteration history |
| `stability_iterations.csv` | Per-iteration selection records |
| `flow_counts.json` | Feature counts at each pipeline stage |

Advanced options:

```bash
python scripts/run_feature_selection.py \
    --fdr-alpha 0.05 \
    --corr-threshold 0.70 \
    --vif-threshold 5.0 \
    --freq-threshold 0.90 \
    --seed 42
```

---

## Step 2 – Model Training

### Single model

```bash
python scripts/run_pipeline.py --model XGBoost
```

### All models sequentially

```bash
python scripts/run_pipeline.py --model all
```

### Skip feature selection (use saved results)

```bash
python scripts/run_pipeline.py --model LightGBM --skip-feature-selection
```

### Disable Optuna (use default hyperparameters)

```bash
python scripts/run_pipeline.py --model RandomForest --no-optuna
```

### Full option reference

```
--model {all,LogisticRegression,RandomForest,XGBoost,LightGBM,CatBoost}
--skip-feature-selection   Load previously saved selected_features.csv
--feature-selection-only   Run feature selection and exit
--no-optuna                Use default hyperparameters
--n-trials INT             Optuna trials per outer fold (default: 50)
--version STR              Output directory version tag (default: v1)
--skip-external            Skip external validation
--bootstrap-ci             Compute bootstrap 95% CIs (slow)
--seed INT                 Global random seed (default: 42)
--debug                    Re-raise exceptions instead of catching them
```

---

## Step 3 – External Validation

```bash
python scripts/run_external_eval.py --model all --bootstrap-ci
```

---

## Batch Training

```bash
bash scripts/run_all_models.sh
bash scripts/run_all_models.sh --quick          # no Optuna
bash scripts/run_all_models.sh --n-trials 100   # more Optuna trials
bash scripts/run_all_models.sh --model XGBoost LightGBM
```

---

## Output Directory Layout

```
outputs/
├── feature_selection/
│   ├── selected_features.csv
│   └── ...
├── {ModelName}_v1/
│   ├── checkpoints/        Serialised fold models (*.pkl)
│   ├── predictions/        cv_all_predictions.csv, external_predictions.csv
│   ├── features/           Fusion feature arrays (*.npy) + index CSVs
│   ├── statistics/         Metrics JSONs, bootstrap CI tables
│   ├── hyperparameters/    Optuna history and best parameters per fold
│   └── logs/               config_snapshot.json, feature_names.json
└── model_comparison_summary.csv
```
