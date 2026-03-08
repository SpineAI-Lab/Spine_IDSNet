"""
SpineIDS Clinical ML - Global Configuration
============================================
Part 1 of a three-part dissertation on AI-based differential diagnosis
of spinal infections (STB / BS / PS) using clinical features.

Configuration sections:
    - Path configuration
    - Data schema
    - Feature selection hyperparameters
    - Training hyperparameters
    - Optuna search spaces and default parameters
    - Statistical analysis settings
    - Output format specification
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

BASE_DIR    = Path("/root/autodl-tmp/05_MachineLearning")
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR  = BASE_DIR / "outputs"

TRAIN_DATA_PATH = DATASET_DIR / "Train_Spinal_Infections.csv"
TEST_DATA_PATH  = DATASET_DIR / "Test_Spinal_Infections.csv"
CNN_FOLDS_PATH  = DATASET_DIR / "706_make_folds_region4_with_folds5_joint.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data Schema
# =============================================================================

LABEL_COL      = "Label"
GLOBAL_ID_COL  = "Global_ID"
PATIENT_ID_COL = "Patient_ID"

CLASS_NAMES  = ['STB', 'BS', 'PS']
NUM_CLASSES  = 3
CLASS_TO_IDX = {'STB': 0, 'BS': 1, 'PS': 2}
IDX_TO_CLASS = {0: 'STB', 1: 'BS', 2: 'PS'}

EXCLUDE_COLS = [
    'Patient_ID',
    'Global_ID',
    'Label',
    'Fold',
]

EXCLUDE_FEATURES = [
    'Occupation_0', 'Occupation_1', 'Occupation_2',  'Occupation_3',
    'Occupation_4', 'Occupation_5', 'Occupation_6',  'Occupation_7',
    'Occupation_8', 'Occupation_9', 'Occupation_10', 'Occupation_11',
]

# =============================================================================
# Feature Selection Configuration
# =============================================================================

FDR_ALPHA             = 0.05   # Benjamini-Hochberg FDR threshold (Step 1)
CORRELATION_THRESHOLD = 0.70   # Spearman |rho| ceiling (Step 2)
VIF_THRESHOLD         = 5.0    # Variance inflation factor ceiling (Step 2)

STABILITY_N_SPLITS            = 5     # Inner CV folds per repeat (Step 3)
STABILITY_N_REPEATS           = 3     # Number of repeats (Step 3)
SELECTION_FREQUENCY_THRESHOLD = 0.90  # Minimum selection frequency to retain a feature
IMPORTANCE_BOOTSTRAP_N        = 1000  # Bootstrap iterations for importance ranking

# =============================================================================
# Training Configuration
# =============================================================================

RANDOM_SEED    = 42
N_FOLDS        = 5    # Outer CV folds (aligned with CNN pipeline)
INNER_CV_FOLDS = 3    # Inner folds for hyperparameter optimisation

OPTUNA_N_TRIALS = 50   # Trials per outer fold
OPTUNA_TIMEOUT  = 600  # Wall-clock timeout per fold (seconds)
OPTUNA_METRIC   = 'AUC_Macro'

# =============================================================================
# Model Registry
# =============================================================================

MODELS_TO_TRAIN = [
    'LogisticRegression',
    'RandomForest',
    'XGBoost',
    'LightGBM',
    'CatBoost',
]

# Models that require feature standardisation prior to training
MODELS_NEED_SCALING = ['LogisticRegression']

# =============================================================================
# Optuna Hyperparameter Search Spaces
# =============================================================================
# Tuple schema: (type, *args)
#   'fixed'       -> (value,)
#   'int'         -> (low, high)
#   'float'       -> (low, high)
#   'float_log'   -> (low, high)      log-uniform sampling
#   'categorical' -> ([choices],)

HYPERPARAM_SPACE = {
    'LogisticRegression': {
        'C':            ('float_log', 0.001, 100),
        'l1_ratio':     ('float', 0.1, 0.9),
        'max_iter':     ('fixed', 5000),
        'solver':       ('fixed', 'saga'),
        'penalty':      ('fixed', 'elasticnet'),
        'class_weight': ('fixed', 'balanced'),
        'random_state': ('fixed', RANDOM_SEED),
    },

    'RandomForest': {
        'n_estimators':      ('int', 100, 500),
        'max_depth':         ('int', 5, 30),
        'min_samples_split': ('int', 2, 20),
        'min_samples_leaf':  ('int', 1, 10),
        'max_features':      ('categorical', ['sqrt', 'log2', None]),
        'class_weight':      ('fixed', 'balanced'),
        'random_state':      ('fixed', RANDOM_SEED),
        'n_jobs':            ('fixed', -1),
    },

    'XGBoost': {
        'n_estimators':     ('int', 100, 500),
        'max_depth':        ('int', 3, 10),
        'learning_rate':    ('float_log', 0.01, 0.3),
        'subsample':        ('float', 0.6, 1.0),
        'colsample_bytree': ('float', 0.6, 1.0),
        'min_child_weight': ('int', 1, 10),
        'gamma':            ('float', 0, 1),
        'reg_alpha':        ('float_log', 1e-8, 10),
        'reg_lambda':       ('float_log', 1e-8, 10),
        'random_state':     ('fixed', RANDOM_SEED),
        'use_label_encoder':('fixed', False),
        'eval_metric':      ('fixed', 'mlogloss'),
        'n_jobs':           ('fixed', -1),
    },

    'LightGBM': {
        'n_estimators':     ('int', 100, 500),
        'num_leaves':       ('int', 15, 127),
        'max_depth':        ('int', 3, 15),
        'learning_rate':    ('float_log', 0.01, 0.3),
        'subsample':        ('float', 0.6, 1.0),
        'colsample_bytree': ('float', 0.6, 1.0),
        'min_child_samples':('int', 5, 100),
        'reg_alpha':        ('float_log', 1e-8, 10),
        'reg_lambda':       ('float_log', 1e-8, 10),
        'class_weight':     ('fixed', 'balanced'),
        'random_state':     ('fixed', RANDOM_SEED),
        'verbose':          ('fixed', -1),
        'n_jobs':           ('fixed', -1),
    },

    'CatBoost': {
        'iterations':          ('int', 100, 500),
        'depth':               ('int', 4, 10),
        'learning_rate':       ('float_log', 0.01, 0.3),
        'l2_leaf_reg':         ('float_log', 1e-8, 10),
        'bagging_temperature': ('float', 0, 1),
        'random_strength':     ('float', 0, 1),
        'auto_class_weights':  ('fixed', 'Balanced'),
        'random_seed':         ('fixed', RANDOM_SEED),
        'verbose':             ('fixed', False),
    },
}

# =============================================================================
# Default Parameters (used when Optuna is disabled)
# =============================================================================

DEFAULT_PARAMS = {
    'LogisticRegression': {
        'C':            1.0,
        'l1_ratio':     0.5,
        'max_iter':     5000,
        'solver':       'saga',
        'penalty':      'elasticnet',
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
    },

    'RandomForest': {
        'n_estimators':      300,
        'max_depth':         15,
        'min_samples_split': 5,
        'min_samples_leaf':  2,
        'class_weight':      'balanced',
        'random_state':      RANDOM_SEED,
        'n_jobs':            -1,
    },

    'XGBoost': {
        'n_estimators':     300,
        'max_depth':        6,
        'learning_rate':    0.05,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'random_state':     RANDOM_SEED,
        'use_label_encoder':False,
        'eval_metric':      'mlogloss',
        'n_jobs':           -1,
    },

    'LightGBM': {
        'n_estimators':  300,
        'num_leaves':    31,
        'max_depth':     10,
        'learning_rate': 0.05,
        'class_weight':  'balanced',
        'random_state':  RANDOM_SEED,
        'verbose':       -1,
        'n_jobs':        -1,
    },

    'CatBoost': {
        'iterations':        300,
        'depth':             6,
        'learning_rate':     0.05,
        'auto_class_weights':'Balanced',
        'random_seed':       RANDOM_SEED,
        'verbose':           False,
    },
}

# =============================================================================
# Statistical Analysis Configuration
# =============================================================================

BOOTSTRAP_N_ITERATIONS = 1000
BOOTSTRAP_CI_LEVEL     = 0.95
CALIBRATION_N_BINS     = 10
DCA_THRESHOLD_MIN      = 0.01
DCA_THRESHOLD_MAX      = 0.99
DCA_THRESHOLD_STEP     = 0.01

# =============================================================================
# Output Format Specification
# =============================================================================

# Column order for prediction CSV files (must match CNN pipeline schema)
PREDICTION_COLS = [
    'RealPID', 'GT', 'Prob_STB', 'Prob_BS', 'Prob_PS',
    'Pred', 'GT_Name', 'Pred_Name', 'Fold',
]
EXTERNAL_PREDICTION_COLS = [
    'RealPID', 'GT', 'Prob_STB', 'Prob_BS', 'Prob_PS',
    'Pred', 'GT_Name', 'Pred_Name',
]

VERSION = "v1"


# =============================================================================
# Directory Helpers
# =============================================================================

def get_model_output_dir(model_name: str, version: str = VERSION) -> Path:
    """Return (and create) the root output directory for a given model run."""
    output_dir = OUTPUT_DIR / f"{model_name}_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_all_subdirs(model_name: str, version: str = VERSION) -> dict:
    """Create and return the standard subdirectory tree for a model run."""
    base_dir = get_model_output_dir(model_name, version)

    subdirs = {
        'checkpoints':     base_dir / 'checkpoints',
        'predictions':     base_dir / 'predictions',
        'features':        base_dir / 'features',
        'statistics':      base_dir / 'statistics',
        'hyperparameters': base_dir / 'hyperparameters',
        'logs':            base_dir / 'logs',
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    return subdirs
