#!/usr/bin/env python3
"""
SpineIDS Clinical ML - Single-Model Training Script
=====================================
Thin wrapper around run_pipeline.py for training a single model
with a streamlined argument interface.

Usage:
    python run_single_model.py LightGBM
    python run_single_model.py XGBoost --no-optuna
    python run_single_model.py RandomForest --n-trials 30
"""

import os
import sys
import argparse
from pathlib import Path

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import main as run_pipeline_main


def main():
    """Parse arguments and delegate to the main pipeline."""
    parser = argparse.ArgumentParser(
        description='Run single model training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'model',
        type=str,
        help='Model name: LogisticRegression | RandomForest | XGBoost | LightGBM | CatBoost'
    )
    
    parser.add_argument(
        '--no-optuna',
        action='store_true',
        help='Disable Optuna; use default hyperparameters'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials per outer fold'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        help='Version tag for output directories'
    )
    
    args = parser.parse_args()
    
    # Build sys.argv for run_pipeline.main()
    sys.argv = [
        'run_pipeline.py',
        '--model', args.model,
        '--version', args.version,
        '--n-trials', str(args.n_trials),
    ]
    
    if args.no_optuna:
        sys.argv.append('--no-optuna')
    
    # Delegate to the pipeline
    run_pipeline_main()


if __name__ == '__main__':
    main()
