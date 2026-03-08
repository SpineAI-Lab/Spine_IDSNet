#!/usr/bin/env python3
"""
SpineIDS Clinical ML - Standalone Feature Selection Script
=======================================
Execute feature selection independently from the full training pipeline.

Usage:
    python run_feature_selection.py
    python run_feature_selection.py --fdr-alpha 0.05
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src_core.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, CNN_FOLDS_PATH,
    OUTPUT_DIR, RANDOM_SEED
)
from src_core.preprocessing import load_and_preprocess_data, DataPreprocessor
from src_core.feature_selection import FeatureSelector
from src_core.utils import setup_logging, set_random_seed, Timer, print_section_header


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Feature Selection')
    
    parser.add_argument('--fdr-alpha', type=float, default=0.05, help='Benjamini-Hochberg FDR significance threshold (default: 0.05)')
    parser.add_argument('--corr-threshold', type=float, default=0.70, help='Spearman |rho| ceiling for collinearity filtering (default: 0.70)')
    parser.add_argument('--vif-threshold', type=float, default=5.0, help='Variance inflation factor ceiling (default: 5.0)')
    parser.add_argument('--freq-threshold', type=float, default=0.90, help='Minimum stability-selection frequency to retain a feature (default: 0.90)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Global random seed (default: 42)')
    
    args = parser.parse_args()
    
    
    set_random_seed(args.seed)
    
    
    output_dir = OUTPUT_DIR / 'feature_selection'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'feature_selection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("SpineIDS Clinical ML - Feature Selection")
    logger.info("=" * 60)
    logger.info(f"Parameters: {vars(args)}")
    
    
    print_section_header("Loading Data")
    
    train_df, test_df, fold_mapping, feature_names = load_and_preprocess_data(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        folds_path=CNN_FOLDS_PATH,
        logger=logger
    )
    

    preprocessor = DataPreprocessor(logger)
    preprocessor.feature_names = feature_names
    X, y, _ = preprocessor.get_feature_matrix(train_df, feature_names)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    
    print_section_header("Running Feature Selection")
    
    selector = FeatureSelector(
        fdr_alpha=args.fdr_alpha,
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
        selection_freq_threshold=args.freq_threshold,
        random_state=args.seed,
        logger=logger
    )
    
    with Timer("Feature selection"):
        selected_features = selector.fit(X, y, feature_names)
    
    
    selector.save_results(output_dir)
    
    
    print_section_header("Results")
    logger.info(f"Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        logger.info(f"  {i}. {feat}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Feature selection complete!")


if __name__ == '__main__':
    main()
