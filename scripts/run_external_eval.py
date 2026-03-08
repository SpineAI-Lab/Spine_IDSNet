#!/usr/bin/env python3
"""
SpineIDS Clinical ML - Standalone External Validation Script
=======================================
Load trained fold models and evaluate them on the held-out external cohort.

Usage:
    python run_external_eval.py --model LightGBM
    python run_external_eval.py --model all
    python run_external_eval.py --model LightGBM --bootstrap-ci
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
    OUTPUT_DIR, MODELS_TO_TRAIN, VERSION
)
from src_core.preprocessing import load_and_preprocess_data, DataPreprocessor
from src_core.inference import ModelInference, FusionFeatureExporter
from src_core.statistics import StatisticalAnalyzer
from src_core.metrics import compute_all_metrics, bootstrap_metrics, format_metrics_report
from src_core.utils import (
    setup_logging, save_json, print_section_header, print_subsection_header
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='External Validation')
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all',
        choices=['all'] + MODELS_TO_TRAIN,
        help='Model to evaluate; use "all" for all registered models'
    )
    
    parser.add_argument(
        '--version', '-v',
        type=str,
        default=VERSION,
        help='Version tag matching the training run (default: v1)'
    )
    
    parser.add_argument(
        '--bootstrap-ci',
        action='store_true',
        help='Compute bootstrap 95% CIs (slow)'
    )
    
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap resamples (default: 1000)'
    )
    
    args = parser.parse_args()
    

    log_file = OUTPUT_DIR / 'logs' / f'external_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("SpineIDS Clinical ML - External Validation")
    logger.info("=" * 60)
    
    
    if args.model == 'all':
        models_to_eval = MODELS_TO_TRAIN
    else:
        models_to_eval = [args.model]
    
    
    print_section_header("Loading Data")
    
    train_df, test_df, fold_mapping, all_features = load_and_preprocess_data(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        folds_path=CNN_FOLDS_PATH,
        logger=logger
    )
    
    
    features_file = OUTPUT_DIR / 'feature_selection' / 'selected_features.csv'
    if features_file.exists():
        selected_features = pd.read_csv(features_file)['feature'].tolist()
        logger.info(f"Loaded {len(selected_features)} selected features")
    else:
        logger.error(f"Features file not found: {features_file}")
        logger.info("Please run feature selection first.")
        return
    

    preprocessor = DataPreprocessor(logger)
    preprocessor.feature_names = all_features
    
    feature_indices = [all_features.index(f) for f in selected_features]
    X_test_all, y_test, test_pids = preprocessor.get_feature_matrix(test_df, all_features)
    X_test = X_test_all[:, feature_indices]
    
    logger.info(f"External test samples: {len(X_test)}")
    
    
    print_section_header("External Validation")
    
    all_results = {}
    all_probs = {}
    
    for model_name in models_to_eval:
        print_subsection_header(f"Evaluating {model_name}")
        
        try:
            
            inference = ModelInference(model_name, args.version, logger)
            results = inference.evaluate(
                X=X_test,
                y=y_test,
                patient_ids=test_pids,
                dataset_name='external',
                save_results=True
            )
            
            all_results[model_name] = results['metrics']
            all_probs[model_name] = results['probs']
            
            logger.info(f"  AUC_Macro: {results['metrics']['AUC_Macro']:.4f}")
            logger.info(f"  Accuracy: {results['metrics']['Accuracy']:.4f}")
            
            
            if args.bootstrap_ci:
                logger.info(f"  Computing Bootstrap CI...")
                analyzer = StatisticalAnalyzer(n_bootstrap=args.n_bootstrap, logger=logger)
                
                output_dir = inference.subdirs['statistics']
                metrics_ci = analyzer.analyze_single_model(
                    y_test, results['probs'], output_dir
                )
            
            
            exporter = FusionFeatureExporter(model_name, args.version, logger)
            exporter.export_external_features(results['predictions'])
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    
    if len(all_results) > 1:
        print_section_header("Model Comparison on External Data")
        
        summary_df = pd.DataFrame(all_results).T
        summary_df = summary_df[['AUC_Macro', 'Accuracy', 'F1_Macro', 'Kappa', 'MCC']]
        summary_df = summary_df.sort_values('AUC_Macro', ascending=False)
        
        logger.info(f"\n{summary_df.round(4).to_string()}")
        
        
        summary_path = OUTPUT_DIR / 'external_validation_summary.csv'
        summary_df.to_csv(summary_path)
        logger.info(f"\nSummary saved to: {summary_path}")
        
        
        if len(all_probs) > 1:
            logger.info("\nRunning statistical comparisons...")
            analyzer = StatisticalAnalyzer(logger=logger)
            comparison = analyzer.compare_models(
                y_test, all_probs,
                output_dir=OUTPUT_DIR / 'comparison'
            )
    
    logger.info("\nExternal validation complete!")


if __name__ == '__main__':
    main()
