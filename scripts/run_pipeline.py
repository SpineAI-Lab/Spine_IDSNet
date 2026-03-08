#!/usr/bin/env python3
"""
SpineIDS Clinical ML - Full Training Pipeline
===========================================
Steps executed:
    1. Data preprocessing
    2. Feature selection
    3. Model training (single model or all models)
    4. External validation
    5. Fusion feature export

Usage:

    python run_pipeline.py --model LightGBM
    

    python run_pipeline.py --model all
    

    python run_pipeline.py --model LightGBM --no-optuna
    

    python run_pipeline.py --model LightGBM --skip-feature-selection
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src_core.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, CNN_FOLDS_PATH,
    OUTPUT_DIR, MODELS_TO_TRAIN, RANDOM_SEED, VERSION
)
from src_core.preprocessing import load_and_preprocess_data, DataPreprocessor
from src_core.feature_selection import FeatureSelector, run_feature_selection
from src_core.trainer import Trainer, train_single_model
from src_core.inference import ModelInference, FusionFeatureExporter, export_all_fusion_features
from src_core.statistics import StatisticalAnalyzer
from src_core.metrics import compute_all_metrics, bootstrap_metrics
from src_core.utils import (
    setup_logging, set_random_seed, save_json, Timer,
    print_section_header, print_subsection_header
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SpineIDS Clinical ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python run_pipeline.py --model LightGBM
  

  python run_pipeline.py --model all
  

  python run_pipeline.py --model RandomForest --no-optuna
  

  python run_pipeline.py --feature-selection-only
        """
    )
    

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all',
        choices=['all'] + MODELS_TO_TRAIN,
        help='Model to train; use "all" to train every registered model (default: all)'
    )
    
    
    parser.add_argument(
        '--skip-feature-selection',
        action='store_true',
        help='Skip feature selection and load a previously saved feature list'
    )
    
    parser.add_argument(
        '--feature-selection-only',
        action='store_true',
        help='Run feature selection only; skip model training'
    )
    
    parser.add_argument(
        '--features-file',
        type=str,
        default=None,
        help='Path to an existing selected_features.csv (overrides default location)'
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
        help='Number of Optuna trials per outer fold (default: 50)'
    )
    
    parser.add_argument(
        '--version', '-v',
        type=str,
        default=VERSION,
        help=f'Version tag appended to output directory names (default: {VERSION})'
    )
    

    parser.add_argument(
        '--skip-external',
        action='store_true',
        help='Skip external validation after training'
    )
    
    parser.add_argument(
        '--bootstrap-ci',
        action='store_true',
        help='Compute bootstrap 95% CIs for all metrics (slow)'
    )
    

    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Global random seed (default: {RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode; re-raise exceptions instead of catching them'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    

    log_dir = OUTPUT_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'
    logger = setup_logging(log_file)
    
    logger.info("=" * 70)
    logger.info("SpineIDS Clinical ML Pipeline")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Determine which models to train
    if args.model == 'all':
        models_to_run = MODELS_TO_TRAIN
    else:
        models_to_run = [args.model]
    
    logger.info(f"Models to train: {models_to_run}")
    
    # ----------------------------------------------------------------
    # Step 1 – Data Preprocessing
    # ----------------------------------------------------------------
    print_section_header("Step 1: Data Preprocessing")
    
    with Timer("Data preprocessing"):
        train_df, test_df, fold_mapping, all_feature_names = load_and_preprocess_data(
            train_path=TRAIN_DATA_PATH,
            test_path=TEST_DATA_PATH,
            folds_path=CNN_FOLDS_PATH,
            logger=logger
        )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    logger.info(f"Initial features: {len(all_feature_names)}")
    
    # Extract arrays from preprocessed DataFrames
    preprocessor = DataPreprocessor(logger)
    preprocessor.feature_names = all_feature_names
    
    X_train_all, y_train, train_pids = preprocessor.get_feature_matrix(train_df, all_feature_names)
    X_test_all, y_test, test_pids = preprocessor.get_feature_matrix(test_df, all_feature_names)
    fold_assignments = train_df['Fold'].values
    
    # ----------------------------------------------------------------
    # Step 2 – Feature Selection
    # ----------------------------------------------------------------
    print_section_header("Step 2: Feature Selection")
    
    feature_selection_dir = OUTPUT_DIR / 'feature_selection'
    feature_selection_dir.mkdir(parents=True, exist_ok=True)
    
    if args.skip_feature_selection:
        # Load previously saved feature list from disk
        if args.features_file:
            features_file = Path(args.features_file)
        else:
            features_file = feature_selection_dir / 'selected_features.csv'
        
        if features_file.exists():
            features_df = pd.read_csv(features_file)
            selected_features = features_df['feature'].tolist()
            logger.info(f"Loaded {len(selected_features)} features from {features_file}")
        else:
            logger.error(f"Features file not found: {features_file}")
            logger.info("Running feature selection...")
            args.skip_feature_selection = False
    
    if not args.skip_feature_selection:
        with Timer("Feature selection"):
            selected_features = run_feature_selection(
                X_train_all, y_train, all_feature_names,
                output_dir=feature_selection_dir,
                logger=logger
            )
        logger.info(f"Selected {len(selected_features)} features")
    
    # Early exit when --feature-selection-only is set
    if args.feature_selection_only:
        logger.info("Feature selection complete. Exiting (--feature-selection-only)")
        return
    
    # Build the reduced feature matrices
    feature_indices = [all_feature_names.index(f) for f in selected_features]
    X_train = X_train_all[:, feature_indices]
    X_test = X_test_all[:, feature_indices]
    
    logger.info(f"Train shape after selection: {X_train.shape}")
    logger.info(f"Test shape after selection: {X_test.shape}")
    
    # ----------------------------------------------------------------
    # Step 3 – Model Training
    # ----------------------------------------------------------------
    print_section_header("Step 3: Model Training")
    
    all_results = {}
    all_trainers = {}
    
    for model_name in models_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            with Timer(f"Training {model_name}"):
                trainer = Trainer(
                    model_name=model_name,
                    version=args.version,
                    use_optuna=not args.no_optuna,
                    n_trials=args.n_trials,
                    random_state=args.seed,
                    logger=logger
                )
                
                results = trainer.train(
                    X=X_train,
                    y=y_train,
                    patient_ids=train_pids,
                    fold_assignments=fold_assignments,
                    feature_names=selected_features
                )
            
            all_results[model_name] = results
            all_trainers[model_name] = trainer
            
            logger.info(f"{model_name} CV AUC_Macro: {results['cv_metrics']['AUC_Macro']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            if args.debug:
                raise
    
    # ----------------------------------------------------------------
    # Step 4 – External Validation
    # ----------------------------------------------------------------
    if not args.skip_external:
        print_section_header("Step 4: External Validation")
        
        for model_name, trainer in all_trainers.items():
            logger.info(f"\nEvaluating {model_name} on external data...")
            
            try:
                ext_predictions = trainer.predict_external(
                    X=X_test,
                    patient_ids=test_pids,
                    y_true=y_test
                )
                
                # Compute external validation metrics
                ext_metrics = compute_all_metrics(
                    y_test,
                    ext_predictions[['Prob_STB', 'Prob_BS', 'Prob_PS']].values
                )
                
                logger.info(f"  External AUC_Macro: {ext_metrics['AUC_Macro']:.4f}")
                logger.info(f"  External Accuracy: {ext_metrics['Accuracy']:.4f}")
                
                # Persist external validation metrics
                ext_metrics_path = trainer.subdirs['statistics'] / 'external_metrics.json'
                save_json(ext_metrics, ext_metrics_path)
                
                all_results[model_name]['external_metrics'] = ext_metrics
                all_results[model_name]['external_predictions'] = ext_predictions
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                if args.debug:
                    raise
    
    # ----------------------------------------------------------------
    # Step 5 – Fusion Feature Export
    # ----------------------------------------------------------------
    print_section_header("Step 5: Export Fusion Features")
    
    for model_name, results in all_results.items():
        logger.info(f"\nExporting fusion features for {model_name}...")
        
        try:
            export_all_fusion_features(
                model_name=model_name,
                version=args.version,
                oof_predictions_df=results['oof_predictions'],
                external_predictions_df=results.get('external_predictions')
            )
        except Exception as e:
            logger.error(f"Error exporting features for {model_name}: {e}")
    
    # ----------------------------------------------------------------
    # Step 6 – Bootstrap Confidence Intervals (optional)
    # ----------------------------------------------------------------
    if args.bootstrap_ci:
        print_section_header("Step 6: Bootstrap Confidence Intervals")
        
        analyzer = StatisticalAnalyzer(n_bootstrap=1000, logger=logger)
        
        for model_name, results in all_results.items():
            logger.info(f"\nComputing CI for {model_name}...")
            
            oof_df = results['oof_predictions']
            y_true = oof_df['GT'].values
            y_prob = oof_df[['Prob_STB', 'Prob_BS', 'Prob_PS']].values
            
            metrics_with_ci = analyzer.analyze_single_model(
                y_true, y_prob,
                output_dir=all_trainers[model_name].subdirs['statistics']
            )
    
    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print_section_header("Summary")
    
    # Build summary table
    summary_data = []
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'CV_AUC_Macro': results['cv_metrics']['AUC_Macro'],
            'CV_Accuracy': results['cv_metrics']['Accuracy'],
            'CV_F1_Macro': results['cv_metrics']['F1_Macro'],
        }
        if 'external_metrics' in results:
            row['Ext_AUC_Macro'] = results['external_metrics']['AUC_Macro']
            row['Ext_Accuracy'] = results['external_metrics']['Accuracy']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('CV_AUC_Macro', ascending=False)
    
    
    summary_path = OUTPUT_DIR / 'model_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info(f"\nSummary saved to: {summary_path}")
    
    # Report the best-performing model
    best_model = summary_df.iloc[0]['Model']
    best_auc = summary_df.iloc[0]['CV_AUC_Macro']
    logger.info(f"\nBest model: {best_model} (AUC_Macro = {best_auc:.4f})")
    
    logger.info(f"\nPipeline completed at: {datetime.now()}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
