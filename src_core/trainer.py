"""
SpineIDS Clinical ML - Training Logic
====================================
Core training procedure:
    1. Nested cross-validation
    2. Optuna hyperparameter optimisation
    3. Out-of-fold (OOF) prediction generation
    4. Per-fold model serialisation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .config import (
    N_FOLDS, INNER_CV_FOLDS, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT,
    RANDOM_SEED, CLASS_NAMES, IDX_TO_CLASS,
    get_model_output_dir, get_all_subdirs
)
from .models import ModelFactory, sample_params_from_space
from .metrics import compute_all_metrics
from .utils import (
    setup_logging, save_pickle, load_pickle, save_json, save_numpy,
    print_section_header, print_subsection_header, Timer,
    label_to_name
)


class Trainer:
    """Trainer"""
    
    def __init__(
        self,
        model_name: str,
        version: str = "v1",
        use_optuna: bool = True,
        n_trials: int = OPTUNA_N_TRIALS,
        timeout: int = OPTUNA_TIMEOUT,
        random_state: int = RANDOM_SEED,
        logger=None
    ):
        self.model_name = model_name
        self.version = version
        self.use_optuna = use_optuna and HAS_OPTUNA
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.logger = logger or setup_logging()
        

        self.output_dir = get_model_output_dir(model_name, version)
        self.subdirs = get_all_subdirs(model_name, version)
        

        self.fold_models = []
        self.fold_imputers = []
        self.fold_scalers = []
        self.fold_params = []
        self.fold_metrics = []
        self.oof_predictions = None
        

        self.needs_scaling = ModelFactory.needs_scaling(model_name)
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_ids: np.ndarray,
        fold_assignments: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run the full nested cross-validation training procedure.
        
        Args:
            X: Feature matrix of shape (N, D).
            y: Integer class labels of shape (N,).
            patient_ids: Patient identifier array of shape (N,).
            fold_assignments: Pre-assigned fold indices of shape (N,), values in {0, …, 4}.
            feature_names: List of feature column names.
        
        Returns:
            results: Results dictionary.
        """
        print_section_header(f"Training {self.model_name}")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Use Optuna: {self.use_optuna}")
        self.logger.info(f"  Needs scaling: {self.needs_scaling}")
        
        n_samples, n_features = X.shape
        unique_folds = np.sort(np.unique(fold_assignments))
        n_folds = len(unique_folds)
        
        self.logger.info(f"  Samples: {n_samples}, Features: {n_features}, Folds: {n_folds}")
        

        oof_probs = np.zeros((n_samples, len(CLASS_NAMES)))
        oof_preds = np.full(n_samples, -1)
        oof_folds = fold_assignments.copy()
        

        for fold in unique_folds:
            print_subsection_header(f"Fold {fold}")
            

            val_mask = fold_assignments == fold
            train_mask = ~val_mask
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            pids_val = patient_ids[val_mask]
            
            self.logger.info(f"    Train: {len(X_train)}, Val: {len(X_val)}")
            

            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)
            self.fold_imputers.append(imputer)
            

            if self.needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                self.fold_scalers.append(scaler)
            else:
                self.fold_scalers.append(None)
            

            if self.use_optuna:
                best_params = self._optuna_tuning(
                    X_train, y_train, fold
                )
            else:
                best_params = ModelFactory.get_default_params(self.model_name)
            
            self.fold_params.append(best_params)
            

            model = ModelFactory.create_model(
                self.model_name,
                params=best_params,
                use_default=False
            )
            
            with Timer(f"Training fold {fold}"):
                model.fit(X_train, y_train)
            
            self.fold_models.append(model)
            

            proba = model.predict_proba(X_val)
            pred = np.argmax(proba, axis=1)
            

            oof_probs[val_mask] = proba
            oof_preds[val_mask] = pred
            

            fold_metrics = compute_all_metrics(y_val, proba, pred)
            self.fold_metrics.append(fold_metrics)
            
            self.logger.info(f"    Fold {fold} AUC_Macro: {fold_metrics['AUC_Macro']:.4f}")
            self.logger.info(f"    Fold {fold} Accuracy: {fold_metrics['Accuracy']:.4f}")
            

            self._save_fold_model(fold)
        

        self.oof_predictions = self._create_oof_dataframe(
            patient_ids, y, oof_probs, oof_preds, fold_assignments
        )
        

        cv_metrics = compute_all_metrics(y, oof_probs, oof_preds)
        
        print_subsection_header("Cross-Validation Results")
        self.logger.info(f"  Overall AUC_Macro: {cv_metrics['AUC_Macro']:.4f}")
        self.logger.info(f"  Overall Accuracy: {cv_metrics['Accuracy']:.4f}")
        self.logger.info(f"  Overall F1_Macro: {cv_metrics['F1_Macro']:.4f}")
        

        self._save_all_results(feature_names, cv_metrics)
        
        return {
            'cv_metrics': cv_metrics,
            'fold_metrics': self.fold_metrics,
            'oof_predictions': self.oof_predictions
        }
    
    def _optuna_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fold: int
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter optimisation for one outer fold."""
        self.logger.info(f"    Running Optuna optimization ({self.n_trials} trials)...")
        
        param_space = ModelFactory.get_param_space(self.model_name)
        
        def objective(trial):

            params = sample_params_from_space(param_space, trial)
            

            inner_cv = StratifiedKFold(
                n_splits=INNER_CV_FOLDS,
                shuffle=True,
                random_state=self.random_state
            )
            
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_inner_train = X_train[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                y_inner_val = y_train[inner_val_idx]
                

                try:
                    model = ModelFactory.create_model(
                        self.model_name,
                        params=params,
                        use_default=False
                    )
                    model.fit(X_inner_train, y_inner_train)
                    

                    proba = model.predict_proba(X_inner_val)
                    

                    metrics = compute_all_metrics(y_inner_val, proba)
                    inner_scores.append(metrics['AUC_Macro'])
                    
                except Exception as e:
                    return 0.0
            
            return np.mean(inner_scores)
        

        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False
            )
        
        best_params = study.best_params
        

        for param_name, spec in param_space.items():
            if spec[0] == 'fixed':
                best_params[param_name] = spec[1]
        
        self.logger.info(f"    Best AUC: {study.best_value:.4f}")
        

        history_path = self.subdirs['hyperparameters'] / f'optuna_history_fold{fold}.csv'
        history_df = study.trials_dataframe()
        history_df.to_csv(history_path, index=False)
        

        params_path = self.subdirs['hyperparameters'] / f'best_params_fold{fold}.json'
        save_json(best_params, params_path)
        
        return best_params
    
    def _create_oof_dataframe(
        self,
        patient_ids: np.ndarray,
        y_true: np.ndarray,
        oof_probs: np.ndarray,
        oof_preds: np.ndarray,
        folds: np.ndarray
    ) -> pd.DataFrame:
        """Assemble an out-of-fold prediction DataFrame in the canonical
        schema shared with the CNN pipeline."""
        df = pd.DataFrame({
            'RealPID': patient_ids,
            'GT': y_true,
            'Prob_STB': oof_probs[:, 0],
            'Prob_BS': oof_probs[:, 1],
            'Prob_PS': oof_probs[:, 2],
            'Pred': oof_preds,
            'GT_Name': [label_to_name(int(l)) for l in y_true],
            'Pred_Name': [label_to_name(int(p)) for p in oof_preds],
            'Fold': folds
        })
        
        return df
    
    def _save_fold_model(self, fold: int):
        """Serialise the model, imputer, and (optional) scaler for one fold."""

        model_path = self.subdirs['checkpoints'] / f'{self.model_name}_fold{fold}_best.pkl'
        save_pickle(self.fold_models[-1], model_path)
        

        imputer_path = self.subdirs['checkpoints'] / f'imputer_fold{fold}.pkl'
        save_pickle(self.fold_imputers[-1], imputer_path)
        

        if self.fold_scalers[-1] is not None:
            scaler_path = self.subdirs['checkpoints'] / f'scaler_fold{fold}.pkl'
            save_pickle(self.fold_scalers[-1], scaler_path)
    
    def _save_all_results(
        self,
        feature_names: List[str],
        cv_metrics: Dict[str, float]
    ):
        """Persist all training artefacts to the configured output directories."""

        oof_path = self.subdirs['predictions'] / 'cv_all_predictions.csv'
        self.oof_predictions.to_csv(oof_path, index=False)
        self.logger.info(f"  OOF predictions saved to {oof_path}")
        

        fold_metrics_df = pd.DataFrame(self.fold_metrics)
        fold_metrics_df['Fold'] = range(len(self.fold_metrics))
        fold_metrics_path = self.subdirs['statistics'] / 'cv_fold_metrics.csv'
        fold_metrics_df.to_csv(fold_metrics_path, index=False)
        

        cv_metrics_path = self.subdirs['statistics'] / 'cv_metrics.json'
        save_json(cv_metrics, cv_metrics_path)
        

        features_path = self.subdirs['logs'] / 'feature_names.json'
        save_json({'features': feature_names}, features_path)
        

        config_snapshot = {
            'model_name': self.model_name,
            'version': self.version,
            'use_optuna': self.use_optuna,
            'n_trials': self.n_trials,
            'needs_scaling': self.needs_scaling,
            'n_folds': N_FOLDS,
            'random_state': self.random_state,
            'timestamp': datetime.now().isoformat()
        }
        config_path = self.subdirs['logs'] / 'config_snapshot.json'
        save_json(config_snapshot, config_path)
        
        self.logger.info(f"  All results saved to {self.output_dir}")
    
    def predict_external(
        self,
        X: np.ndarray,
        patient_ids: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions on an external dataset.
        
        Args:
            X: Feature matrix.
            patient_ids: Patient identifier array.
            y_true: Ground-truth labels (optional).
        
        Returns:
            predictions_df: Prediction results DataFrame.
        """
        if len(self.fold_models) == 0:
            raise ValueError("No trained models. Call train() first.")
        
        n_samples = len(X)
        all_probs = []
        

        for fold_idx, (model, imputer, scaler) in enumerate(
            zip(self.fold_models, self.fold_imputers, self.fold_scalers)
        ):

            X_processed = imputer.transform(X)
            if scaler is not None:
                X_processed = scaler.transform(X_processed)
            

            proba = model.predict_proba(X_processed)
            all_probs.append(proba)
        

        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        

        df = pd.DataFrame({
            'RealPID': patient_ids,
            'GT': y_true if y_true is not None else [-1] * n_samples,
            'Prob_STB': ensemble_probs[:, 0],
            'Prob_BS': ensemble_probs[:, 1],
            'Prob_PS': ensemble_probs[:, 2],
            'Pred': ensemble_preds,
            'GT_Name': [label_to_name(int(l)) if y_true is not None else 'Unknown' for l in (y_true if y_true is not None else [-1] * n_samples)],
            'Pred_Name': [label_to_name(int(p)) for p in ensemble_preds]
        })
        

        ext_path = self.subdirs['predictions'] / 'external_predictions.csv'
        df.to_csv(ext_path, index=False)
        self.logger.info(f"  External predictions saved to {ext_path}")
        
        return df


def train_single_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    fold_assignments: np.ndarray,
    feature_names: List[str],
    version: str = "v1",
    use_optuna: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience wrapper for single-model training.
    
    Returns:
        results: Training results dict and fitted Trainer instance.
    """
    trainer = Trainer(
        model_name=model_name,
        version=version,
        use_optuna=use_optuna,
        **kwargs
    )
    
    results = trainer.train(X, y, patient_ids, fold_assignments, feature_names)
    
    return results, trainer

