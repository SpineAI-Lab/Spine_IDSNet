"""
SpineIDS Clinical ML - Inference and Feature Export
==========================================
Contents:
- Model loading and ensemble inference
- Fusion feature export (schema compatible with CNN pipeline)
- External validation inference
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import (
    CLASS_NAMES, OUTPUT_DIR, 
    get_model_output_dir, get_all_subdirs
)
from .utils import (
    load_pickle, save_numpy, save_json,
    setup_logging, print_section_header, label_to_name,
    validate_real_pid_format
)
from .metrics import compute_all_metrics, bootstrap_metrics


class ModelInference:
    """ModelInference"""
    
    def __init__(
        self,
        model_name: str,
        version: str = "v1",
        logger=None
    ):
        self.model_name = model_name
        self.version = version
        self.logger = logger or setup_logging()
        
        self.output_dir = get_model_output_dir(model_name, version)
        self.subdirs = get_all_subdirs(model_name, version)
        

        self.fold_models = []
        self.fold_imputers = []
        self.fold_scalers = []
        
        self._load_models()
    
    def _load_models(self):
        """Load serialised models, imputers, and scalers for all five folds."""
        checkpoints_dir = self.subdirs['checkpoints']
        
        for fold in range(5):

            model_path = checkpoints_dir / f'{self.model_name}_fold{fold}_best.pkl'
            if model_path.exists():
                self.fold_models.append(load_pickle(model_path))
            else:
                self.logger.warning(f"Model not found: {model_path}")
            

            imputer_path = checkpoints_dir / f'imputer_fold{fold}.pkl'
            if imputer_path.exists():
                self.fold_imputers.append(load_pickle(imputer_path))
            

            scaler_path = checkpoints_dir / f'scaler_fold{fold}.pkl'
            if scaler_path.exists():
                self.fold_scalers.append(load_pickle(scaler_path))
            else:
                self.fold_scalers.append(None)
        
        self.logger.info(f"Loaded {len(self.fold_models)} fold models")
    
    def predict(
        self,
        X: np.ndarray,
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Generate ensemble predictions by averaging over five fold models.
        
        Args:
            X: Feature matrix of shape (N, D).
            return_individual: If True, also return the list of per-fold probability arrays.
        
        Returns:
            ensemble_probs   : Averaged probability array of shape (N, 3).
            individual_probs : Per-fold probability list (only when return_individual=True).
        """
        if len(self.fold_models) == 0:
            raise ValueError("No models loaded")
        
        individual_probs = []
        
        for fold_idx, (model, imputer, scaler) in enumerate(
            zip(self.fold_models, self.fold_imputers, self.fold_scalers)
        ):

            X_processed = imputer.transform(X)
            if scaler is not None:
                X_processed = scaler.transform(X_processed)
            

            proba = model.predict_proba(X_processed)
            individual_probs.append(proba)
        

        ensemble_probs = np.mean(individual_probs, axis=0)
        
        if return_individual:
            return ensemble_probs, individual_probs
        return ensemble_probs
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_ids: np.ndarray,
        dataset_name: str = "test",
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate the ensemble on a labelled dataset.
        
        Args:
            X: Feature matrix.
            y            : Ground-truth labels.
            patient_ids: Patient identifier array.
            dataset_name : Label for the dataset (used in file names, default: 'test').
            save_results : If True, persist predictions and metrics to disk.
        
        Returns:
            Dict with keys: metrics, predictions, probs, preds.
        """
        print_section_header(f"Evaluating on {dataset_name}")
        

        probs = self.predict(X)
        preds = np.argmax(probs, axis=1)
        

        metrics = compute_all_metrics(y, probs, preds)
        
        self.logger.info(f"  AUC_Macro: {metrics['AUC_Macro']:.4f}")
        self.logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
        self.logger.info(f"  F1_Macro: {metrics['F1_Macro']:.4f}")
        

        predictions_df = pd.DataFrame({
            'RealPID': patient_ids,
            'GT': y,
            'Prob_STB': probs[:, 0],
            'Prob_BS': probs[:, 1],
            'Prob_PS': probs[:, 2],
            'Pred': preds,
            'GT_Name': [label_to_name(int(l)) for l in y],
            'Pred_Name': [label_to_name(int(p)) for p in preds]
        })
        
        if save_results:

            pred_path = self.subdirs['predictions'] / f'{dataset_name}_predictions.csv'
            predictions_df.to_csv(pred_path, index=False)
            

            metrics_path = self.subdirs['statistics'] / f'{dataset_name}_metrics.json'
            save_json(metrics, metrics_path)
            
            self.logger.info(f"  Results saved to {self.subdirs['predictions']}")
        
        return {
            'metrics': metrics,
            'predictions': predictions_df,
            'probs': probs,
            'preds': preds
        }


class FusionFeatureExporter:
    """FusionFeatureExporter"""
    
    def __init__(
        self,
        model_name: str,
        version: str = "v1",
        logger=None
    ):
        self.model_name = model_name
        self.version = version
        self.logger = logger or setup_logging()
        
        self.output_dir = get_model_output_dir(model_name, version)
        self.subdirs = get_all_subdirs(model_name, version)
        self.features_dir = self.subdirs['features']
    
    def export_internal_features(
        self,
        oof_predictions_df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Export OOF probability predictions as fusion features for the internal cohort.
        
        Args:
            oof_predictions_df: OOF prediction DataFrame from Trainer.
        
        Returns:
            features: Feature matrix. (N, 3)
            index_df  : Index DataFrame with RealPID and GT columns.
        """
        print_section_header("Exporting Internal Fusion Features")
        

        features = oof_predictions_df[['Prob_STB', 'Prob_BS', 'Prob_PS']].values
        

        index_df = oof_predictions_df[['RealPID', 'GT']].copy()
        

        self._validate_features(features, index_df)
        

        np.save(self.features_dir / 'train_patient_features.npy', features.astype(np.float32))
        index_df.to_csv(self.features_dir / 'train_patient_feature_index.csv', index=False)
        
        self.logger.info(f"  Saved train features: {features.shape}")
        self.logger.info(f"  Path: {self.features_dir}")
        
        return features, index_df
    
    def export_external_features(
        self,
        external_predictions_df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Export ensemble probability predictions as fusion features for the external cohort.
        
        Args:
            external_predictions_df: External prediction DataFrame.
        
        Returns:
            features: Feature matrix. (N, 3)
            index_df  : Index DataFrame with RealPID and GT columns.
        """
        print_section_header("Exporting External Fusion Features")
        

        features = external_predictions_df[['Prob_STB', 'Prob_BS', 'Prob_PS']].values
        

        index_df = external_predictions_df[['RealPID', 'GT']].copy()
        

        self._validate_features(features, index_df)
        

        np.save(self.features_dir / 'external_patient_features.npy', features.astype(np.float32))
        index_df.to_csv(self.features_dir / 'external_patient_feature_index.csv', index=False)
        
        self.logger.info(f"  Saved external features: {features.shape}")
        
        return features, index_df
    
    def _validate_features(self, features: np.ndarray, index_df: pd.DataFrame):
        """Validate the shape, probability range, and RealPID format of fusion features."""

        assert features.shape[1] == 3, f"Expected 3 columns, got {features.shape[1]}"
        assert len(features) == len(index_df), "Feature and index length mismatch"
        

        assert np.all(features >= 0) and np.all(features <= 1), "Probabilities out of range"
        

        row_sums = features.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.01):
            self.logger.warning("Some probability rows don't sum to 1")
        

        invalid_pids = index_df[~index_df['RealPID'].apply(validate_real_pid_format)]
        if len(invalid_pids) > 0:
            self.logger.warning(f"Found {len(invalid_pids)} invalid RealPIDs")
    
    def create_fusion_readme(self):
        """Write a FUSION_README.md describing the exported feature files."""
        readme_content = f"""# Fusion Features - {self.model_name}

## Files

- `train_patient_features.npy`: Internal validation OOF predictions (N_train, 3)
- `train_patient_feature_index.csv`: RealPID and GT for internal samples
- `external_patient_features.npy`: External validation predictions (N_external, 3)
- `external_patient_feature_index.csv`: RealPID and GT for external samples

## Format

### Feature Matrix (npy)
- Shape: (N, 3)
- Column order: [Prob_STB, Prob_BS, Prob_PS]
- dtype: float32

### Index CSV
- Columns: RealPID, GT
- RealPID format: {{STB|BS|PS}}_{{0000}}
- GT: 0=STB, 1=BS, 2=PS

## Usage for Fusion

```python
import numpy as np
import pandas as pd

# Load clinical features
clinical_features = np.load('train_patient_features.npy')
clinical_index = pd.read_csv('train_patient_feature_index.csv')

# Load CNN features (from Part 2)
cnn_features = np.load('path/to/cnn/train_patient_features.npy')
cnn_index = pd.read_csv('path/to/cnn/train_patient_feature_index.csv')

# Merge by RealPID
merged = clinical_index.merge(cnn_index, on='RealPID', suffixes=('_clinical', '_cnn'))

# Align features
clinical_aligned = clinical_features[merged.index]
cnn_aligned = cnn_features[merged.index]

# Late fusion example
alpha = 0.5  # tune this on OOF
fused_probs = alpha * clinical_aligned + (1 - alpha) * cnn_aligned
```

## Notes

- Ensure RealPID format matches exactly with CNN features
- Use OOF predictions for fusion training to avoid data leakage
- External features should use ensemble predictions from 5 fold models
"""
        
        readme_path = self.features_dir / 'FUSION_README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"  Created {readme_path}")


def export_all_fusion_features(
    model_name: str,
    version: str,
    oof_predictions_df: pd.DataFrame,
    external_predictions_df: Optional[pd.DataFrame] = None
):
    """
    Export internal and external fusion features in a single call.
    
    Args:
        model_name             : Registered model name.
        version                : Model version string.
        oof_predictions_df     : OOF prediction DataFrame from training.
        external_predictions_df: External prediction DataFrame (optional).
    """
    exporter = FusionFeatureExporter(model_name, version)
    

    exporter.export_internal_features(oof_predictions_df)
    

    if external_predictions_df is not None:
        exporter.export_external_features(external_predictions_df)
    

    exporter.create_fusion_readme()

