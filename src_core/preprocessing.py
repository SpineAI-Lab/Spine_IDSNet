"""
SpineIDS Clinical ML - Data Preprocessing
======================================
Contents:
- Data Loading
- Patient ID normalisation
- Fold assignment alignment with CNN pipeline
- Missing value imputation
- Data quality checks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, CNN_FOLDS_PATH,
    LABEL_COL, GLOBAL_ID_COL, PATIENT_ID_COL,
    EXCLUDE_COLS, EXCLUDE_FEATURES,
    CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES,
    RANDOM_SEED
)
from .utils import (
    extract_real_pid, standardize_real_pid, validate_real_pid_format,
    setup_logging, print_section_header, print_subsection_header
)


class DataLoader:
    """DataLoader"""
    
    def __init__(
        self,
        train_path: Union[str, Path] = TRAIN_DATA_PATH,
        test_path: Union[str, Path] = TEST_DATA_PATH,
        folds_path: Union[str, Path] = CNN_FOLDS_PATH,
        logger=None
    ):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.folds_path = Path(folds_path)
        self.logger = logger or setup_logging()
        

        self.train_df = None
        self.test_df = None
        self.folds_df = None
        self.feature_names = None
        
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training data, external test data, and fold assignment table.
        
        Returns:
            train_df : Internal training cohort (706 patients).
            test_df  : External test cohort (354 patients).
            folds_df : Fold assignment table from the CNN pipeline.
        """
        print_section_header("Data Loading")
        

        self.logger.info(f"Loading train data from {self.train_path}")
        self.train_df = pd.read_csv(self.train_path)
        self.logger.info(f"  Train data shape: {self.train_df.shape}")
        

        self.logger.info(f"Loading test data from {self.test_path}")
        self.test_df = pd.read_csv(self.test_path)
        self.logger.info(f"  Test data shape: {self.test_df.shape}")
        

        self.logger.info(f"Loading fold assignments from {self.folds_path}")
        self.folds_df = pd.read_csv(self.folds_path)
        self.logger.info(f"  Folds data shape: {self.folds_df.shape}")
        
        return self.train_df, self.test_df, self.folds_df
    
    def get_label_distribution(self, df: pd.DataFrame, name: str = "Data"):
        """Log the class label distribution for *df*."""
        if LABEL_COL in df.columns:
            dist = df[LABEL_COL].value_counts().sort_index()
            self.logger.info(f"\n{name} Label Distribution:")
            for label, count in dist.items():
                cls_name = CLASS_NAMES[label] if isinstance(label, int) else label
                self.logger.info(f"  {cls_name}: {count} ({count/len(df)*100:.1f}%)")


class DataPreprocessor:
    """DataPreprocessor"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logging()
        

        self.train_df = None
        self.test_df = None
        self.patient_fold_mapping = None
        self.feature_names = None
        
    def process(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        folds_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """
        Execute the full preprocessing pipeline.
        
        Args:
            train_df : Raw internal cohort DataFrame.
            test_df  : Raw external cohort DataFrame.
            folds_df : Fold assignment table from the CNN pipeline.
        
        Returns:
            train_df            : Processed internal cohort with RealPID and Fold columns.
            test_df             : Processed external cohort with RealPID column.
            patient_fold_mapping: Mapping from RealPID to fold index.
        """
        print_section_header("Data Preprocessing")
        
        # Step 1: ID Standardisation
        print_subsection_header("Step 1: ID Normalisation")
        train_df = self._standardize_ids(train_df, "train")
        test_df = self._standardize_ids(test_df, "test")
        
        # Step 2: Build Fold Mapping
        print_subsection_header("Step 2: Build Fold Mapping")
        self.patient_fold_mapping = self._create_fold_mapping(folds_df)
        

        print_subsection_header("Step 3: Assign Folds")
        train_df = self._assign_folds(train_df)
        
        # Step 4: Label Encoding
        print_subsection_header("Step 4: Label Encoding")
        train_df = self._encode_labels(train_df)
        test_df = self._encode_labels(test_df)
        

        print_subsection_header("Step 5: Identify Feature Columns")
        self.feature_names = self._get_feature_columns(train_df)
        self.logger.info(f"  Number of features: {len(self.feature_names)}")
        
        # Step 6: Data Quality Check
        print_subsection_header("Step 6: Data Quality Check")
        self._check_data_quality(train_df, "Train")
        self._check_data_quality(test_df, "Test")
        
        self.train_df = train_df
        self.test_df = test_df
        
        return train_df, test_df, self.patient_fold_mapping
    
    def _standardize_ids(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Normalise the Global_ID column and derive RealPID."""
        df = df.copy()
        

        if GLOBAL_ID_COL not in df.columns:
            raise ValueError(f"{name} data missing '{GLOBAL_ID_COL}' column")
        

        df['RealPID'] = df[GLOBAL_ID_COL].apply(extract_real_pid)
        

        invalid_pids = df[~df['RealPID'].apply(validate_real_pid_format)]
        if len(invalid_pids) > 0:
            self.logger.warning(f"  Found {len(invalid_pids)} invalid RealPIDs in {name}")
        
        self.logger.info(f"  {name}: Standardized {len(df)} patient IDs")
        
        return df
    
    def _create_fold_mapping(self, folds_df: pd.DataFrame) -> Dict[str, int]:
        """
        Build a {RealPID -> fold_index} mapping from the CNN fold assignment CSV.
        """

        id_col = None
        for col in ['Global_ID', 'global_id', 'RealPID', 'Patient_ID']:
            if col in folds_df.columns:
                id_col = col
                break
        
        if id_col is None:
            raise ValueError("Folds CSV missing ID column")
        

        fold_col = None
        for col in ['Fold', 'fold', 'FOLD']:
            if col in folds_df.columns:
                fold_col = col
                break
        
        if fold_col is None:
            raise ValueError("Folds CSV missing Fold column")
        

        folds_df = folds_df.copy()
        folds_df['RealPID'] = folds_df[id_col].apply(
            lambda x: extract_real_pid(str(x))
        )
        

        unique_mapping = folds_df.drop_duplicates('RealPID')[['RealPID', fold_col]]
        mapping = dict(zip(unique_mapping['RealPID'], unique_mapping[fold_col]))
        
        self.logger.info(f"  Created fold mapping for {len(mapping)} patients")
        

        fold_counts = pd.Series(list(mapping.values())).value_counts().sort_index()
        for fold, count in fold_counts.items():
            self.logger.info(f"    Fold {fold}: {count} patients")
        
        return mapping
    
    def _assign_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign fold indices to the training DataFrame via the patient fold mapping."""
        df = df.copy()
        
        df['Fold'] = df['RealPID'].map(self.patient_fold_mapping)
        

        unmatched = df[df['Fold'].isna()]
        if len(unmatched) > 0:
            self.logger.warning(f"  {len(unmatched)} patients not found in fold mapping!")
            self.logger.warning(f"  Unmatched PIDs: {unmatched['RealPID'].tolist()[:5]}...")
        
        matched = df[df['Fold'].notna()]
        self.logger.info(f"  Successfully matched {len(matched)}/{len(df)} patients")
        

        df['Fold'] = df['Fold'].astype(int)
        
        return df
    
    def _encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode string class labels to integer indices and append as Label_Numeric."""
        df = df.copy()
        
        if LABEL_COL in df.columns:

            if df[LABEL_COL].dtype == object:
                df['Label_Numeric'] = df[LABEL_COL].map(CLASS_TO_IDX)
                self.logger.info(f"  Encoded string labels to numeric")
            else:
                df['Label_Numeric'] = df[LABEL_COL]
            

            unique_labels = df['Label_Numeric'].unique()
            self.logger.info(f"  Unique labels: {sorted(unique_labels)}")
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Derive the list of usable feature column names by excluding metadata columns."""

        all_cols = df.columns.tolist()
        

        exclude = set(EXCLUDE_COLS + EXCLUDE_FEATURES + [
            'RealPID', 'Fold', 'Label_Numeric',
            'Label', 'Global_ID', 'Patient_ID'
        ])
        
        feature_cols = [col for col in all_cols if col not in exclude]
        
        self.logger.info(f"  Total columns: {len(all_cols)}")
        self.logger.info(f"  Excluded columns: {len(exclude)}")
        self.logger.info(f"  Feature columns: {len(feature_cols)}")
        
        return feature_cols
    
    def _check_data_quality(self, df: pd.DataFrame, name: str):
        """Log missing-value statistics and class-label distribution for *df*."""
        self.logger.info(f"\n  {name} Data Quality Check:")
        self.logger.info(f"    Shape: {df.shape}")
        

        if self.feature_names:
            feature_df = df[self.feature_names]
            missing_counts = feature_df.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            
            if len(missing_cols) > 0:
                self.logger.info(f"    Columns with missing values: {len(missing_cols)}")
                for col, count in missing_cols.items():
                    pct = count / len(df) * 100
                    self.logger.info(f"      {col}: {count} ({pct:.1f}%)")
            else:
                self.logger.info(f"    No missing values in feature columns")
        

        if 'Label_Numeric' in df.columns:
            label_dist = df['Label_Numeric'].value_counts().sort_index()
            self.logger.info(f"    Label distribution:")
            for label, count in label_dist.items():
                cls_name = CLASS_NAMES[label]
                pct = count / len(df) * 100
                self.logger.info(f"      {cls_name}: {count} ({pct:.1f}%)")
    
    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the feature matrix, label vector, and patient ID array.
        
        Args:
            df   : DataFrame with a Fold column.
            feature_names: Column names to extract (defaults to self.feature_names).
        
        Returns:
            X: Feature matrix of shape (N, D).
            y: Integer class labels of shape (N,).
            pids: Patient identifier array.
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        X = df[feature_names].values.astype(np.float32)
        y = df['Label_Numeric'].values.astype(np.int64)
        pids = df['RealPID'].values
        
        return X, y, pids
    
    def get_fold_indices(self, df: pd.DataFrame, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return train and validation integer index arrays for *fold*.
        
        Args:
            df   : DataFrame with a Fold column.
            fold : Zero-based fold index.
        
        Returns:
            train_idx : Indices of training samples.
            val_idx   : Indices of validation samples.
        """
        val_mask = df['Fold'] == fold
        train_mask = ~val_mask
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        
        return train_idx, val_idx


def load_and_preprocess_data(
    train_path: Union[str, Path] = TRAIN_DATA_PATH,
    test_path: Union[str, Path] = TEST_DATA_PATH,
    folds_path: Union[str, Path] = CNN_FOLDS_PATH,
    logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], List[str]]:
    """
    One-call convenience function for data loading and preprocessing.
    
    Returns:
        train_df     : Processed internal cohort.
        test_df      : Processed external cohort.
        fold_mapping : {RealPID -> fold_index} mapping.
        feature_names: Ordered list of feature column names.
    """

    loader = DataLoader(train_path, test_path, folds_path, logger)
    train_df, test_df, folds_df = loader.load_all()
    

    preprocessor = DataPreprocessor(logger)
    train_df, test_df, fold_mapping = preprocessor.process(train_df, test_df, folds_df)
    feature_names = preprocessor.feature_names
    
    return train_df, test_df, fold_mapping, feature_names

