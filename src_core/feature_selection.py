"""
SpineIDS Clinical ML - Feature Selection
====================================
Three-stage feature selection pipeline:
    1. Statistical filtering  : Kruskal-Wallis + Benjamini-Hochberg FDR
    2. Collinearity reduction : Spearman correlation + VIF
    3. Stability selection    : ElasticNet + LightGBM (repeated CV)

v2.0 additions:
    - Persist all intermediate artefacts for downstream visualisation
    - LASSO regularisation path computation
    - VIF iteration history
    - Per-iteration stability selection records
    - Feature-flow counts for Sankey diagram

v2.1 additions:
    - LASSO cross-validation (CV error curve)
    - Output: lasso_cv.csv and lasso_cv_optimal.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
import json

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .config import (
    FDR_ALPHA, CORRELATION_THRESHOLD, VIF_THRESHOLD,
    STABILITY_N_SPLITS, STABILITY_N_REPEATS, 
    SELECTION_FREQUENCY_THRESHOLD, IMPORTANCE_BOOTSTRAP_N,
    RANDOM_SEED, NUM_CLASSES, OUTPUT_DIR
)
from .utils import (
    setup_logging, print_section_header, print_subsection_header,
    save_json, Timer
)


class FeatureSelector:
    """FeatureSelector"""
    
    def __init__(
        self,
        fdr_alpha: float = FDR_ALPHA,
        corr_threshold: float = CORRELATION_THRESHOLD,
        vif_threshold: float = VIF_THRESHOLD,
        selection_freq_threshold: float = SELECTION_FREQUENCY_THRESHOLD,
        n_splits: int = STABILITY_N_SPLITS,
        n_repeats: int = STABILITY_N_REPEATS,
        random_state: int = RANDOM_SEED,
        logger=None
    ):
        self.fdr_alpha = fdr_alpha
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.selection_freq_threshold = selection_freq_threshold
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.logger = logger or setup_logging()
        

        self.step1_results = None
        self.step2_results = None
        self.step3_results = None
        self.selected_features = None
        

        self.correlation_matrix = None
        self.vif_history = []
        self.lasso_path_data = None
        self.stability_iterations = []
        self.flow_counts = {}
        

        self.lasso_cv_data = None
        self.optimal_C = None
        self.optimal_alpha = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """
        Run the full three-stage feature selection pipeline.
        
        Args:
            X: Feature matrix of shape (N, D).
            y: Integer class labels of shape (N,).
            feature_names: List of feature column names.
        
        Returns:
            selected_features: Ordered list of selected feature names.
        """
        print_section_header("Feature Selection")
        

        df = pd.DataFrame(X, columns=feature_names)
        

        self.flow_counts['initial'] = len(feature_names)
        

        print_subsection_header("Step 1: Statistical Filtering (Kruskal-Wallis + FDR)")
        features_step1 = self._step1_statistical_filter(df, y)
        self.flow_counts['after_statistical'] = len(features_step1)
        self.logger.info(f"  Features after Step 1: {len(features_step1)}")
        

        print_subsection_header("Step 2: Collinearity Reduction (Spearman + VIF)")
        features_step2 = self._step2_collinearity_filter(df[features_step1], y)
        self.flow_counts['after_collinearity'] = len(features_step2)
        self.logger.info(f"  Features after Step 2: {len(features_step2)}")
        

        print_subsection_header("Step 3: Embedded Importance (Stability Selection)")
        features_step3 = self._step3_importance_filter(df[features_step2], y, features_step2)
        self.flow_counts['after_importance'] = len(features_step3)
        self.logger.info(f"  Features after Step 3: {len(features_step3)}")
        
        self.selected_features = features_step3
        self.flow_counts['final'] = len(features_step3)
        

        print_subsection_header("Step 4: Compute LASSO Regularisation Path")
        self._calculate_lasso_path(df[features_step2], y, features_step2)
        

        print_subsection_header("Step 5: Compute Final Feature Correlation Matrix")
        self._calculate_final_correlation(df[features_step3])
        

        print_subsection_header("Final Feature Selection Results")
        self.logger.info(f"  Selected {len(self.selected_features)} features:")
        for i, feat in enumerate(self.selected_features, 1):
            self.logger.info(f"    {i}. {feat}")
        

        self.logger.info(f"\n  Feature flow: {self.flow_counts['initial']} → "
                        f"{self.flow_counts['after_statistical']} → "
                        f"{self.flow_counts['after_collinearity']} → "
                        f"{self.flow_counts['final']}")
        
        return self.selected_features
    
    def _step1_statistical_filter(
        self,
        df: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        """
        Step 1: Statistical filtering (Kruskal-Wallis + BH-FDR).
        
        
        """
        results = []
        
        for col in df.columns:

            groups = [df[col][y == i].dropna().values for i in range(NUM_CLASSES)]
            

            if any(len(g) < 3 for g in groups):
                continue
            

            try:
                h_stat, p_value = stats.kruskal(*groups)
                results.append({
                    'feature': col,
                    'H_statistic': h_stat,
                    'p_value': p_value
                })
            except Exception as e:
                self.logger.warning(f"    KW test failed for {col}: {e}")
        

        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            self.logger.warning("  No features passed KW test!")
            return df.columns.tolist()
        

        _, fdr_pvalues, _, _ = multipletests(
            results_df['p_value'], 
            alpha=self.fdr_alpha, 
            method='fdr_bh'
        )
        results_df['FDR_q'] = fdr_pvalues
        

        passed = results_df[results_df['FDR_q'] < self.fdr_alpha]
        
        self.logger.info(f"    Total features tested: {len(results_df)}")
        self.logger.info(f"    Features with FDR < {self.fdr_alpha}: {len(passed)}")
        

        passed = passed.sort_values('H_statistic', ascending=False)
        

        self.step1_results = results_df.sort_values('FDR_q')
        
        return passed['feature'].tolist()
    
    def _step2_collinearity_filter(
        self,
        df: pd.DataFrame,
        y: np.ndarray
    ) -> List[str]:
        """
        Step 2: Collinearity reduction.
        
        
        """
        features = df.columns.tolist()
        

        self.logger.info(f"    2a. Spearman correlation filtering (threshold={self.corr_threshold})")
        

        corr_matrix = df[features].corr(method='spearman')
        

        self.correlation_matrix_step2 = corr_matrix.copy()
        

        features_to_remove = set()
        high_corr_pairs = []
        n_pairs_removed = 0
        
        for i in range(len(features)):
            if features[i] in features_to_remove:
                continue
            for j in range(i + 1, len(features)):
                if features[j] in features_to_remove:
                    continue
                
                corr = abs(corr_matrix.iloc[i, j])
                if corr > self.corr_threshold:


                    if self.step1_results is not None:
                        h_i = self.step1_results[
                            self.step1_results['feature'] == features[i]
                        ]['H_statistic'].values
                        h_j = self.step1_results[
                            self.step1_results['feature'] == features[j]
                        ]['H_statistic'].values
                        
                        h_i = h_i[0] if len(h_i) > 0 else 0
                        h_j = h_j[0] if len(h_j) > 0 else 0
                        
                        if h_i >= h_j:
                            removed = features[j]
                            kept = features[i]
                            features_to_remove.add(features[j])
                        else:
                            removed = features[i]
                            kept = features[j]
                            features_to_remove.add(features[i])
                    else:

                        removed = features[j]
                        kept = features[i]
                        features_to_remove.add(features[j])
                    

                    high_corr_pairs.append({
                        'feature_1': features[i],
                        'feature_2': features[j],
                        'correlation': corr,
                        'removed': removed,
                        'kept': kept
                    })
                    n_pairs_removed += 1
        
        features = [f for f in features if f not in features_to_remove]
        self.logger.info(f"      Removed {len(features_to_remove)} highly correlated features")
        

        self.logger.info(f"    2b. VIF filtering (threshold={self.vif_threshold})")
        

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        

        self.vif_history = []
        

        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations and len(features) > 1:
            try:

                vif_data = []
                for i in range(X_scaled.shape[1]):
                    vif = variance_inflation_factor(X_scaled, i)
                    vif_data.append({
                        'feature': features[i],
                        'VIF': vif
                    })
                
                vif_df = pd.DataFrame(vif_data)
                max_vif = vif_df['VIF'].max()
                

                vif_record = {
                    'iteration': iteration,
                    'n_features': len(features),
                    'max_vif': max_vif,
                    'vif_values': vif_df.to_dict('records')
                }
                self.vif_history.append(vif_record)
                
                if max_vif > self.vif_threshold:

                    worst_feature = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
                    self.logger.info(f"      Iteration {iteration}: Removing {worst_feature} (VIF={max_vif:.2f})")
                    features.remove(worst_feature)
                    

                    X_scaled = scaler.fit_transform(df[features])
                    
                    iteration += 1
                else:
                    self.logger.info(f"      All VIF <= {self.vif_threshold}, stopping")
                    break
                    
            except Exception as e:
                self.logger.warning(f"      VIF calculation error: {e}")
                break
        
        self.logger.info(f"      VIF iterations: {iteration}")
        self.logger.info(f"      Features remaining: {len(features)}")
        

        self.step2_results = {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'removed_by_correlation': list(features_to_remove),
            'vif_iterations': iteration,
            'features_after_correlation': [f for f in df.columns if f not in features_to_remove],
            'features_after_vif': features
        }
        
        return features
    
    def _step3_importance_filter(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """
        Step 3: Stability selection importance filtering.
        - Stability Selection with ElasticNet + LightGBM
                Retains features whose selection frequency meets the configured threshold.
        """
        X = df.values
        n_features = len(feature_names)
        

        enet_importance = np.zeros(n_features)
        lgbm_importance = np.zeros(n_features)
        selection_count = np.zeros(n_features)
        

        self.stability_iterations = []
        

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        
        n_iterations = self.n_splits * self.n_repeats
        self.logger.info(f"    Running {n_iterations} iterations...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            

            iteration_record = {
                'iteration': fold_idx,
                'enet_selected': [],
                'lgbm_selected': [],
                'enet_coef': {},
                'lgbm_importance': {}
            }
            
            # ElasticNet Logistic Regression
            try:
                enet = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=0.95,
                    C=0.05,
                    max_iter=5000,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                enet.fit(X_train_scaled, y_train)
                

                coef_abs = np.abs(enet.coef_).mean(axis=0)
                enet_importance += coef_abs
                

                selected = coef_abs > 1e-6
                selection_count += selected.astype(int)
                

                for i, fname in enumerate(feature_names):
                    if selected[i]:
                        iteration_record['enet_selected'].append(fname)
                    iteration_record['enet_coef'][fname] = float(coef_abs[i])
                
            except Exception as e:
                self.logger.warning(f"      ElasticNet failed at fold {fold_idx}: {e}")
            
            # LightGBM
            if HAS_LIGHTGBM:
                try:
                    lgbm_model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        class_weight='balanced',
                        random_state=self.random_state,
                        verbose=-1,
                        n_jobs=-1
                    )
                    lgbm_model.fit(X_train, y_train)
                    
                    lgbm_importance += lgbm_model.feature_importances_
                    

                    selected = lgbm_model.feature_importances_ > 0
                    selection_count += selected.astype(int)
                    

                    for i, fname in enumerate(feature_names):
                        if selected[i]:
                            iteration_record['lgbm_selected'].append(fname)
                        iteration_record['lgbm_importance'][fname] = float(lgbm_model.feature_importances_[i])
                    
                except Exception as e:
                    self.logger.warning(f"      LightGBM failed at fold {fold_idx}: {e}")
            

            self.stability_iterations.append(iteration_record)
        

        enet_importance /= n_iterations
        lgbm_importance /= n_iterations
        

        max_selections = n_iterations * 2
        selection_freq = selection_count / max_selections
        

        enet_rank = stats.rankdata(enet_importance)
        lgbm_rank = stats.rankdata(lgbm_importance)
        combined_rank = 0.5 * enet_rank + 0.5 * lgbm_rank
        

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'enet_importance': enet_importance,
            'lgbm_importance': lgbm_importance,
            'enet_rank': enet_rank,
            'lgbm_rank': lgbm_rank,
            'combined_rank': combined_rank,
            'selection_freq': selection_freq
        }).sort_values('combined_rank', ascending=False)
        

        selected = importance_df[
            importance_df['selection_freq'] >= self.selection_freq_threshold
        ]
        
        self.logger.info(f"    Features with selection freq >= {self.selection_freq_threshold}: {len(selected)}")
        

        if len(selected) < 5:
            self.logger.warning(f"    Too few features! Relaxing threshold...")

            n_select = min(10, len(importance_df))
            selected = importance_df.head(n_select)
            self.logger.info(f"    Selected top {n_select} features by importance")
        

        self.step3_results = importance_df
        
        return selected['feature'].tolist()
    
    def _calculate_lasso_path(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ):
        """
        Compute the LASSO regularisation path across a log-spaced C grid.
        v2.1 additions: also performs 5-fold LASSO cross-validation.
        
        """
        X = df.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        


        C_values = np.logspace(-4, 2, 50)
        
        lasso_path_records = []
        
        self.logger.info(f"    Computing LASSO path with {len(C_values)} C values...")
        
        for C in C_values:
            try:

                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=C,
                    max_iter=5000,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_scaled, y)
                

                coef_abs = np.abs(model.coef_).mean(axis=0)
                
                record = {
                    'C': C,
                    'alpha': 1.0 / C,
                    'n_nonzero': np.sum(coef_abs > 1e-6)
                }
                

                for i, fname in enumerate(feature_names):
                    record[fname] = float(coef_abs[i])
                
                lasso_path_records.append(record)
                
            except Exception as e:
                self.logger.warning(f"      LASSO path failed for C={C}: {e}")
                continue
        
        self.lasso_path_data = pd.DataFrame(lasso_path_records)
        self.logger.info(f"    LASSO path computed: {len(lasso_path_records)} points")
        
        # =====================================================================

        # =====================================================================
        self.logger.info(f"    Computing LASSO CV (5-fold)...")
        try:

            lasso_cv = LogisticRegressionCV(
                Cs=C_values,
                penalty='l1',
                solver='saga',
                cv=5,
                scoring='neg_log_loss',
                max_iter=5000,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                refit=True
            )
            lasso_cv.fit(X_scaled, y)
            

            # scores_: dict {class_label: (n_folds, n_Cs)}
            cv_scores = lasso_cv.scores_
            

            lasso_cv_records = []
            
            for i, C in enumerate(lasso_cv.Cs_):

                all_scores = []
                for class_label, scores_matrix in cv_scores.items():
                    # scores_matrix: (n_folds, n_Cs)
                    all_scores.extend(scores_matrix[:, i].tolist())
                
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                
                lasso_cv_records.append({
                    'C': float(C),
                    'alpha': float(1.0 / C),
                    'mean_score': float(mean_score),
                    'std_score': float(std_score),
                    'mean_error': float(-mean_score),
                    'std_error': float(std_score)
                })
            
            self.lasso_cv_data = pd.DataFrame(lasso_cv_records)
            


            if hasattr(lasso_cv.C_, '__len__'):
                self.optimal_C = float(lasso_cv.C_[0])
            else:
                self.optimal_C = float(lasso_cv.C_)
            self.optimal_alpha = float(1.0 / self.optimal_C)
            
            self.logger.info(f"    LASSO CV completed:")
            self.logger.info(f"      - CV points: {len(lasso_cv_records)}")
            self.logger.info(f"      - Optimal C: {self.optimal_C:.6f}")
            self.logger.info(f"      - Optimal alpha (1/C): {self.optimal_alpha:.4f}")
            
        except Exception as e:
            self.logger.warning(f"    LASSO CV calculation failed: {e}")
            self.lasso_cv_data = None
            self.optimal_C = None
            self.optimal_alpha = None
    
    def _calculate_final_correlation(self, df: pd.DataFrame):
        """
        Compute the Spearman correlation matrix for the final selected features.
        
        """
        self.correlation_matrix = df.corr(method='spearman')
        self.logger.info(f"    Final correlation matrix: {self.correlation_matrix.shape}")
    
    def save_results(self, output_dir: Union[str, Path]):
        """Persist all feature selection artefacts to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        

        

        if self.step1_results is not None:
            self.step1_results.to_csv(
                output_dir / 'step1_statistical_tests.csv', 
                index=False
            )
            self.logger.info(f"  ✓ step1_statistical_tests.csv")
        

        if self.step3_results is not None:
            self.step3_results.to_csv(
                output_dir / 'step3_feature_importance.csv',
                index=False
            )
            self.logger.info(f"  ✓ step3_feature_importance.csv")
        

        if self.selected_features is not None:
            pd.DataFrame({
                'feature': self.selected_features,
                'rank': range(1, len(self.selected_features) + 1)
            }).to_csv(output_dir / 'selected_features.csv', index=False)
            self.logger.info(f"  ✓ selected_features.csv")
        

        

        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(output_dir / 'correlation_matrix.csv')
            self.logger.info(f"  ✓ correlation_matrix.csv")
        

        if self.vif_history:

            vif_records = []
            for record in self.vif_history:
                iteration = record['iteration']
                for vif_item in record['vif_values']:
                    vif_records.append({
                        'iteration': iteration,
                        'feature': vif_item['feature'],
                        'VIF': vif_item['VIF']
                    })
            pd.DataFrame(vif_records).to_csv(
                output_dir / 'vif_history.csv', 
                index=False
            )
            self.logger.info(f"  ✓ vif_history.csv")
            

            with open(output_dir / 'vif_history.json', 'w') as f:
                json.dump(self.vif_history, f, indent=2)
            self.logger.info(f"  ✓ vif_history.json")
        

        if self.lasso_path_data is not None:
            self.lasso_path_data.to_csv(
                output_dir / 'lasso_path.csv',
                index=False
            )
            self.logger.info(f"  ✓ lasso_path.csv")
        

        

        if self.lasso_cv_data is not None:
            self.lasso_cv_data.to_csv(
                output_dir / 'lasso_cv.csv',
                index=False
            )
            self.logger.info(f"  ✓ lasso_cv.csv")
            

            if self.optimal_C is not None:
                cv_optimal_info = {
                    'optimal_C': self.optimal_C,
                    'optimal_alpha': self.optimal_alpha,
                    'description': 'Optimal regularization parameters from 5-fold CV'
                }
                with open(output_dir / 'lasso_cv_optimal.json', 'w') as f:
                    json.dump(cv_optimal_info, f, indent=2)
                self.logger.info(f"  ✓ lasso_cv_optimal.json")
        

        

        if self.stability_iterations:

            stability_records = []
            all_features = set()
            for record in self.stability_iterations:
                all_features.update(record.get('enet_selected', []))
                all_features.update(record.get('lgbm_selected', []))
            
            for record in self.stability_iterations:
                iteration = record['iteration']
                for fname in all_features:
                    stability_records.append({
                        'iteration': iteration,
                        'feature': fname,
                        'enet_selected': 1 if fname in record.get('enet_selected', []) else 0,
                        'lgbm_selected': 1 if fname in record.get('lgbm_selected', []) else 0,
                        'enet_coef': record.get('enet_coef', {}).get(fname, 0),
                        'lgbm_importance': record.get('lgbm_importance', {}).get(fname, 0)
                    })
            
            pd.DataFrame(stability_records).to_csv(
                output_dir / 'stability_iterations.csv',
                index=False
            )
            self.logger.info(f"  ✓ stability_iterations.csv")
        

        if self.flow_counts:
            with open(output_dir / 'flow_counts.json', 'w') as f:
                json.dump(self.flow_counts, f, indent=2)
            self.logger.info(f"  ✓ flow_counts.json")
        

        if self.step2_results is not None:

            if 'high_corr_pairs' in self.step2_results and self.step2_results['high_corr_pairs']:
                pd.DataFrame(self.step2_results['high_corr_pairs']).to_csv(
                    output_dir / 'high_correlation_pairs.csv',
                    index=False
                )
                self.logger.info(f"  ✓ high_correlation_pairs.csv")
        
        self.logger.info(f"\n  All results saved to {output_dir}")
        

        self.logger.info("\n  === Saved Files ===")
        self.logger.info("  Files required for visualisation:")
        self.logger.info("    Statistical tests (Step 1)    : step1_statistical_tests.csv")
        self.logger.info("    Feature importance (Step 3)   : step3_feature_importance.csv")
        self.logger.info("    Stability iterations          : stability_iterations.csv")
        self.logger.info("    Selected features             : selected_features.csv")
        self.logger.info("    Correlation matrix            : correlation_matrix.csv")
        self.logger.info("    VIF history                   : vif_history.csv")
        self.logger.info("    LASSO regularisation path     : lasso_path.csv")
        self.logger.info("    LASSO CV error curve (v2.1)   : lasso_cv.csv")
        self.logger.info("    Feature-flow counts (Sankey)  : flow_counts.json")


def run_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
    **kwargs
) -> List[str]:
    """
    Convenience wrapper that instantiates FeatureSelector and runs fit().
    
    Args:
        X: Feature matrix.
        y: Integer class label array.
        feature_names: List of feature column names.
        output_dir: Output directory for persisting results (optional).
        **kwargs: Forwarded to FeatureSelector.__init__.
    
    Returns:
        selected_features: Ordered list of selected feature names.
    """
    selector = FeatureSelector(**kwargs)
    selected = selector.fit(X, y, feature_names)
    
    if output_dir is not None:
        selector.save_results(output_dir)
    
    return selected

